"""Step 3 — build the aggregate bin table, re-rank the held-out half, score it.

For each model and each split digit:
  * Adaptation half A -> per-bin HIBP membership rate (the ONLY thing carried over).
  * Evaluation half B ranked two ways:
      baseline  : the model's own generation order (first_index ascending)
      reranked  : by the learned bin membership rate (ties broken by baseline)
  * Metrics at each budget: unique / count-weighted / tail coverage + strength.
  * 95% CIs by bootstrap (partial budgets) and spread across independent splits.

All numbers land in results/E1_results.json.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pyarrow.parquet as pq

import config
import common


def _bin_keys(length, zscore, charclass):
    lb = np.clip(length, config.LEN_MIN_BUCKET, config.LEN_MAX_BUCKET)
    if config.USE_CHARCLASS_MASK:
        return np.char.add(np.char.add(np.char.add(lb.astype(str), "|"),
                           np.char.add(zscore.astype(str), "|")), charclass.astype(str))
    return np.char.add(np.char.add(lb.astype(str), "|"), zscore.astype(str))


def _bootstrap_ci(found_topn: np.ndarray, reps: int, rng: np.random.Generator):
    """95% CI of the coverage COUNT within a top-N set, by resampling members.

    Real resampling when the set is small enough; otherwise the analytic
    bootstrap-equivalent for a sum of i.i.d. 0/1 draws (count = N*p,
    sd = sqrt(N*p*(1-p)))."""
    n = len(found_topn)
    if n == 0:
        return [0.0, 0.0], "empty"
    if n <= 500_000:
        sums = np.empty(reps)
        for r in range(reps):
            sums[r] = found_topn[rng.integers(0, n, n)].sum()
        return [float(np.percentile(sums, 2.5)),
                float(np.percentile(sums, 97.5))], "resample"
    p = float(found_topn.mean())
    sd = (n * p * (1 - p)) ** 0.5
    c = n * p
    return [max(0.0, c - 1.96 * sd), c + 1.96 * sd], "analytic"


def _rank_metrics(order, found, count, zscore, tail, budgets, reps, rng):
    """Given a ranking (indices, best first) compute per-budget metrics."""
    f = found[order]
    c = count[order]
    z = zscore[order]
    tailmask = ((c > 0) & (c <= tail))
    cum_found = np.cumsum(f)
    cum_weight = np.cumsum(c)
    cum_tail = np.cumsum(tailmask)
    n = len(order)
    per = {}
    for frac in budgets:
        N = max(1, int(round(frac * n)))
        idx = N - 1
        prof = np.bincount(z[:N][f[:N] > 0], minlength=len(config.ZSCORE_LEVELS)).tolist()
        ci, method = _bootstrap_ci(f[:N], reps, rng)
        per[f"{frac:.2f}"] = {
            "budget_n": int(N),
            "unique_coverage": int(cum_found[idx]),
            "weighted_coverage": int(cum_weight[idx]),
            "tail_coverage": int(cum_tail[idx]),
            "unique_coverage_ci95": ci,
            "ci_method": method,
            "strength_profile_found": prof,
        }
    return per


def _split_analysis(length, zscore, charclass, first_index, count, half, budgets, reps, rng):
    A = half == "A"
    B = half == "B"
    keys = _bin_keys(length, zscore, charclass)
    found = count > 0

    # --- bin table from A only ---
    ka = keys[A]
    fa = found[A].astype(np.float64)
    uniq, inv = np.unique(ka, return_inverse=True)
    sums = np.bincount(inv, weights=fa, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq))
    rate = sums / np.maximum(cnts, 1)
    bin_rate = dict(zip(uniq.tolist(), rate.tolist()))
    global_rate = float(fa.mean()) if fa.size else 0.0

    # --- rank B ---
    kb = keys[B]
    fb = found[B]
    cb = count[B]
    zb = zscore[B]
    fib = first_index[B]
    rb = np.array([bin_rate.get(k, global_rate) for k in kb], dtype=np.float64)

    baseline_order = np.argsort(fib, kind="stable")               # model's own order
    # rerank: high bin rate first; within a bin keep the model's order
    rerank_order = np.lexsort((fib, -rb))

    res = {
        "n_A": int(A.sum()),
        "n_B": int(B.sum()),
        "found_in_B": int(fb.sum()),
        "baseline": _rank_metrics(baseline_order, fb, cb, zb, config.TAIL_MAX_COUNT,
                                  budgets, reps, rng),
        "reranked": _rank_metrics(rerank_order, fb, cb, zb, config.TAIL_MAX_COUNT,
                                  budgets, reps, rng),
    }
    # deltas (reranked - baseline)
    res["delta"] = {}
    for frac in budgets:
        k = f"{frac:.2f}"
        res["delta"][k] = {
            "unique_coverage": res["reranked"][k]["unique_coverage"]
            - res["baseline"][k]["unique_coverage"],
            "weighted_coverage": res["reranked"][k]["weighted_coverage"]
            - res["baseline"][k]["weighted_coverage"],
            "tail_coverage": res["reranked"][k]["tail_coverage"]
            - res["baseline"][k]["tail_coverage"],
        }
    # small bin table for reporting (primary split only, sorted by rate)
    res["_bin_rate"] = {k: round(v, 6) for k, v in
                        sorted(bin_rate.items(), key=lambda x: -x[1])}
    return res


def analyze_model(model: str) -> dict:
    t0 = time.time()
    prep = pq.read_table(os.path.join(config.WORK_DIR, f"{model}_prepared.parquet"))
    cnt = pq.read_table(os.path.join(config.WORK_DIR, f"{model}_counts.parquet"))
    length = prep.column("length").to_numpy()
    zscore = prep.column("zscore").to_numpy().astype(np.int64)
    charclass = prep.column("charclass").to_numpy()
    first_index = prep.column("first_index").to_numpy()
    count = cnt.column("count").to_numpy()

    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    out = {
        "n_unique": int(len(length)),
        "total_found": int((count > 0).sum()),
        "found_rate": float((count > 0).mean()),
        "splits": {},
    }
    for d in config.SPLIT_DIGITS:
        half = np.array(prep.column(f"half_d{d}").to_pylist())
        out["splits"][f"d{d}"] = _split_analysis(
            length, zscore, charclass, first_index, count, half,
            config.BUDGET_FRACTIONS, config.BOOTSTRAP_REPS, rng)
        print(f"[{model}] split d{d} done ({time.time()-t0:.0f}s)", flush=True)

    # --- summary across splits: mean delta and spread ---
    out["summary"] = {}
    for frac in config.BUDGET_FRACTIONS:
        k = f"{frac:.2f}"
        for metric in ("unique_coverage", "weighted_coverage", "tail_coverage"):
            base = [out["splits"][f"d{d}"]["baseline"][k][metric] for d in config.SPLIT_DIGITS]
            rer = [out["splits"][f"d{d}"]["reranked"][k][metric] for d in config.SPLIT_DIGITS]
            dl = [out["splits"][f"d{d}"]["delta"][k][metric] for d in config.SPLIT_DIGITS]
            out["summary"].setdefault(k, {})[metric] = {
                "baseline_mean": float(np.mean(base)),
                "reranked_mean": float(np.mean(rer)),
                "delta_mean": float(np.mean(dl)),
                "delta_min": int(np.min(dl)),
                "delta_max": int(np.max(dl)),
                "relative_gain_pct": (100.0 * np.mean(dl) / np.mean(base))
                if np.mean(base) else 0.0,
            }
    # keep the primary split's bin table, drop the bulky ones from other splits
    for d in config.SPLIT_DIGITS:
        if d != config.PRIMARY_SPLIT:
            out["splits"][f"d{d}"].pop("_bin_rate", None)
    print(f"[{model}] analysis complete ({time.time()-t0:.0f}s)", flush=True)
    return out


def run():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    results = {
        "experiment": "E1 constraint-first adaptation (aggregate-only re-ranking)",
        "config": {
            "split_digits": config.SPLIT_DIGITS,
            "primary_split": config.PRIMARY_SPLIT,
            "budget_fractions": config.BUDGET_FRACTIONS,
            "tail_max_count": config.TAIL_MAX_COUNT,
            "bins": "length_bucket x zxcvbn" + ("x charclass" if config.USE_CHARCLASS_MASK else ""),
            "len_buckets": [config.LEN_MIN_BUCKET, config.LEN_MAX_BUCKET],
            "bootstrap_reps": config.BOOTSTRAP_REPS,
        },
        "models": {},
    }
    for model in config.MODELS:
        results["models"][model] = analyze_model(model)
    out = os.path.join(config.RESULTS_DIR, "E1_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[analyze] wrote {out}", flush=True)


if __name__ == "__main__":
    run()
