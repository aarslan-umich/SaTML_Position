"""Step 3b — analyze with stronger baselines (no network; reuses work/ parquets).

Compares four rankings of the held-out half B:
  generation : the model's own generation order (the original baseline)
  dup_count  : how many times the model emitted the string (desc) — a stronger,
               model-internal frequency baseline
  zxcvbn     : weakest-first by zxcvbn score alone — uses NO HIBP information;
               tests whether the aggregate bin table adds anything beyond a
               public strength prior
  bin_table  : the aggregate-only re-ranking learned from half A (the method)

Ties in every ranking are broken by generation order.
Writes results/E1_results2.json.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pyarrow.parquet as pq

import config
from analyze import _bin_keys, _rank_metrics

RANKINGS = ["generation", "dup_count", "zxcvbn", "bin_table"]
BASELINES = ["generation", "dup_count", "zxcvbn"]
METRICS = ("unique_coverage", "weighted_coverage", "tail_coverage")


def _orders(fib, dup, zsc, bin_rate_vec):
    """Return the four orderings (indices, best first). Tie-break by gen order."""
    return {
        "generation": np.argsort(fib, kind="stable"),
        "dup_count": np.lexsort((fib, -dup)),
        "zxcvbn": np.lexsort((fib, zsc)),          # weakest strength first
        "bin_table": np.lexsort((fib, -bin_rate_vec)),
    }


def _split_analysis(length, zscore, charclass, first_index, dup_count, count, half,
                    budgets, reps, rng):
    A = half == "A"
    B = half == "B"
    keys = _bin_keys(length, zscore, charclass)
    found = count > 0

    # bin table from adaptation half A only
    ka = keys[A]
    fa = found[A].astype(np.float64)
    uniq, inv = np.unique(ka, return_inverse=True)
    rate = np.bincount(inv, weights=fa, minlength=len(uniq)) / np.maximum(
        np.bincount(inv, minlength=len(uniq)), 1)
    bin_rate = dict(zip(uniq.tolist(), rate.tolist()))
    global_rate = float(fa.mean()) if fa.size else 0.0

    # held-out half B
    kb = keys[B]
    fb = found[B]
    cb = count[B]
    zb = zscore[B]
    fib = first_index[B]
    db = dup_count[B]
    rb = np.array([bin_rate.get(k, global_rate) for k in kb], dtype=np.float64)

    orders = _orders(fib, db, zb, rb)
    res = {"n_A": int(A.sum()), "n_B": int(B.sum()), "found_in_B": int(fb.sum())}
    for name, order in orders.items():
        res[name] = _rank_metrics(order, fb, cb, zb, config.TAIL_MAX_COUNT,
                                  budgets, reps, rng)

    # deltas of the method against each baseline
    res["delta_vs"] = {}
    for base in BASELINES:
        res["delta_vs"][base] = {}
        for frac in budgets:
            k = f"{frac:.2f}"
            res["delta_vs"][base][k] = {
                m: res["bin_table"][k][m] - res[base][k][m] for m in METRICS}
    return res


def analyze_model(model: str) -> dict:
    t0 = time.time()
    prep = pq.read_table(os.path.join(config.WORK_DIR, f"{model}_prepared.parquet"))
    cnt = pq.read_table(os.path.join(config.WORK_DIR, f"{model}_counts.parquet"))
    length = prep.column("length").to_numpy()
    zscore = prep.column("zscore").to_numpy().astype(np.int64)
    charclass = prep.column("charclass").to_numpy()
    first_index = prep.column("first_index").to_numpy()
    dup_count = prep.column("dup_count").to_numpy()
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
            length, zscore, charclass, first_index, dup_count, count, half,
            config.BUDGET_FRACTIONS, config.BOOTSTRAP_REPS, rng)
        print(f"[{model}] split d{d} done ({time.time()-t0:.0f}s)", flush=True)

    # summary: method mean vs each baseline mean, delta with spread across splits
    out["summary"] = {}
    for frac in config.BUDGET_FRACTIONS:
        k = f"{frac:.2f}"
        out["summary"][k] = {}
        for m in METRICS:
            method_vals = [out["splits"][f"d{d}"]["bin_table"][k][m]
                           for d in config.SPLIT_DIGITS]
            entry = {"method_mean": float(np.mean(method_vals))}
            for base in RANKINGS:
                base_vals = [out["splits"][f"d{d}"][base][k][m]
                             for d in config.SPLIT_DIGITS]
                entry[f"{base}_mean"] = float(np.mean(base_vals))
            for base in BASELINES:
                dl = [out["splits"][f"d{d}"]["delta_vs"][base][k][m]
                      for d in config.SPLIT_DIGITS]
                bm = np.mean([out["splits"][f"d{d}"][base][k][m]
                              for d in config.SPLIT_DIGITS])
                entry[f"delta_vs_{base}_mean"] = float(np.mean(dl))
                entry[f"delta_vs_{base}_min"] = int(np.min(dl))
                entry[f"delta_vs_{base}_max"] = int(np.max(dl))
                entry[f"gain_vs_{base}_pct"] = (100.0 * np.mean(dl) / bm) if bm else 0.0
            out["summary"][k][m] = entry
    print(f"[{model}] done ({time.time()-t0:.0f}s)", flush=True)
    return out


def run():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    results = {
        "experiment": "E1 constraint-first adaptation — stronger baselines",
        "rankings": RANKINGS,
        "note": "zxcvbn ranking uses NO HIBP signal; it isolates the value of the "
                "aggregate bin table beyond a public strength prior.",
        "config": {
            "split_digits": config.SPLIT_DIGITS,
            "budget_fractions": config.BUDGET_FRACTIONS,
            "tail_max_count": config.TAIL_MAX_COUNT,
            "bins": "length_bucket x zxcvbn",
            "bootstrap_reps": config.BOOTSTRAP_REPS,
        },
        "models": {},
    }
    for model in config.MODELS:
        results["models"][model] = analyze_model(model)
    out = os.path.join(config.RESULTS_DIR, "E1_results2.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[analyze2] wrote {out}", flush=True)


if __name__ == "__main__":
    run()
