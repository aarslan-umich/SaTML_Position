"""Step 1 — prepare a model's candidate pool.

For one model list we produce a single parquet with one row per *unique* candidate:
    password, dup_count, first_index, length, charclass, zscore, prefix, suffix
plus the split-half label for every configured split digit (half_d0, half_d1, ...).

The expensive part is zxcvbn scoring; it is spread across N_PROC processes.
Nothing here touches the network.
"""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

import config
import common


def _score_chunk(passwords: list[str]) -> list[tuple[int, int, int]]:
    """Worker: return (length, charclass_mask, zxcvbn_score) per password."""
    from zxcvbn import zxcvbn  # imported inside the worker
    out = []
    for p in passwords:
        try:
            z = zxcvbn(p)["score"] if p else 0
        except Exception:
            z = 0
        out.append((len(p), common.charclass_mask(p), int(z)))
    return out


def _read_unique(path: str, sample: int | None):
    """Read the list, preserving first-seen order and per-string duplicate counts."""
    order: dict[str, int] = {}
    dup: dict[str, int] = {}
    idx = 0
    with open(path, "rb") as fh:
        for raw in fh:
            line = raw.rstrip(b"\n")
            if not line:
                continue
            try:
                p = line.decode("utf-8")
            except UnicodeDecodeError:
                p = line.decode("utf-8", "replace")
            if p in dup:
                dup[p] += 1
            else:
                dup[p] = 1
                order[p] = idx
                idx += 1
            if sample and idx >= sample:
                break
    passwords = list(order.keys())
    first_index = [order[p] for p in passwords]
    dup_count = [dup[p] for p in passwords]
    return passwords, first_index, dup_count


def prepare_model(model: str, sample: int | None = None) -> str:
    path = config.MODELS[model]
    os.makedirs(config.WORK_DIR, exist_ok=True)
    t0 = time.time()
    print(f"[{model}] reading {path} ...", flush=True)
    passwords, first_index, dup_count = _read_unique(path, sample)
    n = len(passwords)
    print(f"[{model}] {n:,} unique candidates (read in {time.time()-t0:.1f}s)", flush=True)

    # zxcvbn scoring across processes
    t1 = time.time()
    chunks = [passwords[i:i + config.CHUNK_SIZE] for i in range(0, n, config.CHUNK_SIZE)]
    lengths: list[int] = []
    masks: list[int] = []
    zscores: list[int] = []
    with Pool(processes=config.N_PROC) as pool:
        done = 0
        for res in pool.imap(_score_chunk, chunks, chunksize=1):
            for ln, mk, zs in res:
                lengths.append(ln)
                masks.append(mk)
                zscores.append(zs)
            done += 1
            if done % 50 == 0 or done == len(chunks):
                pct = 100 * done / len(chunks)
                print(f"[{model}] zxcvbn {pct:5.1f}%  ({time.time()-t1:.0f}s)", flush=True)

    # hashing + split labels (cheap, single thread)
    t2 = time.time()
    prefixes: list[str] = []
    suffixes: list[str] = []
    halves = {d: [] for d in config.SPLIT_DIGITS}
    for p in passwords:
        h = common.sha1_hex_upper(p)
        prefixes.append(h[:5])
        suffixes.append(h[5:])
        for d in config.SPLIT_DIGITS:
            halves[d].append(common.half_for_split(h, d))
    print(f"[{model}] hashing done ({time.time()-t2:.0f}s)", flush=True)

    cols = {
        "password": passwords,
        "dup_count": pa.array(dup_count, pa.int32()),
        "first_index": pa.array(first_index, pa.int32()),
        "length": pa.array(lengths, pa.int16()),
        "charclass": pa.array(masks, pa.int8()),
        "zscore": pa.array(zscores, pa.int8()),
        "prefix": prefixes,
        "suffix": suffixes,
    }
    for d in config.SPLIT_DIGITS:
        cols[f"half_d{d}"] = halves[d]
    table = pa.table(cols)

    out = os.path.join(config.WORK_DIR, f"{model}_prepared.parquet")
    pq.write_table(table, out, compression="zstd")
    print(f"[{model}] wrote {out}  ({n:,} rows, total {time.time()-t0:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(config.MODELS), required=True)
    ap.add_argument("--sample", type=int, default=None,
                    help="only the first N unique candidates (smoke test)")
    a = ap.parse_args()
    prepare_model(a.model, a.sample)
