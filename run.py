"""E1 orchestrator: prepare -> hibp -> analyze.

Examples
--------
Smoke test end-to-end on 20k candidates per model (hits the real API a little):
    python run.py --sample 20000

Full run on a many-core machine:
    N_PROC=30 HIBP_CONCURRENCY=200 python run.py

Run a single stage:
    python run.py --only prepare
    python run.py --only hibp
    python run.py --only analyze
"""
from __future__ import annotations

import argparse
import time

import config
import prepare
import hibp
import analyze


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["prepare", "hibp", "analyze"], default=None,
                    help="run just one stage (default: all three in order)")
    ap.add_argument("--sample", type=int, default=None,
                    help="first N unique candidates per model (smoke test)")
    a = ap.parse_args()
    t0 = time.time()

    if a.only in (None, "prepare"):
        for model in config.MODELS:
            prepare.prepare_model(model, a.sample)
    if a.only in (None, "hibp"):
        hibp.run()
    if a.only in (None, "analyze"):
        analyze.run()

    print(f"[run] finished in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
