"""E1: Constraint-First Adaptation Experiment — configuration.

Single source of truth for paths and experiment parameters. Edit here, not in
the step scripts.
"""
from __future__ import annotations

import os

# --- paths -----------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))   # the dir holding run.py

def _find_data_dir() -> str:
    """Prefer data/ next to run.py (server layout); fall back to ../data."""
    override = os.environ.get("DATA_DIR")
    if override:
        return override
    here_data = os.path.join(HERE, "data")
    if os.path.isdir(here_data):
        return here_data
    return os.path.join(os.path.dirname(HERE), "data")

DATA_DIR = _find_data_dir()
WORK_DIR = os.path.join(HERE, "work")
RESULTS_DIR = os.path.join(HERE, "results")

# model name -> input password list (one candidate per line, UTF-8)
MODELS = {
    "passgan": os.path.join(DATA_DIR, "10m_passgan.txt"),
    "passgpt": os.path.join(DATA_DIR, "10m_passgpt.txt"),
}

# --- split -----------------------------------------------------------------
# A leakage-free split: a candidate's SHA-1 hash is a random function of the
# string, so partitioning on a hex digit of the hash is independent of password
# structure and no string lands in both halves.
# For each split index i we look at the i-th hex digit of the full SHA-1:
#   digit in 0..7  -> adaptation half ("A")   (we may read aggregate stats here)
#   digit in 8..F  -> evaluation half  ("B")  (held out; only final scoring)
# Repeating over several digit positions gives a variance estimate.
SPLIT_DIGITS = [0, 1, 2]  # which hex-digit positions to use as independent splits
PRIMARY_SPLIT = 0          # the split reported as the headline number

# --- binning ---------------------------------------------------------------
# The ONLY thing carried from the adaptation half to the evaluation half is this
# aggregate table. Bins = length bucket x zxcvbn strength score.
ZSCORE_LEVELS = [0, 1, 2, 3, 4]        # zxcvbn 0..4
LEN_MIN_BUCKET = 4                      # lengths <= 4 collapse into one bucket
LEN_MAX_BUCKET = 16                     # lengths >= 17 collapse into one bucket
USE_CHARCLASS_MASK = False             # optional finer bins (lower/upper/digit/sym)

# --- evaluation ------------------------------------------------------------
# Budgets = fraction of the evaluation pool we are allowed to "guess".
BUDGET_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
TAIL_MAX_COUNT = 10   # a HIBP occurrence count <= this defines a "tail" (rare) password
BOOTSTRAP_REPS = 500
BOOTSTRAP_SEED = 20260927

# --- HIBP k-anonymity range API -------------------------------------------
# Only the first 5 hex chars of a candidate's SHA-1 ever leave this machine; the
# API returns every suffix under that prefix and we match locally in memory.
HIBP_URL = "https://api.pwnedpasswords.com/range/{prefix}"
# Concurrency: on a fast server the range API tolerates a few hundred in-flight
# requests. Override with env HIBP_CONCURRENCY.
HIBP_CONCURRENCY = int(os.environ.get("HIBP_CONCURRENCY", "200"))
HIBP_ADD_PADDING = True       # ask the API to pad responses (privacy best-practice)
HIBP_MAX_RETRIES = 5
HIBP_TIMEOUT_S = 30
# Identify the client, as the API requires. Replace the placeholder below with your
# own contact before running (the value used in the original run is withheld).
HIBP_USER_AGENT = "E1-research-coverage-study (contact: REPLACE-WITH-YOUR-CONTACT)"

# --- runtime ---------------------------------------------------------------
# Worker processes for the CPU-bound zxcvbn scoring step. Defaults to all cores;
# override with env N_PROC (e.g. 30 on the 64-core box to leave headroom).
N_PROC = int(os.environ.get("N_PROC", str(max(1, (os.cpu_count() or 4)))))
# Rows handed to a worker at a time (bigger = less IPC overhead on many cores).
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "20000"))
