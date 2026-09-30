# E1: Aggregate-Only Adaptation Experiment

Code and aggregate results for experiment E1. The question: can a generator's
candidate list be re-ordered using only bin-level aggregate statistics obtained
through the k-anonymity range API of a breach-corpus service, and how does that
compare with baselines that use no such feedback?

The only signal read from the service is the range API: a 5-character SHA-1
prefix is sent, a list of suffixes with counts comes back, and matching is done
locally in memory. Raw range responses are never written to disk. What is
retained per candidate is one integer occurrence count.

## Files

| file | role |
|------|------|
| `run.py` | orchestrates `prepare` -> `hibp` -> `analyze` (or one stage with `--only`) |
| `config.py` | paths and all experiment parameters (single source of truth) |
| `common.py` | SHA-1 helper, split-half assignment, length buckets, char-class mask |
| `prepare.py` | per model: deduplicate, SHA-1, split labels, length, char-class, zxcvbn score |
| `hibp.py` | one range request per distinct 5-character prefix; resumable; stores counts only |
| `analyze.py` | builds the bin table from the adaptation half, re-ranks the evaluation half, scores it |
| `analyze2.py` | same metrics for four rankings, including two baselines that use no service feedback |
| `results/E1_results.json` | output of `analyze.py` |
| `results/E1_results2.json` | output of `analyze2.py` |

`analyze2.py` is not called by `run.py`. Run it separately with `python analyze2.py`
after the `prepare` and `hibp` stages have produced the intermediate files.

## Method in brief

**Split.** For each split index `d` in `SPLIT_DIGITS` (0, 1, 2) the `d`-th hex digit
of the candidate's SHA-1 decides the half: digit 0-7 is the adaptation half "A",
digit 8-F is the evaluation half "B". No string appears in both halves of a split.

**Bin table.** Bins are (length bucket) x (zxcvbn score 0-4). Lengths are clipped
to the range `LEN_MIN_BUCKET`..`LEN_MAX_BUCKET` (4..16). For each bin, the
fraction of half-A candidates found in the service is computed. This table is the
only information carried from A to B. Bins unseen in A get the global rate of A.

**Rankings of half B** (ties are always broken by generation order):

- `generation`: the model's own generation order.
- `dup_count`: how often the model emitted the string, descending.
- `zxcvbn`: weakest zxcvbn score first; uses no service feedback.
- `bin_table`: descending bin rate learned from half A (the method).

**Metrics** at budgets `BUDGET_FRACTIONS` (fractions of the half-B pool):

- `unique_coverage`: number of distinct top-N candidates found in the service.
- `weighted_coverage`: sum of occurrence counts of the top-N candidates.
- `tail_coverage`: number of top-N candidates with occurrence count between 1 and
  `TAIL_MAX_COUNT` (10).
- `strength_profile_found`: zxcvbn-score histogram of the found candidates.
- `unique_coverage_ci95`: 95% interval for `unique_coverage`, with `ci_method`
  `resample` (bootstrap over members, `BOOTSTRAP_REPS` = 500, seed in `config.py`)
  when the set has at most 500,000 members, otherwise `analytic` (normal
  approximation of a sum of independent 0/1 draws). It reflects resampling of the
  top-N members of a fixed pool, not independent training runs.

At budget 1.00 all rankings see the whole pool, so differences are zero by
construction.

## Inputs (not included)

The candidate pools are withheld because they consist of plaintext strings that
could be reused as guessing material. `config.py` expects one candidate per line,
UTF-8, in:

- `data/10m_passgan.txt`
- `data/10m_passgpt.txt`

Intermediate files in `work/` (prepared tables that contain the candidate strings,
per-candidate counts, and checkpoints) are also not included.

## Running

```bash
pip install pyarrow numpy aiohttp zxcvbn

# full run
N_PROC=30 HIBP_CONCURRENCY=200 python run.py

# or stage by stage (only `hibp` uses the network)
N_PROC=30 python run.py --only prepare
HIBP_CONCURRENCY=200 python run.py --only hibp     # resumable; safe to re-run
python run.py --only analyze
python analyze2.py

# small smoke test on the first 20000 unique candidates per model
python run.py --sample 20000
```

Developed under Python 3.12. Set `HIBP_USER_AGENT` in `config.py` to your own
contact before running: the service requires an identifying User-Agent. The value
used in the original run is withheld and replaced by a placeholder.

Environment variables: `N_PROC` (worker processes for zxcvbn), `HIBP_CONCURRENCY`
(simultaneous requests), `CHUNK_SIZE` (rows per zxcvbn task), `DATA_DIR`
(location of the candidate lists).

## Behavior of the acquisition step

- Each distinct prefix not yet marked done is requested. A prefix gets up to
  `HIBP_MAX_RETRIES` (5) attempts in total: network errors and 5xx responses are
  retried with backoff, and `Retry-After` is honored on 429.
- A prefix whose request still fails after these attempts, or that returns another
  4xx status, is recorded as done and its candidates keep a count of 0. Failures
  are not logged separately, so they are indistinguishable from "not found" in the
  outputs.
- `Add-Padding` is requested (`HIBP_ADD_PADDING = True`).
- Counts reflect the state of the live service at query time. Repeating the
  acquisition later may give different counts.
- `hibp` checkpoints `work/_hibp_done_prefixes.txt` and `work/_hibp_ckpt_*.npy`;
  after an interruption, rerun `--only hibp` to continue.

## Output format

`results/E1_results.json`, per model: `n_unique`, `total_found`, `found_rate`, and
`splits.d0`, `d1`, `d2`, each with `n_A`, `n_B`, `found_in_B`, `baseline` and
`reranked` metrics per budget, and `delta` (`reranked` minus `baseline`). The
primary split (`d0`) also stores `_bin_rate`, the learned bin table. `summary` has
per-budget means across splits with min and max of the deltas.

`results/E1_results2.json`, per model: the same metrics for all four rankings per
split (`generation`, `dup_count`, `zxcvbn`, `bin_table`), `delta_vs` (the method
against each baseline), and a `summary` with means across splits.

The JSON files contain only counts, rates and bin tables. They contain no
candidate strings.

## What can and cannot be reproduced from this release

- **Regenerating the reported tables:** possible from the JSON files.
- **Inspecting and running the pipeline on your own pools:** possible with the code
  and `config.py`.
- **Exact rerun on the original pools:** not possible, because the pools are
  withheld. Live-service counts may also have changed since acquisition.
