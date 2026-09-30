"""Step 2 — resolve HIBP occurrence counts via the k-anonymity range API.

Only the 5-char prefix of each SHA-1 leaves the machine. For every distinct
prefix across both models we make ONE request, match the suffixes we care about
in memory, record each candidate's occurrence count, and discard the range.

We never persist raw HIBP ranges — only the (synthetic candidate -> count)
column, which is the minimal signal the experiment needs and is consistent with
the retention rule (no raw ranges are stored).

Output: work/<model>_counts.parquet with a single int64 column `count`, aligned
row-for-row to <model>_prepared.parquet.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time

import aiohttp
import numpy as np
import pyarrow.parquet as pq

import config


def _load_groups():
    """prefix -> list of (arr_id, row_idx, suffix); plus per-model count arrays."""
    groups: dict[str, list[tuple[int, int, str]]] = {}
    counts = {}
    model_names = list(config.MODELS)
    for arr_id, model in enumerate(model_names):
        path = os.path.join(config.WORK_DIR, f"{model}_prepared.parquet")
        t = pq.read_table(path, columns=["prefix", "suffix"])
        prefixes = t.column("prefix").to_pylist()
        suffixes = t.column("suffix").to_pylist()
        counts[model] = np.zeros(len(prefixes), dtype=np.int64)
        for i, (pf, sf) in enumerate(zip(prefixes, suffixes)):
            groups.setdefault(pf, []).append((arr_id, i, sf))
        print(f"[hibp] {model}: {len(prefixes):,} rows", flush=True)
    return model_names, groups, counts


def _parse_range(text: str) -> dict[str, int]:
    """Parse 'SUFFIX:COUNT' lines into a dict."""
    out: dict[str, int] = {}
    for line in text.splitlines():
        if ":" in line:
            suf, _, cnt = line.partition(":")
            try:
                out[suf.strip()] = int(cnt)
            except ValueError:
                continue
    return out


async def _fetch(session, sem, prefix):
    headers = {}
    if config.HIBP_ADD_PADDING:
        headers["Add-Padding"] = "true"
    url = config.HIBP_URL.format(prefix=prefix)
    delay = 1.0
    for attempt in range(config.HIBP_MAX_RETRIES):
        try:
            async with sem:
                async with session.get(url, headers=headers) as r:
                    if r.status == 200:
                        return await r.text()
                    if r.status == 429:  # rate limited -> honor Retry-After
                        wait = float(r.headers.get("Retry-After", delay))
                        await asyncio.sleep(wait)
                        continue
                    if 500 <= r.status < 600:
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 30)
                        continue
                    return None  # 4xx other than 429: give up on this prefix
        except (aiohttp.ClientError, asyncio.TimeoutError):
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)
    return None


async def _run(model_names, groups, counts, done_path, ckpt_paths):
    prefixes = [p for p in groups if p not in _load_done(done_path)]
    total = len(groups)
    already = total - len(prefixes)
    print(f"[hibp] {total:,} distinct prefixes; {already:,} already done, "
          f"{len(prefixes):,} to fetch", flush=True)

    sem = asyncio.Semaphore(config.HIBP_CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=config.HIBP_TIMEOUT_S)
    connector = aiohttp.TCPConnector(limit=config.HIBP_CONCURRENCY, ttl_dns_cache=300)
    headers = {"User-Agent": config.HIBP_USER_AGENT}

    done_count = already
    t0 = time.time()
    done_file = open(done_path, "a")

    async with aiohttp.ClientSession(timeout=timeout, connector=connector,
                                     headers=headers) as session:
        async def handle(prefix):
            nonlocal done_count
            text = await _fetch(session, sem, prefix)
            if text is not None:
                rng = _parse_range(text)
                for arr_id, idx, suf in groups[prefix]:
                    c = rng.get(suf)
                    if c:
                        counts[model_names[arr_id]][idx] = c
            done_count += 1
            done_file.write(prefix + "\n")
            if done_count % 5000 == 0:
                rate = (done_count - already) / max(1e-9, time.time() - t0)
                eta = (total - done_count) / max(1e-9, rate)
                print(f"[hibp] {done_count:,}/{total:,}  {rate:.0f} req/s  "
                      f"ETA {eta/60:.1f} min", flush=True)
                done_file.flush()
                for m in model_names:
                    np.save(ckpt_paths[m], counts[m])

        # bounded task creation so we don't build 1M coroutines at once
        batch = 20000
        for i in range(0, len(prefixes), batch):
            await asyncio.gather(*(handle(p) for p in prefixes[i:i + batch]))

    done_file.close()
    print(f"[hibp] all prefixes done in {(time.time()-t0)/60:.1f} min", flush=True)


def _load_done(done_path) -> set:
    if os.path.exists(done_path):
        with open(done_path) as f:
            return set(x.strip() for x in f if x.strip())
    return set()


def run():
    os.makedirs(config.WORK_DIR, exist_ok=True)
    model_names, groups, counts = _load_groups()
    done_path = os.path.join(config.WORK_DIR, "_hibp_done_prefixes.txt")
    ckpt_paths = {m: os.path.join(config.WORK_DIR, f"_hibp_ckpt_{m}.npy")
                  for m in model_names}
    # resume partial counts if present
    for m in model_names:
        if os.path.exists(ckpt_paths[m]):
            prev = np.load(ckpt_paths[m])
            if len(prev) == len(counts[m]):
                counts[m] = prev
                print(f"[hibp] resumed {m} counts from checkpoint", flush=True)

    asyncio.run(_run(model_names, groups, counts, done_path, ckpt_paths))

    import pyarrow as pa
    for m in model_names:
        out = os.path.join(config.WORK_DIR, f"{m}_counts.parquet")
        pq.write_table(pa.table({"count": pa.array(counts[m], pa.int64())}),
                       out, compression="zstd")
        found = int((counts[m] > 0).sum())
        print(f"[hibp] {m}: {found:,} candidates found in HIBP -> {out}", flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()  # no args; documents intent
    run()
