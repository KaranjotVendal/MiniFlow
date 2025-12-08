# src/concurrency/concurrency_runner.py
# TODO-3 — Concurrency load testing + concurrency plots

# We’ll add a concurrency runner that supports two modes:

# HTTP mode — repeatedly call an HTTP inference endpoint (e.g., your FastAPI service). Uses requests in a thread pool.

# In-process mode — call process_sample() directly across worker threads (useful for local testing without HTTP).

# It will measure per-request latencies and produce:

# histogram / CDF of latencies

# plot of concurrency level vs mean latency (if running multiple concurrency levels)

# save JSON results per run
#
# Note: this is simple thread-based concurrency using requests.
# For a more accurate async test use httpx.AsyncClient with asyncio; this thread-based approach is robust and works in most environments.
#
import time
import json
from pathlib import Path
from typing import Callable, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import statistics
import math

def _call_http(endpoint: str, audio_bytes: bytes, timeout=60):
    start = time.time()
    r = requests.post(endpoint, data=audio_bytes, headers={"Content-Type":"audio/wav"}, timeout=timeout)
    latency = time.time() - start
    return r.status_code, latency, r.text[:200]

def _call_inproc(fn: Callable, sample, timeout=None):
    start = time.time()
    processed, metrics = fn(sample)
    latency = time.time() - start
    return 200, latency, ""


def run_concurrency_test(
    mode: str,
    target: str,
    samples: List,
    concurrency: int = 4,
    requests_per_worker: int = 10,
    output_dir: Optional[str] = None,
):
    """
    mode: "http" or "inproc"
    target: endpoint URL if mode == "http", or a callable if "inproc" (string path to function not supported)
    samples: list of audio samples or bytes to send. We will cycle through them.
    concurrency: number of parallel workers
    requests_per_worker: total requests per worker
    """
    out_path = Path(output_dir or "concurrency_results")
    out_path.mkdir(parents=True, exist_ok=True)
    run_id = int(time.time())
    results_file = out_path / f"concurrency_{run_id}.jsonl"

    # create worker pool
    total_requests = concurrency * requests_per_worker
    print(f"Starting concurrency test: mode={mode} concurrency={concurrency} total_requests={total_requests}")

    tasks = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = []
        for w in range(concurrency):
            for r in range(requests_per_worker):
                sample = samples[(w * requests_per_worker + r) % len(samples)]
                if mode == "http":
                    futures.append(ex.submit(_call_http, target, sample))
                else:
                    # target must be a callable in this case
                    futures.append(ex.submit(_call_inproc, target, sample))

        # collect results
        latencies = []
        statuses = []
        with open(results_file, "w") as f:
            for fut in as_completed(futures):
                try:
                    status, latency, resp = fut.result()
                    latencies.append(latency)
                    statuses.append(status)
                    rec = {"status": status, "latency": latency, "timestamp": time.time()}
                    f.write(json.dumps(rec) + "\n")
                except Exception as e:
                    rec = {"status": "error", "latency": None, "error": str(e), "timestamp": time.time()}
                    f.write(json.dumps(rec) + "\n")

    # simple aggregation
    clean_lat = [l for l in latencies if l is not None]
    summary = {
        "mode": mode,
        "concurrency": concurrency,
        "num_requests": len(clean_lat),
        "mean_latency": statistics.mean(clean_lat) if clean_lat else None,
        "median_latency": statistics.median(clean_lat) if clean_lat else None,
        "p95": np_percentile(clean_lat, 95) if clean_lat else None,
        "p99": np_percentile(clean_lat, 99) if clean_lat else None,
    }
    with open(out_path / f"concurrency_summary_{run_id}.json", "w") as f:
        json.dump(summary, f, indent=2)

    return results_file, summary


def np_percentile(arr, p):
    if not arr:
        return None
    arr = sorted(arr)
    k = (len(arr)-1) * (p/100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return arr[int(k)]
    d0 = arr[int(f)] * (c-k)
    d1 = arr[int(c)] * (k-f)
    return d0 + d1
