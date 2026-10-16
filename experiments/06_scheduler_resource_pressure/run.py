"""Phase 6 — Scheduler resource pressure.

Extracted from docs/01_raw_transformers.ipynb (Colab cell 23).
Requires a running vLLM server on http://127.0.0.1:8000.
"""

import csv
import json
import re
import time
import threading
import requests

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# Configuration
# ============================================================

BASE_URL = "http://127.0.0.1:8000"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Large single-shot workload.
CONCURRENCY = 128

# Leave enough context room for generation.
TARGET_PROMPT_TOKENS = 3000
MAX_NEW_TOKENS = 100

# High-frequency scheduler sampling.
SAMPLE_INTERVAL = 0.05

OUTPUT_DIR = Path("results/phase-6")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# vLLM metrics
# ============================================================

def get_metrics():
    response = requests.get(
        f"{BASE_URL}/metrics",
        timeout=5,
    )
    response.raise_for_status()

    text = response.text

    def gauge(pattern, default=0.0):
        match = re.search(pattern, text, re.MULTILINE)
        return float(match.group(1)) if match else default

    return {
        "running": gauge(
            r'^vllm:num_requests_running\s+([0-9.eE+-]+)',
            0,
        ),

        "waiting": gauge(
            r'^vllm:num_requests_waiting\s+([0-9.eE+-]+)',
            0,
        ),

        "kv_cache_usage": gauge(
            r'^vllm:kv_cache_usage_perc\s+([0-9.eE+-]+)',
            0,
        ),

        "capacity_waiting": gauge(
            r'^vllm:num_requests_waiting_by_reason\{reason="capacity"\}\s+([0-9.eE+-]+)',
            0,
        ),

        "deferred_waiting": gauge(
            r'^vllm:num_requests_waiting_by_reason\{reason="deferred"\}\s+([0-9.eE+-]+)',
            0,
        ),
    }


# ============================================================
# Prompt construction
# ============================================================

unit = """
Large language model inference systems must manage GPU compute,
memory bandwidth, request scheduling, and KV cache capacity.

During serving, multiple requests compete for finite GPU resources.
The scheduler decides which requests can execute while respecting
available KV cache and execution budgets.

Autoregressive generation maintains a key-value cache for every
transformer layer. As the context grows, each active request consumes
more KV cache memory.

With many concurrent requests, the aggregate KV-cache requirement can
become larger than the available memory reserved for inference.

A production inference engine therefore cannot simply execute every
incoming request at the same time. It continuously decides which
requests can remain in the running set and which requests must wait
for resources to become available.

This creates an important scheduling problem. The inference engine
must balance GPU computation, memory capacity, request latency,
throughput, and fairness between active requests.

Long prompts require substantial prefill computation and KV-cache
allocation. During decoding, every generated token extends the
request's KV cache.

When many long-context requests arrive together, the scheduler may
need to keep some requests waiting until sufficient KV-cache blocks
become available.

The goal of this experiment is to observe this behavior directly
under a large concurrent workload.
"""


# ============================================================
# Tokenize prompt through vLLM
# ============================================================

def tokenize_prompt(text):
    """
    Use vLLM's tokenizer endpoint so the experiment knows
    the actual token count rather than estimating from characters.
    """

    response = requests.post(
        f"{BASE_URL}/tokenize",
        json={
            "model": MODEL,
            "prompt": text,
        },
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    tokens = data.get("tokens", [])

    return tokens


# Build a sufficiently large candidate prompt.
candidate_prompt = unit * 200

print("Checking vLLM server...")

initial_metrics = get_metrics()

print(initial_metrics)

print()
print("Tokenizing workload...")


candidate_tokens = tokenize_prompt(candidate_prompt)

print(f"Candidate tokens: {len(candidate_tokens):,}")


# If the candidate is smaller than the target, repeat it.
while len(candidate_tokens) < TARGET_PROMPT_TOKENS:

    candidate_prompt += unit
    candidate_tokens = tokenize_prompt(candidate_prompt)


# We deliberately stay below max-model-len=4096.
# Target ~3000 input tokens + 100 output tokens.
prompt_tokens = candidate_tokens[:TARGET_PROMPT_TOKENS]


# Decode the selected tokens back into text using the tokenizer API.
# vLLM's detokenize endpoint lets us preserve an exact token target.
response = requests.post(
    f"{BASE_URL}/detokenize",
    json={
        "model": MODEL,
        "tokens": prompt_tokens,
    },
    timeout=30,
)

response.raise_for_status()

prompt = response.json()["prompt"]

# Verify final token count.
final_tokens = tokenize_prompt(prompt)

print(f"Final prompt tokens: {len(final_tokens):,}")
print(f"Prompt characters:    {len(prompt):,}")

if len(final_tokens) + MAX_NEW_TOKENS > 4096:
    raise RuntimeError(
        f"Prompt too large: {len(final_tokens)} input + "
        f"{MAX_NEW_TOKENS} output > 4096 max-model-len"
    )


# ============================================================
# Request worker
# ============================================================

def request_one(request_id):

    start = time.perf_counter()

    try:

        response = requests.post(
            f"{BASE_URL}/v1/completions",
            json={
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": MAX_NEW_TOKENS,
                "temperature": 0,
            },
            timeout=300,
        )

        latency = time.perf_counter() - start

        if not response.ok:

            return {
                "id": request_id,
                "success": False,
                "status_code": response.status_code,
                "error": response.text[:2000],
                "latency": latency,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

        data = response.json()

        usage = data.get("usage", {})

        return {
            "id": request_id,
            "success": True,
            "status_code": response.status_code,
            "error": None,
            "latency": latency,
            "prompt_tokens": usage.get(
                "prompt_tokens",
                len(final_tokens),
            ),
            "completion_tokens": usage.get(
                "completion_tokens",
                0,
            ),
        }

    except Exception as e:

        return {
            "id": request_id,
            "success": False,
            "status_code": None,
            "error": str(e),
            "latency": time.perf_counter() - start,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }


# ============================================================
# Background metrics sampler
# ============================================================

samples = []

stop_sampling = False


def sampler():

    while not stop_sampling:

        try:

            sample = get_metrics()

            sample["time"] = time.perf_counter()

            samples.append(sample)

        except Exception:
            pass

        time.sleep(SAMPLE_INTERVAL)


# ============================================================
# Percentile
# ============================================================

def percentile(values, p):

    if not values:
        return 0.0

    values = sorted(values)

    k = (len(values) - 1) * p / 100

    lower = int(k)
    upper = min(lower + 1, len(values) - 1)

    return (
        values[lower]
        + (values[upper] - values[lower])
        * (k - lower)
    )


# ============================================================
# Warm-up
# ============================================================

print()
print("Running warm-up...")

warmup = requests.post(
    f"{BASE_URL}/v1/completions",
    json={
        "model": MODEL,
        "prompt": "Explain KV cache in one sentence.",
        "max_tokens": 10,
        "temperature": 0,
    },
    timeout=120,
)

if not warmup.ok:
    print("Warm-up failed:")
    print(warmup.text)
    raise RuntimeError("vLLM warm-up request failed.")


print("Warm-up complete.")


# ============================================================
# Experiment
# ============================================================

print()
print("========================================")
print("PHASE 6 RESOURCE PRESSURE EXPERIMENT")
print("========================================")

print(f"Model:                {MODEL}")
print(f"Concurrency:          {CONCURRENCY}")
print(f"Prompt tokens:        {len(final_tokens):,}")
print(f"Max new tokens:       {MAX_NEW_TOKENS}")
print(f"Total token/request:  {len(final_tokens) + MAX_NEW_TOKENS:,}")
print(f"Sampling interval:    {SAMPLE_INTERVAL}s")

print()
print("Starting workload...")
print()


samples.clear()
stop_sampling = False

sampler_thread = threading.Thread(
    target=sampler,
    daemon=True,
)

sampler_thread.start()


wall_start = time.perf_counter()

results = []


with ThreadPoolExecutor(
    max_workers=CONCURRENCY
) as executor:

    futures = [
        executor.submit(
            request_one,
            request_id,
        )
        for request_id in range(CONCURRENCY)
    ]

    for future in as_completed(futures):

        result = future.result()

        results.append(result)

        # Lightweight progress indicator.
        if len(results) % 16 == 0:

            print(
                f"Completed: {len(results)}/{CONCURRENCY}"
            )


wall_time = time.perf_counter() - wall_start


stop_sampling = True

sampler_thread.join(timeout=2)


# Final sample
try:

    final_sample = get_metrics()

    final_sample["time"] = time.perf_counter()

    samples.append(final_sample)

except Exception:
    pass


# ============================================================
# Separate successful / failed requests
# ============================================================

successful = [
    result
    for result in results
    if result["success"]
]

failed = [
    result
    for result in results
    if not result["success"]
]


# ============================================================
# Aggregate measurements
# ============================================================

completed = len(successful)
failed_count = len(failed)

total_prompt_tokens = sum(
    result["prompt_tokens"]
    for result in successful
)

total_output_tokens = sum(
    result["completion_tokens"]
    for result in successful
)

latencies = [
    result["latency"]
    for result in successful
]


mean_latency = (
    sum(latencies) / len(latencies)
    if latencies
    else 0
)

p50_latency = percentile(
    latencies,
    50,
)

p95_latency = percentile(
    latencies,
    95,
)

p99_latency = percentile(
    latencies,
    99,
)

request_throughput = (
    completed / wall_time
    if wall_time > 0
    else 0
)

output_tok_s = (
    total_output_tokens / wall_time
    if wall_time > 0
    else 0
)


# ============================================================
# Peak scheduler metrics
# ============================================================

max_running = max(
    (
        sample["running"]
        for sample in samples
    ),
    default=0,
)

max_waiting = max(
    (
        sample["waiting"]
        for sample in samples
    ),
    default=0,
)

max_capacity_waiting = max(
    (
        sample["capacity_waiting"]
        for sample in samples
    ),
    default=0,
)

max_deferred_waiting = max(
    (
        sample["deferred_waiting"]
        for sample in samples
    ),
    default=0,
)

peak_kv = max(
    (
        sample["kv_cache_usage"]
        for sample in samples
    ),
    default=0,
)


# ============================================================
# Print final results
# ============================================================

print()
print("========================================")
print("RESULTS")
print("========================================")

print()
print("Requests:")
print(
    f"Successful requests:       "
    f"{completed}/{CONCURRENCY}"
)

print(
    f"Failed requests:           "
    f"{failed_count}/{CONCURRENCY}"
)

print(
    f"Wall time:                 "
    f"{wall_time:.3f}s"
)

print(
    f"Request throughput:        "
    f"{request_throughput:.2f} req/s"
)


print()
print("Tokens:")

print(
    f"Total prompt tokens:       "
    f"{total_prompt_tokens:,}"
)

print(
    f"Total output tokens:       "
    f"{total_output_tokens:,}"
)

print(
    f"Aggregate output tok/s:    "
    f"{output_tok_s:.2f}"
)


print()
print("Latency:")

print(
    f"Mean:                       "
    f"{mean_latency:.3f}s"
)

print(
    f"P50:                        "
    f"{p50_latency:.3f}s"
)

print(
    f"P95:                        "
    f"{p95_latency:.3f}s"
)

print(
    f"P99:                        "
    f"{p99_latency:.3f}s"
)


print()
print("Scheduler / resource pressure:")

print(
    f"Max running:                "
    f"{max_running:.0f}"
)

print(
    f"Max waiting:                "
    f"{max_waiting:.0f}"
)

print(
    f"Max capacity waiting:       "
    f"{max_capacity_waiting:.0f}"
)

print(
    f"Max deferred waiting:       "
    f"{max_deferred_waiting:.0f}"
)

print(
    f"Peak KV cache usage:        "
    f"{peak_kv:.2f}%"
)

print(
    f"Metric samples:             "
    f"{len(samples)}"
)


# ============================================================
# Failed request diagnostics
# ============================================================

if failed:

    print()
    print("========================================")
    print("FIRST FAILED REQUEST")
    print("========================================")

    first_failure = failed[0]

    print(
        "Status:",
        first_failure["status_code"],
    )

    print(
        "Error:",
        first_failure["error"],
    )


# ============================================================
# Result JSON
# ============================================================

result = {

    "experiment":
        "phase_6_scheduler_resource_pressure",

    "configuration": {

        "model": MODEL,

        "concurrency":
            CONCURRENCY,

        "prompt_tokens":
            len(final_tokens),

        "max_new_tokens":
            MAX_NEW_TOKENS,

        "max_model_len":
            4096,

        "sample_interval":
            SAMPLE_INTERVAL,
    },

    "results": {

        "successful_requests":
            completed,

        "failed_requests":
            failed_count,

        "wall_time_seconds":
            wall_time,

        "request_throughput":
            request_throughput,

        "total_prompt_tokens":
            total_prompt_tokens,

        "total_output_tokens":
            total_output_tokens,

        "output_tokens_per_second":
            output_tok_s,

        "mean_latency_seconds":
            mean_latency,

        "p50_latency_seconds":
            p50_latency,

        "p95_latency_seconds":
            p95_latency,

        "p99_latency_seconds":
            p99_latency,
    },

    "scheduler_metrics": {

        "max_running":
            max_running,

        "max_waiting":
            max_waiting,

        "max_capacity_waiting":
            max_capacity_waiting,

        "max_deferred_waiting":
            max_deferred_waiting,

        "peak_kv_cache_usage_percent":
            peak_kv,
    },

    "metric_samples":
        samples,
}


# ============================================================
# Save JSON
# ============================================================

json_path = (
    OUTPUT_DIR /
    "scheduler_pressure.json"
)

with open(json_path, "w") as f:

    json.dump(
        result,
        f,
        indent=2,
    )


# ============================================================
# Save metrics CSV
# ============================================================

csv_path = (
    OUTPUT_DIR /
    "scheduler_metrics.csv"
)

with open(
    csv_path,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "time",
            "running",
            "waiting",
            "capacity_waiting",
            "deferred_waiting",
            "kv_cache_usage",
        ],
    )

    writer.writeheader()

    writer.writerows(samples)


print()
print("========================================")
print("FILES SAVED")
print("========================================")

print(json_path)
print(csv_path)