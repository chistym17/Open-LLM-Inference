# Phase 4 — Concurrency Benchmark

## Objective

Measure how vLLM behaves as multiple requests arrive concurrently.

The experiment focuses on:

* aggregate token throughput
* request throughput
* per-request latency
* latency distribution
* KV-cache utilization
* running/waiting requests

The goal is to quantify the throughput/latency tradeoff rather than reimplement vLLM's scheduler or continuous batching.

## Workload

Model:

`Qwen/Qwen2.5-1.5B-Instruct`

Generation:

* max new tokens: 100
* temperature: 0
* same prompt for every request

Concurrency levels:

```text
1
2
4
8
16
32
```

## Architecture

```text
                 ┌── Request 1 ──┐
                 ├── Request 2 ──┤
Workload ────────┼── Request N ──┼──> vLLM Server
                 └───────────────┘
                         │
                         ▼
                  Metrics + Results
```

The benchmark uses a thread pool to submit multiple HTTP requests concurrently to the vLLM server.

## Metrics

### Throughput

**Requests/sec**

```text
completed requests / total wall time
```

**Aggregate output tok/s**

```text
total generated tokens / total wall time
```

This is the primary system-throughput metric for the concurrency experiment.

### Latency

For each request we measure:

* mean latency
* P50 latency
* P95 latency

### vLLM metrics

We also record:

* KV-cache utilization
* running requests
* waiting requests

## Previous exploratory results

Before formalizing this benchmark, concurrency was already tested manually.

The exploratory experiment showed strong throughput scaling from concurrency 1 through 16, with diminishing gains at higher concurrency and increasing latency.

Those measurements are treated as discovery results. The current benchmark exists to make the experiment reproducible and generate standardized repository results.

## Running

Start vLLM:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --dtype float16 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096 \
  --port 8000
```

Then run:

```bash
PYTHONPATH=src python experiments/04_concurrency/run.py
```

Results are written to:

```text
results/phase-4/concurrency.csv
results/phase-4/concurrency.json
```

## Expected Analysis

The important questions are:

1. How much does aggregate throughput increase with concurrency?
2. At what point do throughput gains begin diminishing?
3. How does P95 latency change?
4. Does vLLM begin accumulating waiting requests?
5. How does KV-cache utilization change?

The experiment should not assume a particular saturation point. The GPU measurements determine that.

## Key Engineering Lesson

A single-request benchmark does not describe an inference server's real behavior.

Inference systems are designed to serve many requests concurrently. Increasing concurrency allows the engine to keep the GPU busier and batch work across requests, increasing aggregate throughput.

The tradeoff is that higher concurrency can increase queueing and per-request latency.

This experiment provides the baseline for later work on scheduling, resource constraints, and GPU utilization.
