# Phase 4 — Concurrency Benchmark

## Objective

Measure how vLLM behaves as multiple requests are served concurrently.

The experiment focuses on:

* aggregate output throughput
* request throughput
* per-request latency
* latency distribution
* vLLM runtime metrics

The goal is to quantify the throughput/latency tradeoff under increasing concurrency.

## Model and Configuration

Model:

`Qwen/Qwen2.5-1.5B-Instruct`

vLLM configuration:

* dtype: FP16
* GPU: NVIDIA T4
* GPU memory utilization: 0.8
* maximum model length: 4096 tokens
* max new tokens: 100
* temperature: 0
* same workload for every request

Concurrency levels:

```text
1, 2, 4, 8, 16, 32
```

## Benchmark Architecture

```text
                  ┌── Request 1 ──┐
                  ├── Request 2 ──┤
Workload ─────────┼── Request N ──┼──> vLLM Server
                  └───────────────┘
                           │
                           ▼
                    Metrics + Results
```

The benchmark uses a Python thread pool to submit HTTP requests concurrently to the vLLM server.

## Metrics

### Aggregate Output Throughput

```text
total generated tokens / total wall time
```

This is the primary throughput metric for this experiment because it measures the inference server's total token processing capacity under concurrency.

### Request Throughput

```text
completed requests / total wall time
```

### Latency

For every request:

* mean latency
* P50 latency
* P95 latency

### vLLM Metrics

The benchmark also queries:

* KV-cache utilization
* running requests
* waiting requests

These are supplementary runtime observations rather than the primary performance measurements.

## Results

| Concurrency | Wall Time (s) | Requests/s | Aggregate tok/s | Mean Latency (s) | P50 (s) | P95 (s) |
| ----------: | ------------: | ---------: | --------------: | ---------------: | ------: | ------: |
|           1 |         3.188 |       0.31 |           31.37 |            3.187 |   3.187 |   3.187 |
|           2 |         1.517 |       1.32 |          131.82 |            1.508 |   1.508 |   1.512 |
|           4 |         1.526 |       2.62 |          262.08 |            1.519 |   1.523 |   1.523 |
|           8 |         1.592 |       5.03 |          502.62 |            1.583 |   1.584 |   1.586 |
|          16 |         1.778 |       9.00 |          899.90 |            1.733 |   1.733 |   1.745 |
|          32 |         2.235 |      14.32 |         1431.73 |            2.192 |   2.191 |   2.202 |

All requested requests completed successfully at every concurrency level.

## Observations

### 1. Concurrency dramatically increases aggregate throughput

Single-request throughput was:

```text
31.37 tok/s
```

At concurrency 32:

```text
1431.73 tok/s
```

This represents approximately a **45.6× increase in aggregate throughput**.

The GPU can therefore process substantially more total work when multiple requests are available concurrently.

### 2. Throughput continues increasing with concurrency

Throughput increased at every tested level:

```text
31 → 132 → 262 → 503 → 900 → 1432 tok/s
```

No clear throughput saturation point was observed by concurrency 32.

Therefore, this experiment should not claim that C=32 is the maximum useful concurrency. A higher concurrency level or a more demanding workload would be required to identify the actual saturation point.

### 3. Latency has a different scaling behavior

Latency was relatively stable from C=2 through C=8:

```text
C=2   ~1.51s
C=4   ~1.52s
C=8   ~1.58s
```

It then increased as concurrency grew:

```text
C=16  ~1.73s
C=32  ~2.19s
```

This demonstrates the fundamental inference-server tradeoff:

> Higher concurrency can improve total system throughput while increasing individual request latency.

### 4. The GPU benefits from concurrent work

At concurrency 1, the server has very little work available at each step.

With many active requests, vLLM can combine work across requests during decoding through its continuous batching system. This allows the GPU to perform more useful computation per unit of wall-clock time.

This is why aggregate throughput increases much faster than single-request throughput.

### 5. KV-cache metric interpretation

The benchmark reported:

```text
KV cache after: 0.0%
```

This does **not** mean that KV cache was unused.

The metric is sampled after all requests have completed. Once requests finish, their KV-cache blocks are released, so utilization returns to zero.

Therefore, these boundary measurements cannot be used as peak KV-cache utilization measurements.

The repository records the metric for observability, while dedicated KV-cache experiments are responsible for measuring cache utilization during active inference.

## Reproducibility

Start the vLLM server:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --dtype float16 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096 \
  --port 8000
```

Run the benchmark:

```bash
PYTHONPATH=src python experiments/04_concurrency/run.py
```

Results are saved to:

```text
results/phase-4/concurrency.csv
results/phase-4/concurrency.json
```

## Engineering Takeaway

A single-request benchmark is not sufficient for evaluating an inference server.

Inference engines are designed to serve multiple requests concurrently. Increasing concurrency allows the system to batch work and utilize the GPU more effectively, producing much higher aggregate throughput.

However, this comes with a latency cost as concurrency increases.

This experiment establishes the baseline for later investigation of **scheduling, resource constraints, GPU utilization, and inference-system saturation**.
