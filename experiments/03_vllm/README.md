# Phase 3 — vLLM Integration

## Objective

Integrate vLLM as a production inference backend and compare it against the raw Transformers baseline using the same benchmark workload.

The benchmark layer remains engine-independent.

## Architecture

```text
                    ┌── TransformersEngine
                    │
Workload → Runner ──┤
                    │
                    └── VLLMEngine → vLLM Server
```

## vLLM Configuration

The vLLM server should be started separately:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --dtype float16 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096 \
  --port 8000
```

The benchmark communicates with the server through its OpenAI-compatible API.

## Benchmark Configuration

* Model: `Qwen/Qwen2.5-1.5B-Instruct`
* Precision: FP16
* Workload: same short workload used by the benchmark framework
* Maximum output: 100 tokens
* Temperature: 0
* Warmup: 2 requests
* Measured runs: 5

## Metrics

Common benchmark metrics:

* Input tokens
* Output tokens
* Mean latency
* P50 latency
* P95 latency
* Output tokens/sec

vLLM-specific metrics:

* KV-cache utilization
* Running requests
* Waiting requests

## Measurement Boundary

Transformers runs directly inside the benchmark process.

vLLM runs as a separate inference server and is measured through HTTP requests.

Therefore, the latency measurements include different system boundaries:

```text
Transformers:

benchmark → model → benchmark


vLLM:

benchmark → HTTP → vLLM → HTTP → benchmark
```

For this reason, latency should not be treated as a perfectly apples-to-apples comparison.

Output throughput is the primary comparison metric in this phase.

## Status / Limitation

Phase 3 established the engine integration and measurement boundary, but a finalized apples-to-apples `results/phase-3/` CSV was not archived from the Colab runs.

Later phases provide stronger comparative evidence:

* Phase 4 — vLLM concurrency scaling
* Phase 1 / 7 — Transformers single-request baselines

## Running

Start the vLLM server first, then run:

```bash
PYTHONPATH=src python experiments/03_vllm/run.py
```

Results are written to:

```text
results/phase-3/vllm-baseline.csv
results/phase-3/vllm-baseline.json
```
