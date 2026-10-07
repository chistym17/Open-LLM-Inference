Now finish Phase 3 documentation. We already have the code; don't add more infrastructure.

## 1. Create the Phase 3 README

```bash
nano experiments/03_vllm/README.md
```

Paste:

````markdown
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
````

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

## Expected Result

The experiment should establish how much throughput improvement the optimized inference engine provides over raw Transformers inference for the same model and workload.

Earlier exploratory vLLM testing showed substantially higher single-request generation throughput than the raw Transformers baseline, but the final Phase 3 numbers must come from the standardized benchmark.

## Running

Start the vLLM server first, then run:

```bash
python experiments/03_vllm/run.py
```

Results are written to:

```text
results/phase-3/vllm-baseline.csv
results/phase-3/vllm-baseline.json
```


