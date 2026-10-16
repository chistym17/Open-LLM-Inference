# Experiments Index

Methodology notes and pointers to phase reports. Measured artifacts live under `results/`.

The consolidated V1 engineering report is [`experiments/10_v1_report/README.md`](../experiments/10_v1_report/README.md).

## Common setup

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- GPU: NVIDIA Tesla T4
- Default precision: FP16
- Config defaults: [`configs/default.yaml`](../configs/default.yaml)

## Phase guide

### Phase 1 — Raw Transformers baseline

Establish prefill, end-to-end latency, throughput, and context memory for Hugging Face Transformers.

- Report: [`experiments/01_baseline/README.md`](../experiments/01_baseline/README.md)
- Results: [`results/phase-1-baseline.csv`](../results/phase-1-baseline.csv)

| Metric | Result |
| ------ | -----: |
| Prefill (11 tokens) | 0.137 s |
| Latency (100 new tokens) | 5.469 s |
| Output throughput | 18.29 tok/s |
| Peak GPU memory | 2.891 GB |

### Phase 2 — Benchmark framework

Reusable runner, workloads, and metrics (`src/llm_platform/benchmark/`). No archived phase-2 CSV in V1.

- Report: [`experiments/02_benchmark/README.md`](../experiments/02_benchmark/README.md)

### Phase 3 — vLLM integration

HTTP client to a vLLM server with an engine-agnostic benchmark boundary. Finalized phase-3 archive not shipped; see Phases 1 and 4 for comparative numbers.

- Report: [`experiments/03_vllm/README.md`](../experiments/03_vllm/README.md)

### Phase 4 — Concurrency

vLLM concurrency 1→32. Aggregate throughput ~31 → 1432 tok/s.

- Report: [`experiments/04_concurrency/README.md`](../experiments/04_concurrency/README.md)
- Results: [`results/phase-4/`](../results/phase-4/)

### Phase 5 — Context length and KV cache

Theoretical KV bytes/token (~28 KB) plus Transformers context sweeps and vLLM KV/latency tables.

- Report: [`experiments/05_context_kv/README.md`](../experiments/05_context_kv/README.md)
- Results: [`results/phase-5/`](../results/phase-5/)

### Phase 6 — Scheduler resource pressure

128 concurrent ~3k-token prompts against vLLM. All requests completed; decode dominated runtime.

- Report: [`experiments/06_scheduler_resource_pressure/README.md`](../experiments/06_scheduler_resource_pressure/README.md)
- Results: [`results/phase-6/`](../results/phase-6/)

### Phase 7 — GPU performance fundamentals

Idle baseline, GPU sampling during inference, steady-state latency, Transformers batch scaling.

- Report: [`experiments/07_gpu_performance/README.md`](../experiments/07_gpu_performance/README.md)
- Results: [`results/phase-7/`](../results/phase-7/)

### Phase 8 — Attention and prefill scaling

Prompt length vs SDPA forward latency and extra peak memory.

- Report: [`experiments/08_attention_optimization/README.md`](../experiments/08_attention_optimization/README.md)
- Results: [`results/phase-8/`](../results/phase-8/)

### Phase 9 — Quantization

FP16 vs bitsandbytes 4-bit NF4: large memory win, slower generation on T4.

- Report: [`experiments/09_quantization/README.md`](../experiments/09_quantization/README.md)
- Results: [`results/phase-9/`](../results/phase-9/)

### Phase 10 — V1 report

Cross-phase consolidation, recommendations, and reproducibility.

- Report: [`experiments/10_v1_report/README.md`](../experiments/10_v1_report/README.md)

## Colab sources

- [`docs/01_raw_transformers.ipynb`](01_raw_transformers.ipynb) — Phases 1/5–7 exploration + Phase 4/6 runners
- [`docs/phase8.ipynb`](phase8.ipynb) — Attention / SDPA / prompt-length prefill
- [`docs/phase9.ipynb`](phase9.ipynb) — FP16 vs NF4
