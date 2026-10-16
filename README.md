# Open LLM Inference Platform

Hands-on inference engineering for open-weight LLMs: measure how prefill, decode, KV cache, concurrency, batching, scheduling, attention backends, and quantization behave on real GPUs—not just “serve a model.”

**V1 status:** Phases 1–10 complete. Primary evidence lives under `results/` and `experiments/*/README.md`. The consolidated engineering report is [`experiments/10_v1_report/README.md`](experiments/10_v1_report/README.md).

## What this project is

```text
Understand mechanism
       ↓
Build / reuse minimal code
       ↓
Benchmark on NVIDIA T4 (Colab)
       ↓
Document trade-offs
```

Model under test: **`Qwen/Qwen2.5-1.5B-Instruct`**  
Primary GPU: **NVIDIA Tesla T4 (FP16)**  
Engines: **Hugging Face Transformers** and **vLLM**

## Key findings (V1)

| Finding | Evidence |
| ------- | -------- |
| vLLM aggregate throughput rises ~**45×** from concurrency 1 → 32 | Phase 4 |
| Longer prompts raise prefill latency (~**6.3×** 32→1024) and peak memory (~**32×**) | Phase 8 |
| Explicit Transformers batching raises aggregate tok/s (~12 → 206) | Phase 7 |
| **4-bit NF4** cuts model footprint ~**64%** but is ~**2.2× slower** than FP16 | Phase 9 |
| Warm-up requests distort latency; steady-state needs repeated runs | Phase 7 |
| Flash SDPA unavailable on T4 (CC 7.5) | Phase 8 |

**Quick recommendations for this setup**

- **Lower latency (Transformers):** FP16 + SDPA  
- **Lower memory:** 4-bit NF4  
- **Higher serving throughput:** vLLM with higher concurrency  

Full tables, caveats, and non-comparability notes: [V1 report](experiments/10_v1_report/README.md).

## Repository layout

```text
src/llm_platform/          # reusable engines + benchmark helpers
experiments/01..10/        # phase runners + reports
results/                   # measured CSV/JSON artifacts
configs/default.yaml       # default model / vLLM settings
docs/                      # inference concepts + Colab notebooks
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
# Optional extras used by later phases:
# pip install vllm requests bitsandbytes pandas
```

Requires an NVIDIA GPU and, for vLLM phases, a running server.

## Reproduce flagship experiments

```bash
# Baseline (Transformers)
PYTHONPATH=src python experiments/01_baseline/baseline.py

# Start vLLM (separate process)
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --dtype float16 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096 \
  --port 8000

# Concurrency matrix
PYTHONPATH=src python experiments/04_concurrency/run.py

# GPU / attention / quantization
PYTHONPATH=src python experiments/07_gpu_performance/run.py
PYTHONPATH=src python experiments/08_attention_optimization/run.py
PYTHONPATH=src python experiments/09_quantization/run.py
```

Development notebooks: `docs/01_raw_transformers.ipynb`, `docs/phase8.ipynb`, `docs/phase9.ipynb`.

## Experiment index

| Phase | Directory | Results |
| ----: | --------- | ------- |
| 1 Baseline | [`experiments/01_baseline`](experiments/01_baseline) | [`results/phase-1-baseline.csv`](results/phase-1-baseline.csv) |
| 2 Benchmark framework | [`experiments/02_benchmark`](experiments/02_benchmark) | methodology |
| 3 vLLM integration | [`experiments/03_vllm`](experiments/03_vllm) | integration; archive incomplete |
| 4 Concurrency | [`experiments/04_concurrency`](experiments/04_concurrency) | [`results/phase-4`](results/phase-4) |
| 5 Context / KV | [`experiments/05_context_kv`](experiments/05_context_kv) | [`results/phase-5`](results/phase-5) |
| 6 Scheduler pressure | [`experiments/06_scheduler_resource_pressure`](experiments/06_scheduler_resource_pressure) | [`results/phase-6`](results/phase-6) |
| 7 GPU performance | [`experiments/07_gpu_performance`](experiments/07_gpu_performance) | [`results/phase-7`](results/phase-7) |
| 8 Attention / prefill | [`experiments/08_attention_optimization`](experiments/08_attention_optimization) | [`results/phase-8`](results/phase-8) |
| 9 Quantization | [`experiments/09_quantization`](experiments/09_quantization) | [`results/phase-9`](results/phase-9) |
| 10 V1 report | [`experiments/10_v1_report`](experiments/10_v1_report) | this release |

Also see [`docs/inference.md`](docs/inference.md) and [`docs/experiments.md`](docs/experiments.md).

## Limitations

- Most measurements come from Colab T4 sessions; absolute numbers will differ on other GPUs.
- Not every phase shares identical `max_new_tokens`, prompts, or measurement boundaries.
- Phase 2–3 do not ship finalized comparable CSV archives.
- Phase 6 Prometheus gauge samples were unreliable; trust request-level outcomes there.

## License / intent

Educational inference-engineering portfolio. Not a production serving product.
