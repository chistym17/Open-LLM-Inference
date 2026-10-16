# Open LLM Inference Platform

Hands-on inference engineering for open-weight LLMs: measure how prefill, decode, KV cache, concurrency, batching, scheduling, attention backends, and quantization behave on real GPUs—not just “serve a model.”

**V1 status:** Experiment phases 1–9 are documented; Phase 10 consolidates them. Measured artifacts live under `results/`. Methodology, tables, and caveats live in the V1 report—treat that as the source of truth:

**[experiments/10_v1_report/README.md](experiments/10_v1_report/README.md)**

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

| Item | Value |
| ---- | ----- |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| GPU | NVIDIA Tesla T4 (15,360 MiB, compute capability 7.5) |
| Engines | Hugging Face Transformers, vLLM |
| Primary runtime | Google Colab GPU sessions |

Phases were run across multiple Colab sessions. Headline numbers below are **not one unified leaderboard**—token counts, measurement boundaries, and engines differ. See the V1 report for comparability notes.

## Key findings (V1)

Throughput figures below are **aggregate** (total generated tokens / wall time), not per-request decode rates, unless noted.

| Finding | Evidence |
| ------- | -------- |
| vLLM **aggregate** output throughput rose from **31.37 → 1,431.73 tok/s** (~**45.6×**) as concurrency went from 1 → 32 | [`results/phase-4/concurrency.csv`](results/phase-4/concurrency.csv) |
| In the tested **full-model forward pass**, increasing the prompt from 32 → 1,024 tokens produced ~**6.3×** prefill latency and ~**32×** *additional* peak-memory growth | [`results/phase-8/prompt_length.csv`](results/phase-8/prompt_length.csv) |
| Explicit Transformers batching raised **aggregate** tok/s from ~**12.3 → 206.3** (batch 1 → 8) | [`results/phase-7/batch_scaling.csv`](results/phase-7/batch_scaling.csv) |
| 4-bit NF4 cut model footprint ~**63.7%** (2.875 → 1.045 GiB) but mean generation latency was ~**2.24×** higher than FP16 | [`results/phase-9/quantization.csv`](results/phase-9/quantization.csv) |
| Warm-up requests distort latency; steady-state needs repeated runs after warm-up | Phase 7 report + [`results/phase-7/steady_state.csv`](results/phase-7/steady_state.csv) |
| The tested Flash SDPA kernel was unsupported on the T4 (compute capability 7.5) in this environment | Phase 8 report |

**Baselines for this setup (not universal rules)**

- Transformers latency-oriented baseline: **FP16 + SDPA** (faster than NF4 in Phase 9; still workload-dependent)
- Memory-constrained baseline: **4-bit NF4**
- Serving throughput in this benchmark: **vLLM achieved higher aggregate throughput as concurrency increased**

Full tables, sample counts, and caveats: [V1 report](experiments/10_v1_report/README.md).

## Repository layout

```text
src/llm_platform/          # reusable engines + benchmark helpers
experiments/01..10/        # phase runners + reports
results/                   # measured CSV/JSON artifacts
configs/default.yaml       # default model / vLLM settings
docs/                      # inference concepts + Colab notebooks
```

## Setup

Core package (Transformers experiments):

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
pip install pandas
```

**Environment note:** Phase 4–6 need **vLLM**; Phase 9 needs **bitsandbytes**. These stacks were tested in **separate Colab sessions**, not as one guaranteed combo. Prefer separate virtualenvs if CUDA/package conflicts appear:

```bash
# Serving / concurrency (Phases 4–6) — separate env recommended
pip install vllm requests

# Quantization (Phase 9) — separate env recommended
pip install bitsandbytes
```

Requires an NVIDIA GPU. vLLM phases also need a running server.

Tested reference point (Colab): Tesla T4, PyTorch `2.11.0+cu130`, model FP16 unless noted.

## Reproduce flagship experiments

Paths below exist in this repository. Run from the repo root with `PYTHONPATH=src` where shown.

```bash
# Baseline (Transformers)
PYTHONPATH=src python experiments/01_baseline/baseline.py

# Start vLLM (separate process / env)
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --dtype float16 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096 \
  --port 8000

# Concurrency matrix (needs vLLM server)
PYTHONPATH=src python experiments/04_concurrency/run.py

# GPU fundamentals (Transformers)
PYTHONPATH=src python experiments/07_gpu_performance/run.py

# Prefill / prompt-length sweep (Transformers + SDPA)
PYTHONPATH=src python experiments/08_attention_optimization/run.py

# FP16 vs NF4 (needs bitsandbytes)
PYTHONPATH=src python experiments/09_quantization/run.py
```

Development notebooks: `docs/01_raw_transformers.ipynb`, `docs/phase8.ipynb`, `docs/phase9.ipynb`.

## Experiment index

| Phase | Directory | Results |
| ----: | --------- | ------- |
| 1 Baseline | [`experiments/01_baseline`](experiments/01_baseline) | [`results/phase-1-baseline.csv`](results/phase-1-baseline.csv) |
| 2 Benchmark framework | [`experiments/02_benchmark`](experiments/02_benchmark) | methodology (no archived CSV) |
| 3 vLLM integration | [`experiments/03_vllm`](experiments/03_vllm) | integration; apples-to-apples archive incomplete |
| 4 Concurrency | [`experiments/04_concurrency`](experiments/04_concurrency) | [`results/phase-4`](results/phase-4) |
| 5 Context / KV | [`experiments/05_context_kv`](experiments/05_context_kv) | [`results/phase-5`](results/phase-5) |
| 6 Scheduler pressure | [`experiments/06_scheduler_resource_pressure`](experiments/06_scheduler_resource_pressure) | [`results/phase-6`](results/phase-6) |
| 7 GPU performance | [`experiments/07_gpu_performance`](experiments/07_gpu_performance) | [`results/phase-7`](results/phase-7) |
| 8 Attention / prefill | [`experiments/08_attention_optimization`](experiments/08_attention_optimization) | [`results/phase-8`](results/phase-8) |
| 9 Quantization | [`experiments/09_quantization`](experiments/09_quantization) | [`results/phase-9`](results/phase-9) |
| 10 V1 report | [`experiments/10_v1_report`](experiments/10_v1_report) | [`results/phase-10`](results/phase-10) |

Also see [`docs/inference.md`](docs/inference.md) and [`docs/experiments.md`](docs/experiments.md).

## Limitations

- Most measurements come from Colab T4 sessions; absolute numbers will differ on other GPUs.
- Not every phase shares identical `max_new_tokens`, prompts, or measurement boundaries—do not merge tables without reading the V1 report caveats.
- Phase 2–3 do not ship finalized comparable CSV archives.
- Phase 6 Prometheus gauge samples were unreliable; trust request-level outcomes there.
- Phase 8 memory/latency figures are from a full-model forward pass (`use_cache=False`), not an isolated attention kernel.

## License / intent

Educational inference-engineering portfolio. Not a production serving product.
