# Phase 10 — V1 Production Benchmark Report

## Objective

Consolidate Phases 1–9 into a coherent inference-engineering story: what was measured on an NVIDIA Tesla T4 with `Qwen/Qwen2.5-1.5B-Instruct`, what the trade-offs are, and which configurations to prefer for latency vs memory.

This report **does not invent new numbers**. It standardizes saved results and clearly flags measurements that are not directly comparable.

## Hardware and Software Card

| Item | Value |
| ---- | ----- |
| GPU | NVIDIA Tesla T4 (15,360 MiB, CC 7.5) |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Typical precision | FP16 |
| Engines | Hugging Face Transformers, vLLM |
| Attention (Transformers) | PyTorch SDPA |
| Quantization | bitsandbytes 4-bit NF4 + double quant |
| Primary runtime | Google Colab GPU |

## Evidence Map

| Phase | Topic | Saved results | Notes |
| ----: | ----- | ------------- | ----- |
| 1 | Raw Transformers baseline | `results/phase-1-baseline.csv` | Prefill + context scaling |
| 2 | Benchmark framework | — | Methodology / code; no archived CSV |
| 3 | vLLM integration | — | Engine wired; apples-to-apples archive missing |
| 4 | Concurrency | `results/phase-4/` | Strong vLLM scaling evidence |
| 5 | Context + KV | `results/phase-5/` | Transformers exploratory + vLLM KV tables |
| 6 | Scheduler pressure | `results/phase-6/` | Request metrics real; gauge samples unreliable |
| 7 | GPU fundamentals | `results/phase-7/` | Idle/load/steady-state/batch |
| 8 | Attention / prefill | `results/phase-8/` | Prompt-length prefill + memory |
| 9 | Quantization | `results/phase-9/` | FP16 vs NF4 |

## Headline Findings

### 1. Concurrency is where vLLM wins

vLLM aggregate output throughput scaled from **31.37 → 1,431.73 tok/s** between concurrency 1 and 32 (~**45.6×**), while mean latency rose from ~1.51 s (C=2) to ~2.19 s (C=32).

| Concurrency | Aggregate tok/s | Mean latency (s) |
| ----------: | --------------: | ---------------: |
| 1 | 31.37 | 3.187 |
| 8 | 502.62 | 1.583 |
| 32 | 1431.73 | 2.192 |

**Engineering takeaway:** single-request benchmarks understate serving engines. Continuous batching needs concurrent work.

### 2. Context length raises memory and (often) latency

Transformers peak GPU memory rose with prompt length (~2.93 → 3.21 GB from ~397 → 1,486 tokens).

Under vLLM, longer prompts at fixed concurrency increased both KV utilization and average latency (e.g. C=8: ~2.21 s → 3.43 s from ~500 → ~2,000 tokens).

**Engineering takeaway:** KV cost scales with `sequence_length × active_sequences`.

### 3. Prefill cost grows sharply at longer prompts

With FP16 + SDPA forward passes (`use_cache=False`):

| Prompt tokens | Prefill (ms) | Extra peak memory (MB) |
| ------------: | -----------: | ---------------------: |
| 32 | 58.69 | 23.2 |
| 256 | 70.67 | 187.3 |
| 1024 | 371.51 | 741.9 |

From 32 → 1,024 tokens: latency ≈ **6.3×**, extra peak memory ≈ **32×**.

These are **full forward-pass** costs, not attention-only.

### 4. Explicit Transformers batching improves aggregate throughput

| Batch size | Aggregate tok/s | Peak allocated (GiB) |
| ---------: | --------------: | -------------------: |
| 1 | 12.269 | 2.896 |
| 4 | 105.060 | 2.901 |
| 8 | 206.318 | 2.918 |

Batch size ≠ vLLM request concurrency. This experiment sets Transformers batch size explicitly.

### 5. Warm-up and steady state matter

Early requests during GPU sampling were much slower than later ones. After warm-up, steady-state generation (~200 tokens) averaged **~27.4 tok/s** with ~4.4% throughput CV.

**Engineering takeaway:** exclude warm-up from published latency numbers.

### 6. 4-bit NF4 saves memory, not speed (on this T4)

| Metric | FP16 | 4-bit NF4 |
| ------ | ---- | --------- |
| Model footprint | 2.875 GiB | 1.045 GiB (−63.7%) |
| Peak allocated | 2.890 GiB | 1.089 GiB (−62.3%) |
| Mean generation latency (64 tokens) | 2.291 s | 5.136 s (**2.24× slower**) |

**Engineering takeaway:** quantization is a **memory** optimization here, not an automatic latency win.

### 7. GPU utilization ≠ memory bandwidth proof

During sustained Transformers inference, mean GPU util was ~38%, memory used ~3.1 GiB, power ~48 W. Utilization shows activity; it does not by itself prove a compute- or bandwidth-bound bottleneck. Flash SDPA was unavailable on T4 (CC 7.5).

### 8. Scheduler pressure ≠ failure

At concurrency 128 with ~3k-token prompts, vLLM completed **128/128** requests (~396 output tok/s, ~31 s mean latency). Instantaneous Prometheus gauge scrapes in that run were unreliable (mostly zeros); request-level outcomes remain the trusted signal.

## Comparability Warnings

Do **not** merge these into one fake leaderboard without caveats:

1. `max_new_tokens` differs (64 / 100 / 200).
2. Transformers in-process latency ≠ vLLM HTTP latency.
3. Phase 2–3 lack archived standardized CSVs.
4. Phase 8 measures forward/prefill passes; Phase 7/9 measure `generate()`.
5. Colab GPU state and software versions vary across days.

## Recommended Configurations (this model + T4)

| Goal | Prefer | Why |
| ---- | ------ | --- |
| Lowest single-request Transformers latency | FP16 + SDPA | Faster than NF4 in Phase 9 |
| Lowest model memory | 4-bit NF4 | ~1.05 GiB footprint vs ~2.88 GiB |
| Highest aggregate serving throughput | vLLM + higher concurrency | ~45× tok/s from C=1→32 |
| Studying prefill / prompt cost | Phase 8 prompt sweep | Clear latency + memory vs length |
| Studying batching without a server | Transformers explicit batch | Clear aggregate tok/s gains |

## Reproduction (flagship paths)

```bash
# Install
pip install -e .

# Transformers baseline
PYTHONPATH=src python experiments/01_baseline/baseline.py

# vLLM server (separate terminal)
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --dtype float16 --gpu-memory-utilization 0.8 \
  --max-model-len 4096 --port 8000

# Concurrency matrix
PYTHONPATH=src python experiments/04_concurrency/run.py

# GPU fundamentals / attention / quantization
PYTHONPATH=src python experiments/07_gpu_performance/run.py
PYTHONPATH=src python experiments/08_attention_optimization/run.py
PYTHONPATH=src python experiments/09_quantization/run.py
```

Colab notebooks used during development:

* `docs/01_raw_transformers.ipynb`
* `docs/phase9.ipynb`

## V1 Conclusion

This project demonstrates open-weight LLM inference as a **systems** problem: prefill vs decode, KV growth, concurrency/batching, scheduler admission, GPU activity, attention backends, and precision trade-offs.

For this V1 portfolio on a Tesla T4 with Qwen2.5-1.5B:

* Use **vLLM + concurrency** when aggregate throughput matters.
* Use **FP16** when Transformers latency matters.
* Use **4-bit NF4** when memory is the binding constraint and slower generation is acceptable.
* Always separate **warm-up** from steady-state numbers, and treat utilization metrics as supporting evidence—not proof of a specific bottleneck.

## Status

**Phase 10 / V1 report complete** — consolidated measured trade-offs with explicit limitations.
