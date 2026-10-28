# Phase 4 — Load-testing harness

## What we did

- Built a client-side load harness against the FastAPI gateway.
- Ran closed-loop concurrency sweeps and one open-loop RPS run.
- Saved raw per-request JSON under `results/v2/phase-4/` (gitignored).

## What this is / is not

- **Is:** load generation + measurement (closed/open arrival patterns in the **client**).
- **Is not:** gateway admission queue (Stage 5) or a custom vLLM scheduler.

## Environment

| Item | Value |
| ---- | ----- |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| GPU | NVIDIA T4 (Modal) |
| Path | local FastAPI → Modal vLLM |
| max_tokens | 32 |
| prompt_chars | 80 |
| stream | false |

## Numbers

Measured rows only (warmup excluded from percentiles). All runs: **0 failures**.

| Mode | knobs | n ok/fail | latency p50 (ms) | latency p95 (ms) | output tok/s | wall (s) |
| ---- | ----- | --------- | ---------------: | ---------------: | -----------: | -------: |
| closed | c=1, n=10 | 10/0 | **1127** | **1295** | 27.2 | 11.8 |
| closed | c=4, n=20 | 20/0 | **1336** | **9427** | 43.4 | 14.8 |
| closed | c=8, n=24 | 24/0 | **1433** | **2200** | **155.4** | 4.9 |
| open | rps=2, n=20 | 20/0 | **1214** | **1291** | 60.1 | 10.7 |

### Reading

- **c=1 → c=8:** more concurrency raises p50 a bit (1.13 s → 1.43 s) but **aggregate tok/s rises a lot** (27 → 155) — vLLM batching/sharing the GPU.
- **c=4 p95 ~9.4 s** is a heavy tail (a few multi-second outliers), not the typical request. Treat as variance/queueing noise; don’t use that p95 alone as capacity.
- **Open rps=2:** starts every 0.5 s; with ~1.2 s latency, a few requests overlap. Steady and healthy at this light open-loop rate (no failures).

## Decision

Harness is good enough to reproduce load shapes for Stage 5. Next overload work should push past comfortable concurrency/RPS until latency/errors degrade predictably.

## Limitation

Single short prompt size; no GPU util scrape in this stage. Open-loop only tested at 2 rps. c=4 tail needs more repeats if cited publicly.

## Next step

Stage 5: admission control (bound in-flight work / reject under overload).
