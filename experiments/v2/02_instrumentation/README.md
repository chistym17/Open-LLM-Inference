# Phase 2 — Request lifecycle instrumentation

## What we did

- Added request IDs, structured JSON logs, and split timers (`gateway_ms` / `upstream_ms`).
- Compared direct Modal vLLM vs FastAPI→vLLM with warmup excluded from percentiles.
- Fixed the harness until the comparison was fair: warm both paths, interleaved pairs, persistent httpx keep-alive on both sides.

## What worked

- Instrumentation fields and logs behave as intended.
- After a fair protocol, **direct ≤ gateway** as expected.
- FastAPI handler overhead is negligible vs generation (`gateway_ms ≈ upstream_ms`).

## What failed or stayed unverified

- Early runs were misleading: sequential order (cold direct) and `urllib` without keep-alive inflated direct latency. Those are not used for the final overhead claim.
- `upstream_ms` is still HTTP time to vLLM, not pure GPU time.

## Environment

| Item | Value |
| ---- | ----- |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| GPU | NVIDIA T4 (Modal) |
| Prompt | `Say hello in one short sentence.` |
| Tokens | prompt 36, completion 10 |
| max_tokens | 32 |
| Protocol | warmup both (2 each), then 10 interleaved pairs; persistent httpx |
| Clocks | `time.perf_counter()` |

## Numbers

Raw rows: `results/v2/phase-2/latency_compare.json` (local only).

### Warmup (excluded from p50/p95)

| Path | samples (ms) | mean (ms) |
| ---- | ------------ | --------: |
| Direct | 1849, 622 | 1235 |
| Gateway | 1240, 703 | 972 |

Replica was already warm-ish this session (no 10–20 s cold hits).

### Measured steady-state (n=10 pairs)

| Path | n | client p50 (ms) | client p95 (ms) | mean (ms) |
| ---- | -: | --------------: | --------------: | --------: |
| Direct vLLM | 10 | **672.4** | **720.9** | 668.7 |
| Gateway | 10 | **716.0** | **729.0** | 702.8 |
| Overhead (gateway − direct) | | **+43.6** | **+8.1** | **+34.1** |

Gateway internal (measured):

| Timer | p50 (ms) | p95 (ms) | mean (ms) |
| ----- | -------: | -------: | --------: |
| `gateway_ms` | 710.7 | 724.0 | 697.7 |
| `upstream_ms` | 710.5 | 723.8 | 697.5 |

Handler add-on ≈ **0.2 ms** mean. Most of the ~30–45 ms client overhead is localhost hop + gateway process, still small vs ~0.7 s generation.

## Decision

Keep the instrumented gateway. For this short chat workload, the FastAPI layer adds on the order of **tens of milliseconds** end-to-end, not hundreds. Optimize the engine/path later, not the gateway plumbing.

## Limitation

n=10, one prompt size, Modal network variance. Overhead p95 looking smaller than p50 is just sample noise on a small n.

## Next step

Stage 3: streaming, TTFT, and disconnect/cancellation behavior.
