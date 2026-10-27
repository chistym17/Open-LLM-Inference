# Phase 1 — Basic serving path

## What we did

- Deployed `Qwen/Qwen2.5-1.5B-Instruct` with vLLM on Modal (NVIDIA T4, FP16).
- Kept the engine behind Modal auth (proxy token); no public unauthenticated endpoint.
- Added a local FastAPI gateway that validates input and forwards chat completions upstream.
- Ran a direct Modal smoke, then gateway Stage 1 checks.

## What worked

- Modal `/health` returned 200 after image/tokenizer pins (`vllm==0.11.0`, `transformers==4.57.6`, `tokenizers==0.22.2`).
- Direct Modal chat completion returned 200 with a short reply.
- Direct smoke usage sample: `prompt_tokens=36`, `completion_tokens=10`, `total_tokens=46`.
- Gateway rejects blank messages, oversized prompts (`413`), and out-of-range `max_tokens` (`400`) without crashing.

## What failed or stayed unverified

- First image used unpinned `transformers` and crashed on `Qwen2Tokenizer.all_special_tokens_extended` (fixed by pins above).
- Authenticated endpoint without a proxy token returned `401` (expected).
- Gateway vs direct-vLLM latency split is only a smoke measurement here; Stage 2 will instrument request IDs and repeated samples.
- Cold-start / scale-to-zero cost is not quantified yet (Stage 8).

## Environment

| Item | Value |
| ---- | ----- |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Precision | FP16 |
| GPU | NVIDIA T4 (Modal) |
| Engine | vLLM `0.11.0` |
| Gateway | FastAPI + httpx |
| Config | `gpu_memory_utilization=0.8`, `max_model_len=4096`, scaledown ~5 min |

## Workload

One variable for this phase: introduce a FastAPI gateway in front of an already-working Modal vLLM server.

Smoke prompt: short chat completion, `max_tokens=32`, `temperature=0`.

## Numbers

Raw rows: `results/v2/phase-1/smoke_test.json` (local only, gitignored).

| Check | Expected | Notes |
| ----- | -------- | ----- |
| Direct Modal health | 200 | Verified in `modal run` smoke |
| Direct Modal chat | 200 | 10 completion tokens in sample |
| Gateway health | upstream ok when warm | See local JSON |
| Valid gateway chat | 200 | See local JSON |
| Blank message | 422 | Validation |
| Oversized input | 413 | `GATEWAY_MAX_INPUT_CHARS` |
| Huge max_tokens | 400 | Cap via `GATEWAY_MAX_TOKENS` |
| 3 sequential chats | all 200 | Response isolation smoke |

Sample counts are small (n=1 for direct Modal chat, n=3 sequential gateway chats). Treat as path verification, not a latency benchmark.

## Decision

Keep Modal vLLM private and put FastAPI in front as the only app-facing surface. Enough for Stage 1; do not claim production readiness yet.

## Limitation

`latency_ms` on the gateway today is wall time for the upstream HTTP call (queue + generation + transfer), not pure GPU time.

## Next step

Stage 2: request IDs, structured logs, and repeated direct-vs-gateway latency comparison with p50/p95.
