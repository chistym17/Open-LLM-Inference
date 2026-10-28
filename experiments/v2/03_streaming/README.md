# Phase 3 — Streaming

## What we did

- Extended the gateway chat API with `stream: true` (and `X-Stream: true`).
- Proxied vLLM SSE through FastAPI without buffering the full answer.
- Compared non-stream vs stream (TTFT, total, inter-chunk) plus a disconnect smoke.
- Fixed `httpx` stream open (`timeout` on `build_request`, not `send`).

## What worked

- Stream responses are real SSE: `content-type: text/event-stream`, `gateway_mode=stream`, `saw_data_prefix=true`.
- Client sees incremental content (~9 content chunks per short reply).
- Streaming **TTFT** is clearly lower than non-stream wait-for-full-body.
- Disconnect smoke aborted after the first content chunk.

## What failed or stayed unverified

- Earlier runs returned non-stream JSON (stale process / stream path not hit). Discarded for TTFT claims.
- Disconnect proves **client abort + gateway can stop forwarding**. It does **not** prove vLLM GPU cancel (Stage 6).

## Environment

| Item | Value |
| ---- | ----- |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| GPU | NVIDIA T4 (Modal) |
| Prompt | `Say hello in one short sentence.` |
| Tokens | ~36 prompt / 10 completion |
| max_tokens | 32 |
| Protocol | warmup both modes, 8 interleaved pairs, 1 disconnect abort |
| Clocks | client `perf_counter`; TTFT = first non-empty `delta.content` |

## Numbers

Raw rows: `results/v2/phase-3/streaming_compare.json` (local only). Measured n=8.

| Mode | TTFT p50 (ms) | TTFT p95 (ms) | Total p50 (ms) | Total p95 (ms) | mean content chunks |
| ---- | ------------: | ------------: | -------------: | -------------: | ------------------: |
| Non-stream | 716.0* | 845.2* | **716.0** | **845.2** | 1 |
| Stream | **501.1** | **578.9** | **785.2** | **887.7** | **9** |

\*For non-stream, TTFT is defined as total (full body).

Inter-chunk gaps (stream, 64 gaps): p50 **1.8 ms**, p95 **99.2 ms**, mean **27.9 ms**.

Disconnect smoke: aborted=`true`, chunks=1, TTFT≈514 ms, total≈515 ms.

Warmup (excluded): non-stream first hit ~2.4 s; stream warmups already ~0.7–0.9 s total.

## Decision

Use streaming for interactive UX: first tokens arrive ~**200 ms earlier** (p50 TTFT 501 vs 716) on this short chat. Total generation time stays similar (~0.7–0.8 s). Keep non-stream for simple batch checks.

## Limitation

Short reply (10 completion tokens). Larger outputs will show a bigger TTFT-vs-total gap. GPU cancel on disconnect still unverified.

## Next step

Stage 4: repeatable load harness (concurrency, open/closed loop).
