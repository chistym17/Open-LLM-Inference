# Phase 5 — Admission control

## What we did

- Added per-process non-blocking in-flight admission (`serving/gateway/admission.py`).
- Overload → immediate **429** + `Retry-After` + `X-Error-Class: admission_rejected` (no queue).
- Slots acquired after validation, released in `finally` (JSON + SSE).
- A/B under closed-loop c=16: unlimited vs limits **4 / 6 / 8**, plus recovery c=1.
- Raw rows: `results/v2/phase-5/` (gitignored).

## Decisions (locked)

| # | Decision | Choice |
|---|---|---|
| 1 | In-flight limit | **`GATEWAY_MAX_IN_FLIGHT=6`** (measured vs 4/8; `<=0` = unlimited). |
| 2 | Overload policy | Immediate reject (no queue). |
| 3 | HTTP response | 429 + `Retry-After` + `X-Error-Class: admission_rejected`. |
| 4 | Limit scope | Per application process (single uvicorn worker). |
| 5 | Slot lifecycle | Acquire after validation; release in `finally`. |
| 6 | Success criteria | Under overload: rejects > 0; accepted p95 ≤ unlimited baseline; recovery → 0 rejects. |

## Environment

| Item | Value |
| ---- | ----- |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| GPU | NVIDIA T4 (Modal) |
| Path | local FastAPI → Modal vLLM |
| max_tokens | 32 |
| prompt_chars | 80 |
| stream | false |
| load | closed c=16 n=32 (recovery c=1 n=5) |

## Numbers

Source: `results/v2/phase-5/{baseline,admit4,admit6,admit8}_c16.json`, `recovery_c1.json`.

| Config | limit | load | ok | 429 | accepted p50 | accepted p95 | tok/s |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | off | c=16 | 32 | 0 | 1561 | 2172 | 274.6 |
| controlled | 4 | c=16 | 4 | 28 | 2060 | 2076 | 61.5 |
| controlled | 6 | c=16 | 6 | 26 | 1977 | 1980 | 96.8 |
| controlled | 8 | c=16 | 8 | 24 | 2160 | 2163 | 118.2 |
| recovery | 6 | c=1 | 5 | 0 | 1169 | 1231 | 27.2 |

## What worked

- Admission rejects under overload exactly at the configured cap (ok ≈ limit).
- Accepted p95 at limit 6 (**1980 ms**) stays below unlimited baseline p95 (**2172 ms**).
- Recovery: 0×429 after overload.

## What failed / unverified

- First `admit8` artifact was invalid (looked unlimited); re-run fixed it.
- `/health` `in_flight→0` after recovery not saved as a file (only load JSON).
- Not measured: multi-worker / global admission; streaming under admission; light load c=2 reject rate.

## Decision

**Lock default `GATEWAY_MAX_IN_FLIGHT=6`** (`.env`, `.env.example`, config default). Best accepted p95 among 4/6/8; better tok/s than 4; 8 trades worse accepted latency for more throughput.

## Limitation

Single prompt size, JSON-only, one Modal T4 process, short n. Limit is per gateway process only.

## Next step

Stage 6: timeouts, cancellation, and whether disconnect stops upstream GPU work.
