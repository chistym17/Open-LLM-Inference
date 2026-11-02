# Phase 9 — Production-readiness review

## What we did

Reviewed the integrated gateway and Modal vLLM path against the Stage 9 checklist. No new load was run. Each row cites a result file from Phases 1–8. A row with no file stays unverified.

Model `Qwen/Qwen2.5-1.5B-Instruct`, one NVIDIA T4 (Modal), vLLM 0.11.0, float16, local FastAPI in front. Gateway time is the HTTP path, not GPU-kernel time.

## Checklist

| Area | Verdict | Evidence | Limit |
| --- | --- | --- | --- |
| Correctness | Verified for the smoke | `results/v2/phase-1/smoke_test.json` | n=1 direct chat, n=3 gateway chats. Path check, not a quality eval |
| Streaming | Verified for a short reply | `results/v2/phase-3/streaming_compare.json` | 8 pairs, ~10 completion tokens |
| Cancellation | Partial | `results/v2/phase-7/cancel.json` | Engine running-count dropped. Abort counter did not |
| Latency | Verified for a short chat | `results/v2/phase-2/latency_compare.json` | n=10. Gateway add-on is tens of ms, not GPU time |
| Load | Verified inside this grid | `results/v2/phase-4/closed_c1_n10.json`, `closed_c4_n20.json`, `closed_c8_n24.json`, `open_rps2_n20.json` | One prompt size. c=4 p95 was a heavy tail |
| Overload | Verified at cap 6 | `results/v2/phase-5/admit6_c16.json`, `recovery_c1.json` | Per process, JSON only, n=32 |
| Reliability | Partial | `results/v2/phase-6/upstream_timeout.json`, `request_deadline.json`; `results/v2/phase-7/upstream_down.json` | Gateway restart with requests in flight was not run |
| Observability | Verified for counts | `results/v2/phase-7/known_load.json`, `overload.json` | Latency mean did not match p50 on the cold known-load run |
| Cost | Verified for one burst | `results/v2/phase-8/session.json` | One 25-request lifecycle. Always-on hour was not purchased |
| Reproducibility | Verified as docs | Phase READMEs under `experiments/v2/` plus the raw JSON above | Results are gitignored. Another person needs Modal credentials and a T4 |
| Security | Partial | Phase 1 file for 413/400; code for the proxy token and `.gitignore` | No caller auth on the local gateway. No abuse test |

## What the files show

**Correctness.** Phase 1 smoke: direct health 200, direct chat 200 (10 completion tokens), gateway chat 200, blank message 422, oversized input 413, `max_tokens` over the cap 400, three sequential chats all 200.

**Streaming.** Phase 3, n=8. Stream TTFT p50 **501 ms** versus non-stream total p50 **716 ms**. Mean **9** content chunks. Inter-chunk gap p50 **1.8 ms**.

**Cancellation.** Phase 7 cancel: gateway disconnect **+1**, vLLM running **0 → 1 → 0** within 1 s, `finished_reason="abort"` **+0**. Phase 6 slot recovery: after an abort, a burst of 6 succeeded with **0** rejects (`slot_recovery.json`). The request left the engine. The abort counter is not the proof on this build. A CUDA kernel already running was not shown to stop mid-step.

**Latency.** Phase 2, n=10 interleaved pairs. Gateway client p50 **716 ms**, direct vLLM p50 **672 ms**, overhead p50 **+44 ms**. Inside the gateway, `gateway_ms` p50 **711 ms** and `upstream_ms` p50 **711 ms**. The handler itself is about **0.2 ms**.

**Load.** Phase 4, all successes. Closed c=1 n=10: p50 **1127 ms**, p95 **1295 ms**, 27 tok/s. Closed c=4 n=20: p50 **1336 ms**, p95 **9427 ms**, 43 tok/s. Closed c=8 n=24: p50 **1433 ms**, p95 **2200 ms**, **155 tok/s**. Open 2 rps n=20: p50 **1214 ms**, p95 **1291 ms**. The c=4 p95 is a tail, not the typical request.

**Overload.** Phase 5, closed c=16 n=32, cap **6**. Accepted **6**, rejected **26**, accepted p50 **1977 ms**, p95 **1980 ms**, against an unlimited p95 of **2172 ms**. Recovery at c=1: **5/5** ok, **0** rejects. Rejects are immediate 429s. There is no queue. The cap is one gateway process, not a fleet.

**Reliability.** A 3 s hop timeout returned **504** `upstream_timeout` in **3011 ms** and released the slot. A 3 s deadline with a 120 s hop returned **504** `request_deadline` in **3014 ms**. With vLLM pointed at a closed port, `/health/live` stayed **200** and `/health/ready` returned **503**. Defaults stay hop timeout **120 s** and deadline **off**. A gateway crash mid-request was not measured.

**Observability.** Eight sequential successes incremented the gateway ok counter by **8**. Sixteen concurrent requests with the cap at 6 produced **6** oks and **10** rejects, and both counters matched. In-flight returned to **0**.

**Cost.** Phase 8 session, 25/25 ok. Modal report delta **$0.089** (T4 $0.085, CPU $0.0018, memory $0.0025, egress $0.0001). Idle until the container disappeared was **447 s** and about **83%** of that bill. On-demand cost **$0.0036** per success. The same sustained pace held for an hour would be about **$0.00012** per success. That hour was not run.

## Security

- The Modal server is not public. `unauthenticated=False`. The gateway sends `MODAL_PROXY_TOKEN` on the upstream hop.
- `.env` is gitignored. Result files are gitignored.
- Request size is bounded and was smoked: input over `GATEWAY_MAX_INPUT_CHARS` (8000) returns 413. `max_tokens` over `GATEWAY_MAX_TOKENS` (256) returns 400.
- The local gateway does not check caller identity. Binding it beyond localhost would expose the T4. That was not penetration-tested.

## Decision

Ready for one operator, one local gateway process, and one Modal T4 serving `Qwen2.5-1.5B-Instruct`, for short chats (`max_tokens` 32, and one measured 128-token cancel stream), concurrency up to the in-flight cap of 6, with excess calls rejected 429, a dead upstream reported as not-ready, and the GPU scaled to zero between bursts.

Not ready as a multi-user public API. Not ready for more than one gateway worker sharing one cap. Not ready to claim that disconnect aborts a CUDA kernel already on the GPU. Not ready to quote the always-on hourly price as a measured bill.

## What stayed unverified

- Gateway process restart while requests are in flight.
- Caller authentication on the gateway.
- Multi-worker or multi-GPU admission.
- A purchased always-on hour.
- Output quality beyond "a completion came back."

## Next step

V2's serving phases are complete for the scope above. Further work is a new scope: public auth, more than one worker, or a longer always-on cost run.
