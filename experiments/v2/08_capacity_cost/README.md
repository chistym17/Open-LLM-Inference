# Phase 8 — Capacity and deployment cost

## What we did

- Ran one on-demand lifecycle of `open-llm-v2-vllm` on a Modal T4: cold start from zero containers, a light warm load, a sustained warm load, then waited until Modal reported zero containers again.
- Priced that interval from the Modal billing report (after minus before), not from the sticker price alone.
- Compared that on-demand burst with a counterfactual hour kept as busy as the sustained run. The hour was not purchased.

Raw file: `results/v2/phase-8/session.json` (gitignored).

## Environment

| Item | Value |
| ---- | ----- |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| GPU | NVIDIA T4 (Modal), one GPU |
| vLLM | 0.11.0, float16 |
| Path | local FastAPI → Modal vLLM |
| T4 list price | $0.59 per GPU-hour |
| CPU list price | $0.0473 per CPU-hour |
| Memory list price | $0.008 per GiB-hour |
| Configured scale-down | 300 s with no requests |
| Timers | cold start is the first `/health/ready` poll until ready. Request latency is the client HTTP round trip. Idle is the last response until `modal container list` is empty. None of these are GPU-kernel time. |

## Workload

One variable across the lifecycle: arrival pattern. Same prompt, `max_tokens` 32, temperature 0.

| Segment | Shape |
| --- | --- |
| cold start | container count was 0, then health until ready |
| light | 5 requests, one at a time |
| sustained | 20 requests, concurrency 4 |
| idle | no further requests until the container was gone |

## How dollars are counted

- **Session dollars** are the Modal report delta for this app over the run: T4 + CPU + memory + network egress. Source: `modal_billing_report`. The uptime estimate was $0.092; the report was $0.089. The report is the number used below.
- **Per successful request, on demand** = session dollars / 25 successes. Cold start and idle are in the numerator.
- **Per output token** = the same session dollars / 525 completion tokens. Prompt tokens are not in the denominator.
- **Active-only** drops the idle share. It is the cost of the seconds spent starting and serving.
- **Always-on counterfactual** = one GPU-hour at $0.59 times 1.0473. The 4.73% is this app's CPU+memory bill divided by its T4 bill before the run. It is not an extra hour of spend.

## Numbers

| Segment | seconds | share of $0.089198 |
| --- | ---: | ---: |
| cold start | 47.1 | $0.0078 |
| light | 30.7 | $0.0051 |
| sustained | 13.9 | $0.0023 |
| idle | 447.1 | $0.0740 |

Bill delta: T4 $0.0849, CPU $0.0018, memory $0.0025, egress $0.0001.

| Load | n ok / fail | p50 (ms) | output tokens | output tok/s |
| --- | ---: | ---: | ---: | ---: |
| light, warm, sequential | 5 / 0 | 927 | 105 | 3.4 |
| sustained, warm, c=4 | 20 / 0 | 1017 | 420 | 30.2 |

| Cost figure | Value |
| --- | ---: |
| Session | $0.0892 |
| Per successful request, including idle | $0.00357 |
| Per output token, including idle | $0.000170 |
| Per successful request, active seconds only | $0.00061 |
| One always-on GPU-hour at this overhead | $0.618 |
| Per request if that hour stayed as busy as c=4 | $0.000119 |

Cold start was 47.1 s from the first ready check. The container was gone 447 s after the last response, longer than the configured 300 s window.

## What worked

- 25/25 requests succeeded. No timeouts.
- The billing report moved during the run and matched the clock: about 518 s of T4 at $0.59 versus 539 s measured, including poll delay.
- Sustained concurrency raised aggregate output from 3.4 tok/s to 30 tok/s. p50 stayed near 1 s.

## What failed or stayed unverified

- Idle until the container disappeared was 447 s, not the configured 300 s. The report still matches the longer clock.
- p95 is not reported. Latency is not the claim of this phase.
- The always-on figure is arithmetic. We did not leave the GPU up for an hour.
- One burst only. A second cold start after this scale-down was not run.

## Decision

For a 25-request burst, idle is about **83%** of the bill ($0.074 of $0.089). On-demand cost is **$0.0036 per success**. The same sustained pace, held for a full hour, would be about **$0.00012 per success**. Scale this T4 to zero between bursts. Keeping it up is the cheaper path only when work keeps arriving.

## Limitation

Single T4, one short prompt, 32 output tokens, concurrency 4. Egress is in the session total and is negligible. Volume storage did not move.

## How to reproduce

From `open-llm-inference-platform`, with the app deployed and no container already running:

```bash
python -m serving.clients.capacity_cost --out-name session
```

## Next step

Stage 9: production-readiness review across the checks from Stages 1–8.
