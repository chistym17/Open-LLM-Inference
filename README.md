# Open LLM Inference Platform

A measured inference system for an open-weight model: how generation actually uses the GPU, then how a serving path behaves under load, failure, and cost.

Same model throughout: `Qwen/Qwen2.5-1.5B-Instruct` on an NVIDIA T4. V1 studies the engine (Transformers and vLLM on Colab). V2 puts vLLM behind a FastAPI gateway on Modal and treats it like a small production service.

This is not a chat demo. The point is the systems behavior: prefill versus decode, KV cache, batching, admission, streaming, cancellation, and what an idle GPU costs.

## What was built

```text
client
  → FastAPI gateway (localhost)
      admission cap, timeouts, streaming, metrics
  → vLLM on one Modal T4
```

The gateway is the only app-facing surface. The GPU stays private. When the in-flight cap is full, extra requests are rejected immediately. There is no queue. A client disconnect frees the gateway slot. The GPU scales to zero after the burst, and that idle window is most of the bill.

## Two parts

**V1 — engine behavior.** Prefill cost, KV growth, concurrency, batching, the scheduler, attention, and FP16 versus 4-bit. These explain why a single-request benchmark lies about a serving engine.

**V2 — serving behavior.** A real request path, then one change at a time: timings, streaming, load, admission, timeouts and cancellation, metrics, and cost. Each phase has a writeup and raw results. The [readiness review](experiments/v2/09_production_readiness/README.md) says what those runs do and do not prove.

## Core results

Numbers are for this T4 and these workloads. They are not a general leaderboard.

| What it shows | Result |
| --- | --- |
| Concurrent work is why vLLM is fast | Aggregate output **31 → 1,432 tok/s** from concurrency 1 to 32 (~46×). Latency stayed around 2 s. |
| Long prompts are expensive before decode starts | Prompt 32 → 1,024 tokens: prefill latency ~**6.3×**, extra peak memory ~**32×**. |
| Quantization is a trade, not a free win | 4-bit NF4 cut the footprint ~**64%** (2.88 → 1.05 GiB) and made generation ~**2.2×** slower than FP16. |
| A gateway is cheap if you measure it apart from the GPU | On a ~700 ms chat, FastAPI added about **44 ms** p50. Almost all of the time was the upstream call. |
| Streaming helps time-to-first-token | First token p50 **501 ms** streamed versus **716 ms** waiting for the full body. |
| Overload must be explicit | Cap of **6** in flight. At 16 clients: **6 accepted, 26 rejected** with 429. Accepted p95 stayed under the uncapped run. |
| Cancel is two different events | The gateway frees its slot, and vLLM's running-request count drops. vLLM's abort counter did not move, so a mid-kernel GPU stop is not claimed. |
| On-demand GPUs bill the quiet time | A 25-request burst cost **$0.089**. About **83%** of that was idle time until the T4 scaled to zero. Busy for a full hour, the same pace would be about **$0.00012** per request instead of **$0.0036**. |

## Decisions worth knowing

- Reject when full. Do not queue unbound work in the gateway.
- Keep the upstream timeout separate from an overall request deadline. A client giving up does not, by itself, prove the GPU stopped.
- Split liveness from readiness. The process can be up while vLLM is down.
- Count cost from the provider bill (GPU + CPU + memory + egress), and include scale-down idle.

## Where to read

| | |
| --- | --- |
| Engine measurements and caveats | [V1 report](experiments/10_v1_report/README.md) |
| Serving review and its limits | [V2 readiness review](experiments/v2/09_production_readiness/README.md) |
| Per-phase serving notes | `experiments/v2/01` … `09` |
| Gateway and how to run it | [serving/README.md](serving/README.md) |

Raw CSVs and JSON are under `results/`. Serving runs are local artifacts and are not committed.

## Layout

```text
src/llm_platform/     V1 engines and benchmark helpers
experiments/01..10    V1 phase notes
experiments/v2/       V2 serving notes
serving/              Modal vLLM, FastAPI gateway, load and probes
results/              measured rows
```

## Scope

Ready as a single-operator system: one gateway process, one T4, short chats, concurrency up to 6, excess traffic rejected, GPU off between bursts.

Not a multi-tenant public API. The local gateway does not authenticate callers. Admission is per process. Disconnect was shown to drop the request off the engine scheduler, not to interrupt a CUDA kernel already running.

## Run the gateway

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
pip install -r serving/requirements.txt
cp .env.example .env   # Modal URL and proxy token

modal deploy serving/modal/vllm_server.py
uvicorn serving.gateway.app:app --host 127.0.0.1 --port 8081
```

V1 engine scripts need a local NVIDIA GPU and were run on Colab. See the V1 report before comparing those tables with the Modal serving numbers.
