# Open LLM Inference Platform

A measured inference system for an open-weight model: how generation uses the GPU, then how a serving path behaves under **load, failure, and cost**.

**Model:** `Qwen/Qwen2.5-1.5B-Instruct`  
**GPU:** NVIDIA T4

**V1** studies the engine with Transformers and vLLM on Colab. **V2** puts vLLM behind a FastAPI gateway on Modal and runs it as a small production service.

The work is the systems behavior: **prefill vs decode**, **KV cache**, **batching**, **admission**, **streaming**, **cancellation**, and **what an idle GPU costs**.

## What was built

```text
client
  → FastAPI gateway (localhost)
      admission cap, timeouts, streaming, metrics
  → vLLM on one Modal T4
```

The **gateway** is the only app-facing surface. The **GPU stays private**.

- Full in-flight cap → **immediate reject**. No queue.
- Client disconnect → **gateway slot freed**.
- After the burst the GPU **scales to zero**. That idle window is **most of the bill**.

## Two parts

### V1 — engine behavior

Prefill cost, KV growth, concurrency, batching, the scheduler, attention, and **FP16 vs 4-bit**. A single-request benchmark hides how a serving engine actually runs.

### V2 — serving behavior

A real request path, then one change at a time: timings, streaming, load, admission, timeouts and cancellation, metrics, and cost. The [readiness review](experiments/v2/09_production_readiness/README.md) states what those runs prove.

## Core results

Numbers are for **this T4 and these workloads**.

### Engine

| | |
| --- | --- |
| **Concurrency is the serving win** | Aggregate output **31 → 1,432 tok/s** from concurrency 1 to 32 (**~46×**). Latency stayed around **2 s**. |
| **Long prompts cost before decode** | Prompt 32 → 1,024 tokens: prefill latency **~6.3×**, extra peak memory **~32×**. |
| **4-bit is a trade** | NF4 cut the footprint **~64%** (2.88 → 1.05 GiB) and made generation **~2.2×** slower than FP16. |

### Serving

| | |
| --- | --- |
| **Gateway overhead is small** | On a ~700 ms chat, FastAPI added about **44 ms** p50. Almost all of the time was the upstream call. |
| **Streaming cuts time-to-first-token** | First token p50 **501 ms** streamed vs **716 ms** for the full body. |
| **Overload is a hard cap** | Cap of **6** in flight. At 16 clients: **6 accepted, 26 rejected** (429). Accepted p95 stayed under the uncapped run. |
| **Cancel has two signals** | The gateway frees its slot and vLLM's **running-request count drops**. The abort counter did not move, so a mid-kernel GPU stop is not claimed. |

### Cost

| | |
| --- | --- |
| **Idle time dominates a short burst** | 25 requests cost **$0.089**. About **83%** was idle until the T4 scaled to zero. |
| **A busy hour is a different price** | Same pace, held for an hour: about **$0.00012** per request, versus **$0.0036** on demand. That hour was not purchased. |

## Decisions

- **Reject when full.** Extra work does not wait in a gateway queue.
- **Two clocks.** The upstream timeout and the overall deadline are separate. A client giving up does not prove the GPU stopped.
- **Live is not ready.** The process can be up while vLLM is down.
- **Bill the whole container.** GPU + CPU + memory + egress, including scale-down idle.

## Where to read

| | |
| --- | --- |
| **Engine measurements** | [V1 report](experiments/10_v1_report/README.md) |
| **What serving runs prove** | [V2 readiness review](experiments/v2/09_production_readiness/README.md) |
| **Phase notes** | `experiments/v2/01` … `09` |
| **How to run the gateway** | [serving/README.md](serving/README.md) |

Raw CSVs and JSON are under `results/`. Serving runs stay local and are not committed.

## Layout

```text
src/llm_platform/     V1 engines and benchmark helpers
experiments/01..10    V1 phase notes
experiments/v2/       V2 serving notes
serving/              Modal vLLM, FastAPI gateway, load and probes
results/              measured rows
```

## Scope

### Ready for

One operator, **one gateway process**, **one T4**, short chats, concurrency up to **6**, excess traffic **rejected**, GPU **off between bursts**.

### Outside this claim

A multi-tenant public API. Caller login on the local gateway. A shared cap across multiple workers. Proof that disconnect interrupts a CUDA kernel already on the GPU.

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

V1 engine scripts need a local NVIDIA GPU and were run on Colab. Read the V1 report before comparing those tables with the Modal serving numbers.
