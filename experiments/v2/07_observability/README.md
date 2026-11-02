# Phase 7 — Observability

## What we did

- Split health into liveness and readiness. `GET /health/live` does not call vLLM. `GET /health/ready` is 200 only when vLLM `/health` is 200. `GET /health` keeps the old body and adds `live` / `ready`.
- Exposed in-process Prometheus text on `GET /metrics`: request count by status and error class, latency histogram, in-flight gauge, upstream failures, client disconnects, admission rejects.
- Scraped vLLM `/metrics` directly for `vllm:num_requests_running` and `vllm:request_success_total{finished_reason="abort"}`.
- Ran five probes. Raw JSON: `results/v2/phase-7/` (gitignored). One variable per run.

## Decisions (locked)

| # | Decision | Choice |
|---|---|---|
| 1 | Liveness | `GET /health/live` returns 200 and does not call vLLM. |
| 2 | Readiness | `GET /health/ready` is 200 only when vLLM is healthy. At-cap in-flight is still ready. |
| 3 | Combined health | `GET /health` keeps the old shape and adds `live` / `ready`. |
| 4 | Gateway metrics | Prometheus text on `GET /metrics`. No Prometheus server, no Grafana. |
| 5 | Engine cancel signal | Running-request gauge. The abort finish-reason counter did not move on this build. |

## Environment

| Item | Value |
| ---- | ----- |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| GPU | NVIDIA T4 (Modal) |
| vLLM | 0.11.0 |
| Gateway | 0.7.0, `prometheus-client` 0.26.0 |
| Precision | float16 |
| Path | local FastAPI → Modal vLLM |
| Gateway timer | chat handler entry to response (histogram `gateway_request_latency_ms`) |
| Client timer | local HTTP round trip. Not GPU time. |

## Workload

| Experiment | Variable | Load |
| --- | --- | --- |
| idle | no chat | health + both metric scrapes |
| known_load | 8 sequential JSON chats | max_tokens 32 |
| overload | 16 concurrent JSON chats | max_tokens 32, in-flight cap 6 |
| upstream_down | `VLLM_BASE_URL` pointed at a closed port | health only |
| cancel | client closes after the first content chunk | 1 stream, max_tokens = min_tokens 128 |

## Numbers

Source files: `results/v2/phase-7/{idle,known_load,overload,upstream_down,cancel}.json`.

| Experiment | n | result | pass? |
| --- | ---: | --- | --- |
| idle | 0 chats | live 200, ready 200, gateway and vLLM `/metrics` both 200. running 0, abort 0 | yes |
| known_load | 8 | gateway ok count **8** = client ok **8**. client p50 **820 ms**. gateway mean **4178 ms** | count yes, mean no |
| overload | 16 | ok **6** = success counter 6. 429 **10** = admission counter 10. in-flight after **0** | yes |
| upstream_down | 0 chats | live **200**, ready **503** | yes |
| cancel | 1 stream | gateway disconnect **+1**. vLLM running **0 → 1 → 0** within the 1 s scrape. abort counter **+0** | gauge yes, abort counter no |

The known-load mean is not the typical request. The first two chats were still cold (gateway **17.8 s** and **10.9 s**). The other six were about **0.7–0.9 s**, which is why p50 is 820 ms.

## What worked

- Liveness stays 200 when vLLM is unreachable. Readiness returns 503.
- Under a known load, the gateway success counter matches the client (8/8).
- Under overload, accepted and 429 counts match the counters, and the in-flight gauge returns to 0.
- On client disconnect, the gateway records one disconnect (metric status 499). vLLM's running-request gauge shows the request arrive (1) and leave (0).

## What failed or stayed unverified

- `vllm:request_success_total{finished_reason="abort"}` stayed 0 after a disconnect that cleared the running gauge. That counter is not evidence of cancellation on vLLM 0.11.0 here.
- Known-load latency mean does not match client p50, because two cold requests dominate the mean. p95 was not the claim and is not reported.
- The running gauge shows the request left the engine. It does not prove a CUDA kernel already on the GPU was interrupted mid-step.
- No Prometheus server, no dashboard. Metrics are per gateway process.

## Decision

Use `/health/live` and `/health/ready` as the two health checks. Trust the gateway counters for request count, 429s, and disconnects. For engine cancellation, use `vllm:num_requests_running` (observed 0 → 1 → 0). Do not use the abort finish-reason counter on this build.

## Limitation

Single process, one T4, short JSON load plus one forced 128-token stream. Gateway latency is gateway-to-client time, not GPU execution time.

## How to reproduce

From `open-llm-inference-platform`, Modal warm. Restart the gateway after code or env changes.

```bash
uvicorn serving.gateway.app:app --host 127.0.0.1 --port 8081
```

```bash
python -m serving.clients.observability_probes --experiment idle --out-name idle
```

```bash
python -m serving.clients.observability_probes --experiment known_load --requests 8 --max-tokens 32 --out-name known_load
```

```bash
python -m serving.clients.observability_probes --experiment overload --concurrency 16 --max-tokens 32 --out-name overload
```

```bash
VLLM_BASE_URL=http://127.0.0.1:9 uvicorn serving.gateway.app:app --host 127.0.0.1 --port 8081
```

```bash
python -m serving.clients.observability_probes --experiment upstream_down --out-name upstream_down
```

```bash
uvicorn serving.gateway.app:app --host 127.0.0.1 --port 8081
```

```bash
python -m serving.clients.observability_probes --experiment cancel --max-tokens 128 --out-name cancel
```

## Next step

Stage 8: capacity and deployment cost.
