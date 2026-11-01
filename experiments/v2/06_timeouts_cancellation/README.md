# Phase 6 — Timeouts, cancellation, and resilience

## Decisions (locked for measurement)

| # | Decision | Choice |
|---|---|---|
| 1 | Upstream hop timeout | `GATEWAY_UPSTREAM_TIMEOUT_S` (FastAPI → vLLM httpx). Default 120. |
| 2 | Overall request deadline | `GATEWAY_REQUEST_DEADLINE_S` wall clock from chat handler. **`<=0` disables** (default 0 until measured). |
| 3 | Classification | Hop timeout alone → `504` + `upstream_timeout`. Deadline-bound → `504` + `request_deadline`. |
| 4 | Client disconnect | Detect in SSE loop; `aclose()` upstream; release admission in `finally`. |
| 5 | Retries | Gateway **does not** auto-retry chat/completions. |
| 6 | GPU cancel claim | **Not** inferred from gateway alone. `gpu_cancel_proxy` is a latency hint only. |

## What we built

- `serving/gateway/deadline.py` — remaining time / effective hop timeout.
- Per-call httpx timeouts on JSON + stream open.
- Stream loop: disconnect **and** deadline checks; always `upstream.aclose()`.
- `/health` exposes `upstream_timeout_s`, `request_deadline_s`.
- Probe runner: `python -m serving.clients.resilience_probes --experiment …`

## How to measure

Warm Modal. Restart gateway when env changes. One experiment at a time from `open-llm-inference-platform`.

### 1 — Upstream timeout

```bash
GATEWAY_UPSTREAM_TIMEOUT_S=3 GATEWAY_REQUEST_DEADLINE_S=0 uvicorn serving.gateway.app:app --host 127.0.0.1 --port 8081
```

```bash
python -m serving.clients.resilience_probes --experiment upstream_timeout --max-tokens 256 --out-name upstream_timeout
```

### 2 — Overall deadline

```bash
GATEWAY_UPSTREAM_TIMEOUT_S=120 GATEWAY_REQUEST_DEADLINE_S=3 uvicorn serving.gateway.app:app --host 127.0.0.1 --port 8081
```

```bash
python -m serving.clients.resilience_probes --experiment request_deadline --max-tokens 256 --out-name request_deadline
```

### 3 — Client disconnect

```bash
GATEWAY_UPSTREAM_TIMEOUT_S=120 GATEWAY_REQUEST_DEADLINE_S=0 uvicorn serving.gateway.app:app --host 127.0.0.1 --port 8081
```

```bash
python -m serving.clients.resilience_probes --experiment client_disconnect --max-tokens 128 --out-name client_disconnect
```

### 4 — GPU cancel proxy (hint only)

```bash
python -m serving.clients.resilience_probes --experiment gpu_cancel_proxy --max-tokens 256 --out-name gpu_cancel_proxy
```

### 5 — Slot recovery

```bash
python -m serving.clients.resilience_probes --experiment slot_recovery --max-tokens 128 --out-name slot_recovery
```

```bash
curl -s http://127.0.0.1:8081/health
```

## Numbers

Model `Qwen/Qwen2.5-1.5B-Instruct`, Modal T4. Source: `results/v2/phase-6/*.json`.

| Experiment | client result | error_class | in_flight after | pass? | notes |
| --- | --- | --- | --- | --- | --- |
| upstream_timeout | 504 @ 3011 ms | upstream_timeout | 0 | yes | hop timeout 3s |
| request_deadline | 504 @ 3014 ms | request_deadline | 0 | yes | deadline 3s, hop 120s |
| client_disconnect | aborted after 1 chunk @ 391 ms | — | 0 | yes | slot released |
| gpu_cancel_proxy | follow-up ok | — | 0 | yes | followup/baseline **1.88** (hint only) |
| slot_recovery | burst 6 ok / 0×429 | — | 0 | yes | after abort, capacity free |

## Decision

- Keep **`GATEWAY_REQUEST_DEADLINE_S=0`** (off) by default; enable when a product SLA needs an overall wall clock.
- Keep **`GATEWAY_UPSTREAM_TIMEOUT_S=120`** as the hop guard; short values (e.g. 3s) verified to return `504` + `upstream_timeout` and free the slot.
- Client disconnect closes the upstream HTTP stream and frees admission; **do not claim GPU cancel** — proxy ratio 1.88 only suggests the follow-up was not badly stalled.

## Limitation

No vLLM internal cancel API / metrics in this stage. Modal GPU util not scraped. Probe `pass` is a checklist, not a production SLA.

## Next step

Stage 7: observability and operational readiness.
