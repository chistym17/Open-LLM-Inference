# Serving (V2)

```text
serving/
  modal/vllm_server.py   # Modal T4 + vLLM
  gateway/               # FastAPI + admission control
  clients/               # smoke + latency + streaming benchmarks
  load/                  # Stage 4/5 load harness
  clients/resilience_probes.py  # Stage 6 timeout/cancel probes
  requirements.txt
experiments/v2/          # phase writeups (committed)
results/v2/              # raw measurements (gitignored)
```

Secrets live in `.env` (gitignored). Copy `.env.example`.

## Stage 1 — serving path

```bash
cd open-llm-inference-platform
source venv/bin/activate
pip install -r serving/requirements.txt

modal deploy serving/modal/vllm_server.py
# .env: VLLM_BASE_URL + MODAL_PROXY_TOKEN

uvicorn serving.gateway.app:app --host 127.0.0.1 --port 8081
python serving/clients/smoke_test.py
```

## Stage 2 — instrumentation

```bash
python -m serving.clients.benchmark_latency --warmup 2 --runs 10 --max-tokens 32
```

## Stage 3 — streaming

```bash
python -m serving.clients.benchmark_streaming --warmup 2 --runs 8 --max-tokens 32
```

## Stage 4 — load harness

```bash
python -m serving.load.harness --mode closed --concurrency 4 --requests 20 --warmup 2
python -m serving.load.harness --mode open --rps 2 --requests 20 --warmup 2
```

## Stage 5 — admission control

Per-process in-flight cap. Immediate **429** when full (no queue). **Locked default: 6.** `GATEWAY_MAX_IN_FLIGHT<=0` disables.

```bash
OUT=results/v2/phase-5

# baseline
GATEWAY_MAX_IN_FLIGHT=0 uvicorn serving.gateway.app:app --host 127.0.0.1 --port 8081
python -m serving.load.harness --mode closed --concurrency 16 --requests 32 --warmup 2 \
  --out-dir "$OUT" --out-name baseline_c16

# controlled (restart gateway; try 4, 6, 8)
GATEWAY_MAX_IN_FLIGHT=6 uvicorn serving.gateway.app:app --host 127.0.0.1 --port 8081
python -m serving.load.harness --mode closed --concurrency 16 --requests 32 --warmup 2 \
  --out-dir "$OUT" --out-name admit6_c16
curl -s http://127.0.0.1:8081/health
```

Fill `experiments/v2/05_admission_control/README.md` Numbers from the JSONs.

## Stage 6 — timeouts and cancellation

Separate knobs: `GATEWAY_UPSTREAM_TIMEOUT_S` (hop) vs `GATEWAY_REQUEST_DEADLINE_S` (overall, `<=0` off). Disconnect closes upstream HTTP and frees the admission slot; that does **not** prove GPU cancel.

```bash
# example: upstream timeout probe (restart gateway with short hop timeout first)
GATEWAY_UPSTREAM_TIMEOUT_S=3 GATEWAY_REQUEST_DEADLINE_S=0 uvicorn serving.gateway.app:app --host 127.0.0.1 --port 8081
python -m serving.clients.resilience_probes --experiment upstream_timeout --max-tokens 256 --out-name upstream_timeout
```

Experiments: `upstream_timeout`, `request_deadline`, `client_disconnect`, `gpu_cancel_proxy`, `slot_recovery`. See `experiments/v2/06_timeouts_cancellation/README.md`.

Do not commit `.env`, tokens, endpoint URLs, or `results/v2/`.
