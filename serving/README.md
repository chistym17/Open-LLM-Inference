# Serving (V2)

```text
serving/
  modal/vllm_server.py   # Modal T4 + vLLM
  gateway/               # FastAPI app-facing service
  clients/               # smoke + latency benchmark
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

# other terminal
python serving/clients/smoke_test.py
```

## Stage 2 — instrumentation

Warm Modal first (avoid cold-start 503), keep uvicorn running, then:

```bash
python -m serving.clients.benchmark_latency --warmup 2 --runs 10 --max-tokens 32
```

Protocol: warmup **both** paths, then **interleaved** measured pairs (direct, gateway, …).  
Writes `results/v2/phase-2/latency_compare.json`. Update `experiments/v2/02_instrumentation/README.md` with the new p50/p95.

Do not commit `.env`, tokens, endpoint URLs, or `results/v2/`.
