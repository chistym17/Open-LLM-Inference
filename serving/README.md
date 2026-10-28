# Serving (V2)

```text
serving/
  modal/vllm_server.py   # Modal T4 + vLLM
  gateway/               # FastAPI app-facing service
  clients/               # smoke + latency + streaming benchmarks
  load/                  # Stage 4 load harness
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

Writes `results/v2/phase-4/*.json`. Update `experiments/v2/04_load_harness/README.md`.

Do not commit `.env`, tokens, endpoint URLs, or `results/v2/`.
