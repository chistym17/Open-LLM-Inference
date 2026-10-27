# Serving (V2)

```text
serving/
  modal/vllm_server.py   # Modal T4 + vLLM
  gateway/               # FastAPI app-facing service
  clients/               # local smoke / phase runners
  requirements.txt
experiments/v2/          # phase writeups (committed)
results/v2/              # raw measurements (gitignored)
```

Secrets live in `.env` (gitignored). Copy `.env.example`.

## Stage 1

```bash
cd open-llm-inference-platform
source venv/bin/activate
pip install -r serving/requirements.txt

# durable Modal endpoint
modal deploy serving/modal/vllm_server.py

# .env needs:
#   VLLM_BASE_URL=https://...modal.direct
#   MODAL_PROXY_TOKEN=wk-....ws-....

uvicorn serving.gateway.app:app --host 127.0.0.1 --port 8081

# another terminal
python serving/clients/smoke_test.py
```

Do not commit `.env`, tokens, endpoint URLs, or `results/v2/`.
