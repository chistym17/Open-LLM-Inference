"""Modal T4 deployment for Qwen2.5-1.5B-Instruct via vLLM."""

from __future__ import annotations

import json
import subprocess
from typing import Any

import modal

APP_NAME = "open-llm-v2-vllm"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

GPU = "T4"
N_GPU = 1
VLLM_PORT = 8000
MINUTES = 60

GPU_MEMORY_UTILIZATION = "0.8"
MAX_MODEL_LEN = "4096"
DTYPE = "float16"
FAST_BOOT = True

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.11.0",
        "transformers==4.57.6",
        "tokenizers==0.22.2",
    )
    .env({"VLLM_LOGGING_LEVEL": "INFO"})
)

hf_cache = modal.Volume.from_name("open-llm-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("open-llm-vllm-cache", create_if_missing=True)

app = modal.App(APP_NAME)


@app.server(
    image=vllm_image,
    gpu=f"{GPU}:{N_GPU}",
    port=VLLM_PORT,
    startup_timeout=10 * MINUTES,
    scaledown_window=5 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    target_concurrency=16,
    unauthenticated=False,
)
class VLLMServer:
    @modal.enter()
    def start(self) -> None:
        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--dtype",
            DTYPE,
            "--gpu-memory-utilization",
            GPU_MEMORY_UTILIZATION,
            "--max-model-len",
            MAX_MODEL_LEN,
            "--served-model-name",
            MODEL_NAME,
            "--tensor-parallel-size",
            str(N_GPU),
            "--uvicorn-log-level=info",
        ]
        cmd.append("--enforce-eager" if FAST_BOOT else "--no-enforce-eager")
        print("starting:", " ".join(cmd), flush=True)
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()


def _proxy_auth_headers() -> dict[str, str]:
    import os
    from pathlib import Path

    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except ImportError:
        pass

    token = os.environ.get("MODAL_PROXY_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "Set MODAL_PROXY_TOKEN in the environment or .env. "
            "Create one with: modal workspace proxy-tokens create"
        )
    return {"Authorization": f"Bearer {token}"}


@app.local_entrypoint()
async def smoke(prompt: str = "Say hello in one short sentence.") -> None:
    """Spin a warm replica, check /health, then run one chat completion."""
    import asyncio
    import time

    import aiohttp

    headers = _proxy_auth_headers()
    url = await VLLMServer.get_url.aio()
    deadline = time.time() + 10 * MINUTES

    async with aiohttp.ClientSession(base_url=url, headers=headers) as session:
        print(f"waiting for health at {url}")
        while time.time() < deadline:
            try:
                async with session.get(
                    "/health",
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 200:
                        break
                    if resp.status == 503:
                        await asyncio.sleep(2)
                        continue
                    raise RuntimeError(f"health check failed: HTTP {resp.status}")
            except (aiohttp.ClientError, asyncio.TimeoutError):
                await asyncio.sleep(2)
        else:
            raise RuntimeError("timed out waiting for /health")

        print("health ok")
        payload: dict[str, Any] = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
            "temperature": 0,
        }
        async with session.post("/v1/chat/completions", json=payload) as resp:
            body = await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"chat failed: HTTP {resp.status}\n{body}")
            data = json.loads(body)

        message = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        print("response:", message)
        print("usage:", json.dumps(usage))
