from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from serving.gateway.config import settings
from serving.gateway.schemas import ChatRequest, ChatResponse
from serving.gateway.upstream import UpstreamError, VLLMClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = VLLMClient(settings)
    app.state.vllm = client
    try:
        yield
    finally:
        await client.aclose()


app = FastAPI(title="open-llm-gateway", version="0.1.0", lifespan=lifespan)


def _input_size(messages: list[Any]) -> int:
    return sum(len(message.content) for message in messages)


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    client: VLLMClient = request.app.state.vllm
    try:
        status, _ = await client.health()
        upstream_ok = status == 200
    except UpstreamError:
        upstream_ok = False
    return {
        "status": "ok" if upstream_ok else "degraded",
        "gateway": "ok",
        "upstream": "ok" if upstream_ok else "down",
        "model": settings.model_name,
    }


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(body: ChatRequest, request: Request) -> ChatResponse:
    if _input_size(body.messages) > settings.max_input_chars:
        raise HTTPException(
            status_code=413,
            detail=f"input exceeds {settings.max_input_chars} characters",
        )

    max_tokens = body.max_tokens or 64
    if max_tokens < 1 or max_tokens > settings.max_tokens:
        raise HTTPException(
            status_code=400,
            detail=f"max_tokens must be between 1 and {settings.max_tokens}",
        )

    payload = {
        "model": body.model or settings.model_name,
        "messages": [message.model_dump() for message in body.messages],
        "max_tokens": max_tokens,
        "temperature": body.temperature,
    }

    client: VLLMClient = request.app.state.vllm
    t0 = time.perf_counter()
    try:
        data = await client.chat_completions(payload)
    except UpstreamError as exc:
        mapped = 502 if exc.status_code >= 500 else exc.status_code
        if mapped in {401, 403}:
            mapped = 502
        raise HTTPException(status_code=mapped, detail="upstream request failed") from exc
    latency_ms = (time.perf_counter() - t0) * 1000

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="malformed upstream response") from exc

    return ChatResponse(
        id=str(data.get("id", "")),
        model=str(data.get("model", settings.model_name)),
        content=content,
        usage=data.get("usage") or {},
        latency_ms=round(latency_ms, 1),
    )
