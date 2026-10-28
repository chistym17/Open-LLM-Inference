from __future__ import annotations

import os


def build_prompt(target_chars: int) -> str:
    """Build a deterministic prompt near target_chars (not exact token count)."""
    seed = (
        "Write a clear technical explanation. "
        "Keep going with related detail until the reply limit. "
    )
    if target_chars <= len(seed):
        return seed[: max(1, target_chars)]
    repeats = (target_chars // len(seed)) + 1
    return (seed * repeats)[:target_chars]


def chat_payload(
    *,
    prompt: str,
    max_tokens: int,
    stream: bool,
    model: str | None = None,
) -> dict:
    return {
        "model": model or os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }
