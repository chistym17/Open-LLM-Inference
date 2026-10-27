from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    vllm_base_url: str
    modal_proxy_token: str
    model_name: str
    max_input_chars: int
    max_tokens: int
    upstream_timeout_s: float

    @classmethod
    def from_env(cls) -> Settings:
        base = os.environ.get("VLLM_BASE_URL", "").rstrip("/")
        token = os.environ.get("MODAL_PROXY_TOKEN", "").strip()
        if not base:
            raise RuntimeError("VLLM_BASE_URL is not set")
        if not token:
            raise RuntimeError("MODAL_PROXY_TOKEN is not set")
        return cls(
            vllm_base_url=base,
            modal_proxy_token=token,
            model_name=os.environ.get(
                "MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"
            ),
            max_input_chars=int(os.environ.get("GATEWAY_MAX_INPUT_CHARS", "8000")),
            max_tokens=int(os.environ.get("GATEWAY_MAX_TOKENS", "256")),
            upstream_timeout_s=float(
                os.environ.get("GATEWAY_UPSTREAM_TIMEOUT_S", "120")
            ),
        )


settings = Settings.from_env()
