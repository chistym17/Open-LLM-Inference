from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = None
    temperature: float = 0.0
    model: str | None = None

    @field_validator("messages")
    @classmethod
    def content_not_blank(cls, messages: list[ChatMessage]) -> list[ChatMessage]:
        for message in messages:
            if not message.content.strip():
                raise ValueError("message content must not be blank")
        return messages


class ChatResponse(BaseModel):
    id: str
    model: str
    content: str
    usage: dict[str, Any]
    latency_ms: float
