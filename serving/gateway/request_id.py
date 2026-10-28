from __future__ import annotations

from uuid import uuid4

from starlette.requests import Request


HEADER = "X-Request-ID"


def resolve_request_id(request: Request) -> str:
    incoming = request.headers.get(HEADER, "").strip()
    return incoming or str(uuid4())
