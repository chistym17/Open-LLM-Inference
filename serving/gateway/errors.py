from __future__ import annotations

from fastapi import HTTPException

from serving.gateway.upstream import UpstreamError


def map_upstream_error(exc: UpstreamError) -> HTTPException:
    if exc.status_code == 504:
        status = 504
        error_class = "upstream_timeout"
    elif exc.status_code in {401, 403} or exc.status_code >= 500:
        status = 502
        error_class = "upstream_error"
    else:
        status = exc.status_code
        error_class = "upstream_error"
    return HTTPException(
        status_code=status,
        detail="upstream request failed",
        headers={"X-Error-Class": error_class},
    )


def request_deadline_error() -> HTTPException:
    return HTTPException(
        status_code=504,
        detail="request deadline exceeded",
        headers={"X-Error-Class": "request_deadline"},
    )
