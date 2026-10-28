from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

import httpx

from serving.load.request import run_chat

RequestFactory = Callable[[int, str], dict[str, Any]]


async def _one(
    client: httpx.AsyncClient,
    gateway: str,
    payload: dict[str, Any],
    timeout_s: float,
    index: int,
    phase: str,
) -> dict[str, Any]:
    return await run_chat(
        client,
        gateway,
        payload,
        timeout_s=timeout_s,
        request_index=index,
        phase=phase,
    )


async def run_warmup(
    client: httpx.AsyncClient,
    gateway: str,
    make_payload: RequestFactory,
    *,
    count: int,
    timeout_s: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(count):
        row = await _one(
            client, gateway, make_payload(i, "warmup"), timeout_s, i, "warmup"
        )
        rows.append(row)
        if not row["ok"]:
            raise RuntimeError(f"warmup {i} failed: {row}")
    return rows


async def run_closed_loop(
    client: httpx.AsyncClient,
    gateway: str,
    make_payload: RequestFactory,
    *,
    concurrency: int,
    requests: int,
    timeout_s: float,
) -> list[dict[str, Any]]:
    """Workers pull the next request only after finishing the previous one."""
    queue: asyncio.Queue[int | None] = asyncio.Queue()
    for i in range(requests):
        queue.put_nowait(i)
    for _ in range(concurrency):
        queue.put_nowait(None)

    rows: list[dict[str, Any]] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            index = await queue.get()
            if index is None:
                return
            row = await _one(
                client,
                gateway,
                make_payload(index, "measured"),
                timeout_s,
                index,
                "measured",
            )
            async with lock:
                rows.append(row)

    t0 = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(concurrency)])
    wall_s = time.perf_counter() - t0
    for row in rows:
        row["run_wall_s"] = round(wall_s, 3)
    return rows


async def run_open_loop(
    client: httpx.AsyncClient,
    gateway: str,
    make_payload: RequestFactory,
    *,
    rps: float,
    requests: int,
    timeout_s: float,
) -> list[dict[str, Any]]:
    """Schedule request starts at a fixed rate, independent of completions."""
    if rps <= 0:
        raise ValueError("rps must be > 0")
    interval = 1.0 / rps
    tasks: list[asyncio.Task[dict[str, Any]]] = []
    start = time.perf_counter()

    for i in range(requests):
        target = start + i * interval
        delay = target - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)

        launch = time.perf_counter()
        slip_ms = round((launch - (start + i * interval)) * 1000, 1)

        async def _scheduled(index: int = i, slip: float = slip_ms) -> dict[str, Any]:
            row = await _one(
                client,
                gateway,
                make_payload(index, "measured"),
                timeout_s,
                index,
                "measured",
            )
            row["schedule_slip_ms"] = slip
            return row

        tasks.append(asyncio.create_task(_scheduled()))

    rows = await asyncio.gather(*tasks)
    wall_s = time.perf_counter() - start
    out = list(rows)
    for row in out:
        row["run_wall_s"] = round(wall_s, 3)
    return out
