"""Gateway smoke test. Requires a running local gateway. Writes results/v2/phase-1/."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "v2" / "phase-1"
GATEWAY = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8081").rstrip("/")


def _request(method: str, path: str, payload: dict | None = None, timeout: float = 300):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{GATEWAY}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
            return {
                "ok": True,
                "status": resp.status,
                "body": json.loads(body) if body else {},
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            }
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {"raw": raw[:300]}
        return {
            "ok": False,
            "status": exc.code,
            "body": parsed,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
    except urllib.error.URLError as exc:
        return {
            "ok": False,
            "status": 0,
            "body": {"detail": str(exc.reason)},
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    health = _request("GET", "/health", timeout=60)
    rows.append({"case": "health", **health})
    health_body = health.get("body") or {}
    if health.get("status") != 200 or "gateway" not in health_body:
        print(
            "Gateway not reached. Start it first:\n"
            "  uvicorn serving.gateway.app:app --host 127.0.0.1 --port 8081\n"
            f"Current GATEWAY_URL={GATEWAY}",
            file=sys.stderr,
        )
        out_path = OUT / "smoke_test.json"
        out_path.write_text(
            json.dumps(
                {
                    "gateway_url_set": bool(GATEWAY),
                    "error": "gateway_not_reached",
                    "cases": rows,
                },
                indent=2,
            )
            + "\n"
        )
        return 1

    valid = _request(
        "POST",
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
            "max_tokens": 32,
            "temperature": 0,
        },
    )
    rows.append({"case": "valid_request", **valid})

    invalid = _request(
        "POST",
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "   "}], "max_tokens": 16},
    )
    rows.append({"case": "invalid_blank_message", **invalid})

    too_big = _request(
        "POST",
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "x" * 9000}],
            "max_tokens": 16,
        },
    )
    rows.append({"case": "oversized_input", **too_big})

    bad_tokens = _request(
        "POST",
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 99999,
        },
    )
    rows.append({"case": "max_tokens_rejected", **bad_tokens})

    multi = []
    for i in range(3):
        multi.append(
            _request(
                "POST",
                "/v1/chat/completions",
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Reply with exactly: ok-{i}",
                        }
                    ],
                    "max_tokens": 16,
                    "temperature": 0,
                },
            )
        )
    rows.append(
        {
            "case": "sequential_multi",
            "ok": all(item.get("ok") for item in multi),
            "status": [item.get("status") for item in multi],
            "latency_ms": [item.get("latency_ms") for item in multi],
            "bodies": [item.get("body") for item in multi],
        }
    )

    summary = {
        "gateway_url_set": bool(GATEWAY),
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "gpu": "NVIDIA T4 (Modal)",
        "cases": rows,
    }
    out_path = OUT / "smoke_test.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"wrote": str(out_path), "results": [
        {"case": r["case"], "ok": r.get("ok"), "status": r.get("status")}
        for r in rows
    ]}, indent=2))

    required = {
        "health": True,
        "valid_request": True,
        "invalid_blank_message": False,
        "oversized_input": False,
        "max_tokens_rejected": False,
        "sequential_multi": True,
    }
    by_case = {r["case"]: r for r in rows}
    failed = []
    for case, expect_ok in required.items():
        row = by_case[case]
        if bool(row.get("ok")) != expect_ok:
            failed.append(case)
        if case == "oversized_input" and row.get("status") != 413:
            failed.append("oversized_input_status")
        if case == "max_tokens_rejected" and row.get("status") != 400:
            failed.append("max_tokens_status")
        if case == "invalid_blank_message" and row.get("status") not in {400, 422}:
            failed.append("invalid_status")

    if failed:
        print("FAILED:", ", ".join(failed), file=sys.stderr)
        return 1
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
