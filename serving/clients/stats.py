from __future__ import annotations


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "n": float(len(values)),
        "mean_ms": round(sum(values) / len(values), 1),
        "p50_ms": round(percentile(values, 50), 1),
        "p95_ms": round(percentile(values, 95), 1),
        "min_ms": round(min(values), 1),
        "max_ms": round(max(values), 1),
    }
