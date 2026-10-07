from statistics import mean


def percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate percentile of empty data.")

    values = sorted(values)

    index = (len(values) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    if lower == upper:
        return values[lower]

    fraction = index - lower

    return values[lower] + (values[upper] - values[lower]) * fraction


def summarize_latencies(latencies: list[float]) -> dict:
    if not latencies:
        raise ValueError("No latency measurements provided.")

    return {
        "latency_mean_s": mean(latencies),
        "latency_p50_s": percentile(latencies, 0.50),
        "latency_p95_s": percentile(latencies, 0.95),
    }


def calculate_output_throughput(
    output_tokens: int,
    latency_s: float,
) -> float:
    if latency_s <= 0:
        raise ValueError("Latency must be greater than zero.")

    return output_tokens / latency_s


def calculate_request_throughput(
    completed_requests: int,
    elapsed_s: float,
) -> float:
    if elapsed_s <= 0:
        raise ValueError("Elapsed time must be greater than zero.")

    return completed_requests / elapsed_s
