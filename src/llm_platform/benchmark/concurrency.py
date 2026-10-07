from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time
from statistics import mean

from .metrics import percentile
from .workloads import Workload


@dataclass
class ConcurrencyResult:
    concurrency: int
    completed_requests: int
    wall_time_s: float
    requests_per_s: float
    total_output_tokens: int
    aggregate_output_tok_s: float
    latency_mean_s: float
    latency_p50_s: float
    latency_p95_s: float
    kv_cache_before: float | None
    kv_cache_after: float | None
    running_before: float | None
    waiting_before: float | None


def run_concurrency_benchmark(
    generate,
    metrics,
    workload: Workload,
    concurrency_levels: list[int],
) -> list[ConcurrencyResult]:

    results = []

    for concurrency in concurrency_levels:
        print(f"\n=== Concurrency {concurrency} ===")

        before = metrics()

        latencies = []
        output_tokens = []

        def run_one():
            start = time.perf_counter()

            _, output_count = generate(
                workload.prompt,
                workload.max_new_tokens,
                workload.temperature,
            )

            latency = time.perf_counter() - start

            return latency, output_count

        start_wall = time.perf_counter()

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(run_one)
                for _ in range(concurrency)
            ]

            for future in as_completed(futures):
                latency, output_count = future.result()
                latencies.append(latency)
                output_tokens.append(output_count)

        wall_time = time.perf_counter() - start_wall

        after = metrics()

        completed = len(latencies)
        total_output = sum(output_tokens)

        result = ConcurrencyResult(
            concurrency=concurrency,
            completed_requests=completed,
            wall_time_s=wall_time,
            requests_per_s=completed / wall_time,
            total_output_tokens=total_output,
            aggregate_output_tok_s=total_output / wall_time,
            latency_mean_s=mean(latencies),
            latency_p50_s=percentile(latencies, 0.50),
            latency_p95_s=percentile(latencies, 0.95),
            kv_cache_before=before["kv_cache_usage"],
            kv_cache_after=after["kv_cache_usage"],
            running_before=before["running_requests"],
            waiting_before=before["waiting_requests"],
        )

        results.append(result)

        print(f"Completed:          {completed}")
        print(f"Wall time:          {wall_time:.3f}s")
        print(f"Requests/sec:       {result.requests_per_s:.2f}")
        print(f"Aggregate tok/s:    {result.aggregate_output_tok_s:.2f}")
        print(f"Mean latency:       {result.latency_mean_s:.3f}s")
        print(f"P50 latency:        {result.latency_p50_s:.3f}s")
        print(f"P95 latency:        {result.latency_p95_s:.3f}s")
        print(f"KV cache after:     {result.kv_cache_after}%")

    return results