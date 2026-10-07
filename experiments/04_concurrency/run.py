import os

from llm_platform.benchmark.concurrency import (
    run_concurrency_benchmark,
)
from llm_platform.benchmark.results import save_csv, save_json
from llm_platform.benchmark.workloads import DEFAULT_WORKLOAD
from llm_platform.inference.vllm import VLLMEngine


def main():
    engine = VLLMEngine()

    concurrency_levels = [1, 2, 4, 8, 16, 32]

    print("Checking vLLM server...")
    metrics = engine.metrics()

    print(f"KV cache: {metrics['kv_cache_usage']}%")
    print(f"Running:  {metrics['running_requests']}")
    print(f"Waiting:  {metrics['waiting_requests']}")

    results = run_concurrency_benchmark(
        generate=engine.generate,
        metrics=engine.metrics,
        workload=DEFAULT_WORKLOAD,
        concurrency_levels=concurrency_levels,
    )

    os.makedirs("results/phase-4", exist_ok=True)

    save_csv(
        results,
        "results/phase-4/concurrency.csv",
    )

    save_json(
        results,
        "results/phase-4/concurrency.json",
    )

    print("\nResults saved:")
    print("results/phase-4/concurrency.csv")
    print("results/phase-4/concurrency.json")


if __name__ == "__main__":
    main()