import csv
import json
from dataclasses import asdict

from .runner import BenchmarkResult


def save_json(results: list[BenchmarkResult], path: str):
    data = [asdict(result) for result in results]

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_csv(results: list[BenchmarkResult], path: str):
    if not results:
        return

    rows = [asdict(result) for result in results]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)
