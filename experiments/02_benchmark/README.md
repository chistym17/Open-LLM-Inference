# Phase 2 — Benchmark Framework

## Objective

Convert the Phase 1 measurement process into a reusable benchmark framework.

The framework separates:

- Workloads
- Inference engines
- Benchmark execution
- Metric calculations
- Result storage

This allows the same benchmark to be used with different inference engines.

## Architecture

```text
Workload
   ↓
Benchmark Runner
   ↓
Inference Engine
   ↓
BenchmarkResult
   ↓
CSV / JSON
