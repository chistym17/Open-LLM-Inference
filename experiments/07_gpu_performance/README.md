# Phase 7 — GPU Performance Fundamentals

## Objective

Measure GPU activity during LLM inference, quantify warm-up and steady-state behavior, and examine explicit batch-size scaling using Transformers.

## Environment

- GPU: NVIDIA Tesla T4
- VRAM: 15,360 MiB
- Compute capability: 7.5
- PyTorch: 2.11.0+cu130
- CUDA reported by `nvidia-smi`: 13.0
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Precision: FP16

## Experiments and Results

### 1. Idle GPU baseline

Ten samples were collected at one-second intervals.

| Metric             | Mean       |
| ------------------ | ---------- |
| GPU utilization    | 0%         |
| Memory utilization | 0%         |
| GPU memory used    | 3 MiB      |
| GPU memory total   | 15,360 MiB |
| Board power        | 13.161 W   |
| Temperature        | 37.6°C     |

### 2. Model loading and initial inference

- Model loading time: 45.16 seconds
- GPU memory allocated: 2.88 GiB
- Peak GPU memory: 2.89 GiB
- Input prompt: 23 tokens
- Output: 100 tokens per request
- Five-run mean latency: 3.631 seconds
- Mean output throughput: 27.60 tokens/s

### 3. GPU activity during sustained inference

Eight sequential requests generated 200 tokens each. GPU metrics were sampled approximately every 0.5 seconds, producing 153 samples.

| Metric             | Mean      | Median    | Min       | Max       |
| ------------------ | --------- | --------- | --------- | --------- |
| GPU utilization    | 38.34%    | 41%       | 0%        | 56%       |
| Memory utilization | 31.93%    | 31%       | 0%        | 49%       |
| GPU memory used    | 3,143 MiB | 3,143 MiB | 3,143 MiB | 3,143 MiB |
| Board power        | 48.03 W   | 40.02 W   | 28.55 W   | 72.83 W   |
| Temperature        | 62.71°C   | 59°C      | 54°C      | 75°C      |

The first three requests were considerably slower than the final five. The results suggest warm-up or runtime-state effects, but the cause was not isolated.

The original 153 individual samples were not retained in the pasted notebook output. Their aggregate statistics are preserved in `gpu_activity_summary.json`. Future runs of `run.py` save the raw samples to `gpu_samples.csv`.

### 4. Steady-state inference

After three additional warm-up requests, ten requests generated 200 tokens each.

| Metric                              | Result             |
| ----------------------------------- | ------------------ |
| Mean latency                        | 7.315 s            |
| Median latency                      | 7.466 s            |
| Latency standard deviation          | 0.316 s            |
| Mean output throughput              | 27.388 tokens/s    |
| Throughput standard deviation       | 1.201 tokens/s     |
| Throughput coefficient of variation | Approximately 4.4% |

These measurements illustrate why warm-up should be excluded from steady-state benchmarks and why repeated runs are more informative than a single measurement.

### 5. Explicit Transformers batch-size scaling

Each batch contains copies of the same prompt, and each sequence generates 50 tokens. Values are averages from three timed runs after a warm-up run.

| Batch size | Mean batch latency | Aggregate output throughput | Approx. tokens per sequence divided by batch latency | Peak allocated memory |
| ---------- | ------------------ | --------------------------- | ---------------------------------------------------- | --------------------- |
| 1          | 4.075 s            | 12.269 tok/s                | 12.269 tok/s                                         | 2.896 GiB             |
| 2          | 3.066 s            | 32.617 tok/s                | 16.308 tok/s                                         | 2.896 GiB             |
| 4          | 1.904 s            | 105.060 tok/s               | 26.265 tok/s                                         | 2.901 GiB             |
| 8          | 1.939 s            | 206.318 tok/s               | 25.790 tok/s                                         | 2.918 GiB             |

Aggregate throughput increased approximately 16.8 times between batch sizes 1 and 8. Batch completion time was similar at batch sizes 4 and 8, allowing the larger batch to nearly double aggregate throughput.

The approximate per-sequence column is calculated from output tokens divided by batch completion time; it is not an independently measured per-request generation rate.

## Interpretation

1. GPU memory allocation and GPU utilization measure different things.
2. GPU utilization, power, and temperature rose above idle during inference.
3. Initial requests were slower, making warm-up control important.
4. Explicit batching substantially improved aggregate throughput in this Transformers experiment.
5. Batch size is distinct from vLLM request concurrency. vLLM dynamically schedules requests, while this experiment explicitly sets the Transformers batch size.
6. Results are consistent with improved parallelism and weight reuse at larger batches, but they do not prove a specific compute or memory-bandwidth bottleneck.
7. `nvidia-smi` memory utilization is not a direct measurement of achieved GPU memory bandwidth. Detailed bottleneck diagnosis requires more specialized profiling.

## Reproducibility

In a CUDA-enabled environment:

1. Select a GPU runtime.
2. Install dependencies: `pip install torch transformers accelerate pandas`
3. Run `experiments/07_gpu_performance/run.py`.
4. Inspect the generated files under `results/phase-7/`.

The script requires an NVIDIA GPU and access to `nvidia-smi`.

## Status

Phase 7 complete: GPU activity measurement, steady-state benchmarking, and explicit batch-size scaling.
