# Experiments

This document records the methodology used for inference experiments.

## Phase 1 — Raw Transformers Baseline

### Goal

Establish a baseline for raw Hugging Face Transformers inference before introducing optimized inference engines.

### Model

`Qwen/Qwen2.5-1.5B-Instruct`

### Hardware

- GPU: NVIDIA T4
- Precision: FP16

### Generation Configuration

- Greedy decoding
- `do_sample=False`
- 100 maximum new tokens
- One warmup generation before measurement

### Measurements

The experiments measure:

- Input token count
- Output token count
- Prefill latency
- End-to-end generation latency
- Output tokens/second
- Peak GPU memory

CUDA synchronization is used around timed GPU operations so that asynchronous CUDA execution does not distort measurements.

Peak GPU memory is measured using PyTorch CUDA memory statistics.

### Phase 1 Results

The short baseline produced:

| Input | Output | Latency | Output tok/s | Peak GPU |
|---:|---:|---:|---:|---:|
| 11 | 100 | 5.469 s | 18.29 | 2.891 GB |

Prefill for the 11-token prompt:

```text
0.137 s