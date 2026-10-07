# Phase 1 — Raw Transformers Baseline

## Objective

Establish a raw Hugging Face Transformers inference baseline for
`Qwen/Qwen2.5-1.5B-Instruct` on an NVIDIA T4 GPU.

The baseline measures:

- Input token count
- Output token count
- End-to-end generation latency
- Output token throughput
- Peak GPU memory
- Prefill latency
- Context-length impact

## Setup

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Engine: Hugging Face Transformers
- Precision: FP16
- GPU: NVIDIA T4
- Generation: greedy decoding (`do_sample=False`)
- Maximum new tokens: 100
- Warmup: 1 generation before measurement

## Baseline Results

| Metric | Result |
|---|---:|
| Input tokens | 11 |
| Output tokens | 100 |
| Prefill time | 0.137 s |
| Generation latency | 5.469 s |
| Output throughput | 18.29 tok/s |
| Peak GPU memory | 2.891 GB |

## Context Scaling

Using the same model and 100 generated tokens:

| Input tokens | Latency | Output tok/s | Peak GPU |
|---:|---:|---:|---:|
| 397 | 1.575 s | 17.15 | 2.929 GB |
| 760 | 1.697 s | 15.91 | 2.996 GB |
| 1486 | 1.618 s | 16.68 | 3.214 GB |

## Observations

1. GPU memory increased as the input context became larger.
2. Decode throughput remained in a similar range for these measurements.
3. The short baseline reached 18.29 output tokens/sec.
4. The experiment confirms that context length affects GPU memory usage even when the number of generated tokens is fixed.
5. These measurements are a baseline for later comparison against optimized inference engines such as vLLM.

## Reproduction

The experiment was initially executed in Google Colab on an NVIDIA T4.

The reusable implementation is:

```bash
python experiments/01_baseline/baseline.py