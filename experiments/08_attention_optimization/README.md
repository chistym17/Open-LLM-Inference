# Phase 8: Attention and Inference Optimization

## Objective

Investigate how prompt length affects prefill latency and GPU memory usage when running Qwen2.5-1.5B-Instruct using FP16 and PyTorch SDPA.

## Environment

- GPU: NVIDIA Tesla T4 (15 GB class, compute capability 7.5)
- PyTorch: 2.11.0+cu130
- Transformers: 5.18.0
- Model: Qwen/Qwen2.5-1.5B-Instruct
- Model dtype: FP16
- Attention implementation: SDPA

## Prompt-length experiment

| Prompt tokens | Mean prefill latency (ms) | Extra peak GPU memory (MB) |
| ------------- | ------------------------- | -------------------------- |
| 32            | 58.69                     | 23.2                       |
| 64            | 62.65                     | 47.0                       |
| 128           | 61.86                     | 93.6                       |
| 256           | 70.67                     | 187.3                      |
| 512           | 154.11                    | 370.9                      |
| 1024          | 371.51                    | 741.9                      |

Each measurement used one warmup pass followed by three timed forward passes. All tested outputs contained finite logits.

## Backend investigation

- Flash SDPA was unavailable for the tested operation because the T4 uses compute capability 7.5, while the installed PyTorch Flash SDPA kernel required a newer architecture.
- Memory-efficient and math SDPA calls both returned finite outputs in the isolated test.
- The isolated timing comparison favored the memory-efficient request, but backend dispatch was not conclusively verified. These timings are not treated as a confirmed kernel performance comparison.
- FP16 eager attention produced NaNs in the initial model comparison. Eager FP16 was therefore not used as a correctness reference.

## Findings

Longer prompts increased both prefill latency and peak GPU memory. Latency rose particularly sharply between 256 and 1,024 tokens.

These measurements represent the complete model forward pass, not attention alone. Peak memory includes other allocations, including model activations and the vocabulary-sized logits tensor. The results therefore do not establish the isolated memory complexity or speedup of an attention algorithm.

## Limitations

Only three timed passes were collected per prompt length. Results may be affected by GPU state, runtime variability, and allocation behavior. This experiment did not isolate attention-only memory or establish a statistically robust performance comparison between kernels.

## Conclusion

The experiment provides a baseline for prompt-length-dependent inference cost on a Tesla T4. Further optimization claims require controlled repetitions and more targeted measurements.
