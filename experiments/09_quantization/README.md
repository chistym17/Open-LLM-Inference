# Phase 9: 4-bit Quantization

## Objective

Compare FP16 inference with 4-bit NF4 quantization for Qwen2.5-1.5B-Instruct on an NVIDIA Tesla T4.

## Environment

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- GPU: NVIDIA Tesla T4
- FP16 baseline: PyTorch SDPA
- Quantized configuration: bitsandbytes NF4, double quantization enabled, FP16 compute
- Decoding: greedy, maximum 64 new tokens
- Generation timing: three measured runs after warmup

## Results

| Metric                    | FP16      | 4-bit NF4 |
| ------------------------- | --------- | --------- |
| Model footprint           | 2.875 GiB | 1.045 GiB |
| Peak allocated GPU memory | 2.890 GiB | 1.089 GiB |
| Mean generation latency   | 2.291 s   | 5.136 s   |

### Memory reduction

- Model footprint decreased by 63.7%.
- Peak allocated GPU memory decreased by approximately 62.3%.
- The quantized model used approximately 1.83 GiB less memory for its model footprint.

### Generation performance

The 4-bit configuration took approximately 2.24 times as long as FP16 in this experiment. Quantization reduced memory consumption but did not improve generation speed on this setup.

### Output quality

Both configurations generated coherent introductory explanations with similar structure. This was a qualitative smoke test, not a comprehensive quality evaluation.

## Limitations

Only three timed runs were measured per configuration. Latency includes prompt processing and token generation, rather than measuring decode throughput independently. Results may vary with GPU state, software versions, prompt length, and quantization kernels.

## Conclusion

4-bit NF4 is useful when reducing model memory is the priority, but it was slower than FP16 in this test. For this model and GPU, FP16 is preferable when latency matters more than memory footprint; 4-bit NF4 is a viable alternative when the reduced memory requirement is more important.

These conclusions apply to the tested configuration and should not be generalized to every GPU or model.
