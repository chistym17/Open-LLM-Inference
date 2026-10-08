# Phase 6 — Scheduler and Resource Pressure

## Objective

Observe how vLLM handles a large number of concurrent requests under
substantial context/KV-cache demand.

The goal was not to implement a scheduler, but to study the behavior
of a production inference scheduler under resource pressure.

## Workload

- Model: Qwen/Qwen2.5-1.5B-Instruct
- Concurrency: 128
- Prompt length: ~3,000 tokens/request
- Maximum generation: 100 tokens/request
- Sampling interval: 50 ms
- vLLM GPU memory utilization: 0.8
- KV cache capacity: 269,968 tokens
- KV block size: 16 tokens
- Theoretical max concurrency at 4096 tokens: 65.91 requests

## Results

| Metric | Result |
|---|---:|
| Requests | 128 |
| Successful | 128 |
| Failed | 0 |
| Prompt tokens | 384,000 |
| Output tokens | 12,800 |
| Wall time | 32.341 s |
| Request throughput | 3.96 req/s |
| Output throughput | 395.79 tok/s |
| Mean latency | 30.901 s |
| P50 latency | 31.136 s |
| P95 latency | 31.688 s |
| P99 latency | 31.710 s |
| Preemptions | 0 |

## vLLM Request Lifecycle Metrics

The vLLM Prometheus metrics provide more useful lifecycle information
than instantaneous gauges alone.

Across the completed requests:

- Queue time remained extremely small.
- Aggregate inference time was approximately 3595 s.
- Aggregate prefill time was approximately 79 s.
- Aggregate decode time was approximately 3516 s.
- No request preemptions occurred.

The decode phase therefore dominated the total inference time for this
workload.

## Scheduler Observations

The workload submitted 128 requests simultaneously with approximately
3,100 tokens of total requested sequence length per request.

vLLM completed all requests successfully rather than requiring all
128 sequences to remain resident in the KV cache simultaneously.

This demonstrates an important distinction:

The theoretical KV-cache concurrency figure describes how many requests
could simultaneously occupy the cache at a specified maximum sequence
length. It does not represent the maximum number of requests that can
be submitted to the server.

Requests can be admitted and processed dynamically as resources become
available.

## Metric Sampling Limitation

The benchmark sampled:

- `vllm:num_requests_running`
- `vllm:num_requests_waiting`
- `vllm:num_requests_waiting_by_reason`
- `vllm:kv_cache_usage_perc`

at 50 ms intervals.

The sampled instantaneous gauges did not capture meaningful non-zero
values during this run. This should not be interpreted as evidence that
the scheduler performed no batching or resource management.

The completed-request histograms provide stronger evidence about the
actual request lifecycle. In particular, queue, prefill, inference,
decode, and preemption metrics were populated after the workload.

This is an important observability lesson: instantaneous gauges and
request-level aggregate histograms answer different questions.

## Key Takeaways

1. vLLM dynamically manages concurrent requests rather than requiring
   every submitted request to remain resident simultaneously.

2. KV-cache capacity limits simultaneous active state, not total
   workload size.

3. High concurrency can substantially increase end-to-end latency even
   when the server successfully completes every request.

4. Decode dominated this workload, accounting for the vast majority of
   aggregate inference time.

5. No preemption occurred under this configuration.

6. Production scheduler behavior is best studied using both
   instantaneous scheduler gauges and completed-request lifecycle
   metrics.

## Status

Phase 6 complete.

The experiment focused on observing vLLM's production scheduler and
resource management rather than implementing a toy scheduler.