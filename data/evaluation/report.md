# Evaluation Report

> Generated: 2026-03-27 22:31 UTC
> Sessions evaluated: 50

## Metric Comparison

| Metric | Value |
|--------|-------|
| consistency_score | 0.9965 |
| context_reuse_rate | 1.0000 |
| token_efficiency | 0.9619 |
| retrieval_precision | 0.0000 |
| latency_impact | baseline_avg: 0.0 | baseline_p50: 0.0 | baseline_p95: 0.0 | memory_avg: 0.0 | memory_p50: 0.0 | memory_p95: 0.0 | overhead_ms: 0.0 |

## Baseline Pipeline

| Metric | Value |
|--------|-------|
| total_tokens | 7018 |
| avg_latency | 0.0000 |

## Memory-Aware Pipeline

| Metric | Value |
|--------|-------|
| consistency_score | 0.9965 |
| context_reuse_rate | 1.0000 |
| token_efficiency | 0.9619 |
| retrieval_precision | 0.0000 |
| latency_impact | baseline_avg: 0.0 | baseline_p50: 0.0 | baseline_p95: 0.0 | memory_avg: 0.0 | memory_p50: 0.0 | memory_p95: 0.0 | overhead_ms: 0.0 |

## Per-Session Details

| Session | Topic | BL Tokens | Mem Tokens | BL Latency (ms) | Mem Latency (ms) |
|---------|-------|-----------|-----------|-----------------|-------------------|
| f10d5ebf... | pix_fees | 166 | 174 | 0.0 | 0.0 |
| 48e7324d... | pix_fees | 90 | 94 | 0.0 | 0.0 |
| 01f39d27... | pix_fees | 106 | 110 | 0.0 | 0.0 |
| 4ed7001b... | pix_fees | 96 | 100 | 0.0 | 0.0 |
| 833ef737... | pix_fees | 126 | 131 | 0.0 | 0.0 |
| c421983f... | pix_fees | 114 | 119 | 0.0 | 0.0 |
| 6eff2554... | pix_fees | 86 | 90 | 0.0 | 0.0 |
| 9664c8d4... | pix_fees | 98 | 103 | 0.0 | 0.0 |
| f405d265... | pix_fees | 94 | 98 | 0.0 | 0.0 |
| 96ef3572... | pix_fees | 92 | 96 | 0.0 | 0.0 |
| 37629a4d... | pix_deadlines | 206 | 214 | 0.0 | 0.0 |
| 90865690... | pix_deadlines | 99 | 103 | 0.0 | 0.0 |
| 72fd5e96... | pix_deadlines | 199 | 207 | 0.0 | 0.0 |
| 80b78251... | pix_deadlines | 81 | 84 | 0.0 | 0.0 |
| 368bdbbe... | pix_deadlines | 84 | 87 | 0.0 | 0.0 |
| aed2d354... | pix_deadlines | 186 | 194 | 0.0 | 0.0 |
| c04377ad... | pix_deadlines | 98 | 102 | 0.0 | 0.0 |
| 4adbd3c6... | pix_deadlines | 107 | 111 | 0.0 | 0.0 |
| da604eaf... | pix_deadlines | 188 | 195 | 0.0 | 0.0 |
| 3c3f8425... | pix_deadlines | 102 | 106 | 0.0 | 0.0 |
| ... | (30 more) | | | | |

## Methodology

- **Consistency Score**: Pairwise cosine similarity of responses grouped by topic
- **Context Reuse Rate**: Fraction of turns where memory was read and influenced response
- **Token Efficiency**: Ratio of baseline tokens / memory tokens (>1.0 = memory saves tokens)
- **Retrieval Precision**: Relevant documents / retrieved documents per query, averaged
- **Latency Impact**: Avg, P50, P95 latency comparison between pipelines
