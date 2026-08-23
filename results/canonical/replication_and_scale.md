# Canonical-order replication and model scale

Accuracy is reported on balanced 600-case sets. Intervals are Wilson 95% confidence intervals.

| Study | Submitted | Comparative | Cue-free | Submitted to Cue-free |
|---|---:|---:|---:|---:|
| Qwen2.5-VL-3B primary | 20.17% | 24.67% | 28.50% | +8.33 pp |
| Qwen2.5-VL-3B same-source holdout | 21.00% | 24.17% | 29.33% | +8.33 pp |
| Qwen2.5-VL-7B primary | 25.50% | 31.33% | 35.17% | +9.67 pp |

The same prompt ordering was observed on the non-overlapping holdout and with the larger Qwen checkpoint. These are canonical-order comparisons; the full five-way cyclic audit was run only for Qwen2.5-VL-3B.
