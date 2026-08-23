# InternVL3-9B Cross-Family Anchor Validation

Prespecified sample: 200 MedReason MCQ cases; five schema-example labels per case.

| Cue | Accuracy (95% CI) | Follow-cue rate (95% CI) | A/B/C/D/E | Parser fallback |
|---|---:|---:|---:|---:|
| A | 24.00% [18.61, 30.37] | 89.00% [83.91, 92.62] | 178/6/4/5/7 | 0 |
| B | 21.50% [16.37, 27.70] | 98.00% [94.97, 99.22] | 0/196/2/1/1 | 0 |
| C | 22.00% [16.82, 28.24] | 97.50% [94.28, 98.93] | 1/2/195/1/1 | 0 |
| D | 21.00% [15.93, 27.16] | 98.00% [94.97, 99.22] | 0/1/2/196/1 | 0 |
| E | 20.00% [15.05, 26.09] | 98.50% [95.68, 99.49] | 1/0/1/1/197 | 0 |

- Chi-square(16) = 3631.667, p = 0; Cramer's V = 0.953.
- Cases changed by at least one cue: 200/200.
- Cases unchanged across all cues: 0/200.
- Total parser fallbacks: 0/1000.
- Minimum expected contingency-cell count: 36.00.
