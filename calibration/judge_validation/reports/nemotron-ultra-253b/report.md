# Judge validation: `nvidia/Llama-3_1-Nemotron-Ultra-253B-v1`

- Rows total: 6853  (judged: 6852)
- Concordant / discordant: 5241 / 1612

## Agreement frames

| Frame | n | Agreement |
|---|---:|---:|
| Human ↔ human (baseline) | 7777 pairs | **77.6%** (κ 0.366) |
| LLM ↔ single human | 14160 pairs | **78.8%** |
| LLM ↔ consensus (concordant) | 5240 rows | **87.4%** |

## Rubber-stamp detector

- Judge met-rate: **83.3%** (physicians: 77.0%)
- Agreement on physician-met rows: **92.9%** (4466 rows)
- Agreement on physician-not-met rows: **55.3%** (774 rows)

## Per-category (concordant agreement)

| Category | n_concordant | Agreement | Human baseline |
|---|---:|---:|---:|
| CHILD_HEALTH | 2462 | **86.8%** | 76.8% |
| MATERNAL | 1777 | **88.4%** | 79.1% |
| NEONATAL | 291 | **85.2%** | 77.6% |
| SEXUAL_AND_REPRODUCTIVE_HEALTH | 710 | **87.6%** | 76.3% |

95% CI on LLM ↔ single human: [78.0%, 79.6%]
95% CI on LLM ↔ consensus:    [86.4%, 88.2%]
