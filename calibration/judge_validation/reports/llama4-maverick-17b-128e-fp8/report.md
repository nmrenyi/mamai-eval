# Judge validation: `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`

- Rows total: 6853  (judged: 6853)
- Concordant / discordant: 5241 / 1612

## Agreement frames

| Frame | n | Agreement |
|---|---:|---:|
| Human ↔ human (baseline) | 7777 pairs | **77.6%** (κ 0.366) |
| LLM ↔ single human | 14162 pairs | **79.8%** |
| LLM ↔ consensus (concordant) | 5241 rows | **88.7%** |

## Rubber-stamp detector

- Judge met-rate: **88.8%** (physicians: 77.0%)
- Agreement on physician-met rows: **96.3%** (4467 rows)
- Agreement on physician-not-met rows: **44.6%** (774 rows)

## Per-category (concordant agreement)

| Category | n_concordant | Agreement | Human baseline |
|---|---:|---:|---:|
| CHILD_HEALTH | 2463 | **88.5%** | 76.8% |
| MATERNAL | 1777 | **88.1%** | 79.1% |
| NEONATAL | 291 | **89.7%** | 77.6% |
| SEXUAL_AND_REPRODUCTIVE_HEALTH | 710 | **90.4%** | 76.3% |

95% CI on LLM ↔ single human: [79.0%, 80.6%]
95% CI on LLM ↔ consensus:    [87.8%, 89.5%]
