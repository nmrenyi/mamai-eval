# Judge validation: `gpt-5`

- Rows total: 6853  (judged: 6853)
- Concordant / discordant: 5241 / 1612

## Agreement frames

| Frame | n | Agreement |
|---|---:|---:|
| Human ↔ human (baseline) | 7777 pairs | **77.6%** (κ 0.366) |
| LLM ↔ single human | 14162 pairs | **66.7%** |
| LLM ↔ consensus (concordant) | 5241 rows | **71.8%** |

## Rubber-stamp detector

- Judge met-rate: **58.3%** (physicians: 77.0%)
- Agreement on physician-met rows: **70.1%** (4467 rows)
- Agreement on physician-not-met rows: **81.9%** (774 rows)

## Per-category (concordant agreement)

| Category | n_concordant | Agreement | Human baseline |
|---|---:|---:|---:|
| CHILD_HEALTH | 2463 | **71.5%** | 76.8% |
| MATERNAL | 1777 | **72.5%** | 79.1% |
| NEONATAL | 291 | **72.2%** | 77.6% |
| SEXUAL_AND_REPRODUCTIVE_HEALTH | 710 | **71.1%** | 76.3% |

95% CI on LLM ↔ single human: [65.8%, 67.8%]
95% CI on LLM ↔ consensus:    [70.5%, 73.1%]
