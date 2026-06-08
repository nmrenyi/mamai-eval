# Judge validation: `gpt-5-mini`

- Rows total: 6853  (judged: 6853)
- Concordant / discordant: 5241 / 1612

## Agreement frames

| Frame | n | Agreement |
|---|---:|---:|
| Human ↔ human (baseline) | 7777 pairs | **77.6%** (κ 0.366) |
| LLM ↔ single human | 14162 pairs | **72.5%** |
| LLM ↔ consensus (concordant) | 5241 rows | **79.0%** |

## Rubber-stamp detector

- Judge met-rate: **70.0%** (physicians: 77.0%)
- Agreement on physician-met rows: **80.7%** (4467 rows)
- Agreement on physician-not-met rows: **69.5%** (774 rows)

## Per-category (concordant agreement)

| Category | n_concordant | Agreement | Human baseline |
|---|---:|---:|---:|
| CHILD_HEALTH | 2463 | **78.2%** | 76.8% |
| MATERNAL | 1777 | **80.6%** | 79.1% |
| NEONATAL | 291 | **82.8%** | 77.6% |
| SEXUAL_AND_REPRODUCTIVE_HEALTH | 710 | **76.6%** | 76.3% |

95% CI on LLM ↔ single human: [71.6%, 73.4%]
95% CI on LLM ↔ consensus:    [77.9%, 80.0%]
