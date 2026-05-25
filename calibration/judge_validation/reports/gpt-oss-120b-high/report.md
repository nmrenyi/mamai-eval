# Judge validation: `openai/gpt-oss-120b`

- Rows total: 6853  (judged: 6844)
- Concordant / discordant: 5241 / 1612

## Agreement frames

| Frame | n | Agreement |
|---|---:|---:|
| Human ↔ human (baseline) | 7777 pairs | **77.6%** (κ 0.366) |
| LLM ↔ single human | 14144 pairs | **71.2%** |
| LLM ↔ consensus (concordant) | 5234 rows | **77.5%** |

## Rubber-stamp detector

- Judge met-rate: **68.6%** (physicians: 77.0%)
- Agreement on physician-met rows: **78.9%** (4460 rows)
- Agreement on physician-not-met rows: **69.6%** (774 rows)

## Per-category (concordant agreement)

| Category | n_concordant | Agreement | Human baseline |
|---|---:|---:|---:|
| CHILD_HEALTH | 2459 | **76.8%** | 76.8% |
| MATERNAL | 1775 | **79.0%** | 79.1% |
| NEONATAL | 290 | **77.9%** | 77.6% |
| SEXUAL_AND_REPRODUCTIVE_HEALTH | 710 | **76.2%** | 76.3% |

95% CI on LLM ↔ single human: [70.3%, 72.2%]
95% CI on LLM ↔ consensus:    [76.3%, 78.6%]
