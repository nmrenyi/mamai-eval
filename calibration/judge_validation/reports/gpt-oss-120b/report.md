# Judge validation: `openai/gpt-oss-120b`

- Rows total: 6853  (judged: 6753)
- Concordant / discordant: 5241 / 1612

## Agreement frames

| Frame | n | Agreement |
|---|---:|---:|
| Human ↔ human (baseline) | 7777 pairs | **77.6%** (κ 0.366) |
| LLM ↔ single human | 13962 pairs | **74.3%** |
| LLM ↔ consensus (concordant) | 5173 rows | **81.6%** |

## Rubber-stamp detector

- Judge met-rate: **73.2%** (physicians: 77.1%)
- Agreement on physician-met rows: **84.0%** (4410 rows)
- Agreement on physician-not-met rows: **67.9%** (763 rows)

## Per-category (concordant agreement)

| Category | n_concordant | Agreement | Human baseline |
|---|---:|---:|---:|
| CHILD_HEALTH | 2432 | **80.0%** | 76.8% |
| MATERNAL | 1750 | **82.7%** | 79.1% |
| NEONATAL | 286 | **80.4%** | 77.6% |
| SEXUAL_AND_REPRODUCTIVE_HEALTH | 705 | **84.8%** | 76.3% |

95% CI on LLM ↔ single human: [73.4%, 75.2%]
95% CI on LLM ↔ consensus:    [80.5%, 82.7%]
