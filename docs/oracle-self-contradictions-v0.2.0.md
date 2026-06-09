# Oracle context self-contradictions — mamaretrieval v0.2.0 (top-20 union)

*Filed 2026-06-09, from the Gemma 4 E4B faithfulness run
`configs/config-v0.2.0/results/generator/gemma4-e4b/20260524T144321/`.*

## What this is

A fraction of Lynx-flagged `FAIL`s are not model hallucinations but artefacts
of the retrieved oracle context disagreeing with itself: two of the merged
mamaretrieval chunks state conflicting clinical facts about the same thing,
the answer matches one, the judge faults it against the other. v0.2.0 uses a
richer oracle (top-20 union, mean 7.45 chunks/query vs ~2.8 on v0.1.0) so
both the count and the shape of these self-contradictions differ from v0.1.0.

## How they were found

All 43 cases that gpt-5 placed in the `contradiction` bucket (of 174 v0.2.0
FAILs) were audited against their literal context. A case is confirmed only
when two verbatim passages from the same context conflict on the same
clinical fact. Every quote below was substring-verified against the raw
`context` field.

**Result: 16 confirmed of 43 contradiction-bucket FAILs** (27 excluded — see
§4). The headline faithfulness count should treat these 16 as **false FAILs**.
This is a data-quality finding about the guideline corpus / chunker, not
about Gemma 4.

---

## The 16 confirmed cases

### 1. q_00459 — ECP vomit re-dose window

*"How should high-dose emergency contraceptive pills be dosed and managed if vomiting occurs?"*

- **A:** "If a woman vomits within **3 hours** of taking the pills, she should take the same dose again."
- **B:** "If the woman vomits within **2 hours** after taking progestin-only or combined ECPs, she should take another dose."
- Conflict: 3 h vs 2 h re-dose window for combined ECPs (the 50-mcg-EE high-dose pills in this context are combined ECPs).
- Gemma answered **2 hours** → matches B.

### 2. q_00474 — adrenaline for allergic shock / anaphylaxis

- **A (Allergic shock):** "inject 1:1000, **0.5 ml** adrenaline … under the skin, **1 time only** (subcutaneous injection)"
- **B (Anaphylaxis):** "Administer adrenaline **0.3–0.5 mg IM** immediately, and repeat every **10 to 15 minutes**, as needed."
- Conflict: SC 0.5 ml once vs IM 0.3–0.5 mg with repeats — for the same clinical entity. **Clinically significant.**
- Gemma answered IM with repeats → matches B.

### 3. q_00560 — active-phase / partograph dilation threshold

- **A:** "The active phase of the first stage of labour starts when the cervix is **4 cm** dilated"
- **B:** "active phase of the first stage of labour (**5 cm or more** cervical dilatation)"
- Conflict: 4 cm vs 5 cm cut-off. Same pattern as v0.1.0 q_01302 (5 vs 6 cm).
- Gemma answered **5 cm** → matches B.

### 4. q_00761 — retained-placenta time threshold

- **A:** "the placenta is retained if it has not been delivered within **one hour** of the delivery of the baby."
- **B:** "Diagnose a prolonged third stage … within **30 minutes** of the birth with active management or within **60 minutes** … with physiological management."
- Conflict: 1-hour-flat vs management-dependent 30/60-min.
- Gemma answered 30/60-min → matches B.

### 5. q_00780 — total breech extraction for cephalic second twin

- **A:** "Total breech extraction should be used **only for a non-cephalic** second twin"
- **B:** "Delivery of a second twin in **cephalic presentation** or transverse lie: version … and allow a total breech extraction"
- Conflict: cephalic excluded vs cephalic included (after version).
- Gemma listed cephalic → matches B.

### 6. q_00846 — Bishop's score thresholds

- **A:** "An unfavourable cervix … **4 or less** … score of **8 or more** [favourable]"
- **B:** "favourable to induction, if the score is **6 or greater**"
- Conflict: ≤4/≥8 vs ≥6 cut-offs.
- Gemma answered ≥6 → matches B.

### 7. q_00919 — antihypertensive threshold in essential hypertension

- **A:** "Treatment is administered if systolic blood pressure is **≥ 160 mmHg** or if diastolic blood pressure is **≥ 110 mmHg**."
- **B (essential / chronic HTN section):** "Antihypertensive drugs may be prescribed if the **diastolic blood pressure exceeds 100 mmHg**"
- Conflict: ≥160/≥110 vs DBP >100. **Clinically significant** (different drug-start triggers).
- Gemma answered DBP >100 → matches B.

### 8. q_01224 — GDM induction-of-labour gestation

- **A (ACOG):** "can wait until **40 weeks**' gestation to consider induction of labor"
- **B (NICE-style):** "Advise women with gestational diabetes to give birth no later than **40+6 weeks**"
- Conflict: 40 vs 40+6 wk.
- Gemma answered **40 weeks** → matches A.

### 9. q_01447 — first-stage FHR recording interval

- **A:** "**once every hour during the latent phase**, **once every 30 minutes during the active phase**"
- **B:** "every **half hour in the first stage of labour**" (i.e. throughout both latent and active)
- Conflict: phase-specific vs flat first-stage frequency (latent specifically).
- Gemma answered every-half-hour-first-stage → matches B.

### 10. q_02218 — neonatal weight-loss threshold

- **A:** "Weight loss of **>7%** in the first week of life is an indication of possible insufficient milk intake … **>10%** should be a cause for concern."
- **B (NICE 1.4.1):** "If a baby loses **more than 10%** of their birth weight … measure their weight again at appropriate intervals"
- Conflict: >7% vs >10% as the action-trigger threshold.
- Gemma answered **>7%** → matches A.

### 11. q_02878 — non-absorbable suture removal *(recurs from v0.1.0)*

- **A:** "For non-absorbable sutures: remove the stitches **between the 5 and 8 day**"
- **B:** "Remove skin sutures (if not absorbable) **on D7**."
- Conflict: D5–8 range vs D7-flat. Same conflict as v0.1.0 case 6.
- Gemma answered **D5–D8** → matches A.

### 12. q_02932 — direction of finger insertion during VE

- **A (Table 2.2):** "gently introduce lubricated fingers in **downward and backward** direction"
- **B (Skill section):** "insert them very gently into the vagina, following the direction of the vagina, **upwards and backwards**"
- Conflict: literally opposite vertical directions for the same step.
- Gemma answered **upwards and backwards** → matches B.

### 13. q_03002 — aspirin secondary prophylaxis duration

- **A:** "acetylsalicylic acid PO: 75 to 150 mg once daily starting at 12 weeks LMP and continuing **until 36 weeks LMP**"
- **B:** "Give 75 mg to every pregnant woman at risk of developing pre-eclampsia from 12 weeks **until delivery**."
- Conflict: stop at 36 wk vs continue until delivery.
- Gemma answered **until the baby is born** → matches B.

### 14. q_03024 — neonatal head position for resuscitation

- **A:** "The baby is placed on its back, on the resuscitaire, with its **head facing downwards**."
- **B:** "Lay the neonate on the back with the head in a **neutral position** … avoid flexion or hyperextension"
- Conflict: "head facing downwards" vs neutral. A even contradicts itself in the following sentences. **Clinically significant.**
- Gemma answered **head facing downwards** → matches A.

### 15. q_03090 — IV ergometrine dose for PPH

- **A (table):** "IM / IV: **0.2 mg**" (rendered as `I M / I V: 0. 2 m g`)
- **B (prose):** "Intravenous ergometrine **0.25–0.5 mg**"
- Conflict: 0.2 mg vs 0.25–0.5 mg for IV ergometrine in PPH. Classic prose-vs-table mismatch.
- Gemma answered IV **0.2 mg** → matches A.

### 16. q_03144 — maternal heart-rate alert threshold

- **A (causes table):** "Heart rate | **>130 /minute**" (high risk)
- **B (LCG pulse table):** "Alert: <60, **≥120**"
- Conflict: >130 vs ≥120 bpm for the alerting threshold.
- Gemma answered **≥120** as high-risk alert → matches B.

---

## Root-cause patterns

1. **Cross-guideline disagreement** (cases 1, 4, 6, 7, 8, 10, 13, 16): two
   different source guidelines (ACOG vs NICE, RCOG vs WHO, two editions of the
   same book) each retrieved in their own chunk, each stating a different
   number. This is the dominant v0.2.0 pattern and is a direct consequence of
   the larger chunk pool — it replaces v0.1.0's "duplicate-section merging".
2. **Prose-vs-table conflict in one source** (cases 11, 15, partly 6) —
   identical pattern to v0.1.0 cases 4–6.
3. **Within-passage contradiction** (case 14) — one chunk says "head facing
   downwards" *and* "neutral position".
4. **Opposite-direction conflict** (case 12) — opposite directions for the
   same examination step.

In 14 of 16 cases Gemma's wording matches a verbatim passage the judge ignored.

## Cross-reference with v0.1.0

| v0.1.0 query | v0.2.0 status |
|---|---|
| q_00410 (pre-eclampsia monitoring) | Dropped from contradiction bucket. |
| q_01073 (neonate HSV) | Dropped. |
| q_01302 (active-stage dilation: >5 vs >6 cm) | Dropped, but the same dilation-threshold conflict reappears as v0.2.0 q_00560 (4 vs 5 cm) — corpus issue still present. |
| q_01618 (misoprostol 13–22 wk) | Dropped. |
| q_02317 (misoprostol 13–22 wk) | Dropped. |
| q_02878 (suture D5–8 vs D7) | **Recurs verbatim** as v0.2.0 case 11. |

Net: 1 recurs, 5 drop, 15 are new. The richer chunk pool retires some old
conflicts (likely reclassified or now offset by a third concordant chunk) and
exposes many more by mixing more independent sources.

## Impact

16 of 174 v0.2.0 FAILs (≈9%) and ≈37% of the contradiction bucket are false
FAILs. Several conflicts (cases 2, 7, 14) are clinically meaningful and worth
a guideline owner's attention. The gpt-oss-120b judge-validation failure
should be re-read with this floor in mind.

## Recommendation

1. Report the 16 query IDs and their passages upstream to the
   mamaretrieval / `mamai-medical-guidelines` owners.
2. The prose-vs-table conflicts (cases 11, 15) and the within-passage
   contradiction (case 14) need a guideline owner to decide which value is
   authoritative.
3. The cross-guideline conflicts (cases 1, 4, 6, 7, 8, 10, 13, 16) are not
   chunker bugs — they reflect different source guidelines disagreeing.
   Faithfulness scoring should distinguish "answer disagrees with context"
   from "context disagrees with itself".
4. Pattern (1) implies the contradiction count will scale with chunk-pool
   size; re-audit on v0.3.0.
