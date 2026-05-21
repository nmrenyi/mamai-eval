# Oracle context self-contradictions — mamaretrieval v0.1.0 (top-3 union)

*Filed 2026-05-21, from the Gemma 4 E4B faithfulness run
`configs/config-v0.2.0/results/generator/gemma4-e4b/20260520T094749/`.*

## What this is

During the generator-faithfulness evaluation we found that some answers the
Lynx judge marked `FAIL` are **not model hallucinations** — they are artifacts
of the *retrieved context disagreeing with itself*. A query's oracle context is
its top-3 mamaretrieval chunks concatenated; in these cases two of those chunks
state **conflicting clinical facts about the same thing**. The model's answer
matches one chunk; the judge faults it against the other.

## How they were found

All 145 `FAIL` cases were audited one-by-one against their literal context.
A case is "confirmed" only when **two verbatim passages from the same context
genuinely conflict**. Every quote below was substring-verified against the raw
`context` field — none is paraphrased.

**Result: 6 confirmed of 145 FAILs** (3 further cases were inconclusive and
excluded — alternative drug routes or subset/superset lists, not true
conflicts). The headline faithfulness count should treat these 6 as **false
FAILs**.

This is a **data-quality finding about the guideline corpus / mamaretrieval**,
not about Gemma 4. Several of the conflicts are clinically meaningful and worth
a guideline owner's attention in their own right.

---

## The 6 confirmed cases

### 1. q_00410 — pre-eclampsia monitoring frequency
*"What is the management for mild pre-eclampsia with diastolic BP 90–110 mmHg?"*
Chunks: `e6524db057a225c1`, `0e4ac9d5db4aea09`, `d361a1ef800855ff`

- **A:** "The woman is usually managed as an outpatient and followed up **weekly** at home or at a local clinic"
- **B:** "**see the woman twice a week** to monitor blood pressure, urine (for proteinuria) and fetal condition"
- **Conflict:** weekly follow-up vs twice-weekly.
- **Gemma answered:** "You must follow up with her **weekly**." → matches A; Lynx faulted it against B.

### 2. q_01073 — neonate exposed to maternal genital herpes
*"Immediate treatment protocol for a neonate born to a mother with genital herpes?"*
Chunks: `b4288e1c15f26263`, `3fee281f9689a10b`

- **A:** "Immediately apply one dose of 3% aciclovir (b) eye ointment … **Observe for 5 days**: If the neonate becomes symptomatic: refer … If … asymptomatic: discharge"
- **B:** "Immediately apply one dose of 3% aciclovir (a) eye ointment … **Refer to neonatal care unit for IV aciclovir treatment**." (immediate, unconditional)
- **Conflict:** observe-5-days-then-refer-if-symptomatic vs refer-for-IV-aciclovir-immediately. The context contains two near-identical "10.4.3 Genital herpes" blocks (variant (a) vs (b)) with different protocols.
- **Gemma answered:** "Observe the newborn for **5 days**…" → matches A. **Clinically significant divergence.**

### 3. q_01302 — prolonged active first stage of labour
*"Clinical criteria for diagnosing prolonged labour?"*
Chunks: `49194689634c4a40`, `73bf83eb621d9d10`, `91ec2fbb2606efc8`

- **A:** "Prolonged active first stage (**>5cm dilated**)"
- **B:** "Prolonged active first stage (**>6cm dilated**)"
- **Conflict:** >5 cm vs >6 cm dilation threshold — two near-duplicate "diagnosis of unsatisfactory progress" tables.
- **Gemma answered:** "more than **5 cm dilated** … no cervical change for 4 hours" → matches A.

### 4. q_01618 — misoprostol dosing, 13–22 weeks
*"Misoprostol regimen … for pregnancy expulsion between 13 and 22 weeks?"*
Chunks: `77d5cfad794cc6e7`, `53ccbe2f8fc8e22d`, `a459c5f885a366be`

- **A (protocol table 12.2.2):** "misoprostol sublingually or vaginally: 400 micrograms **every 3 hours** until foetal and placental [expulsion]"
- **B (prose regimen):** "400 micrograms sublingually or intravaginally … **every 4 to 6 hours** until labour starts"
- **Conflict:** every 3 h vs every 4–6 h, same drug, same gestational window.
- **Gemma answered:** "every **3 hours** until the foetus and placenta are expelled" → matches A. **Clinically significant.**

### 5. q_02317 — misoprostol dosing, 13–22 weeks (same conflict)
*"Misoprostol dosage … before 13 weeks versus 13-22 weeks?"*
Chunks: `53ccbe2f8fc8e22d`, `a459c5f885a366be`

- Same prose-vs-table conflict as case 4 (every 3 h vs every 4–6 h).
- **Gemma answered:** "400 micrograms … **every 3 hours** until expulsion starts" → matches A.

### 6. q_02878 — non-absorbable suture removal timing
*"When should … non-absorbable skin sutures be removed?"*
Chunks: `05eedf50f2d9782f`, `65bac50dd027e694`, `1ba0f5da7125bb69`

- **A:** "For non-absorbable sutures: remove the stitches **between the 5 and 8 day**"
- **B:** "Remove skin sutures (if not absorbable) **on D7**"
- **Conflict:** day 5–8 vs day 7 (a third passage adds "often 4–5 days after the surgery").
- **Gemma answered:** "between **Day 5 and Day 8**" → matches A.

---

## Root-cause patterns

1. **Prose-vs-table conflict within one document** (cases 4, 5, 6, partly 3) —
   a guideline's protocol table and its prose regimen give different numbers.
   This is an inconsistency in the **source guideline itself**.
2. **Duplicate-section merging by the chunker** (cases 1, 2, 3) — the context
   concatenates two retrieved chunks covering the *same* guideline section,
   authored with different specifics (e.g. two "10.4.3 Genital herpes" blocks,
   two "diagnosis of unsatisfactory progress" tables).
3. **Recurring topic:** second-trimester misoprostol dosing frequency accounts
   for 2 of the 6.

In all 6, Gemma picked the **table / numeric** value and Lynx scored it against
the **prose** value — so these are contradictory-context artifacts, not
hallucinations.

## Impact

- **On this eval:** 6 of 145 `FAIL`s are false. Minor numerically, but they
  inflate the `contradiction` bucket and Lynx's apparent error rate.
- **Beyond this eval:** cases 2 and 4/5 are clinically meaningful conflicts in
  the guideline corpus. Any RAG system over this corpus can retrieve either
  side and answer differently. This is exactly the failure mode §3.3
  (guideline contradictions) is meant to catch — surfaced here by accident.

## Recommendation

1. **Report upstream** to the mamaretrieval / `mamai-medical-guidelines` owners
   — the 6 query IDs and chunk IDs above pinpoint the conflicting chunks.
2. The prose-vs-table conflicts (cases 4–6) likely need a **guideline owner /
   clinician** to decide which value is authoritative.
3. The duplicate-section merges (cases 1–3) are a **chunking** issue — the same
   section retrieved twice in slightly different editions.
4. When the faithfulness eval is rerun on mamaretrieval v0.2.0 (top-20 union),
   re-audit: a richer pool may or may not still surface these.
