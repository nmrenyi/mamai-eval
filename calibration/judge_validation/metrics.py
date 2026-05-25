"""Pure metric functions for judge validation.

All functions are pure: no I/O, no API calls. They take rows (loaded from
obgyn_meta_eval.jsonl, tagged with `_orig_idx`) and, where applicable, a dict
of judge verdicts keyed by ORIGINAL row index (a sparse mapping is fine —
unjudged rows are silently skipped).

Glossary
--------
- A row = one (prompt, completion, single rubric criterion) triple plus
  physician `binary_labels` (one bool per physician on that row).
- Concordant row = all physicians on the row gave the same verdict.
- Discordant row = physicians disagreed.
- Human-human inter-rater agreement = pairwise, pooled across all rows
  with >=2 physicians. Cohen-style kappa at the pair level uses the global
  physician met-rate as the chance base rate.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path


# ── Loading ──────────────────────────────────────────────────────────────────


def load_rows(jsonl_path: Path | str) -> list[dict]:
    """Load all rows from the calibration JSONL, tagged with `_orig_idx`."""
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                obj = json.loads(line)
                obj["_orig_idx"] = i
                rows.append(obj)
    return rows


# ── Concordance split ────────────────────────────────────────────────────────


def split_concordant(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split rows into (concordant, discordant) by physician unanimity.

    Concordant = single distinct value across `binary_labels`.
    """
    concordant, discordant = [], []
    for r in rows:
        if len(set(r["binary_labels"])) == 1:
            concordant.append(r)
        else:
            discordant.append(r)
    return concordant, discordant


# ── Human-human baseline ─────────────────────────────────────────────────────


def human_human_agreement(rows: list[dict]) -> dict:
    """Pairwise pooled physician agreement (the ceiling for any judge).

    For each row with >=2 physicians, every unordered pair contributes one
    comparison. Cohen-style kappa is computed at the pair level using the
    global physician met-rate as the chance base rate. (Fleiss kappa would
    require a fixed number of raters per row; our calibration has rotating
    physician panels of size 2/3/4, so pairwise pooled is the natural
    choice.)
    """
    pairs_eq = 0
    pairs_tot = 0
    all_labels: list[bool] = []
    for r in rows:
        labels = r["binary_labels"]
        all_labels.extend(labels)
        for a, b in combinations(labels, 2):
            pairs_tot += 1
            if a == b:
                pairs_eq += 1
    if pairs_tot == 0:
        return {"pairs": 0, "agreement": None, "kappa": None}

    po = pairs_eq / pairs_tot
    p = sum(all_labels) / len(all_labels) if all_labels else 0.0
    pe = p * p + (1 - p) * (1 - p)
    kappa = (po - pe) / (1 - pe) if pe < 1.0 else 0.0
    return {
        "pairs": pairs_tot,
        "agreement": round(po, 4),
        "physician_met_rate": round(p, 4),
        "chance_agreement": round(pe, 4),
        "kappa": round(kappa, 4),
    }


# ── Judge vs single human (apples-to-apples with human-human) ────────────────


def llm_vs_single_human(rows: list[dict], verdicts: dict[int, bool]) -> dict:
    """Compare judge verdict against EACH individual physician label.

    Pairs are pooled: every (row, physician) with a verdict contributes one.
    This is the apples-to-apples match to `human_human_agreement` — both are
    single-rater vs single-rater.
    """
    pairs_eq = 0
    pairs_tot = 0
    for r in rows:
        idx = r["_orig_idx"]
        if idx not in verdicts:
            continue
        j = verdicts[idx]
        for h in r["binary_labels"]:
            pairs_tot += 1
            if j == h:
                pairs_eq += 1
    if pairs_tot == 0:
        return {"pairs": 0, "agreement": None}
    return {
        "pairs": pairs_tot,
        "agreement": round(pairs_eq / pairs_tot, 4),
    }


# ── Judge vs physician consensus (concordant rows only) ──────────────────────


def llm_vs_consensus(concordant_rows: list[dict], verdicts: dict[int, bool]) -> dict:
    """Judge vs unanimous physician verdict on concordant rows.

    Also reports per-class agreement (the rubber-stamp detector): a judge
    that always says met will hit high overall agreement on the imbalanced
    (~85% met) concordant subset, but `agreement_on_not_met_rows` will be
    near 0 and `judge_met_rate` will be near 1.
    """
    n = 0
    n_match = 0
    judge_met = 0
    met_n = 0
    met_match = 0
    notmet_n = 0
    notmet_match = 0

    for r in concordant_rows:
        idx = r["_orig_idx"]
        if idx not in verdicts:
            continue
        consensus = r["binary_labels"][0]  # unanimous on concordant rows
        j = verdicts[idx]
        n += 1
        if j:
            judge_met += 1
        if j == consensus:
            n_match += 1
        if consensus:
            met_n += 1
            if j == consensus:
                met_match += 1
        else:
            notmet_n += 1
            if j == consensus:
                notmet_match += 1

    if n == 0:
        return {"n": 0, "agreement": None}
    return {
        "n": n,
        "agreement": round(n_match / n, 4),
        "judge_met_rate": round(judge_met / n, 4),
        "n_met_rows": met_n,
        "n_not_met_rows": notmet_n,
        "agreement_on_met_rows": round(met_match / met_n, 4) if met_n else None,
        "agreement_on_not_met_rows": round(notmet_match / notmet_n, 4) if notmet_n else None,
    }


# ── Judge prediction distribution (top-level rubber-stamp summary) ───────────


def prediction_distribution(rows: list[dict], verdicts: dict[int, bool]) -> dict:
    """Judge met-rate vs physician met-rate across all judged rows.

    Read alongside `llm_vs_consensus.agreement_on_not_met_rows` to detect
    rubber-stamp judges.
    """
    if not verdicts:
        return {"judge_met_rate": None, "physician_met_rate": None}
    judge_met = sum(1 for v in verdicts.values() if v)
    judge_total = len(verdicts)
    all_phys: list[bool] = []
    for r in rows:
        if r["_orig_idx"] in verdicts:
            all_phys.extend(r["binary_labels"])
    return {
        "judge_met_rate": round(judge_met / judge_total, 4),
        "physician_met_rate": (
            round(sum(all_phys) / len(all_phys), 4) if all_phys else None
        ),
        "n_judge_verdicts": judge_total,
        "n_physician_labels": len(all_phys),
    }


# ── Per-category breakdown ───────────────────────────────────────────────────


def per_category(rows: list[dict], verdicts: dict[int, bool], fn) -> dict:
    """Apply `fn(rows_subset, verdicts_subset)` per `mamabench_obgyn_category`.

    `verdicts_subset` is restricted to rows in that category.
    """
    cats: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        cats[r["mamabench_obgyn_category"]].append(r)
    out = {}
    for cat, rs in cats.items():
        idxs = {r["_orig_idx"] for r in rs}
        v_sub = {i: v for i, v in verdicts.items() if i in idxs}
        out[cat] = fn(rs, v_sub)
    return out


# ── Bootstrap CI ─────────────────────────────────────────────────────────────


def bootstrap_ci(
    rows: list[dict],
    verdicts: dict[int, bool],
    fn,
    key: str = "agreement",
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Row-resample bootstrap CI for any metric returning a dict with `key`.

    Verdicts stay keyed by original index — resampled rows keep their
    `_orig_idx` so the same verdicts dict works without re-keying.
    """
    rng = random.Random(seed)
    n = len(rows)
    if n == 0:
        return (math.nan, math.nan)
    vals: list[float] = []
    for _ in range(n_resamples):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        result = fn(sample, verdicts)
        v = result.get(key)
        if v is not None:
            vals.append(v)
    if not vals:
        return (math.nan, math.nan)
    vals.sort()
    lo = vals[int((1 - ci) / 2 * len(vals))]
    hi = vals[int((1 + ci) / 2 * len(vals))]
    return (round(lo, 4), round(hi, 4))


# ── Full report ──────────────────────────────────────────────────────────────


def full_report(
    rows: list[dict],
    verdicts: dict[int, bool],
    judge_model: str = "?",
    bootstrap: bool = False,
    n_resamples: int = 1000,
) -> dict:
    """Compute the full metric set for one candidate judge."""
    concordant, discordant = split_concordant(rows)

    report = {
        "judge_model": judge_model,
        "n_rows_total": len(rows),
        "n_rows_judged": sum(1 for r in rows if r["_orig_idx"] in verdicts),
        "n_concordant": len(concordant),
        "n_discordant": len(discordant),
        "human_human_baseline": human_human_agreement(rows),
        "llm_vs_single_human": llm_vs_single_human(rows, verdicts),
        "llm_vs_consensus": llm_vs_consensus(concordant, verdicts),
        "prediction_distribution": prediction_distribution(rows, verdicts),
        "per_category_consensus": per_category(
            rows,
            verdicts,
            lambda rs, vs: llm_vs_consensus(
                [r for r in rs if len(set(r["binary_labels"])) == 1], vs
            ),
        ),
        "per_category_human_baseline": per_category(
            rows, verdicts, lambda rs, _vs: human_human_agreement(rs)
        ),
    }

    if bootstrap:
        report["ci_95_llm_vs_single_human"] = bootstrap_ci(
            rows, verdicts, llm_vs_single_human, key="agreement",
            n_resamples=n_resamples,
        )
        report["ci_95_llm_vs_consensus"] = bootstrap_ci(
            concordant, verdicts, llm_vs_consensus, key="agreement",
            n_resamples=n_resamples,
        )

    return report


# ── Markdown rendering ───────────────────────────────────────────────────────


def render_markdown(report: dict) -> str:
    """Human-readable summary. Used in bake-off side-by-side comparison."""
    h = report["human_human_baseline"]
    s = report["llm_vs_single_human"]
    c = report["llm_vs_consensus"]
    p = report["prediction_distribution"]

    lines = [
        f"# Judge validation: `{report['judge_model']}`",
        "",
        f"- Rows total: {report['n_rows_total']}  "
        f"(judged: {report['n_rows_judged']})",
        f"- Concordant / discordant: {report['n_concordant']} / {report['n_discordant']}",
        "",
        "## Agreement frames",
        "",
        "| Frame | n | Agreement |",
        "|---|---:|---:|",
        f"| Human ↔ human (baseline) | {h['pairs']} pairs | "
        f"**{h['agreement']:.1%}** (κ {h['kappa']:.3f}) |",
        f"| LLM ↔ single human | {s['pairs']} pairs | **{s['agreement']:.1%}** |",
        f"| LLM ↔ consensus (concordant) | {c['n']} rows | **{c['agreement']:.1%}** |",
        "",
        "## Rubber-stamp detector",
        "",
        f"- Judge met-rate: **{p['judge_met_rate']:.1%}** "
        f"(physicians: {p['physician_met_rate']:.1%})",
        f"- Agreement on physician-met rows: "
        f"**{c['agreement_on_met_rows']:.1%}** ({c['n_met_rows']} rows)",
        f"- Agreement on physician-not-met rows: "
        f"**{c['agreement_on_not_met_rows']:.1%}** ({c['n_not_met_rows']} rows)",
        "",
        "## Per-category (concordant agreement)",
        "",
        "| Category | n_concordant | Agreement | Human baseline |",
        "|---|---:|---:|---:|",
    ]
    for cat in sorted(report["per_category_consensus"]):
        cc = report["per_category_consensus"][cat]
        hh = report["per_category_human_baseline"][cat]
        agr = f"{cc['agreement']:.1%}" if cc.get("agreement") is not None else "n/a"
        hag = f"{hh['agreement']:.1%}" if hh.get("agreement") is not None else "n/a"
        lines.append(f"| {cat} | {cc.get('n', 0)} | **{agr}** | {hag} |")

    if "ci_95_llm_vs_single_human" in report:
        lo, hi = report["ci_95_llm_vs_single_human"]
        lines += ["", f"95% CI on LLM ↔ single human: [{lo:.1%}, {hi:.1%}]"]
    if "ci_95_llm_vs_consensus" in report:
        lo, hi = report["ci_95_llm_vs_consensus"]
        lines += [f"95% CI on LLM ↔ consensus:    [{lo:.1%}, {hi:.1%}]"]

    return "\n".join(lines) + "\n"
