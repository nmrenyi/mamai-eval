"""Compare device vs cluster MCQ results on the same row set.

Reads result JSONs from two run dirs (cluster + device), cross-joins by row
`id`, and writes a calibration report (Markdown + CSV) measuring how close
the two runtimes agree.

Usage:
    python compare_calibration.py \
        --cluster configs/config-v0.2.0/results/end_to_end_eval/gemma4-e4b/<ts>/ \
        --device configs/config-v0.2.0/results/end_to_end_eval/gemma4-e4b-device/<ts>/ \
        --manifest configs/config-v0.2.0/calibration/mcq_300.json \
        --output configs/config-v0.2.0/reports/calibration-mcq-<date>.md
"""

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


# ── Loaders ──────────────────────────────────────────────────────────────────

def _load_run_dir(d: Path) -> dict[str, dict]:
    """Walk a run dir, load every <dataset>.json result file, key all rows by id."""
    by_id = {}
    for f in sorted(d.glob("*.json")):
        if f.name.endswith(".eval_input.json") or f.name.endswith(".eval_output.json"):
            continue
        data = json.loads(f.read_text())
        dataset = data.get("metadata", {}).get("dataset", f.stem)
        for row in data.get("results", []):
            by_id[row["id"]] = {**row, "_dataset": dataset, "_source_file": f.name}
    return by_id


# ── Stats helpers ────────────────────────────────────────────────────────────

def _accuracy(rows: list[dict], pred_field: str) -> tuple[float, int, int]:
    if not rows:
        return 0.0, 0, 0
    correct = sum(1 for r in rows if r[pred_field] == r["ground_truth"])
    return correct / len(rows), correct, len(rows)


def _bootstrap_ci(rows: list[dict], pred_field: str,
                  n_iter: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Bootstrap 95% CI on accuracy."""
    if len(rows) < 2:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(rows)
    correct = [1 if r[pred_field] == r["ground_truth"] else 0 for r in rows]
    accs = []
    for _ in range(n_iter):
        sample = [correct[rng.randrange(n)] for _ in range(n)]
        accs.append(sum(sample) / n)
    accs.sort()
    lo = accs[int(0.025 * n_iter)]
    hi = accs[int(0.975 * n_iter)]
    return lo, hi


def _cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    """Cohen's kappa on the per-row (device, cluster) predicted-letter pairs."""
    if not pairs:
        return 0.0
    n = len(pairs)
    p_o = sum(1 for a, b in pairs if a == b) / n
    classes = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    p_a = {c: sum(1 for a, _ in pairs if a == c) / n for c in classes}
    p_b = {c: sum(1 for _, b in pairs if b == c) / n for c in classes}
    p_e = sum(p_a[c] * p_b[c] for c in classes)
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def _agreement(pairs: list[tuple[str, str]]) -> tuple[float, int, int]:
    if not pairs:
        return 0.0, 0, 0
    agree = sum(1 for a, b in pairs if a == b)
    return agree / len(pairs), agree, len(pairs)


# ── Report ───────────────────────────────────────────────────────────────────

def _render_report(rows: list[dict], manifest: dict, args) -> str:
    lines = []
    lines.append(f"# MCQ device-vs-cluster calibration\n")
    lines.append(f"- **Manifest**: `{args.manifest}` "
                 f"(name=`{manifest.get('name')}`, "
                 f"seed={manifest.get('seed')}, "
                 f"created={manifest.get('created_at_utc')})")
    lines.append(f"- **Cluster run**: `{args.cluster}`")
    lines.append(f"- **Device run**: `{args.device}`")
    lines.append(f"- **HF dataset**: `{manifest.get('hf_repo')}@{manifest.get('revision')}`")
    lines.append(f"- **Sampled**: {manifest.get('n_per_config')} per config × "
                 f"{len(manifest.get('configs', []))} configs = "
                 f"{manifest.get('total')} rows")
    lines.append(f"- **Joined**: {len(rows)} overlapping rows scored on both venues\n")

    # Runtime + precision
    lines.append("## Runtime\n")
    lines.append("| Venue | Model artifact | Inference runtime | Hardware backend | Numeric precision |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| Device | {args.device_artifact} | {args.device_runtime} | "
                 f"{args.device_backend} | **{args.device_precision}** |")
    lines.append(f"| Cluster | {args.cluster_artifact} | {args.cluster_runtime} | "
                 f"{args.cluster_backend} | **{args.cluster_precision}** |")
    lines.append("")
    lines.append("> The two venues run the same model family but very different precision regimes:")
    lines.append(f"> device uses LiteRT-LM's GPU FP16 attention (the default — known FP32 escape via")
    lines.append("> the artifact's `prefer_activation_type=float32` metadata key is **not** set on")
    lines.append("> this artifact), while cluster uses llama-cpp-python's GGUF Q4_0 (integer 4-bit")
    lines.append("> quantisation) on CUDA. Any accuracy gap below the ±5 pp noise floor at n=300")
    lines.append("> is dominated by precision differences, not model differences.\n")

    # Aggregate
    lines.append("## Aggregate accuracy\n")
    device_acc, d_c, d_n = _accuracy(rows, "device_pred")
    cluster_acc, c_c, c_n = _accuracy(rows, "cluster_pred")
    d_lo, d_hi = _bootstrap_ci(rows, "device_pred", seed=42)
    c_lo, c_hi = _bootstrap_ci(rows, "cluster_pred", seed=43)
    lines.append(f"| Venue | Accuracy | 95% CI | n |")
    lines.append(f"|---|---:|---|---:|")
    lines.append(f"| Device (LiteRT, .litertlm) | **{device_acc:.1%}** "
                 f"({d_c}/{d_n}) | [{d_lo:.1%}, {d_hi:.1%}] | {d_n} |")
    lines.append(f"| Cluster (GGUF Q4_0, llama-cpp) | **{cluster_acc:.1%}** "
                 f"({c_c}/{c_n}) | [{c_lo:.1%}, {c_hi:.1%}] | {c_n} |")
    delta = device_acc - cluster_acc
    lines.append(f"| **Δ (device − cluster)** | **{delta:+.1%}** | | |\n")

    pairs = [(r["device_pred"], r["cluster_pred"]) for r in rows]
    agree_rate, n_agree, _ = _agreement(pairs)
    kappa = _cohen_kappa(pairs)
    lines.append("## Per-row agreement\n")
    lines.append(f"- **Same predicted letter on both venues**: {n_agree}/{len(pairs)} "
                 f"({agree_rate:.1%})")
    lines.append(f"- **Cohen's κ** (chance-corrected agreement, 4-letter classes): **{kappa:.3f}**")
    if kappa >= 0.81:
        lines.append("  - κ ≥ 0.81 → \"almost perfect\" agreement (Landis & Koch).")
    elif kappa >= 0.61:
        lines.append("  - κ ≥ 0.61 → \"substantial\" agreement.")
    elif kappa >= 0.41:
        lines.append("  - κ ≥ 0.41 → \"moderate\" agreement.")
    else:
        lines.append("  - κ < 0.41 → \"fair-or-worse\" agreement — runtimes diverge meaningfully.")
    lines.append("")

    # Per-config breakdown
    lines.append("## Per-config breakdown\n")
    lines.append("| Config | n | Device acc | Cluster acc | Δ | Agree | κ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[r["dataset"]].append(r)
    for cfg in sorted(by_cfg):
        cfg_rows = by_cfg[cfg]
        d_a, d_c2, d_n2 = _accuracy(cfg_rows, "device_pred")
        c_a, c_c2, c_n2 = _accuracy(cfg_rows, "cluster_pred")
        cfg_pairs = [(r["device_pred"], r["cluster_pred"]) for r in cfg_rows]
        ag, n_ag, _ = _agreement(cfg_pairs)
        k = _cohen_kappa(cfg_pairs)
        lines.append(f"| {cfg} | {d_n2} | {d_a:.1%} ({d_c2}) | {c_a:.1%} ({c_c2}) "
                     f"| {d_a - c_a:+.1%} | {ag:.1%} ({n_ag}) | {k:.3f} |")
    lines.append("")

    # Confusion patterns
    lines.append("## Disagreement patterns\n")
    disagreements = [r for r in rows if r["device_pred"] != r["cluster_pred"]]
    lines.append(f"- {len(disagreements)} rows ({len(disagreements) / len(rows):.1%}) "
                 f"where device and cluster predict different letters.")
    lines.append(f"- Of those, **device-correct & cluster-wrong**: "
                 f"{sum(1 for r in disagreements if r['device_pred'] == r['ground_truth'])}")
    lines.append(f"- **cluster-correct & device-wrong**: "
                 f"{sum(1 for r in disagreements if r['cluster_pred'] == r['ground_truth'])}")
    lines.append(f"- **both wrong, different letters**: "
                 f"{sum(1 for r in disagreements if r['device_pred'] != r['ground_truth'] and r['cluster_pred'] != r['ground_truth'])}")
    lines.append("")

    # 4×4 confusion (cluster → device)
    classes = sorted({r["device_pred"] for r in rows} | {r["cluster_pred"] for r in rows})
    cnt: dict = defaultdict(int)
    for r in rows:
        cnt[(r["cluster_pred"], r["device_pred"])] += 1
    lines.append("### Confusion matrix (rows=cluster prediction, cols=device prediction)\n")
    header = "| cluster ↓ \\ device → | " + " | ".join(classes) + " |"
    lines.append(header)
    lines.append("|---" + "|---" * len(classes) + "|")
    for c in classes:
        row_str = f"| **{c}** | " + " | ".join(str(cnt.get((c, d), 0)) for d in classes) + " |"
        lines.append(row_str)
    lines.append("")

    lines.append("## Interpretation\n")
    if abs(delta) <= 0.03:
        lines.append("- Accuracy delta is within ±3 pp — the two runtimes are **effectively interchangeable** "
                     "at this sample size. Cluster runs alone are sufficient for the rest of the pilot.")
    elif abs(delta) <= 0.05:
        lines.append("- Accuracy delta is in ±5 pp — borderline. The headline numbers from one venue "
                     "are *probably* a good proxy for the other, but worth re-running on the full pilot.")
    else:
        lines.append(f"- Accuracy delta is **{abs(delta):.1%} pp** — meaningful precision gap between "
                     "runtimes. Device runs are load-bearing for the headline accuracy if the deployed "
                     "model is the LiteRT variant.")
    return "\n".join(lines)


def _write_csv(rows: list[dict], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "dataset", "ground_truth",
                                          "device_pred", "cluster_pred",
                                          "device_correct", "cluster_correct",
                                          "agree"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "id": r["id"],
                "dataset": r["dataset"],
                "ground_truth": r["ground_truth"],
                "device_pred": r["device_pred"],
                "cluster_pred": r["cluster_pred"],
                "device_correct": int(r["device_pred"] == r["ground_truth"]),
                "cluster_correct": int(r["cluster_pred"] == r["ground_truth"]),
                "agree": int(r["device_pred"] == r["cluster_pred"]),
            })


# ── Driver ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", required=True, help="Cluster run dir (has *.json results)")
    parser.add_argument("--device", required=True, help="Device run dir (has *.json results)")
    parser.add_argument("--manifest", required=True, help="Path to calibration manifest JSON")
    parser.add_argument("--output", required=True, help="Output Markdown report path")
    # Runtime descriptors — surfaced prominently in the report so the precision regime
    # of each venue is documented alongside the accuracy delta.
    parser.add_argument("--device-artifact", default="gemma-4-E4B-it.litertlm")
    parser.add_argument("--device-runtime", default="LiteRT-LM")
    parser.add_argument("--device-backend", default="GPU (Android, OpenCL)")
    parser.add_argument("--device-precision", default="FP16")
    parser.add_argument("--cluster-artifact", default="google_gemma-4-E4B-it-Q4_0.gguf")
    parser.add_argument("--cluster-runtime", default="llama-cpp-python")
    parser.add_argument("--cluster-backend", default="CUDA (NVIDIA A100 80GB)")
    parser.add_argument("--cluster-precision", default="Q4_0 (4-bit integer quant)")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    expected_ids = set(manifest["ids"])

    cluster_rows = _load_run_dir(Path(args.cluster))
    device_rows = _load_run_dir(Path(args.device))
    print(f"Cluster: {len(cluster_rows)} rows loaded")
    print(f"Device:  {len(device_rows)} rows loaded")
    print(f"Manifest: {len(expected_ids)} ids expected")

    rows = []
    for rid in sorted(expected_ids):
        if rid not in cluster_rows or rid not in device_rows:
            continue
        c = cluster_rows[rid]
        d = device_rows[rid]
        rows.append({
            "id": rid,
            "dataset": c.get("_dataset") or d.get("_dataset"),
            "ground_truth": c["ground_truth"],
            "device_pred": d["extracted_answer"],
            "cluster_pred": c["extracted_answer"],
        })
    print(f"Joined: {len(rows)} rows on both venues\n")

    report = _render_report(rows, manifest, args)
    out_md = Path(args.output)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report + "\n")
    print(f"Report:  {out_md}")

    out_csv = out_md.with_suffix(".csv")
    _write_csv(rows, out_csv)
    print(f"Per-row: {out_csv}")


if __name__ == "__main__":
    main()
