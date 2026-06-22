#!/bin/bash
set -euo pipefail
# Track A aggregation — per-cell recall / harm / dangerous / deflection (kenya SAQ)
# + mean_weighted_met (healthbench rubric), recomputed from the judged run-dirs
# using the repo's OWN _agg_dataset functions so methodology matches the pipeline
# exactly. CPU-only (GPU_REQUEST=0). Emits one delimited JSON block per cell plus
# a combined matrix block for local reconstruction.
#
# Submit AFTER all Track A judges finish:
#   GPU_REQUEST=0 CPU_REQUEST=2 MEMORY_REQUEST=8G RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-mxa-agg run_cluster_matrix_trackA_agg.sh

REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
REPO_REF="${REPO_REF:-feat/g1-prompt-fix-20260611}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
A="${A:-/lightscratch/users/yiren/eval_output/gen_prompt_matrix_20260622/A}"
CELLS="${CELLS:-g4_baseline g4_g1 g4_g1g2 3n_baseline 3n_g1 3n_g1g2}"

echo "=== DEPS ==="
apt-get update -qq && apt-get install -y -qq python3.10 git > /dev/null
ln -sf /usr/bin/python3.10 /usr/bin/python3
echo "=== DEPS DONE ==="
rm -rf "$WORKTREE"; git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WORKTREE"; cd "$WORKTREE"

A="$A" CELLS="$CELLS" python3 - <<'PY'
import json, os, sys
sys.path.insert(0, ".")
from end_to_end_eval.rescore_open_v2 import _agg_dataset as agg_open
from end_to_end_eval.rescore_rubric import _agg_dataset as agg_rubric

A = os.environ["A"]; CELLS = os.environ["CELLS"].split()
matrix = {}
for c in CELLS:
    out = {}
    # kenya SAQ — recall / harm / dangerous / deflection
    try:
        kd = json.load(open(f"{A}/{c}/run/kenya.json"))
        a = agg_open(kd.get("results", []))
        sd = a.get("safety_distribution", {}) or {}
        out["kenya"] = {
            "n": a.get("n_judged"),
            "recall": a.get("mean_key_fact_recall"),
            "harm_rate": a.get("harm_rate"),
            "dangerous": sd.get("dangerous", 0),
            "potentially_harmful": sd.get("potentially_harmful", 0),
            "deflection_rate": a.get("deflection_rate"),
            "safety_distribution": sd,
            "behavior_distribution": a.get("behavior_distribution", {}),
        }
    except Exception as e:
        out["kenya"] = {"error": repr(e)}
    # healthbench rubric — overall + positive/negative split + per-axis
    try:
        hd = json.load(open(f"{A}/{c}/run/healthbench_oss_eval.json"))
        a = agg_rubric(hd.get("results", []))
        out["healthbench"] = {
            "n": a.get("n_scored"),
            "mean_weighted_met": a.get("mean_weighted_met"),
            "mean_positive_score": a.get("mean_positive_score"),  # completeness (+ criteria)
            "mean_penalty_rate": a.get("mean_penalty_rate"),      # harm/penalty (- criteria triggered) ↓ better
            "per_axis_mean": a.get("per_axis_mean"),
        }
    except Exception as e:
        out["healthbench"] = {"error": repr(e)}
    matrix[c] = out
    print(f"@@@CELL {c}@@@"); print(json.dumps(out, indent=2)); print("@@@END@@@")

print("@@@MATRIX@@@"); print(json.dumps(matrix, indent=2)); print("@@@MATRIXEND@@@")
PY
echo "=== AGG DONE ==="
