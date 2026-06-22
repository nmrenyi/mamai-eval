#!/bin/bash
set -euo pipefail
# Faithfulness F3+F4 — gpt-5 categorize + calibrate over Lynx-scored run-dirs.
# CPU-only (GPU_REQUEST=0): the semantic step is the OpenAI Batch API, no GPU.
# Per arm run-dir it runs the full Option-B pipeline in place:
#   extract -> gpt-5 categorize (Batch) -> aggregate
#   sample  -> gpt-5 calibrate  (Batch) -> score
# then prints each arm's lynx_fail_summary.json + calibration_report.json to
# stdout (delimited) so the small summaries can be reconstructed locally without
# pulling the large oracle_responses.json off the PVC.
#
# OPENAI_API_KEY is auto-forwarded by submit_job.sh from the local shell.
# Submit:
#   GPU_REQUEST=0 CPU_REQUEST=4 MEMORY_REQUEST=16G RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-mxb-gpt5 run_cluster_faith_gpt5.sh \
#       'RUN_DIRS=/lightscratch/users/yiren/eval_output/gen_prompt_matrix_20260622/B/g4_baseline /lightscratch/users/yiren/eval_output/gen_prompt_matrix_20260622/B/g4_g1 /lightscratch/users/yiren/eval_output/gen_prompt_matrix_20260622/B/g4_g1g2'

REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
REPO_REF="${REPO_REF:-feat/g1-prompt-fix-20260611}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
OUT_BASE="${OUT_BASE:-/lightscratch/users/yiren/eval_output/gen_prompt_matrix_20260622}"
RUN_DIRS="${RUN_DIRS:-$OUT_BASE/B/g4_baseline $OUT_BASE/B/g4_g1 $OUT_BASE/B/g4_g1g2}"
CAT_MAX_TOK="${CAT_MAX_TOK:-4096}"     # categorize: small JSON, medium effort
CAL_MAX_TOK="${CAL_MAX_TOK:-16384}"    # calibrate: high effort -> reasoning tokens (4096 truncated before)

[ -n "${OPENAI_API_KEY:-}" ] || { echo "ERROR: OPENAI_API_KEY not set in container env"; exit 1; }
# Defensive: ensure we hit the public OpenAI API, never an inherited local vLLM endpoint.
unset OPENAI_BASE_URL || true

echo "=== DEPS ==="
apt-get update -qq && apt-get install -y -qq python3.10 python3-pip git > /dev/null
ln -sf /usr/bin/python3.10 /usr/bin/python3
pip3 install --no-cache-dir -q --retries 10 --upgrade pip > /dev/null
pip3 install --no-cache-dir -q --retries 10 openai > /dev/null
echo "=== DEPS DONE ==="

rm -rf "$WORKTREE"; git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WORKTREE"; cd "$WORKTREE"

FAILED=0
for d in $RUN_DIRS; do
  name=$(basename "$d")
  echo "########## FAITH-GPT5 arm=$name ($d) ##########"
  if [ ! -f "$d/lynx_scores.json" ] || [ ! -f "$d/oracle_responses.json" ]; then
    echo "ERROR: missing lynx_scores.json or oracle_responses.json in $d — skip"; FAILED=1; continue
  fi
  python3 -m generator_eval.analyze_lynx_fails extract "$d"
  python3 -m generator_eval.score_openai_batch_judge categorize \
    --input "$d/lynx_fail_cases.json" --output "$d/lynx_fail_categories.json" \
    --max-output-tokens "$CAT_MAX_TOK"
  python3 -m generator_eval.analyze_lynx_fails aggregate "$d" --categories "$d/lynx_fail_categories.json"
  python3 -m generator_eval.calibrate sample "$d"
  python3 -m generator_eval.score_openai_batch_judge calibrate \
    --input "$d/calibration_blind.json" --output "$d/calibration_verdicts.json" \
    --max-output-tokens "$CAL_MAX_TOK"
  python3 -m generator_eval.calibrate score "$d" --verdicts "$d/calibration_verdicts.json"
done

echo "=== ALL ARMS DONE — emitting summaries for local reconstruction ==="
for d in $RUN_DIRS; do
  name=$(basename "$d")
  for f in lynx_fail_summary calibration_report; do
    echo "@@@SUMMARY_BEGIN $name $f@@@"
    cat "$d/$f.json" 2>/dev/null || echo "{\"error\":\"missing $f.json\"}"
    echo ""
    echo "@@@SUMMARY_END@@@"
  done
done
echo "=== EMIT DONE (failures=$FAILED) ==="
