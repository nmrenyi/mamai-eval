"""
On-device MCQ evaluation runner for mamabench v0.2.

Pushes a set of MCQ rows to the connected Android device, triggers
BenchmarkForegroundService's eval-mode branch via ADB, and pulls the
per-row response file back. Outputs are scored with the same
extract_letters + score_mcq pipeline as run_eval.py so accuracy is
directly comparable to Mac / cluster GGUF runs.

The on-device LiteRT runtime uses .litertlm artifacts with a different
quantisation path than the .gguf files run by llama-cpp-python — this
runner is the calibration anchor that lets us measure the precision
gap between the two.

Usage:
    python run_eval_device.py --config config-v0.2.0 \\
        --datasets afrimedqa --max-questions 20
    python run_eval_device.py --config config-v0.2.0 \\
        --datasets afrimedqa --device-serial 9da65e62

Currently MCQ-only (open_ended / open_ended_rubric to follow).

Requires:
  - Connected device with the eval-mode-enabled mamai APK installed
    (see ~/Downloads/mamai BenchmarkForegroundService.kt:runMcqEval).
  - adb in PATH.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Resolve --config before any prompts imports ──────────────────────────────
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--config", required=True)
_pre_args, _ = _pre.parse_known_args()
os.environ["MAMAI_EVAL_CONFIG"] = _pre_args.config
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.prompts import (CONFIG_VERSION, MCQ_SYSTEM_PROMPT, PROMPT_VERSION,
                            PROTOCOL_VERSION, SPEC_SHA256, DATASET_HF_REPO,
                            DATASET_REVISION, TEMPERATURE, TOP_P, TOP_K, N_CTX,
                            _params as _active_params)
from shared.scoring import _parse_answer_set, extract_letters, score_mcq
from shared.dataset_loader import HF_CONFIGS, _load_dataset

PACKAGE = "com.example.app"
ACTIVITY = f"{PACKAGE}/.BenchmarkActivity"
FILES_DIR = f"/storage/emulated/0/Android/data/{PACKAGE}/files"
INPUT_PATH_ON_DEVICE = f"{FILES_DIR}/eval_input.json"
OUTPUT_PATH_ON_DEVICE = f"{FILES_DIR}/eval_output.json"
LOG_TAG = "mam-ai-bench"
COMPLETE_MARKER = "[BENCHMARK] COMPLETE"
FAILED_MARKER = "[BENCHMARK] FAILED"


# ── ADB helpers ───────────────────────────────────────────────────────────────

def _adb(serial: str | None) -> list[str]:
    cmd = ["adb"]
    if serial:
        cmd += ["-s", serial]
    return cmd


def _adb_run(serial: str | None, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(_adb(serial) + list(args), capture_output=True, text=True, check=check)


def _check_device(serial: str | None) -> None:
    result = _adb_run(serial, "devices")
    devices = [
        line.split()[0] for line in result.stdout.strip().splitlines()[1:]
        if line.strip() and "device" in line
    ]
    if not devices:
        sys.exit("ERROR: No Android device connected (adb devices).")
    if serial and serial not in devices:
        sys.exit(f"ERROR: requested --device-serial {serial} not in adb devices: {devices}")
    print(f"Device connected: {devices[0] if not serial else serial}")

    result = _adb_run(serial, "shell", "pm", "list", "packages", PACKAGE)
    if PACKAGE not in result.stdout:
        sys.exit(f"ERROR: {PACKAGE} is not installed on the device. "
                 f"Run `cd ~/Downloads/mamai/app && flutter build apk --release && "
                 f"adb install -r build/app/outputs/flutter-apk/app-release.apk` first.")
    print(f"App installed: {PACKAGE}")


def _push(serial: str | None, local: str, remote: str) -> None:
    print(f"adb push {local} -> {remote}")
    subprocess.run(_adb(serial) + ["push", local, remote], check=True)


def _pull(serial: str | None, remote: str, local: str) -> None:
    print(f"adb pull {remote} -> {local}")
    result = subprocess.run(
        _adb(serial) + ["pull", remote, local], capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.exit(f"ERROR: adb pull failed: {result.stderr}")


def _clear_logcat(serial: str | None) -> None:
    subprocess.run(_adb(serial) + ["logcat", "-c"], capture_output=True, check=False)


def _launch_eval(serial: str | None) -> None:
    cmd = _adb(serial) + [
        "shell", "am", "start",
        "-n", ACTIVITY,
        "--ez", "eval_mode", "true",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    if "Error" in result.stderr:
        sys.exit(f"ERROR: failed to launch eval activity: {result.stderr}")
    print("BenchmarkActivity launched in eval mode. Monitoring logcat for progress...")


def _wait_for_completion(serial: str | None, timeout_s: int = 3600) -> bool:
    cmd = _adb(serial) + ["logcat", "-s", f"{LOG_TAG}:W", "--format", "brief"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    start = time.time()
    try:
        for line in proc.stdout:
            line = line.strip()
            if "[BENCHMARK]" in line:
                print(f"  {line}")
            if COMPLETE_MARKER in line:
                proc.terminate()
                return True
            if FAILED_MARKER in line:
                proc.terminate()
                return False
            if time.time() - start > timeout_s:
                proc.terminate()
                print(f"ERROR: device run timed out after {timeout_s}s")
                return False
    except KeyboardInterrupt:
        proc.terminate()
        print("\nInterrupted by user.")
        return False
    proc.terminate()
    return False


def _device_apk_info(serial: str | None) -> dict:
    """Capture build SHA + version info from the installed APK for provenance."""
    result = _adb_run(serial, "shell", "dumpsys", "package", PACKAGE, check=False)
    out = result.stdout
    fields = {}
    for needle, key in [("versionName=", "version_name"), ("versionCode=", "version_code")]:
        idx = out.find(needle)
        if idx >= 0:
            chunk = out[idx + len(needle):].split()[0].strip()
            fields[key] = chunk
    return fields


# ── Eval input + output handling ──────────────────────────────────────────────

def _build_eval_input(rows: list[dict]) -> dict:
    """Build the eval_input.json the on-device service consumes.

    Each user_message matches build_mcq_messages(question, choices_formatted)
    in prompts.py — same wire format as Mac/cluster runs.
    """
    payload_rows = []
    for r in rows:
        user_message = f"Question: {r['question']}\nOptions:\n{r['choices_formatted']}"
        payload_rows.append({"id": r["id"], "user_message": user_message})
    return {
        "system_prompt": MCQ_SYSTEM_PROMPT,
        "rows": payload_rows,
    }


def _score_results(rows: list[dict], device_rows: list[dict]) -> tuple[list[dict], dict]:
    """Build the canonical per-row results list + aggregate scores.

    `rows` are the normalised input rows (with ground_truth_letter); `device_rows`
    are what the device returned ({id, response_text, inference_time_ms, error}).
    Joined by `id`.
    """
    by_id = {r["id"]: r for r in device_rows}
    results = []
    predictions = []
    ground_truth = []
    for row in rows:
        dr = by_id.get(row["id"])
        response = (dr or {}).get("response_text", "") or ""
        inference_ms = (dr or {}).get("inference_time_ms", 0)
        error = (dr or {}).get("error")
        if error and error != "null" and not isinstance(error, type(None)):
            print(f"  WARNING: row {row['id']} returned error: {error}")

        extracted_set = extract_letters(response)
        extracted = ",".join(sorted(extracted_set)) if extracted_set else ""
        gt = row["ground_truth_letter"]
        predictions.append(extracted)
        ground_truth.append(gt)

        results.append({
            "id": row["id"],
            "question": row["question"],
            "options": row["choices_formatted"],
            "ground_truth": gt,
            "answer_index": row["answer_index"],
            "rag_context": "",
            "model_response": response,
            "extracted_answer": extracted,
            "extracted_answers": sorted(extracted_set),
            "correct": extracted_set == _parse_answer_set(gt),
            "inference_time_s": round((inference_ms or 0) / 1000.0, 2),
            "device_error": error if (error and not isinstance(error, type(None))) else None,
        })

    scores = score_mcq(predictions, ground_truth)
    return results, scores


# ── Driver ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                        help="Config version (e.g. config-v0.2.0)")
    parser.add_argument("--datasets", required=True,
                        help=f"Comma-separated dataset names. MCQ only: "
                             f"{','.join(n for n, (_, t) in HF_CONFIGS.items() if t == 'mcq')}")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit questions per dataset")
    parser.add_argument("--row-ids", default=None,
                        help="Path to a calibration manifest JSON. When set, only rows whose "
                             "`id` appears in manifest['ids'] are evaluated. Pair with the "
                             "same flag on run_eval.py to produce a device-vs-cluster "
                             "calibration comparison.")
    parser.add_argument("--revision", default=None,
                        help="HF dataset revision (default: dataset.revision from params.json)")
    parser.add_argument("--hf-repo", default=None,
                        help="HF dataset repo (default: dataset.hf_repo from params.json)")
    parser.add_argument("--device-serial", default=None,
                        help="Specific ADB device serial (default: first available)")
    parser.add_argument("--device-model-tag", default="gemma4-e4b-device",
                        help="Result-dir subfolder name to distinguish device runs from "
                             "Mac/cluster GGUF runs (default: gemma4-e4b-device)")
    parser.add_argument("--timeout-s", type=int, default=3600,
                        help="Max wall-clock seconds to wait for a device run")
    parser.add_argument("--output-dir", default=None,
                        help="Output dir (default: configs/<config>/results/end_to_end_eval)")
    args = parser.parse_args()

    revision = args.revision or DATASET_REVISION or "v0.2"
    hf_repo = args.hf_repo or DATASET_HF_REPO or "nmrenyi/mamabench"
    output_dir = args.output_dir or str(
        Path(__file__).resolve().parents[1] / "configs" / args.config / "results" / "end_to_end_eval"
    )
    os.makedirs(output_dir, exist_ok=True)

    dataset_names = [d.strip() for d in args.datasets.split(",")]
    for name in dataset_names:
        if name not in HF_CONFIGS:
            parser.error(f"Unknown dataset: {name}. Available: {list(HF_CONFIGS.keys())}")
        set_type = HF_CONFIGS[name][1]
        if set_type != "mcq":
            parser.error(f"{name} is set_type={set_type}; run_eval_device.py currently "
                         f"supports MCQ only.")

    row_ids_filter: set[str] | None = None
    if args.row_ids:
        manifest = json.loads(Path(args.row_ids).read_text())
        row_ids_filter = set(manifest["ids"])
        print(f"Row-ids filter: {len(row_ids_filter)} ids from "
              f"{args.row_ids} ({manifest.get('name', '?')})")

    _check_device(args.device_serial)
    apk_info = _device_apk_info(args.device_serial)

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = os.path.join(output_dir, args.device_model_tag, run_timestamp)
    os.makedirs(run_dir, exist_ok=True)

    summary = []
    for ds_name in dataset_names:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}  |  Config: {CONFIG_VERSION}  |  Device: {args.device_serial or 'default'}")
        print(f"{'='*60}")

        rows, _ = _load_dataset(ds_name, revision, hf_repo, args.max_questions,
                                row_ids=row_ids_filter)
        if not rows:
            print(f"SKIP: {ds_name} produced 0 normalized rows")
            continue

        # Stage input
        local_input = os.path.join(run_dir, f"{ds_name}.eval_input.json")
        Path(local_input).write_text(json.dumps(_build_eval_input(rows), ensure_ascii=False, indent=2))
        _push(args.device_serial, local_input, INPUT_PATH_ON_DEVICE)

        # Clear stale logcat, kick off eval, watch for completion
        subprocess.run(_adb(args.device_serial) + ["shell", "am", "force-stop", PACKAGE],
                       capture_output=True, check=False)
        _clear_logcat(args.device_serial)
        t0 = time.time()
        _launch_eval(args.device_serial)
        ok = _wait_for_completion(args.device_serial, timeout_s=args.timeout_s)
        elapsed = time.time() - t0
        if not ok:
            print(f"ERROR: eval did not complete cleanly for {ds_name}")
            continue

        # Pull output
        local_output = os.path.join(run_dir, f"{ds_name}.eval_output.json")
        _pull(args.device_serial, OUTPUT_PATH_ON_DEVICE, local_output)
        device_payload = json.loads(Path(local_output).read_text())
        device_rows = device_payload.get("rows", [])
        if len(device_rows) != len(rows):
            print(f"  WARNING: device returned {len(device_rows)} rows, expected {len(rows)}")

        # Score
        results, scores = _score_results(rows, device_rows)

        metadata = {
            "model": args.device_model_tag,
            "inference_venue": "device",
            "device_serial": args.device_serial or "default",
            "device_payload": {
                "eval_version": device_payload.get("eval_version"),
                "device": device_payload.get("device"),
                "total_time_ms": device_payload.get("total_time_ms"),
                "n_rows_reported": device_payload.get("n_rows"),
            },
            "apk_info": apk_info,
            "dataset": ds_name,
            "dataset_type": "mcq",
            "hf_repo": hf_repo,
            "hf_revision": revision,
            "config_version": CONFIG_VERSION,
            "n_questions": len(rows),
            "timestamp": run_timestamp,
            "protocol_version": PROTOCOL_VERSION,
            "prompt_version": PROMPT_VERSION,
            "spec_sha256": SPEC_SHA256,
            "rag": False,
            "generation_params": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "n_ctx": N_CTX,
                "max_tokens_note": "on-device max set by runtime_config.json engine.max_num_tokens",
            },
            "total_inference_time_s": round(elapsed, 1),
            "avg_time_per_question_s": round(elapsed / len(results), 2) if results else 0,
        }

        out_path = os.path.join(run_dir, f"{ds_name}.json")
        Path(out_path).write_text(json.dumps(
            {"metadata": metadata, "aggregate_scores": scores, "results": results},
            indent=2, ensure_ascii=False,
        ))
        print(f"Saved: {out_path}")
        acc = scores.get("accuracy", 0)
        partial = scores.get("partial_credit_accuracy", acc)
        print(f"Accuracy: {acc:.1%} ({scores.get('correct', 0)}/{scores.get('total', 0)})")
        if partial != acc:
            print(f"Partial credit: {partial:.1%}")
        summary.append(f"  {ds_name}: {acc:.1%} (partial: {partial:.1%})")

    print(f"\n{'='*60}")
    print(f"SUMMARY — device ({args.device_model_tag})  |  config: {CONFIG_VERSION}")
    print(f"{'='*60}")
    for line in summary:
        print(line)


if __name__ == "__main__":
    main()
