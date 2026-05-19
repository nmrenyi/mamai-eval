"""Build the deterministic MCQ calibration manifest.

Samples N rows from each MCQ config in mamabench v0.2 with a fixed seed,
writes the chosen row IDs to a JSON manifest. The manifest is committed
to the repo so the calibration set is reproducible from source for the
lifetime of the v0.2 dataset.

Usage:
    python prepare_calibration_set.py
        --output configs/config-v0.2.0/calibration/mcq_300.json
        --n-per-config 100 --seed 42

Reads from `nmrenyi/mamabench@v0.2` by default; override via --hf-repo / --revision.
"""

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

MCQ_CONFIGS = ("afrimedqa", "medqa_usmle", "medmcqa")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="configs/config-v0.2.0/calibration/mcq_300.json")
    parser.add_argument("--hf-repo", default="nmrenyi/mamabench")
    parser.add_argument("--revision", default="v0.2")
    parser.add_argument("--n-per-config", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="mcq_calibration_300")
    args = parser.parse_args()

    from datasets import load_dataset

    chosen_per_config = {}
    chosen_ids: list[str] = []

    for cfg in MCQ_CONFIGS:
        print(f"Loading {args.hf_repo}/{cfg}@{args.revision}")
        ds = load_dataset(args.hf_repo, cfg, revision=args.revision, split="test")
        # Sort IDs lexicographically before sampling so the choice is
        # independent of HF row order (which could change across releases
        # even within the same revision if files are repacked).
        all_ids = sorted(row["id"] for row in ds)
        if len(all_ids) < args.n_per_config:
            sys.exit(f"ERROR: {cfg} has only {len(all_ids)} rows, "
                     f"can't sample {args.n_per_config}")

        rng = random.Random(args.seed)
        picked = rng.sample(all_ids, args.n_per_config)
        # Sort the chosen IDs for stable manifest output.
        picked = sorted(picked)
        chosen_per_config[cfg] = picked
        chosen_ids.extend(picked)
        print(f"  {cfg}: {len(all_ids)} total → sampled {len(picked)}")

    manifest = {
        "schema_version": 1,
        "name": args.name,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hf_repo": args.hf_repo,
        "revision": args.revision,
        "seed": args.seed,
        "n_per_config": args.n_per_config,
        "total": len(chosen_ids),
        "stratify_by": "source.dataset",
        "configs": list(MCQ_CONFIGS),
        "ids_per_config": chosen_per_config,
        "ids": chosen_ids,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {out_path} ({len(chosen_ids)} ids, "
          f"{out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
