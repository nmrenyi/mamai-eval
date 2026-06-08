"""Fetch the OBGYN-scoped HealthBench grader meta-eval set.

This is judge-calibration data — physician-labeled (conversation, completion,
single rubric criterion) triples used to validate an LLM judge against human
ground truth. Shipped as a side-file (not a loadable HF config) under
mamabench@v0.2.1/calibration/, so we fetch by direct URL rather than
load_dataset().
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

DEFAULT_REVISION = "v0.2.1"
DEFAULT_HF_REPO = "nmrenyi/mamabench"
DEFAULT_FILENAME = "obgyn_meta_eval.jsonl"
DEFAULT_CACHE = Path(__file__).resolve().parent / "data"


def calibration_url(
    hf_repo: str = DEFAULT_HF_REPO,
    revision: str = DEFAULT_REVISION,
    filename: str = DEFAULT_FILENAME,
) -> str:
    return (
        f"https://huggingface.co/datasets/{hf_repo}"
        f"/resolve/{revision}/calibration/{filename}"
    )


def fetch_calibration(
    cache_dir: Path | str = DEFAULT_CACHE,
    revision: str = DEFAULT_REVISION,
    hf_repo: str = DEFAULT_HF_REPO,
    filename: str = DEFAULT_FILENAME,
    force: bool = False,
) -> Path:
    """Download obgyn_meta_eval.jsonl to cache_dir. Idempotent (skips if present).

    Returns the local path.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / filename
    if local_path.exists() and not force:
        return local_path

    url = calibration_url(hf_repo, revision, filename)
    print(f"Downloading {url}\n        -> {local_path}", file=sys.stderr)
    urllib.request.urlretrieve(url, str(local_path))
    return local_path
