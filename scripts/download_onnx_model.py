"""
scripts/download_onnx_model.py
==============================
Downloads and verifies the official, pinned unquantized ONNX model artifact
and tokenizer for sentence-transformers/all-MiniLM-L6-v2.

Used during container builds (Dockerfile) and local environment setup.
Verifies SHA-256 checksums to ensure reproducible, tamper-proof artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

# Pinned repository and revision
PINNED_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"
PINNED_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

ARTIFACTS = {
    "onnx/model.onnx": {
        "dest_name": "model.onnx",
        "sha256": "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
        "size": 90405214,
    },
    "tokenizer.json": {
        "dest_name": "tokenizer.json",
        "sha256": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
        "size": 466247,
    },
}


def verify_sha256(file_path: Path, expected_sha: str) -> bool:
    """Verifies that a file's SHA-256 hash matches the expected value."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    actual_sha = h.hexdigest()
    if actual_sha != expected_sha:
        print(f"ERROR: SHA-256 mismatch for {file_path}!")
        print(f"  Expected: {expected_sha}")
        print(f"  Actual:   {actual_sha}")
        return False
    return True


def download_artifacts(target_dir: Path | None = None) -> dict[str, Path]:
    """
    Downloads the pinned ONNX model and tokenizer using huggingface_hub,
    verifies SHA-256 checksums, and optionally copies them to target_dir.
    """
    from huggingface_hub import hf_hub_download

    print(f"Downloading pinned model: {PINNED_REPO_ID} (revision: {PINNED_REVISION})")
    downloaded_paths: dict[str, Path] = {}

    for repo_file, meta in ARTIFACTS.items():
        print(f"Fetching {repo_file}...")
        downloaded = hf_hub_download(
            repo_id=PINNED_REPO_ID,
            filename=repo_file,
            revision=PINNED_REVISION,
        )
        src_path = Path(downloaded)

        print(f"Verifying SHA-256 checksum for {repo_file}...")
        if not verify_sha256(src_path, meta["sha256"]):
            raise ValueError(f"Checksum verification failed for {repo_file}")

        if target_dir:
            target_dir.mkdir(parents=True, exist_ok=True)
            dest_path = target_dir / meta["dest_name"]
            if not dest_path.exists() or dest_path.stat().st_size != meta["size"]:
                shutil.copy2(src_path, dest_path)
                print(f"Copied to {dest_path}")
            downloaded_paths[meta["dest_name"]] = dest_path
        else:
            downloaded_paths[repo_file] = src_path

    print("All pinned ONNX model artifacts downloaded and verified successfully.")
    return downloaded_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify pinned ONNX embedding model.")
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help="Optional directory to copy model.onnx and tokenizer.json into.",
    )
    args = parser.parse_args()

    try:
        download_artifacts(args.target_dir)
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
