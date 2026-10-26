#!/usr/bin/env python3
"""Verify canonical outputs or print configured nnU-Net environment variables."""
import argparse
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiment_registry import ArtifactRoots
from src.layout import verify_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "env"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.action == "env":
        roots = ArtifactRoots(repo)
        for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
            print(f"export {name}={shlex.quote(str(roots.resolve('nnunet://' + name)))}")
    else:
        experiments, outputs = verify_outputs(repo)
        print(f"Verified {experiments} experiments / {outputs} outputs: split hashes, held-out fold coverage and metrics paths.")


if __name__ == "__main__":
    main()
