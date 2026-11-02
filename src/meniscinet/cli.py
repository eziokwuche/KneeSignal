"""Small, dependency-light commands for validating competition artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .constants import TARGETS
from .data.audit import audit_competition_data


def validate_submission(path: Path, expected_rows: int | None = None) -> None:
    frame = pd.read_csv(path)
    expected = ["StudyInstanceUID", *TARGETS]
    if list(frame.columns) != expected:
        raise ValueError(f"Expected columns {expected}; found {list(frame.columns)}")
    if frame.StudyInstanceUID.isna().any() or frame.StudyInstanceUID.duplicated().any():
        raise ValueError("StudyInstanceUID must be present and unique.")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows; found {len(frame)}.")
    scores = frame[list(TARGETS)]
    if scores.isna().any().any() or not scores.apply(lambda x: x.between(0, 1).all()).all():
        raise ValueError("All predictions must be finite probabilities in [0, 1].")
    print(f"Valid submission: {len(frame)} studies and {len(TARGETS)} targets.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="meniscinet")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate-submission")
    check.add_argument("path", type=Path)
    check.add_argument("--expected-rows", type=int)
    audit = commands.add_parser("audit-data", help="Validate the official Kaggle input layout.")
    audit.add_argument("data_dir", type=Path)
    audit.add_argument("--scan-dicoms", action="store_true", help="Also check every series folder for DICOM files.")
    audit.add_argument("--output", type=Path, help="Optional path for the JSON audit report.")
    args = parser.parse_args()
    if args.command == "validate-submission":
        validate_submission(args.path, args.expected_rows)
    if args.command == "audit-data":
        report = audit_competition_data(args.data_dir, scan_dicoms=args.scan_dicoms)
        text = json.dumps(report, indent=2, sort_keys=True)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)


if __name__ == "__main__":
    main()
