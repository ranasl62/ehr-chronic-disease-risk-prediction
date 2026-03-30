#!/usr/bin/env python3
"""Validate a CSV before training. Exits 1 on blocking errors.

  PYTHONPATH=. python scripts/validate_training_data.py --format longitudinal data/raw/ehr_data.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from preprocessing.data_quality import assert_no_blocking_errors, summarize_csv


def main() -> int:
    p = argparse.ArgumentParser(description="Data quality checks for training CSVs.")
    p.add_argument("csv", type=Path)
    p.add_argument("--format", choices=["longitudinal", "tabular"], required=True)
    args = p.parse_args()
    issues = summarize_csv(args.csv, data_format=args.format)
    for i in issues:
        tag = i.get("level", "info").upper()
        print(f"[{tag}] {i.get('code', '')}: {i.get('message', '')}")
    try:
        assert_no_blocking_errors(issues)
    except ValueError as e:
        print("BLOCKED:", e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
