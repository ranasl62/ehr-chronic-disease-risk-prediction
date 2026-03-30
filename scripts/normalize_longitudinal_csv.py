#!/usr/bin/env python3
"""Normalize external CSV column names to repo longitudinal schema. Usage:
  PYTHONPATH=. python scripts/normalize_longitudinal_csv.py in.csv -o data/processed/normalized.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from preprocessing.canonical_schema import assert_longitudinal_minimum, rename_to_canonical_longitudinal
from preprocessing.ehr_loader import load_data


def main() -> None:
    p = argparse.ArgumentParser(description="Map aliases to canonical longitudinal EHR columns.")
    p.add_argument("input_csv", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    args = p.parse_args()
    df = load_data(args.input_csv)
    df = rename_to_canonical_longitudinal(df)
    assert_longitudinal_minimum(df, need_label=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
