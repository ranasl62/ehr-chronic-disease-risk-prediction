#!/usr/bin/env python3
"""
Generate a leakage-safe synthetic longitudinal cohort for paper CI / method verification.

Not clinical data. Includes index_time, post-index outcome rows, sex, and age_band.

  PYTHONPATH=. python scripts/generate_paper_synthetic_cohort.py \\
    -o data/raw/paper_synthetic_cohort.csv --n-patients 400 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd


def generate_cohort(n_patients: int = 400, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    groups: list[dict] = []
    base = pd.Timestamp("2019-01-01")

    for i in range(n_patients):
        pid = 10_000 + i
        sex = "F" if rng.random() < 0.52 else "M"
        age = int(rng.integers(30, 80))
        age_band = "lt50" if age < 50 else ("50_64" if age < 65 else "ge65")
        # Latent risk drives glucose trajectory and outcome.
        risk = 0.15 + 0.01 * max(0, age - 45) + (0.05 if sex == "M" else 0.0) + rng.normal(0, 0.08)
        risk = float(np.clip(risk, 0.02, 0.85))
        n_pre = int(rng.integers(3, 7))
        index_offset = int(rng.integers(200, 900))
        index_time = base + pd.Timedelta(days=index_offset)

        for j in range(n_pre):
            t = index_time - pd.Timedelta(days=int(rng.integers(5, 170)) * (n_pre - j) // n_pre)
            glucose = 90 + 40 * risk + rng.normal(0, 8)
            bp = 110 + 25 * risk + rng.normal(0, 6)
            chol = 170 + 50 * risk + rng.normal(0, 12)
            rows.append(
                {
                    "patient_id": pid,
                    "timestamp": t.strftime("%Y-%m-%d"),
                    "index_time": index_time.strftime("%Y-%m-%d"),
                    "icd_code": "Z00.00",
                    "lab_value": round(glucose - 5 + rng.normal(0, 2), 1),
                    "vital_signs": round(70 + rng.normal(0, 4), 1),
                    "glucose": round(float(glucose), 1),
                    "blood_pressure": round(float(bp), 1),
                    "cholesterol": round(float(chol), 1),
                    "age": age,
                    "sex": sex,
                    "age_band": age_band,
                    "label": 0,
                }
            )

        y = int(rng.random() < risk)
        # Post-index observation (outcome or negative follow-up) within 365d.
        outcome_day = int(rng.integers(30, 360))
        t_out = index_time + pd.Timedelta(days=outcome_day)
        glucose_out = 90 + 55 * risk + (25 if y else 0) + rng.normal(0, 8)
        rows.append(
            {
                "patient_id": pid,
                "timestamp": t_out.strftime("%Y-%m-%d"),
                "index_time": index_time.strftime("%Y-%m-%d"),
                "icd_code": "E11.9" if y else "Z00.00",
                "lab_value": round(glucose_out - 5, 1),
                "vital_signs": round(72 + rng.normal(0, 4), 1),
                "glucose": round(float(glucose_out), 1),
                "blood_pressure": round(float(110 + 30 * risk), 1),
                "cholesterol": round(float(170 + 60 * risk), 1),
                "age": age,
                "sex": sex,
                "age_band": age_band,
                "label": y,
            }
        )
        groups.append({"patient_id": pid, "sex": sex, "age_band": age_band, "age": age})

    return pd.DataFrame(rows), pd.DataFrame(groups)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", type=Path, default=Path("data/raw/paper_synthetic_cohort.csv"))
    ap.add_argument("--groups-out", type=Path, default=Path("data/raw/paper_synthetic_groups.csv"))
    ap.add_argument("--n-patients", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    events, groups = generate_cohort(n_patients=args.n_patients, seed=args.seed)
    args.o.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.o, index=False)
    groups.to_csv(args.groups_out, index=False)
    print(f"Wrote {args.o} ({len(events)} rows, {events['patient_id'].nunique()} patients)")
    print(f"Wrote {args.groups_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
