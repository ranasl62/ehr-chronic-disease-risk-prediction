"""
Probability calibration for clinical risk models (isotonic / Platt-style via sklearn).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV


def calibrate_model(
    estimator: Any,
    X_train,
    y_train,
    *,
    method: str = "isotonic",
    cv: int = 3,
) -> Any:
    """
    Fit CalibratedClassifierCV around an **unfitted** base `estimator`.

    Refits the base learner inside each CV fold, then fits calibrators.
    """
    n = len(y_train)
    if n < 6:
        estimator.fit(X_train, y_train)
        return estimator
    cv_eff = min(cv, max(2, n // 3))
    try:
        calibrated = CalibratedClassifierCV(
            estimator=estimator,
            method=method,
            cv=cv_eff,
        )
        calibrated.fit(X_train, y_train)
        return calibrated
    except ValueError:
        estimator.fit(X_train, y_train)
        return estimator


def compute_lead_time_days(pred_times: Any, diagnosis_times: Any) -> pd.Series:
    """
    Lead time = time from prediction (risk flag) to diagnosis (or confirmatory event).

    Parameters
    ----------
    pred_times, diagnosis_times :
        Sequences parseable by pandas.to_datetime, aligned row-wise.
    """
    p = pd.to_datetime(pd.Series(pred_times), utc=False)
    d = pd.to_datetime(pd.Series(diagnosis_times), utc=False)
    delta = d - p
    return delta.dt.days
