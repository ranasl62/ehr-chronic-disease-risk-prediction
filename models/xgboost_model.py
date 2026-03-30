import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score


def xgb_scale_pos_weight(y_train) -> float:
    y = np.asarray(y_train).astype(int).ravel()
    if len(y) == 0:
        return 1.0
    n_pos = max(int(y.sum()), 1)
    n_neg = max(int(len(y) - y.sum()), 1)
    return float(n_neg / n_pos)


def make_xgb_classifier(*, scale_pos_weight: float = 1.0) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )


def train_xgb(X_train, y_train) -> xgb.XGBClassifier:
    model = make_xgb_classifier(scale_pos_weight=xgb_scale_pos_weight(y_train))
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test) -> dict[str, float]:
    preds = model.predict_proba(X_test)[:, 1]
    y = np.asarray(y_test)
    if len(np.unique(y)) < 2:
        return {"AUC": float("nan")}
    auc = roc_auc_score(y_test, preds)
    return {"AUC": float(auc)}


def train_model(X_train, y_train, **kwargs):
    """Alias used by optional callers; forwards kwargs to XGBClassifier when needed."""
    return train_xgb(X_train, y_train)
