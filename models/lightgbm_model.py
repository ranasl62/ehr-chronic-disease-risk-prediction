"""Optional LightGBM classifier (install `lightgbm` extra)."""


def train_lgb(X_train, y_train, **kwargs):
    try:
        import lightgbm as lgb
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Install lightgbm: pip install lightgbm") from e

    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=-1,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        **kwargs,
    )
    model.fit(X_train, y_train)
    return model


def make_lgbm_estimator():
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=-1,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
