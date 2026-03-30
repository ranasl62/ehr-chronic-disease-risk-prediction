from sklearn.ensemble import RandomForestClassifier


def train_rf(X_train, y_train, random_state: int = 42) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)
    return model
