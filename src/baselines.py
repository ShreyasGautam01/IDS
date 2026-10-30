"""Dedicated novelty baseline for open-set comparison."""

from sklearn.ensemble import IsolationForest


def fit_isolation_forest(X_train, seed: int):
    model = IsolationForest(
        n_estimators=200,
        max_samples=min(256, len(X_train)),
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def anomaly_score(model, X):
    # Higher score = more anomalous/uncertain.
    return -model.decision_function(X)
