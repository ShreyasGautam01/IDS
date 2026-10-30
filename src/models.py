"""
Machine Learning Baseline Classifiers for Network Intrusion Detection.
Primary Model: Random Forest (with access to individual tree ensembles).
Secondary Baselines: Decision Tree, Logistic Regression.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

class IDSClassifier:
    """Wrapper around scikit-learn models tailored for uncertainty quantification."""
    
    def __init__(self, model_type: str = "rf", n_estimators: int = 100, max_depth: int = 15, random_state: int = 42):
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
        if self.model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                n_jobs=-1,
                random_state=self.random_state,
                class_weight="balanced_subsample"
            )
        elif self.model_type == "dt":
            self.model = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=self.random_state,
                class_weight="balanced"
            )
        elif self.model_type == "lr":
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight="balanced"
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit model to training data."""
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (0: Benign, 1: Attack)."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probability distribution [P(Benign), P(Attack)]."""
        return self.model.predict_proba(X)

    def get_individual_tree_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Extract predictions from every tree in the Random Forest ensemble.
        Returns:
            tree_probs: shape (n_estimators, n_samples, n_classes)
        """
        if self.model_type != "rf":
            raise NotImplementedError("Tree predictions are only available for Random Forest ensembles.")
        
        # Array of predictions from each individual estimator
        # Each tree predict_proba produces shape (n_samples, n_classes)
        tree_probs = np.array([tree.predict_proba(X) for tree in self.model.estimators_])
        return tree_probs
