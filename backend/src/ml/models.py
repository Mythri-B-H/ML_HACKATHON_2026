"""
PathSnap - The 5 Machine Learning Models for Candidate Road Classification
1. Logistic Regression
2. Random Forest
3. Histogram Gradient Boosting (GBDT)
4. Multi-Layer Perceptron (MLP)
5. Support Vector Classifier (Linear/RBF SVM with Calibrated Probabilities)
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import joblib
import numpy as np
from typing import Dict, Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from backend.src.ml.features import FEATURE_NAMES

class BaseRoadModel:
    def __init__(self, name: str):
        self.name = name
        self.scaler = StandardScaler()
        self.model = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)
        
    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"name": self.name, "scaler": self.scaler, "model": self.model, "is_fitted": self.is_fitted}, filepath)
        
    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.name = data["name"]
        self.scaler = data["scaler"]
        self.model = data["model"]
        self.is_fitted = data["is_fitted"]

class LogisticRegressionModel(BaseRoadModel):
    def __init__(self):
        super().__init__("Logistic Regression")
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

class RandomForestModel(BaseRoadModel):
    def __init__(self):
        super().__init__("Random Forest")
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_split=5,
            class_weight="balanced", n_jobs=1, random_state=42
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

class HistGradientBoostingModel(BaseRoadModel):
    def __init__(self):
        super().__init__("HistGradientBoosting")
        self.model = HistGradientBoostingClassifier(
            max_iter=100, max_depth=8, learning_rate=0.1,
            class_weight="balanced", random_state=42
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

class MLPClassifierModel(BaseRoadModel):
    def __init__(self):
        super().__init__("Multi-Layer Perceptron")
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu",
            max_iter=300, random_state=42, early_stopping=True
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

class SupportVectorClassifierModel(BaseRoadModel):
    def __init__(self):
        super().__init__("Support Vector Machine")
        # Use CalibratedClassifierCV with LinearSVC for fast, calibrated probabilities
        base_svc = LinearSVC(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
        self.model = CalibratedClassifierCV(estimator=base_svc, cv=3)
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

def get_all_models() -> Dict[str, BaseRoadModel]:
    return {
        "logistic_regression": LogisticRegressionModel(),
        "random_forest": RandomForestModel(),
        "gradient_boosting": HistGradientBoostingModel(),
        "mlp": MLPClassifierModel(),
        "svm": SupportVectorClassifierModel()
    }
