"""
Unit Tests for the 5 ML Models
"""

import os
import pytest
import numpy as np
from backend.src.ml.models import get_all_models

def test_all_five_models_train_and_predict():
    # Synthetic dataset
    np.random.seed(42)
    X = np.random.randn(200, 9).astype(np.float32)
    # Target: 1 if feature 0 is small and feature 1 is small
    y = ((X[:, 0] < 0.5) & (X[:, 1] < 0.5)).astype(np.int32)
    
    models = get_all_models()
    assert len(models) == 5
    expected_keys = {"logistic_regression", "random_forest", "gradient_boosting", "mlp", "svm"}
    assert set(models.keys()) == expected_keys
    
    for key, model in models.items():
        model.fit(X, y)
        assert model.is_fitted
        
        probs = model.predict_proba(X[:10])
        assert probs.shape == (10, 2)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
        
        preds = model.predict(X[:10])
        assert len(preds) == 10
        assert set(preds).issubset({0, 1})
