"""
Thermodynamic Regression Models for Gas Turbine Ambient Derating.

Implements a systematic hierarchy of classical statistical and machine learning models:
1. Linear Regression (Baseline thermodynamic sensitivity gradient)
2. Polynomial Ridge Regression (capturing 2nd-order thermodynamic interaction effects)
3. Random Forest Regressor (Nonlinear ensemble bagging)
4. XGBoost Regressor (Gradient-boosted decision trees on residual errors)
5. Multi-Layer Perceptron Regressor (MLP neural network representation)
"""

from __future__ import annotations

import sys
from typing import Dict, Any, Optional

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def get_all_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Initializes and returns the complete benchmark suite of regression models.
    """
    models: Dict[str, Any] = {}

    # 1. Baseline Ordinary Least Squares
    models["Linear Regression (OLS)"] = LinearRegression()

    # 2. Polynomial Ridge (Degree 2 interactions: AT^2, AT*AP, AT*V, RH*V)
    models["Polynomial Ridge (Degree 2)"] = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("ridge", Ridge(alpha=10.0, random_state=random_state))
    ])

    # 3. Random Forest (Ensemble Bagging)
    models["Random Forest Regressor"] = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state
    )

    # 4. Multi-Layer Perceptron (Neural Baseline)
    models["Multi-Layer Perceptron (MLP)"] = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.01,
        learning_rate_init=0.005,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state
    )

    # 5. XGBoost (Gradient Boosted Trees)
    if HAS_XGBOOST:
        models["XGBoost Regressor"] = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.06,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.05,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=random_state
        )

    return models


def get_model(name: str, random_state: int = 42) -> Any:
    """
    Returns a single model instance by name.
    """
    all_models = get_all_models(random_state=random_state)
    if name not in all_models:
        raise KeyError(f"Model '{name}' not found. Available: {list(all_models.keys())}")
    return all_models[name]
