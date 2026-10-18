"""
Performance metrics and thermodynamic model benchmark evaluation.

Calculates:
- R^2 (Coefficient of Determination)
- RMSE (Root Mean Squared Error, MW)
- MAE (Mean Absolute Error, MW)
- MAPE (Mean Absolute Percentage Error, %)
- Max Absolute Error (MW)
"""

from __future__ import annotations

import sys
from typing import Dict, Any, List
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, Any]:
    """
    Computes regression performance metrics between true and predicted power output.
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()

    diff = y_t - y_p
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))
    max_err = np.max(np.abs(diff))

    # MAPE (%)
    mape = np.mean(np.abs(diff / y_t)) * 100.0

    # R^2 score
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "model_name": model_name,
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "max_err": float(max_err)
    }


def format_metrics_table(metrics_list: List[Dict[str, Any]]) -> str:
    """
    Renders an ASCII / Markdown table comparing regression models.
    """
    header = "| Model Name                     | R² Score | RMSE (MW) | MAE (MW) | MAPE (%) | Max Err (MW) |"
    divider = "|:-------------------------------|:--------:|:---------:|:--------:|:--------:|:------------:|"
    rows = []
    for m in metrics_list:
        rows.append(
            f"| {m['model_name']:<30} | {m['r2']:.4f}   | {m['rmse']:.3f}     | {m['mae']:.3f}    | {m['mape']:.2f}%    | {m['max_err']:.3f}        |"
        )
    return "\n".join([header, divider] + rows)


if __name__ == "__main__":
    y_dummy_true = np.array([450.0, 460.0, 470.0, 480.0])
    y_dummy_pred = np.array([451.2, 459.1, 471.5, 479.2])
    print(format_metrics_table([calculate_metrics(y_dummy_true, y_dummy_pred, "Test Linear Model")]))
