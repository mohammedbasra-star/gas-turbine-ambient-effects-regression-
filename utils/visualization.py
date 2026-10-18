"""
Publication-grade visualization suite for Gas Turbine Ambient Derating.

Generates 300 DPI publication plots:
1. Derating Curve: Ambient Temperature vs Net Power Output with thermodynamic physics annotations.
2. Parity Plots: Predicted vs Actual Power Output across candidate regression models with ±1% error bands.
3. Thermodynamic Feature Importance: Relative influence of compressor inlet temperature, pressure, vacuum, humidity.
4. Residual Analysis: Error distribution and homoscedasticity across the operating temperature envelope.
5. Thermodynamic Correlation Matrix: Correlation heatmap among operational parameters.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Configure publication-grade styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_derating_curve(
    df: pd.DataFrame,
    model_predictions: Optional[Dict[str, np.ndarray]] = None,
    save_path: Optional[str | Path] = None
) -> plt.Figure:
    """
    Plots the ambient temperature derating curve showing compressor density loss.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # Scatter of empirical plant data
    scatter = ax.scatter(
        df["AT"], df["PE"],
        c=df["V"], cmap="plasma",
        alpha=0.35, s=18, edgecolors="none",
        label="Plant Observations (color = Vacuum)"
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Exhaust Vacuum V (cm Hg)", rotation=270, labelpad=15)

    # If predictions provided, overlay smooth curve
    if model_predictions is not None and "XGBoost" in model_predictions:
        sort_idx = np.argsort(df["AT"].values)
        ax.plot(
            df["AT"].values[sort_idx],
            model_predictions["XGBoost"][sort_idx],
            color="#00E676", linewidth=2.5,
            label="XGBoost Non-linear Derating Fit"
        )

    # Physical annotations
    ax.axvline(15.0, color="#FF5252", linestyle="--", alpha=0.85, linewidth=1.5, label="ISO Standard Rating Point (15°C)")
    ax.annotate(
        "Compressor Density Loss Derating:\nΔPE / ΔT ≈ -2.1 MW / °C",
        xy=(24, 445), xytext=(22, 475),
        arrowprops=dict(arrowstyle="->", color="#D50000", lw=1.8),
        fontsize=10.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFF9C4", ec="#FBC02D", lw=1.2)
    )

    ax.set_xlabel("Ambient Air Temperature AT (°C)", fontweight="bold")
    ax.set_ylabel("Net Electrical Power Output PE (MW)", fontweight="bold")
    ax.set_title("Gas Turbine Ambient Derating Curve (Thermodynamic Loss Profile)", fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.92)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"[Visualization] Saved Derating Curve -> {save_path}")

    return fig


def plot_parity(
    y_test: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    metrics_dict: Dict[str, Dict[str, Any]],
    save_path: Optional[str | Path] = None
) -> plt.Figure:
    """
    Plots multi-model parity charts (y_true vs y_pred) with ±1% error envelope.
    """
    n_models = len(predictions_dict)
    cols = min(2, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 5.0 * rows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    min_val = min(y_test.min(), min(p.min() for p in predictions_dict.values())) - 2
    max_val = max(y_test.max(), max(p.max() for p in predictions_dict.values())) + 2

    # Identity reference line
    id_line = np.linspace(min_val, max_val, 100)

    colors = ["#1976D2", "#7B1FA2", "#388E3C", "#E65100"]

    for idx, (m_name, y_pred) in enumerate(predictions_dict.items()):
        ax = axes[idx]
        c = colors[idx % len(colors)]

        # Parity scatter
        ax.scatter(y_test, y_pred, color=c, alpha=0.35, s=14, edgecolors="none")

        # 1:1 perfect parity
        ax.plot(id_line, id_line, "k-", linewidth=1.5, label="Ideal Parity (y = x)")
        ax.plot(id_line, id_line * 1.01, "k--", alpha=0.6, linewidth=1.0, label="±1% Error Band")
        ax.plot(id_line, id_line * 0.99, "k--", alpha=0.6, linewidth=1.0)
        ax.fill_between(id_line, id_line * 0.99, id_line * 1.01, color="#E0E0E0", alpha=0.4)

        m = metrics_dict.get(m_name, {})
        r2 = m.get("r2", 0.0)
        rmse = m.get("rmse", 0.0)
        mape = m.get("mape", 0.0)

        stats_text = f"R² = {r2:.4f}\nRMSE = {rmse:.2f} MW\nMAPE = {mape:.2f}%"
        ax.text(
            0.05, 0.92, stats_text,
            transform=ax.transAxes, verticalalignment="top",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9)
        )

        ax.set_title(m_name, fontweight="bold")
        ax.set_xlabel("Actual Net Power Output (MW)")
        ax.set_ylabel("Predicted Net Power Output (MW)")
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=8.5)

    # Hide extra subplots if any
    for j in range(n_models, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Model Parity Comparison on Unseen Test Operational Data", fontweight="bold", y=1.00)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"[Visualization] Saved Parity Plots -> {save_path}")

    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    model_name: str = "XGBoost",
    save_path: Optional[str | Path] = None
) -> plt.Figure:
    """
    Plots horizontal bar chart of thermodynamic feature importance with physical explanations.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    # Sort descending
    sort_idx = np.argsort(importances)
    sorted_features = [feature_names[i] for i in sort_idx]
    sorted_imps = importances[sort_idx]

    feature_descriptions = {
        "AT": "Ambient Temperature (Compressor Air Density Loss)",
        "V": "Exhaust Vacuum (Condenser Backpressure & Rankine Expansion)",
        "AP": "Ambient Pressure (Compressor Mass Flow Elevation)",
        "RH": "Relative Humidity (Air Psychrometrics & Heat Capacity Ratio)"
    }
    labels = [f"{f} : {feature_descriptions.get(f, f)}" for f in sorted_features]

    colors = ["#42A5F5", "#26A69A", "#FFA726", "#EF5350"]
    bars = ax.barh(labels, sorted_imps, color=colors[:len(labels)], edgecolor="black", linewidth=0.8, height=0.6)

    # Add percentage text on each bar
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.01, bar.get_y() + bar.get_height() / 2.0,
            f"{width * 100:.1f}%",
            va="center", ha="left", fontsize=10.5, fontweight="bold"
        )

    ax.set_xlim(0, max(sorted_imps) * 1.18)
    ax.set_xlabel("Normalized Feature Importance (Thermodynamic Contribution)", fontweight="bold")
    ax.set_title(f"Thermodynamic Parameter Sensitivity ({model_name})", fontweight="bold", pad=12)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"[Visualization] Saved Feature Importance -> {save_path}")

    return fig


def plot_residuals(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    feature_names: List[str],
    model_name: str = "XGBoost",
    save_path: Optional[str | Path] = None
) -> plt.Figure:
    """
    Analyzes residuals (e = y_true - y_pred) across ambient temperature and error distribution.
    """
    residuals = y_test - y_pred
    at_idx = feature_names.index("AT") if "AT" in feature_names else 0
    at_vals = X_test[:, at_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.5))

    # Scatter vs Ambient Temp
    ax1.scatter(at_vals, residuals, color="#0288D1", alpha=0.4, s=15, edgecolors="none")
    ax1.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax1.set_xlabel("Ambient Temperature AT (°C)", fontweight="bold")
    ax1.set_ylabel("Residual Error e = y_true - y_pred (MW)", fontweight="bold")
    ax1.set_title("Homoscedasticity Check vs Temperature", fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Histogram & KDE of residuals
    sns.histplot(residuals, kde=True, color="#7B1FA2", ax=ax2, bins=35, stat="density")
    mean_err = np.mean(residuals)
    std_err = np.std(residuals)
    ax2.axvline(mean_err, color="black", linestyle="--", label=f"Mean = {mean_err:.2f} MW")
    ax2.set_xlabel("Residual Error (MW)", fontweight="bold")
    ax2.set_ylabel("Probability Density", fontweight="bold")
    ax2.set_title(f"Error Distribution (σ = {std_err:.2f} MW)", fontweight="bold")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(f"Residual and Error Analysis for {model_name}", fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"[Visualization] Saved Residuals -> {save_path}")

    return fig


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Optional[str | Path] = None) -> plt.Figure:
    """
    Plots thermodynamic Pearson correlation matrix across variables.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    corr = df[["AT", "V", "AP", "RH", "PE"]].corr()

    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, square=True, linewidths=1.0,
        cbar_kws={"shrink": 0.82, "label": "Pearson Correlation Coefficient"},
        ax=ax
    )
    ax.set_title("Thermodynamic Parameter Correlation Matrix", fontweight="bold", pad=12)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"[Visualization] Saved Correlation Heatmap -> {save_path}")

    return fig
