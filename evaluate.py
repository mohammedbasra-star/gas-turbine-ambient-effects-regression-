"""
Inference and Operational Verification CLI for Gas Turbine Derating Model.

Allows engineering operators and researchers to:
1. Verify the saved best model on test operational data.
2. Query specific ambient operating conditions (AT, V, AP, RH) to estimate power output and derating penalty.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import load_gas_turbine_data
from utils.metrics import calculate_metrics

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Standard ISO reference: 15°C, 1013.25 mbar, 60% RH
ISO_BASE_POWER_MW = 485.0


def evaluate_custom_condition(
    at: float,
    v: float,
    ap: float,
    rh: float,
    model_dir: Path = PROJECT_ROOT / "models"
) -> None:
    """
    Evaluates power output and derating penalty for a single operating point.
    """
    model_file = model_dir / "best_model.joblib"
    scaler_file = model_dir / "scaler.joblib"

    if not model_file.exists() or not scaler_file.exists():
        print(f"[Error] Model artifacts not found in {model_dir}. Run `python train.py` first.")
        sys.exit(1)

    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)

    x_raw = np.array([[at, v, ap, rh]])
    x_scaled = scaler.transform(x_raw)
    pred_power = float(model.predict(x_scaled)[0])

    derating = pred_power - ISO_BASE_POWER_MW
    derating_pct = (derating / ISO_BASE_POWER_MW) * 100.0

    print("=" * 65)
    print("  GAS TURBINE OPERATIONAL CONDITION INFERENCE")
    print("=" * 65)
    print(f"  Ambient Temperature (AT) : {at:6.2f} °C")
    print(f"  Exhaust Vacuum (V)       : {v:6.2f} cm Hg")
    print(f"  Ambient Pressure (AP)    : {ap:6.2f} mbar")
    print(f"  Relative Humidity (RH)   : {rh:6.2f} %")
    print("-" * 65)
    print(f"  Predicted Net Power (PE) : {pred_power:6.2f} MW")
    if derating < 0:
        print(f"  Thermal Derating Penalty : {derating:6.2f} MW ({derating_pct:.2f}% vs ISO 485 MW)")
    else:
        print(f"  Cold Air Surplus Output  : +{derating:6.2f} MW (+{derating_pct:.2f}% vs ISO 485 MW)")
    print("=" * 65)


def run_evaluation_benchmark(
    data_path: Path = PROJECT_ROOT / "data" / "gas_turbine_ambient_dataset.csv",
    model_dir: Path = PROJECT_ROOT / "models"
) -> None:
    """
    Evaluates saved model on full dataset.
    """
    model_file = model_dir / "best_model.joblib"
    scaler_file = model_dir / "scaler.joblib"
    meta_file = model_dir / "metadata.json"

    if not model_file.exists():
        print(f"[Error] Model artifact not found at {model_file}. Run `python train.py` first.")
        sys.exit(1)

    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    df = load_gas_turbine_data(data_path)

    features = ["AT", "V", "AP", "RH"]
    X = scaler.transform(df[features].values)
    y_true = df["PE"].values
    y_pred = model.predict(X)

    metrics = calculate_metrics(y_true, y_pred, model_name="Best Model")
    print("\n" + "=" * 65)
    print("  SAVED MODEL VERIFICATION REPORT")
    print("=" * 65)
    if meta_file.exists():
        with open(meta_file, "r") as f:
            meta = json.load(f)
        print(f"  Architecture:  {meta.get('best_model')}")
        print(f"  Trained date:  {meta.get('timestamp')}")
    print(f"  R² Score:      {metrics['r2']:.4f}")
    print(f"  RMSE:          {metrics['rmse']:.3f} MW")
    print(f"  MAE:           {metrics['mae']:.3f} MW")
    print(f"  MAPE:          {metrics['mape']:.2f}%")
    print(f"  Max Error:     {metrics['max_err']:.3f} MW")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gas Turbine Derating Model")
    parser.add_argument("--at", type=float, default=None, help="Ambient Temperature (°C)")
    parser.add_argument("--v", type=float, default=None, help="Exhaust Vacuum (cm Hg)")
    parser.add_argument("--ap", type=float, default=None, help="Ambient Pressure (mbar)")
    parser.add_argument("--rh", type=float, default=None, help="Relative Humidity (%)")
    parser.add_argument("--benchmark", action="store_true", help="Run full evaluation benchmark")

    args = parser.parse_args()

    if args.at is not None and args.v is not None and args.ap is not None and args.rh is not None:
        evaluate_custom_condition(args.at, args.v, args.ap, args.rh)
    else:
        run_evaluation_benchmark()


if __name__ == "__main__":
    main()
