"""Compare ARIMA configurations with rolling one-day-ahead backtesting.

Run from the forecasting folder:
    python evaluate_arima_orders.py production.csv

Each test prediction uses only records dated before its target date.
"""

from __future__ import annotations
import csv
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from statsmodels.tsa.arima.model import ARIMA

DATE_COLUMNS = ("date", "production_date", "ds")
EGG_COLUMNS = ("eggs", "eggs_produced", "y")
EGGS_PER_TRAY = 30
ARIMA_ORDERS = [(1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 1), (1, 1, 2)]


def load_daily_totals(path: Path):
    totals = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"{path} has no header row.")
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        date_key = next((fields[n] for n in DATE_COLUMNS if n in fields), None)
        egg_key = next((fields[n] for n in EGG_COLUMNS if n in fields), None)
        if date_key is None or egg_key is None:
            raise SystemExit("CSV needs a date column and an eggs column.")
        for row in reader:
            raw_date = (row.get(date_key) or "").strip()
            raw_eggs = (row.get(egg_key) or "").strip()
            if not raw_date or not raw_eggs:
                continue
            day = datetime.strptime(raw_date[:10], "%Y-%m-%d")
            eggs = float(raw_eggs)
            if eggs < 0:
                raise SystemExit(f"Negative egg total found on {day:%Y-%m-%d}.")
            totals[day] = totals.get(day, 0.0) + eggs
    rows = sorted(totals.items())
    if len(rows) < 60:
        raise SystemExit("At least 60 daily records are recommended.")
    return rows


def round_to_tray(value: float) -> int:
    return int(round(max(0.0, value) / EGGS_PER_TRAY) * EGGS_PER_TRAY)


def predict_one_step(history, order):
    values = [eggs for _, eggs in history]
    fitted = ARIMA(
        values, order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()
    forecast = fitted.forecast(steps=1)
    return float(round_to_tray(float(forecast[0])))


def calculate_metrics(actuals, predictions):
    errors = [a - p for a, p in zip(actuals, predictions)]
    mae = mean(abs(e) for e in errors)
    rmse = (mean(e * e for e in errors)) ** 0.5
    mape = mean(abs(a - p) / abs(a) * 100
                for a, p in zip(actuals, predictions) if a != 0)
    avg_actual = mean(actuals)
    ss_tot = sum((a - avg_actual) ** 2 for a in actuals)
    ss_res = sum(e * e for e in errors)
    r2 = float("nan") if ss_tot == 0 else 1 - ss_res / ss_tot
    return mae, rmse, mape, r2


def main(argv):
    if len(argv) != 2:
        print("Usage: python evaluate_arima_orders.py production.csv", file=sys.stderr)
        return 1
    csv_path = Path(argv[1])
    if not csv_path.is_file():
        print(f"File not found: {csv_path}", file=sys.stderr)
        return 1

    rows = load_daily_totals(csv_path)
    split_index = int(len(rows) * 0.8)
    train_count = split_index
    test_rows = rows[split_index:]
    if train_count < 30 or not test_rows:
        raise SystemExit("Not enough records for chronological 80/20 split.")

    output_rows = []
    summaries = []
    for order in ARIMA_ORDERS:
        name = f"ARIMA{order}"
        actuals, predictions = [], []
        failed = 0
        print(f"Evaluating {name}...", flush=True)
        for index, (target, actual) in enumerate(test_rows, start=1):
            history = rows[:split_index + index - 1]
            try:
                predicted = predict_one_step(history, order)
            except Exception as exc:
                failed += 1
                print(f"  Warning: {target:%Y-%m-%d} failed: {type(exc).__name__}", flush=True)
                continue
            actuals.append(actual)
            predictions.append(predicted)
            output_rows.append((name, target.strftime("%Y-%m-%d"), actual, predicted))
        if actuals:
            summaries.append((name, len(actuals), failed,
                              calculate_metrics(actuals, predictions)))

    if not summaries:
        raise SystemExit("All ARIMA configurations failed.")

    out = Path("arima_order_comparison.csv")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "date", "actual_eggs", "predicted_eggs"])
        writer.writerows(output_rows)

    print(f"\nRecords: {len(rows)}")
    print(f"Training: {rows[0][0]:%Y-%m-%d} to {rows[split_index-1][0]:%Y-%m-%d} ({train_count})")
    print(f"Testing: {test_rows[0][0]:%Y-%m-%d} to {test_rows[-1][0]:%Y-%m-%d} ({len(test_rows)})")
    print("\nARIMA comparison:")
    print(f"{'Model':<16} {'Tested':>7} {'Failed':>7} {'MAE':>10} {'RMSE':>10} {'MAPE':>9} {'R²':>9}")
    for name, tested, failed, metrics in summaries:
        mae, rmse, mape, r2 = metrics
        print(f"{name:<16} {tested:>7} {failed:>7} {mae:>10.2f} {rmse:>10.2f} {mape:>8.2f}% {r2:>9.4f}")
    print(f"\nSaved detailed predictions: {out.resolve()}")
    print("Lower MAE/RMSE/MAPE and higher R² are preferred. Check failed counts too.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
