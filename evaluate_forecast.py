"""Compare NB Poultry Farm egg forecasting models using chronological backtesting.

Usage (from the forecasting project folder):
    python evaluate_forecast.py production.csv
    python evaluate_forecast.py production.csv --start 2026-10-01 --end 2026-10-31

Models:
- three_year_linear_regression: mirrors the primary model in src/service.py
- arima: ARIMA(1, 1, 1) comparison baseline
- legacy_linear_regression: optional reference model using recent daily trend

Outputs:
- evaluation_results.csv
- actual_vs_predicted.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Callable

import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

from src.regression import fit_line, predict_y

DATE_COLUMNS = ("date", "production_date", "ds")
EGG_COLUMNS = ("eggs", "eggs_produced", "y")
EGGS_PER_TRAY = 30
HISTORY_YEARS = 3
ARIMA_ORDER = (1, 1, 1)


def load_daily_totals(path: Path) -> list[tuple[datetime, float]]:
    totals: dict[datetime, float] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"{path} has no header row.")

        fields = {name.strip().lower(): name for name in reader.fieldnames}
        date_key = next((fields[name] for name in DATE_COLUMNS if name in fields), None)
        egg_key = next((fields[name] for name in EGG_COLUMNS if name in fields), None)
        if date_key is None or egg_key is None:
            raise SystemExit(
                f"{path} needs a date column and an eggs column. "
                f"Found: {', '.join(reader.fieldnames)}"
            )

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
    if len(rows) < 30:
        raise SystemExit(f"{path} needs at least 30 daily totals for backtesting.")
    return rows


def parse_iso_date(value: str, flag: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise SystemExit(f"{flag} must be YYYY-MM-DD, got {value!r}.") from exc


def parse_cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chronological backtesting for NB Poultry Farm forecasting."
    )
    parser.add_argument("csv_path", help="Path to production.csv")
    parser.add_argument("--start", metavar="YYYY-MM-DD", help="First test date (inclusive)")
    parser.add_argument("--end", metavar="YYYY-MM-DD", help="Last test date (inclusive)")
    return parser.parse_args(argv[1:])


def select_test_window(
    rows: list[tuple[datetime, float]],
    start: datetime | None,
    end: datetime | None,
) -> tuple[int, int]:
    """Return (first_test_index, stop_index) for the scored window."""
    if start is None and end is None:
        first = int(len(rows) * 0.8)
        return first, len(rows)

    if start is None or end is None:
        raise SystemExit("Pass both --start and --end, or neither.")
    if start > end:
        raise SystemExit("--start must be on or before --end.")

    first = next((index for index, (day, _) in enumerate(rows) if day >= start), None)
    if first is None:
        raise SystemExit(f"No records on or after --start {start:%Y-%m-%d}.")
    last = next((index for index in range(len(rows) - 1, -1, -1) if rows[index][0] <= end), None)
    if last is None or last < first:
        raise SystemExit(f"No records between {start:%Y-%m-%d} and {end:%Y-%m-%d}.")
    if first == 0:
        raise SystemExit("Need harvest days before --start so the models have history.")
    return first, last + 1


def round_to_tray(value: float) -> int:
    value = max(0.0, value)
    return int(round(value / EGGS_PER_TRAY) * EGGS_PER_TRAY)


def predict_three_year(
    history: list[tuple[datetime, float]], target: datetime
) -> float | None:
    """Mirror the primary service model's same-calendar-day year trend."""
    matching: dict[int, float] = {}
    for day, eggs in history:
        if day < target and day.month == target.month and day.day == target.day:
            matching[day.year] = eggs

    samples: list[tuple[int, float]] = []
    if matching:
        years = sorted(matching)[-HISTORY_YEARS:]
        samples = [(year, matching[year]) for year in years]
    else:
        # Match service.py fallback: average nearby day-of-year rows per year,
        # within +/- 3 days of the target's day-of-year.
        target_doy = target.timetuple().tm_yday
        nearby: dict[int, list[float]] = {}
        for day, eggs in history:
            if day >= target:
                continue
            if abs(day.timetuple().tm_yday - target_doy) <= 3:
                nearby.setdefault(day.year, []).append(eggs)
        years = sorted(nearby)[-HISTORY_YEARS:]
        samples = [(year, mean(nearby[year])) for year in years]

    if not samples:
        return None
    if len(samples) == 1:
        return float(round_to_tray(samples[0][1]))

    xs = [float(year) for year, _ in samples]
    ys = [eggs for _, eggs in samples]
    slope, intercept = fit_line(xs, ys)
    return float(round_to_tray(predict_y(slope, intercept, float(target.year))))


def predict_legacy(
    history: list[tuple[datetime, float]], target: datetime, window: int = 7
) -> float | None:
    if len(history) < 2:
        return None
    train = history[-window:]
    origin = train[0][0]
    xs = [(day - origin).days for day, _ in train]
    ys = [eggs for _, eggs in train]
    slope, intercept = fit_line([float(x) for x in xs], ys)
    target_x = (target - origin).days
    return float(predict_y(slope, intercept, float(target_x)))


def predict_arima(
    history: list[tuple[datetime, float]], target: datetime
) -> float | None:
    if len(history) < 30:
        return None
    values = [eggs for _, eggs in history]
    try:
        fitted = ARIMA(
            values,
            order=ARIMA_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()
        forecast = float(fitted.forecast(steps=1)[0])
        return float(round_to_tray(forecast))
    except Exception:
        return None


def calculate_metrics(actuals: list[float], predictions: list[float]) -> dict[str, float | int]:
    if not actuals:
        return {"tested": 0, "mae": float("nan"), "rmse": float("nan"),
                "mape": float("nan"), "r2": float("nan")}

    errors = [actual - predicted for actual, predicted in zip(actuals, predictions)]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = (sum(error * error for error in errors) / len(errors)) ** 0.5
    nonzero = [(actual, predicted) for actual, predicted in zip(actuals, predictions) if actual != 0]
    mape = (
        sum(abs(actual - predicted) / abs(actual) * 100 for actual, predicted in nonzero)
        / len(nonzero)
        if nonzero else float("nan")
    )
    avg_actual = mean(actuals)
    ss_tot = sum((actual - avg_actual) ** 2 for actual in actuals)
    ss_res = sum(error * error for error in errors)
    r2 = float("nan") if ss_tot == 0 else 1 - ss_res / ss_tot
    return {"tested": len(actuals), "mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def main(argv: list[str]) -> int:
    args = parse_cli(argv)
    csv_path = Path(args.csv_path)
    if not csv_path.is_file():
        print(f"File not found: {csv_path}", file=sys.stderr)
        return 1

    start = parse_iso_date(args.start, "--start") if args.start else None
    end = parse_iso_date(args.end, "--end") if args.end else None

    rows = load_daily_totals(csv_path)
    split_index, stop_index = select_test_window(rows, start, end)
    if start is None and (split_index < 30 or split_index >= len(rows)):
        raise SystemExit("Not enough records for the chronological 80/20 split.")
    if not rows[split_index:stop_index]:
        raise SystemExit("The selected date range has no harvest days to score.")

    test_rows = rows[split_index:stop_index]
    model_functions: dict[str, Callable] = {
        "Three-Year Linear Regression": predict_three_year,
        "ARIMA(1,1,1)": predict_arima,
        "Legacy Linear Regression": predict_legacy,
    }

    results: dict[str, list[tuple[str, float, float]]] = {}
    metrics_by_model: dict[str, dict[str, float | int]] = {}

    for model_name, predictor in model_functions.items():
        model_rows: list[tuple[str, float, float]] = []
        for target, actual in test_rows:
            # Only records dated before the target are made available to the model.
            history = [(day, eggs) for day, eggs in rows if day < target]
            predicted = predictor(history, target)
            if predicted is not None:
                model_rows.append((target.strftime("%Y-%m-%d"), actual, predicted))

        results[model_name] = model_rows
        metrics_by_model[model_name] = calculate_metrics(
            [item[1] for item in model_rows],
            [item[2] for item in model_rows],
        )

    output_csv = Path("evaluation_results.csv")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "date", "actual_eggs", "predicted_eggs"])
        for model_name, model_rows in results.items():
            for date, actual, predicted in model_rows:
                writer.writerow([model_name, date, actual, round(predicted, 2)])

    plt.figure(figsize=(14, 6))
    plt.plot(
        [day.strftime("%Y-%m-%d") for day, _ in test_rows],
        [eggs for _, eggs in test_rows],
        label="Actual production",
        linewidth=2,
    )
    for model_name, model_rows in results.items():
        if model_rows:
            plt.plot(
                [item[0] for item in model_rows],
                [item[2] for item in model_rows],
                label=model_name,
                alpha=0.8,
            )
    plt.title("Actual vs. Predicted Daily Egg Production")
    plt.xlabel("Date")
    plt.ylabel("Eggs")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    graph_path = Path("actual_vs_predicted.png")
    plt.savefig(graph_path, dpi=200)
    plt.close()

    print(f"Records: {len(rows)}")
    print(f"Training period: {rows[0][0]:%Y-%m-%d} to {rows[split_index - 1][0]:%Y-%m-%d}")
    print(f"Testing period: {test_rows[0][0]:%Y-%m-%d} to {test_rows[-1][0]:%Y-%m-%d}")
    print("\nEvaluation results:")
    print(f"{'Model':<30} {'Tested':>7} {'MAE':>12} {'RMSE':>12} {'MAPE':>10} {'R²':>10}")
    for model_name, metrics in metrics_by_model.items():
        print(
            f"{model_name:<30} {metrics['tested']:>7} "
            f"{metrics['mae']:>12.2f} {metrics['rmse']:>12.2f} "
            f"{metrics['mape']:>9.2f}% {metrics['r2']:>10.4f}"
        )

    print(f"\nSaved predictions: {output_csv.resolve()}")
    print(f"Saved graph: {graph_path.resolve()}")
    print("\nNote: ARIMA is a comparison model. Do not replace the system model "
          "until the results and methodology have been reviewed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
