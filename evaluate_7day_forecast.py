"""Seven-day-ahead backtesting for NB Poultry Farm forecasting.

Run from the forecasting folder:
    python evaluate_7day_forecast.py production.csv
    python evaluate_7day_forecast.py production.csv --start 2026-10-01 --end 2026-10-07

Without --start/--end, the last 20% of the file is the test period.
With a date range, only that window is scored. History is still every
record dated before each 7-day block starts.

The test period is split into consecutive, non-overlapping 7-day blocks.
For each block, ARIMA(2,1,1) is given only records dated before the block starts.
No actual values from the seven forecast dates are used to generate predictions.

The live forecasting model in src/service.py is ARIMA(2,1,1).
This script scores that same ARIMA(2,1,1) model.

Outputs:
- seven_day_backtest_results.csv
- seven_day_actual_vs_predicted.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

DATE_COLUMNS = ("date", "production_date", "ds")
EGG_COLUMNS = ("eggs", "eggs_produced", "y")
EGGS_PER_TRAY = 30
ARIMA_ORDER = (2, 1, 1)
HORIZON_DAYS = 7


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
    if len(rows) < 60:
        raise SystemExit("At least 60 daily records are recommended.")
    # The source dataset is expected to contain one record for every calendar day.
    for previous, current in zip(rows, rows[1:]):
        if (current[0] - previous[0]).days != 1:
            raise SystemExit(
                f"Missing calendar date between {previous[0]:%Y-%m-%d} "
                f"and {current[0]:%Y-%m-%d}. Fill/check the dataset before backtesting."
            )
    return rows


def parse_iso_date(value: str, flag: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise SystemExit(f"{flag} must be YYYY-MM-DD, got {value!r}.") from exc


def parse_cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seven-day-ahead backtesting for NB Poultry Farm forecasting."
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
        first = ((first + HORIZON_DAYS - 1) // HORIZON_DAYS) * HORIZON_DAYS
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


def predict_arima_seven_days(
    history: list[tuple[datetime, float]]
) -> list[float]:
    values = [eggs for _, eggs in history]
    fitted = ARIMA(
        values,
        order=ARIMA_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()
    raw_forecast = fitted.forecast(steps=HORIZON_DAYS)
    return [float(round_to_tray(float(value))) for value in raw_forecast]


def calculate_metrics(actuals: list[float], predictions: list[float]) -> dict[str, float]:
    errors = [actual - predicted for actual, predicted in zip(actuals, predictions)]
    mae = mean(abs(error) for error in errors)
    rmse = (mean(error * error for error in errors)) ** 0.5
    nonzero_pairs = [
        (actual, predicted)
        for actual, predicted in zip(actuals, predictions)
        if actual != 0
    ]
    mape = mean(
        abs(actual - predicted) / abs(actual) * 100
        for actual, predicted in nonzero_pairs
    ) if nonzero_pairs else float("nan")
    avg_actual = mean(actuals)
    ss_tot = sum((actual - avg_actual) ** 2 for actual in actuals)
    ss_res = sum(error * error for error in errors)
    r2 = float("nan") if ss_tot == 0 else 1 - ss_res / ss_tot
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def main(argv: list[str]) -> int:
    args = parse_cli(argv)
    csv_path = Path(args.csv_path)
    if not csv_path.is_file():
        print(f"File not found: {csv_path}", file=sys.stderr)
        return 1

    start = parse_iso_date(args.start, "--start") if args.start else None
    end = parse_iso_date(args.end, "--end") if args.end else None

    rows = load_daily_totals(csv_path)
    first_test_index, stop_index = select_test_window(rows, start, end)
    if start is None and (first_test_index < 30 or first_test_index >= len(rows)):
        raise SystemExit("Not enough data for an 80/20 split and seven-day backtesting.")

    test_rows = rows[first_test_index:stop_index]
    full_blocks = len(test_rows) // HORIZON_DAYS
    if full_blocks == 0:
        raise SystemExit("The test period must contain at least seven days.")

    result_rows: list[tuple[str, str, float, float, int]] = []
    model_actuals: dict[str, list[float]] = {
        f"ARIMA{ARIMA_ORDER}": [],
    }
    model_predictions: dict[str, list[float]] = {
        f"ARIMA{ARIMA_ORDER}": [],
    }
    failed_blocks = {name: 0 for name in model_actuals}

    for block_number in range(full_blocks):
        start = first_test_index + block_number * HORIZON_DAYS
        block = rows[start:start + HORIZON_DAYS]
        origin_history = rows[:start]
        target_dates = [day for day, _ in block]
        actuals = [eggs for _, eggs in block]
        print(
            f"Evaluating block {block_number + 1}/{full_blocks}: "
            f"{target_dates[0]:%Y-%m-%d} to {target_dates[-1]:%Y-%m-%d}",
            flush=True,
        )

        # The seven target-day actuals are not included in origin_history.
        model_name = f"ARIMA{ARIMA_ORDER}"
        try:
            arima_predictions = predict_arima_seven_days(origin_history)
            for target, actual, predicted in zip(target_dates, actuals, arima_predictions):
                model_actuals[model_name].append(actual)
                model_predictions[model_name].append(predicted)
                result_rows.append((model_name, target.strftime("%Y-%m-%d"), actual, predicted, block_number + 1))
        except Exception as exc:
            failed_blocks[model_name] += 1
            print(
                f"  ARIMA block failed ({type(exc).__name__}): {exc}",
                flush=True,
            )

    output_csv = Path("seven_day_backtest_results.csv")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "date", "actual_eggs", "predicted_eggs", "forecast_block"])
        writer.writerows(result_rows)

    scored_end = rows[first_test_index + full_blocks * HORIZON_DAYS - 1][0]
    print(f"\nTotal historical records: {len(rows)}")
    print(f"Training records before test period: {first_test_index}")
    print(f"Testing period: {test_rows[0][0]:%Y-%m-%d} to {scored_end:%Y-%m-%d}")
    print(f"Complete seven-day forecast blocks: {full_blocks}")
    print(f"Unused trailing test days: {len(test_rows) - full_blocks * HORIZON_DAYS}")
    print("\nSeven-day backtesting results:")
    print(f"{'Model':<32} {'Days':>7} {'Failed blocks':>14} {'MAE':>10} {'RMSE':>10} {'MAPE':>9} {'R²':>9}")

    summary_metrics = {}
    for model_name in model_actuals:
        actuals = model_actuals[model_name]
        predictions = model_predictions[model_name]
        if not actuals:
            print(f"{model_name:<32} No successful predictions")
            continue
        metrics = calculate_metrics(actuals, predictions)
        summary_metrics[model_name] = metrics
        print(
            f"{model_name:<32} {len(actuals):>7} {failed_blocks[model_name]:>14} "
            f"{metrics['mae']:>10.2f} {metrics['rmse']:>10.2f} "
            f"{metrics['mape']:>8.2f}% {metrics['r2']:>9.4f}"
        )

    # Plot actual values once and each model's predictions.
    dates_for_plot = [day for day, _ in rows[first_test_index:first_test_index + full_blocks * HORIZON_DAYS]]
    actual_for_plot = [eggs for _, eggs in rows[first_test_index:first_test_index + full_blocks * HORIZON_DAYS]]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates_for_plot, actual_for_plot, label="Actual production", linewidth=2)
    for model_name, predictions in model_predictions.items():
        if len(predictions) == len(dates_for_plot):
            ax.plot(dates_for_plot, predictions, label=model_name, alpha=0.85)
    ax.set_title("Actual vs. Predicted Egg Production — 7-Day Backtesting")
    ax.set_xlabel("Date")
    ax.set_ylabel("Eggs")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    ax.legend()
    fig.tight_layout()
    graph_path = Path("seven_day_actual_vs_predicted.png")
    fig.savefig(graph_path, dpi=200)
    plt.close(fig)

    print(f"\nSaved predictions: {output_csv.resolve()}")
    print(f"Saved graph: {graph_path.resolve()}")
    print(
        "\nMethod: non-overlapping seven-day forecast blocks. Each block is "
        "predicted using only records dated before that block starts."
    )
    print(
        "Important: this scores ARIMA(2,1,1) totals from src/service.py. "
        "Temperature-based egg-size allocation is not evaluated because "
        "historical temperature data is not available in production.csv."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
