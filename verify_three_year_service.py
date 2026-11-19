"""Verify that saved ARIMA(2,1,1) backtest rows match src.service.run_forecast.

Run from the forecasting folder:
    python verify_three_year_service.py production.csv

This does not modify service.py or any existing files.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

from src.service import run_forecast

DATE_COLUMNS = ("date", "production_date", "ds")
EGG_COLUMNS = ("eggs", "eggs_produced", "y")
HISTORY_YEARS = 3
EGGS_PER_TRAY = 30
HORIZON_DAYS = 7


def load_rows(path: Path) -> list[tuple[datetime, float]]:
    totals: dict[datetime, float] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"{path} has no header row.")
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        date_key = next((fields[name] for name in DATE_COLUMNS if name in fields), None)
        egg_key = next((fields[name] for name in EGG_COLUMNS if name in fields), None)
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


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python verify_three_year_service.py production.csv", file=sys.stderr)
        return 1

    path = Path(argv[1])
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    rows = load_rows(path)
    horizon = int(len(rows) * 0.8)
    horizon = ((horizon + HORIZON_DAYS - 1) // HORIZON_DAYS) * HORIZON_DAYS
    test_rows = rows[horizon:]
    blocks = len(test_rows) // HORIZON_DAYS
    if blocks == 0:
        raise SystemExit("Not enough complete seven-day test blocks.")

    compared = 0
    mismatches: list[tuple[str, int, int]] = []

    for block_index in range(blocks):
        start = horizon + block_index * HORIZON_DAYS
        history = rows[:start]
        target_dates = [
            day.strftime("%Y-%m-%d")
            for day, _ in rows[start:start + HORIZON_DAYS]
        ]
        history_payload = [
            {"date": day.strftime("%Y-%m-%d"), "eggs": eggs}
            for day, eggs in history
        ]

        result = run_forecast(
            history_payload,
            predict_dates=target_dates,
            model="arima_2_1_1",
            history_years=HISTORY_YEARS,
            eggs_per_tray=EGGS_PER_TRAY,
        )

        service_predictions = [
            int(point["eggs"]) for point in result.get("series", [])
        ]

        if len(service_predictions) != len(target_dates):
            print(
                f"Block {block_index + 1}: service returned "
                f"{len(service_predictions)} predictions; expected {len(target_dates)}."
            )
            return 2

        # The service output is the reference for this direct comparison.
        # The backtest CSV is checked separately if it exists.
        backtest_path = Path("seven_day_backtest_results.csv")
        if backtest_path.is_file():
            with backtest_path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                eval_map = {
                    row["date"]: int(round(float(row["predicted_eggs"])))
                    for row in reader
                    if row.get("model") == "ARIMA(2, 1, 1)"
                }
            for date, service_value in zip(target_dates, service_predictions):
                if date in eval_map:
                    compared += 1
                    if service_value != eval_map[date]:
                        mismatches.append((date, service_value, eval_map[date]))

    print(f"Complete seven-day blocks checked: {blocks}")
    print(f"Predictions compared with backtest CSV: {compared}")
    print(f"Mismatches: {len(mismatches)}")
    if mismatches:
        print("\nFirst mismatches (date, service.py, evaluation script):")
        for date, service_value, eval_value in mismatches[:20]:
            print(f"{date}: {service_value} vs {eval_value}")
        print(
            "\nThe implementations do not fully match. Do not finalize the model "
            "comparison until the differences are investigated."
        )
        return 2

    if compared == 0:
        print(
            "\nNo matching ARIMA(2, 1, 1) rows were found in seven_day_backtest_results.csv. "
            "Make sure that file is in the current forecasting folder."
        )
        return 2

    print(
        "\nPASS: ARIMA(2,1,1) predictions from service.py match the saved backtest "
        "predictions for all compared dates."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
