"""
HTTP endpoint for egg forecasting.

Local:
    python -m flask --app api.index run --port 8001

Vercel: this file is the serverless entry (Flask `app`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, request

from src.service import (
    DEFAULT_HISTORY_YEARS,
    DEFAULT_NEUTRAL_TEMP,
    DEFAULT_TEMP_SHIFT,
    DEFAULT_TEMP_SHIFT_CAP,
    DEFAULT_WINDOW,
    EGGS_PER_TRAY,
    run_forecast,
)

app = Flask(__name__)


def _authorized() -> bool:
    expected = os.environ.get("FORECAST_API_KEY", "").strip()
    if not expected:
        return True
    provided = request.headers.get("X-Forecast-Key", "")
    if provided == expected:
        return True
    auth = request.headers.get("Authorization", "")
    return auth == f"Bearer {expected}"


def _handle_forecast():
    if request.method == "GET":
        return jsonify(
            {
                "ok": True,
                "service": "nb-farm-forecasting",
                "method": "arima_2_1_1",
                "alternative": "three_year_linear_regression",
                "legacy": "linear_regression",
                "note": (
                    "Primary model is ARIMA(2,1,1); "
                    "three-year linear regression and legacy "
                    "7-day OLS are available as alternatives."
                ),
            }
        )

    if not _authorized():
        return jsonify({"error": "Unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    history = payload.get("history") or []
    predict_dates = payload.get("predict_dates") or []
    temperatures = payload.get("temperatures") or {}

    # ARIMA(2,1,1) is now the default forecasting model.
    model = payload.get("model", "arima_2_1_1")

    window = payload.get("window", DEFAULT_WINDOW)
    history_years = payload.get("history_years", DEFAULT_HISTORY_YEARS)
    neutral_temperature = payload.get(
        "neutral_temperature", DEFAULT_NEUTRAL_TEMP
    )
    temp_shift_per_degree = payload.get(
        "temp_shift_per_degree", DEFAULT_TEMP_SHIFT
    )
    temp_shift_cap = payload.get("temp_shift_cap", DEFAULT_TEMP_SHIFT_CAP)
    eggs_per_tray = payload.get("eggs_per_tray", EGGS_PER_TRAY)
    archive_start = payload.get("archive_start")
    archive_end = payload.get("archive_end")

    try:
        window = int(window)
    except (TypeError, ValueError):
        window = DEFAULT_WINDOW

    try:
        history_years = int(history_years)
    except (TypeError, ValueError):
        history_years = DEFAULT_HISTORY_YEARS

    try:
        neutral_temperature = float(neutral_temperature)
        temp_shift_per_degree = float(temp_shift_per_degree)
        temp_shift_cap = float(temp_shift_cap)
        eggs_per_tray = int(eggs_per_tray)
    except (TypeError, ValueError):
        return jsonify(
            {"error": "Invalid temperature/options payload."}
        ), 400

    if archive_start is not None:
        archive_start = str(archive_start)[:10] or None

    if archive_end is not None:
        archive_end = str(archive_end)[:10] or None

    if not isinstance(history, list):
        return jsonify(
            {"error": "history must be a list of daily harvest rows"}
        ), 400

    if predict_dates and not isinstance(predict_dates, list):
        return jsonify(
            {"error": "predict_dates must be a list of YYYY-MM-DD"}
        ), 400

    if temperatures and not isinstance(temperatures, dict):
        return jsonify(
            {"error": "temperatures must be an object of date -> °C"}
        ), 400

    result = run_forecast(
        history,
        predict_dates or None,
        window=window,
        model=str(model),
        temperatures=temperatures,
        history_years=history_years,
        neutral_temperature=neutral_temperature,
        temp_shift_per_degree=temp_shift_per_degree,
        temp_shift_cap=temp_shift_cap,
        eggs_per_tray=eggs_per_tray,
        archive_start=archive_start,
        archive_end=archive_end,
    )

    status = 200 if not result.get("error") else 422
    return jsonify(result), status


@app.get("/health")
@app.get("/api/health")
def health():
    return jsonify({"ok": True, "service": "nb-farm-forecasting"})


@app.route("/forecast", methods=["GET", "POST"])
@app.route("/api/forecast", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
@app.route("/api/index", methods=["GET", "POST"])
def forecast():
    return _handle_forecast()


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", 8001))
    )