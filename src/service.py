"""Egg forecasting: ARIMA primary, 3-year archive alternative, legacy 7-day OLS kept."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from statsmodels.tsa.arima.model import ARIMA

from src.regression import fit_line, predict_y

DEFAULT_WINDOW = 7
MIN_POINTS = 2
EGGS_PER_TRAY = 30
DEFAULT_HISTORY_YEARS = 3
DEFAULT_NEUTRAL_TEMP = 28.0
DEFAULT_TEMP_SHIFT = 0.015
DEFAULT_TEMP_SHIFT_CAP = 0.20
ARIMA_ORDER = (2, 1, 1)
ARIMA_MIN_POINTS = 30
SIZE_KEYS = ("small", "medium", "large", "extra_large", "cracked")


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value[:10], "%Y-%m-%d")


def _normalize_history(history: list[dict[str, Any]]) -> list[tuple[datetime, float]]:
    rows: dict[str, float] = {}
    for item in history:
        if not isinstance(item, dict):
            continue
        date_raw = item.get("date")
        eggs = item.get("eggs")
        if date_raw is None or eggs is None:
            continue
        try:
            day = _parse_date(str(date_raw))
            total = float(eggs)
        except (TypeError, ValueError):
            continue
        if total < 0:
            total = 0.0
        key = day.strftime("%Y-%m-%d")
        rows[key] = rows.get(key, 0.0) + total

    ordered = sorted(
        ((_parse_date(k), v) for k, v in rows.items()),
        key=lambda pair: pair[0],
    )
    return ordered


def _normalize_archive(history: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for item in history:
        if not isinstance(item, dict):
            continue
        date_raw = item.get("date")
        eggs = item.get("eggs")
        if date_raw is None or eggs is None:
            continue
        try:
            day = _parse_date(str(date_raw))
            total = float(eggs)
        except (TypeError, ValueError):
            continue
        if total <= 0:
            continue
        key = day.strftime("%Y-%m-%d")
        bucket = rows.setdefault(
            key,
            {
                "eggs": 0.0,
                "small": 0.0,
                "medium": 0.0,
                "large": 0.0,
                "extra_large": 0.0,
                "cracked": 0.0,
            },
        )
        bucket["eggs"] += total
        for size in SIZE_KEYS:
            try:
                bucket[size] += max(0.0, float(item.get(size, 0) or 0))
            except (TypeError, ValueError):
                continue
    return rows


def _allocate_by_mix(total: int, mix: dict[str, float]) -> dict[str, int]:
    empty = {key: 0 for key in SIZE_KEYS}
    if total <= 0:
        return empty

    weight_sum = sum(max(0.0, float(mix.get(key, 0) or 0)) for key in SIZE_KEYS)
    if weight_sum <= 0:
        empty["medium"] = total
        return empty

    parts: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    assigned = 0
    for key in SIZE_KEYS:
        exact = total * (max(0.0, float(mix.get(key, 0) or 0)) / weight_sum)
        base = int(exact)
        parts[key] = base
        assigned += base
        remainders.append((exact - base, key))

    left = total - assigned
    remainders.sort(key=lambda item: (-item[0], item[1]))
    for _, key in remainders:
        if left <= 0:
            break
        parts[key] += 1
        left -= 1
    return parts


def _shift_mix_for_temperature(
    mix: dict[str, float],
    temperature: float | None,
    neutral: float,
    per_degree: float,
    cap: float,
) -> dict[str, float]:
    result = {key: float(mix.get(key, 0) or 0) for key in SIZE_KEYS}
    if temperature is None:
        return result

    shift = max(-cap, min(cap, (float(temperature) - neutral) * per_degree))
    if abs(shift) < 1e-9:
        return result

    transferable = result["small"] + result["large"] + result["extra_large"]
    if transferable <= 0:
        return result

    move = int(round(transferable * abs(shift)))
    if move <= 0:
        return result

    if shift > 0:
        from_large = min(int(result["large"]), int(round(move * 0.7)))
        from_xl = min(int(result["extra_large"]), move - from_large)
        result["large"] -= from_large
        result["extra_large"] -= from_xl
        result["small"] += from_large + from_xl
    else:
        from_small = min(int(result["small"]), move)
        result["small"] -= from_small
        result["large"] += from_small

    return result


def _same_calendar_samples(
    archive: dict[str, dict[str, float]],
    target: datetime,
    history_years: int,
) -> list[dict[str, float]]:
    by_year: dict[int, dict[str, float]] = {}
    for key, row in archive.items():
        day = _parse_date(key)
        if day >= target:
            continue
        if day.month != target.month or day.day != target.day:
            continue
        by_year[day.year] = dict(row)

    if not by_year:
        target_doy = target.timetuple().tm_yday
        nearby: dict[int, list[dict[str, float]]] = {}
        for key, row in archive.items():
            day = _parse_date(key)
            if day >= target:
                continue
            if abs(day.timetuple().tm_yday - target_doy) > 3:
                continue
            nearby.setdefault(day.year, []).append(row)
        for year, rows in nearby.items():
            count = len(rows)
            by_year[year] = {
                size: sum(item[size] for item in rows) / count
                for size in ("eggs",) + SIZE_KEYS
            }

    years = sorted(by_year.keys(), reverse=True)[: max(1, history_years)]
    return [by_year[year] for year in years]


def _same_calendar_dated_samples(
    archive: dict[str, dict[str, float]],
    target: datetime,
    history_years: int,
) -> list[tuple[datetime, dict[str, float]]]:
    """Same month/day rows across years, oldest→newest, capped by history_years."""
    by_year: dict[int, tuple[datetime, dict[str, float]]] = {}
    for key, row in archive.items():
        day = _parse_date(key)
        if day >= target:
            continue
        if day.month != target.month or day.day != target.day:
            continue
        by_year[day.year] = (day, dict(row))

    if by_year:
        years = sorted(by_year.keys())[-max(1, history_years) :]
        return [by_year[year] for year in years]

    # Nearby day-of-year fallback: one point per recent year.
    nearby = _same_calendar_samples(archive, target, history_years)
    dated: list[tuple[datetime, dict[str, float]]] = []
    for index, sample in enumerate(reversed(nearby)):
        year = target.year - (index + 1)
        dated.append((datetime(year, target.month, min(target.day, 28)), sample))
    dated.reverse()
    return dated


def _predict_total_for_day(
    dated_samples: list[tuple[datetime, dict[str, float]]],
    target: datetime,
    eggs_per_tray: int,
) -> tuple[int, float, float, int]:
    """
    Linear regression across the same calendar day in past years.
    X = year, Y = eggs. Predict for the target year.
    """
    if not dated_samples:
        return 0, 0.0, 0.0, 0

    if len(dated_samples) == 1:
        eggs = int(round(dated_samples[0][1]["eggs"] / eggs_per_tray) * eggs_per_tray)
        return eggs, 0.0, float(dated_samples[0][1]["eggs"]), 1

    xs = [float(day.year) for day, _ in dated_samples]
    ys = [float(row["eggs"]) for _, row in dated_samples]
    slope, intercept = fit_line(xs, ys)
    predicted = predict_y(slope, intercept, float(target.year))
    predicted = int(round(predicted / eggs_per_tray) * eggs_per_tray)
    return predicted, slope, intercept, len(dated_samples)
    if predicted <= 0:
        avg = sum(ys) / len(ys)
        predicted = int(round(avg / eggs_per_tray) * eggs_per_tray)
    return predicted, slope, intercept, len(dated_samples)


def _predict_one_day(
    archive: dict[str, dict[str, float]],
    target: datetime,
    temperature: float | None,
    history_years: int,
    neutral: float,
    per_degree: float,
    cap: float,
    eggs_per_tray: int,
    predicted_total: int | None = None,
) -> dict[str, Any]:
    dated = _same_calendar_dated_samples(archive, target, history_years)
    samples = [row for _, row in dated]
    predicted, slope, intercept, sample_days = _predict_total_for_day(dated, target, eggs_per_tray)
    if predicted_total is not None:
        predicted = int(round(max(0, predicted_total) / eggs_per_tray) * eggs_per_tray)

    base_mix = {key: 0.0 for key in SIZE_KEYS}
    if samples:
        for sample in samples:
            for key in SIZE_KEYS:
                base_mix[key] += float(sample.get(key, 0) or 0)
    elif predicted > 0:
        base_mix["medium"] = float(predicted)

    shifted = _shift_mix_for_temperature(base_mix, temperature, neutral, per_degree, cap)
    sizes = _allocate_by_mix(predicted, shifted)

    return {
        "date": target.strftime("%Y-%m-%d"),
        "label": target.strftime("%a"),
        "eggs": predicted,
        "temperature": temperature,
        "sample_days": sample_days,
        "sizes": sizes,
        "slope": round(slope, 6),
        "intercept": round(intercept, 6),
    }


def _clip_archive(
    archive: dict[str, dict[str, float]],
    archive_start: str | None,
    archive_end: str | None,
) -> dict[str, dict[str, float]]:
    """Keep only rows inside the client PDF window when bounds are provided."""
    if not archive_start and not archive_end:
        return archive

    clipped: dict[str, dict[str, float]] = {}
    for key, row in archive.items():
        if archive_start and key < archive_start[:10]:
            continue
        if archive_end and key > archive_end[:10]:
            continue
        clipped[key] = row
    return clipped


def run_archive_temperature_forecast(
    history: list[dict[str, Any]],
    predict_dates: list[str] | None = None,
    temperatures: dict[str, Any] | None = None,
    history_years: int = DEFAULT_HISTORY_YEARS,
    neutral_temperature: float = DEFAULT_NEUTRAL_TEMP,
    temp_shift_per_degree: float = DEFAULT_TEMP_SHIFT,
    temp_shift_cap: float = DEFAULT_TEMP_SHIFT_CAP,
    eggs_per_tray: int = EGGS_PER_TRAY,
    archive_start: str | None = None,
    archive_end: str | None = None,
) -> dict[str, Any]:
    """
    Primary client model:
    - for each forecast day, linear regression on that same calendar day across up to 3 years
    - size mix from those matching days, shifted by the provided temperature
      (Laravel sends tomorrow's temperature for all live forecast days)
    """
    archive = _clip_archive(_normalize_archive(history), archive_start, archive_end)
    if not archive:
        return {
            "predicted_eggs": 0,
            "forecast_date": None,
            "labels": [],
            "dates": [],
            "data": [],
            "series": [],
            "sizes": {key: 0 for key in SIZE_KEYS},
            "sample_days": 0,
            "temperature": None,
            "slope": 0.0,
            "intercept": 0.0,
            "model": "three_year_linear_regression",
            "history_years": history_years,
            "error": "Need archive harvest rows with eggs.",
        }

    temps = temperatures or {}
    if not predict_dates:
        last = max(_parse_date(key) for key in archive)
        predict_dates = [(last + timedelta(days=1)).strftime("%Y-%m-%d")]

    series = []
    labels = []
    dates = []
    data = []
    primary = None

    for raw in predict_dates:
        try:
            day = _parse_date(str(raw))
        except (TypeError, ValueError):
            continue
        iso = day.strftime("%Y-%m-%d")
        temp_raw = temps.get(iso, temps.get(str(raw)))
        try:
            temperature = float(temp_raw) if temp_raw is not None else None
        except (TypeError, ValueError):
            temperature = None

        point = _predict_one_day(
            archive,
            day,
            temperature,
            history_years,
            neutral_temperature,
            temp_shift_per_degree,
            temp_shift_cap,
            eggs_per_tray,
        )
        if primary is None:
            primary = point
        series.append(point)
        labels.append(point["label"])
        dates.append(point["date"])
        data.append(point["eggs"])

    if primary is None:
        primary = {
            "eggs": 0,
            "date": None,
            "sizes": {key: 0 for key in SIZE_KEYS},
            "sample_days": 0,
            "temperature": None,
            "slope": 0.0,
            "intercept": 0.0,
        }

    return {
        "predicted_eggs": int(primary["eggs"]),
        "forecast_date": primary["date"],
        "labels": labels,
        "dates": dates,
        "data": data,
        "series": series,
        "sizes": primary["sizes"],
        "sample_days": int(primary["sample_days"]),
        "temperature": primary["temperature"],
        "slope": primary.get("slope", 0.0),
        "intercept": primary.get("intercept", 0.0),
        "window": int(primary["sample_days"]),
        "model": "three_year_linear_regression",
        "history_years": history_years,
        "error": None,
    }


def _arima_totals_before_each_date(
    archive: dict[str, dict[str, float]],
    parsed_dates: list[datetime],
) -> dict[str, float | None]:
    """One-step (or N-step) ARIMA using only harvest days strictly before each target.

    Past chart days walk forward with actuals so the line does not flatten.
    Future days keep a multi-step forecast from the last actual day.
    """
    ordered = sorted(archive)
    parsed_history = [_parse_date(key) for key in ordered]
    totals: dict[str, float | None] = {}
    fitted = None
    fitted_until: datetime | None = None

    for day in sorted(parsed_dates):
        iso = day.strftime("%Y-%m-%d")
        train_end = day - timedelta(days=1)
        train_keys = [
            key for key, hist_day in zip(ordered, parsed_history)
            if hist_day <= train_end
        ]
        if len(train_keys) < ARIMA_MIN_POINTS:
            totals[iso] = None
            fitted = None
            fitted_until = None
            continue

        last_train = _parse_date(train_keys[-1])
        try:
            if fitted is None or fitted_until is None or fitted_until > last_train:
                values = [float(archive[key]["eggs"]) for key in train_keys]
                fitted = ARIMA(
                    values,
                    order=ARIMA_ORDER,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit()
                fitted_until = last_train
            elif fitted_until < last_train:
                for key, hist_day in zip(ordered, parsed_history):
                    if fitted_until < hist_day <= last_train:
                        fitted = fitted.append([float(archive[key]["eggs"])])
                        fitted_until = hist_day

            steps = (day - fitted_until).days
            if steps <= 0:
                totals[iso] = None
                continue
            raw = fitted.forecast(steps=steps)
            totals[iso] = round(max(0.0, float(raw[steps - 1])), 2)
        except Exception:
            totals[iso] = None
            fitted = None
            fitted_until = None

    return totals


def run_arima_temperature_forecast(
    history: list[dict[str, Any]],
    predict_dates: list[str] | None = None,
    temperatures: dict[str, Any] | None = None,
    history_years: int = DEFAULT_HISTORY_YEARS,
    neutral_temperature: float = DEFAULT_NEUTRAL_TEMP,
    temp_shift_per_degree: float = DEFAULT_TEMP_SHIFT,
    temp_shift_cap: float = DEFAULT_TEMP_SHIFT_CAP,
    eggs_per_tray: int = EGGS_PER_TRAY,
    archive_start: str | None = None,
    archive_end: str | None = None,
) -> dict[str, Any]:
    """Forecast total daily eggs with ARIMA(2,1,1), retaining historical size mix."""
    archive = _clip_archive(_normalize_archive(history), archive_start, archive_end)
    if not archive:
        return {
            "predicted_eggs": 0, "forecast_date": None, "labels": [],
            "dates": [], "data": [], "series": [],
            "sizes": {key: 0 for key in SIZE_KEYS}, "sample_days": 0,
            "temperature": None, "model": "arima_2_1_1",
            "history_years": history_years,
            "error": "Need archive harvest rows with eggs.",
        }

    ordered_dates = sorted(archive)
    values = [float(archive[key]["eggs"]) for key in ordered_dates]
    if len(values) < 30:
        return {
            "predicted_eggs": 0, "forecast_date": None, "labels": [],
            "dates": [], "data": [], "series": [],
            "sizes": {key: 0 for key in SIZE_KEYS}, "sample_days": 0,
            "temperature": None, "model": "arima_2_1_1",
            "history_years": history_years,
            "error": "ARIMA forecasting requires at least 30 historical daily records.",
        }

    if not predict_dates:
        last = _parse_date(ordered_dates[-1])
        predict_dates = [
            (last + timedelta(days=offset)).strftime("%Y-%m-%d")
            for offset in range(1, 8)
        ]

    parsed_dates: list[datetime] = []
    for raw in predict_dates:
        try:
            parsed_dates.append(_parse_date(str(raw)))
        except (TypeError, ValueError):
            continue
    if not parsed_dates:
        return {
            "predicted_eggs": 0, "forecast_date": None, "labels": [],
            "dates": [], "data": [], "series": [],
            "sizes": {key: 0 for key in SIZE_KEYS}, "sample_days": 0,
            "temperature": None, "model": "arima_2_1_1",
            "history_years": history_years,
            "error": "No valid forecast dates were supplied.",
        }

    try:
        predicted_by_day = _arima_totals_before_each_date(archive, parsed_dates)
    except Exception as exc:
        return {
            "predicted_eggs": 0, "forecast_date": None, "labels": [],
            "dates": [], "data": [], "series": [],
            "sizes": {key: 0 for key in SIZE_KEYS}, "sample_days": 0,
            "temperature": None, "model": "arima_2_1_1",
            "history_years": history_years,
            "error": f"ARIMA model fitting failed: {type(exc).__name__}: {exc}",
        }

    if not any(total is not None for total in predicted_by_day.values()):
        return {
            "predicted_eggs": 0, "forecast_date": None, "labels": [],
            "dates": [], "data": [], "series": [],
            "sizes": {key: 0 for key in SIZE_KEYS}, "sample_days": 0,
            "temperature": None, "model": "arima_2_1_1",
            "history_years": history_years,
            "error": "ARIMA forecasting requires at least 30 historical daily records before each date.",
        }

    temps = temperatures or {}
    series: list[dict[str, Any]] = []
    for day in parsed_dates:
        iso = day.strftime("%Y-%m-%d")
        total = predicted_by_day.get(iso)
        if total is None:
            continue
        temp_raw = temps.get(iso)
        try:
            temperature = float(temp_raw) if temp_raw is not None else None
        except (TypeError, ValueError):
            temperature = None

        point = _predict_one_day(
            archive, day, temperature, history_years,
            neutral_temperature, temp_shift_per_degree, temp_shift_cap,
            eggs_per_tray, predicted_total=total,
        )
        series.append(point)

    if not series:
        return {
            "predicted_eggs": 0, "forecast_date": None, "labels": [],
            "dates": [], "data": [], "series": [],
            "sizes": {key: 0 for key in SIZE_KEYS}, "sample_days": 0,
            "temperature": None, "model": "arima_2_1_1",
            "history_years": history_years,
            "error": "ARIMA forecasting requires at least 30 historical daily records before each date.",
        }

    primary = series[0]
    return {
        "predicted_eggs": int(primary["eggs"]),
        "forecast_date": primary["date"],
        "labels": [point["label"] for point in series],
        "dates": [point["date"] for point in series],
        "data": [point["eggs"] for point in series],
        "series": series,
        "sizes": primary["sizes"],
        "sample_days": int(primary["sample_days"]),
        "temperature": primary["temperature"],
        "model": "arima_2_1_1",
        "history_years": history_years,
        "error": None,
    }


def run_forecast(
    history: list[dict[str, Any]],
    predict_dates: list[str] | None = None,
    window: int = DEFAULT_WINDOW,
    model: str = "three_year_temperature",
    temperatures: dict[str, Any] | None = None,
    history_years: int = DEFAULT_HISTORY_YEARS,
    neutral_temperature: float = DEFAULT_NEUTRAL_TEMP,
    temp_shift_per_degree: float = DEFAULT_TEMP_SHIFT,
    temp_shift_cap: float = DEFAULT_TEMP_SHIFT_CAP,
    eggs_per_tray: int = EGGS_PER_TRAY,
    archive_start: str | None = None,
    archive_end: str | None = None,
) -> dict[str, Any]:
    normalized_model = str(model).lower()
    if normalized_model in {"arima", "arima_2_1_1", "arima(2,1,1)"}:
        return run_arima_temperature_forecast(
            history,
            predict_dates=predict_dates,
            temperatures=temperatures,
            history_years=history_years,
            neutral_temperature=neutral_temperature,
            temp_shift_per_degree=temp_shift_per_degree,
            temp_shift_cap=temp_shift_cap,
            eggs_per_tray=eggs_per_tray,
            archive_start=archive_start,
            archive_end=archive_end,
        )

    if normalized_model in {
        "three_year_temperature",
        "three_year_linear_regression",
        "archive_temperature",
        "temperature",
    }:
        return run_archive_temperature_forecast(
            history,
            predict_dates=predict_dates,
            temperatures=temperatures,
            history_years=history_years,
            neutral_temperature=neutral_temperature,
            temp_shift_per_degree=temp_shift_per_degree,
            temp_shift_cap=temp_shift_cap,
            eggs_per_tray=eggs_per_tray,
            archive_start=archive_start,
            archive_end=archive_end,
        )

    # Legacy: ordinary least squares on the last `window` harvest days.
    if window < MIN_POINTS:
        window = DEFAULT_WINDOW

    points = _normalize_history(history)
    if len(points) < MIN_POINTS:
        return {
            "predicted_eggs": 0,
            "forecast_date": None,
            "labels": [],
            "dates": [],
            "data": [],
            "series": [],
            "slope": 0.0,
            "intercept": 0.0,
            "window": window,
            "trained_on": [],
            "model": "linear_regression",
            "error": "Need at least 2 daily totals to fit linear regression.",
        }

    train = points[-window:]
    first_date = train[0][0]
    x = [(row[0] - first_date).days for row in train]
    y = [row[1] for row in train]
    slope, intercept = fit_line(x, y)

    last_train_date = train[-1][0]
    next_date = last_train_date + timedelta(days=1)
    next_x = (next_date - first_date).days
    predicted_eggs = predict_y(slope, intercept, next_x)

    if not predict_dates:
        predict_dates = [next_date.strftime("%Y-%m-%d")]

    series = []
    labels = []
    dates = []
    data = []
    for raw in predict_dates:
        try:
            day = _parse_date(str(raw))
        except (TypeError, ValueError):
            continue
        xi = (day - first_date).days
        eggs = predict_y(slope, intercept, xi)
        label = day.strftime("%a")
        iso = day.strftime("%Y-%m-%d")
        series.append({"date": iso, "label": label, "eggs": eggs})
        labels.append(label)
        dates.append(iso)
        data.append(eggs)

    return {
        "predicted_eggs": predicted_eggs,
        "forecast_date": next_date.strftime("%Y-%m-%d"),
        "labels": labels,
        "dates": dates,
        "data": data,
        "series": series,
        "slope": round(slope, 6),
        "intercept": round(intercept, 6),
        "window": len(train),
        "trained_on": [row[0].strftime("%Y-%m-%d") for row in train],
        "model": "linear_regression",
        "error": None,
    }