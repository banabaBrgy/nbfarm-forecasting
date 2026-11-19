"""Egg forecasting model: ARIMA(2,1,1).

Daily totals are forecast with ARIMA(2,1,1). Historical size mix only
splits that ARIMA total into egg sizes.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from statsmodels.tsa.arima.model import ARIMA

DEFAULT_WINDOW = 7
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


def _predict_one_day(
    archive: dict[str, dict[str, float]],
    target: datetime,
    temperature: float | None,
    history_years: int,
    neutral: float,
    per_degree: float,
    cap: float,
    eggs_per_tray: int,
    predicted_total: int | float,
) -> dict[str, Any]:
    """Apply an ARIMA(2,1,1) total and split it by historical size mix."""
    dated = _same_calendar_dated_samples(archive, target, history_years)
    samples = [row for _, row in dated]
    predicted = int(round(max(0.0, float(predicted_total)) / eggs_per_tray) * eggs_per_tray)
    sample_days = len(samples)

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


def _fit_arima(values: list[float]):
    return ARIMA(
        values,
        order=ARIMA_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()


def _forecast_steps(fitted, last_train: datetime, targets: list[datetime]) -> dict[str, float | None]:
    totals: dict[str, float | None] = {}
    max_steps = (targets[-1] - last_train).days
    if max_steps <= 0:
        return {day.strftime("%Y-%m-%d"): None for day in targets}

    raw = fitted.forecast(steps=max_steps)
    for day in targets:
        iso = day.strftime("%Y-%m-%d")
        steps = (day - last_train).days
        if steps <= 0:
            totals[iso] = None
            continue
        totals[iso] = round(max(0.0, float(raw[steps - 1])), 2)
    return totals


def _arima_totals_before_each_date(
    archive: dict[str, dict[str, float]],
    parsed_dates: list[datetime],
) -> tuple[dict[str, float | None], list[str]]:
    """ARIMA(2,1,1) totals that never use the harvest being compared.

    Past chart weeks are non-overlapping 7-day blocks. Each block is fitted
    only on days strictly before that week, matching evaluate_7day_forecast.py.
    Days after the last harvest stay one multi-step forecast from the full
    archive so Seven-Day and a mixed past/future chart share those totals.
    """
    ordered = sorted(archive)
    parsed_history = [_parse_date(key) for key in ordered]
    last_archive = parsed_history[-1]
    totals: dict[str, float | None] = {}
    warnings: list[str] = []
    future_fitted = None
    dates = sorted(parsed_dates)
    index = 0

    while index < len(dates):
        day = dates[index]
        iso = day.strftime("%Y-%m-%d")

        if day > last_archive:
            if len(ordered) < ARIMA_MIN_POINTS:
                totals[iso] = None
                index += 1
                continue
            try:
                if future_fitted is None:
                    values = [float(archive[key]["eggs"]) for key in ordered]
                    future_fitted = _fit_arima(values)
                steps = (day - last_archive).days
                if steps <= 0:
                    totals[iso] = None
                else:
                    raw = future_fitted.forecast(steps=steps)
                    totals[iso] = round(max(0.0, float(raw[steps - 1])), 2)
            except (ValueError, ArithmeticError, IndexError) as exc:
                totals[iso] = None
                future_fitted = None
                warnings.append(f"{iso}: {type(exc).__name__}: {exc}")
            index += 1
            continue

        origin = day
        block_end = origin + timedelta(days=DEFAULT_WINDOW - 1)
        block: list[datetime] = []
        while (
            index < len(dates)
            and dates[index] <= block_end
            and dates[index] <= last_archive
        ):
            block.append(dates[index])
            index += 1

        train_keys = [
            key for key, hist_day in zip(ordered, parsed_history)
            if hist_day < origin
        ]
        if len(train_keys) < ARIMA_MIN_POINTS:
            for target in block:
                totals[target.strftime("%Y-%m-%d")] = None
            continue

        last_train = _parse_date(train_keys[-1])
        try:
            values = [float(archive[key]["eggs"]) for key in train_keys]
            fitted = _fit_arima(values)
            totals.update(_forecast_steps(fitted, last_train, block))
        except (ValueError, ArithmeticError, IndexError) as exc:
            for target in block:
                totals[target.strftime("%Y-%m-%d")] = None
            warnings.append(
                f"{origin.strftime('%Y-%m-%d')}: {type(exc).__name__}: {exc}"
            )

    return totals, warnings


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
        predicted_by_day, fit_warnings = _arima_totals_before_each_date(archive, parsed_dates)
    except Exception as exc:
        return {
            "predicted_eggs": 0, "forecast_date": None, "labels": [],
            "dates": [], "data": [], "series": [],
            "sizes": {key: 0 for key in SIZE_KEYS}, "sample_days": 0,
            "temperature": None, "model": "arima_2_1_1",
            "history_years": history_years,
            "warnings": [],
            "error": f"ARIMA model fitting failed: {type(exc).__name__}: {exc}",
        }

    if not any(total is not None for total in predicted_by_day.values()):
        error = (
            f"ARIMA model fitting failed: {fit_warnings[0]}"
            if fit_warnings
            else "ARIMA forecasting requires at least 30 historical daily records before each date."
        )
        return {
            "predicted_eggs": 0, "forecast_date": None, "labels": [],
            "dates": [], "data": [], "series": [],
            "sizes": {key: 0 for key in SIZE_KEYS}, "sample_days": 0,
            "temperature": None, "model": "arima_2_1_1",
            "history_years": history_years,
            "warnings": fit_warnings,
            "error": error,
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
        error = (
            f"ARIMA model fitting failed: {fit_warnings[0]}"
            if fit_warnings
            else "ARIMA forecasting requires at least 30 historical daily records before each date."
        )
        return {
            "predicted_eggs": 0, "forecast_date": None, "labels": [],
            "dates": [], "data": [], "series": [],
            "sizes": {key: 0 for key in SIZE_KEYS}, "sample_days": 0,
            "temperature": None, "model": "arima_2_1_1",
            "history_years": history_years,
            "warnings": fit_warnings,
            "error": error,
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
        "warnings": fit_warnings,
        "error": None,
    }


def run_forecast(
    history: list[dict[str, Any]],
    predict_dates: list[str] | None = None,
    window: int = DEFAULT_WINDOW,
    model: str = "arima_2_1_1",
    temperatures: dict[str, Any] | None = None,
    history_years: int = DEFAULT_HISTORY_YEARS,
    neutral_temperature: float = DEFAULT_NEUTRAL_TEMP,
    temp_shift_per_degree: float = DEFAULT_TEMP_SHIFT,
    temp_shift_cap: float = DEFAULT_TEMP_SHIFT_CAP,
    eggs_per_tray: int = EGGS_PER_TRAY,
    archive_start: str | None = None,
    archive_end: str | None = None,
) -> dict[str, Any]:
    """Forecast with ARIMA(2,1,1).

    ``model`` and ``window`` are accepted so older callers keep working.
    They do not select another model. The returned model is always
    ``arima_2_1_1``.
    """
    del window, model
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