import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

from src.service import run_forecast


class ForecastTests(unittest.TestCase):
    def _history(self, days=40, eggs=7000):
        start = datetime(2026, 1, 1)
        return [
            {
                "date": (start + timedelta(days=offset)).strftime("%Y-%m-%d"),
                "eggs": eggs,
            }
            for offset in range(days)
        ]

    def test_run_forecast_uses_arima(self):
        result = run_forecast(
            self._history(),
            predict_dates=["2026-02-15"],
            window=14,
            model="arima_2_1_1",
        )
        self.assertIsNone(result["error"])
        self.assertEqual(result["model"], "arima_2_1_1")
        self.assertGreater(result["predicted_eggs"], 0)
        self.assertNotIn("slope", result)
        self.assertNotIn("intercept", result)

    def test_archive_end_excludes_later_rows_from_arima(self):
        start = datetime(2025, 11, 1)
        history = [
            {
                "date": (start + timedelta(days=offset)).strftime("%Y-%m-%d"),
                "eggs": 6000,
            }
            for offset in range(40)
        ]
        history.append({"date": "2026-01-15", "eggs": 20000})
        clipped = run_forecast(
            history,
            predict_dates=["2026-02-01"],
            model="arima_2_1_1",
            archive_end="2025-12-31",
        )
        full = run_forecast(
            history,
            predict_dates=["2026-02-01"],
            model="arima_2_1_1",
        )
        self.assertEqual(clipped["model"], "arima_2_1_1")
        self.assertEqual(full["model"], "arima_2_1_1")
        self.assertIsNone(clipped["error"])
        self.assertIsNone(full["error"])
        self.assertNotEqual(clipped["predicted_eggs"], full["predicted_eggs"])

    def test_hotter_temperature_shifts_toward_small(self):
        history = [
            {
                "date": "2023-10-05",
                "eggs": 6000,
                "small": 600,
                "medium": 1800,
                "large": 3000,
                "extra_large": 600,
                "cracked": 0,
            },
            {
                "date": "2024-10-05",
                "eggs": 6000,
                "small": 600,
                "medium": 1800,
                "large": 3000,
                "extra_large": 600,
                "cracked": 0,
            },
            {
                "date": "2025-10-05",
                "eggs": 6000,
                "small": 600,
                "medium": 1800,
                "large": 3000,
                "extra_large": 600,
                "cracked": 0,
            },
        ]
        history.extend(self._history())
        cool = run_forecast(
            history,
            predict_dates=["2026-10-05"],
            temperatures={"2026-10-05": 22},
            model="arima_2_1_1",
        )
        hot = run_forecast(
            history,
            predict_dates=["2026-10-05"],
            temperatures={"2026-10-05": 34},
            model="arima_2_1_1",
        )
        self.assertEqual(cool["predicted_eggs"], hot["predicted_eggs"])
        self.assertGreater(hot["sizes"]["small"], cool["sizes"]["small"])
        self.assertLess(hot["sizes"]["large"], cool["sizes"]["large"])

    def test_short_history_returns_an_arima_error(self):
        result = run_forecast(
            [{"date": "2026-08-01", "eggs": 100}],
            window=14,
            model="arima_2_1_1",
        )
        self.assertEqual(result["model"], "arima_2_1_1")
        self.assertEqual(result["predicted_eggs"], 0)
        self.assertIn("ARIMA", result["error"])

    def test_arima_backtests_a_past_date_from_prior_days(self):
        start = datetime(2026, 1, 1)
        history = [
            {
                "date": (start + timedelta(days=offset)).strftime("%Y-%m-%d"),
                "eggs": 7000 + (offset % 5) * 30,
            }
            for offset in range(45)
        ]
        result = run_forecast(
            history,
            predict_dates=["2026-02-10"],
            model="arima_2_1_1",
        )
        self.assertIsNone(result["error"])
        self.assertEqual(result["model"], "arima_2_1_1")
        self.assertGreater(result["predicted_eggs"], 0)
        self.assertEqual(result["dates"], ["2026-02-10"])

    def test_arima_future_total_matches_with_or_without_past_chart_days(self):
        start = datetime(2026, 1, 1)
        history = [
            {
                "date": (start + timedelta(days=offset)).strftime("%Y-%m-%d"),
                "eggs": 7000 + (offset % 5) * 30,
            }
            for offset in range(45)
        ]
        future = "2026-02-20"
        only_future = run_forecast(
            history,
            predict_dates=[future],
            model="arima_2_1_1",
        )
        with_past = run_forecast(
            history,
            predict_dates=["2026-02-10", future],
            model="arima_2_1_1",
        )
        self.assertIsNone(only_future["error"])
        self.assertIsNone(with_past["error"])
        self.assertEqual(only_future["predicted_eggs"], with_past["data"][-1])

    def test_arima_rejects_too_little_history(self):
        history = [
            {"date": f"2026-01-{day:02d}", "eggs": 7000}
            for day in range(1, 10)
        ]
        result = run_forecast(
            history,
            predict_dates=["2026-01-20"],
            model="arima_2_1_1",
        )
        self.assertEqual(result["predicted_eggs"], 0)
        self.assertIsNotNone(result["error"])

    def test_seven_day_block_ignores_actuals_inside_the_week(self):
        start = datetime(2026, 1, 1)
        week = [
            (start + timedelta(days=offset)).strftime("%Y-%m-%d")
            for offset in range(40, 47)
        ]

        def history(days=55, spike_on=None):
            rows = []
            for offset in range(days):
                day = start + timedelta(days=offset)
                iso = day.strftime("%Y-%m-%d")
                eggs = 20000 if spike_on == iso else 7000 + (offset % 5) * 30
                rows.append({"date": iso, "eggs": eggs})
            return rows

        plain = run_forecast(history(), predict_dates=week, model="arima_2_1_1")
        spiked = run_forecast(
            history(spike_on="2026-02-14"),
            predict_dates=week,
            model="arima_2_1_1",
        )
        clipped = run_forecast(
            [row for row in history() if row["date"] < week[0]],
            predict_dates=week,
            model="arima_2_1_1",
        )

        self.assertIsNone(plain["error"])
        self.assertIsNone(spiked["error"])
        self.assertIsNone(clipped["error"])
        self.assertEqual(plain["dates"], week)
        self.assertEqual(plain["data"], spiked["data"])
        self.assertEqual(plain["data"], clipped["data"])

    def test_arima_fit_failure_is_returned_as_error(self):
        start = datetime(2026, 1, 1)
        history = [
            {
                "date": (start + timedelta(days=offset)).strftime("%Y-%m-%d"),
                "eggs": 7000,
            }
            for offset in range(40)
        ]

        class Boom:
            def fit(self):
                raise ValueError("singular matrix")

        with patch("src.service.ARIMA", return_value=Boom()):
            result = run_forecast(
                history,
                predict_dates=["2026-02-10"],
                model="arima_2_1_1",
            )
        self.assertEqual(result["predicted_eggs"], 0)
        self.assertIn("ValueError", result["error"])
        self.assertTrue(result["warnings"])


if __name__ == "__main__":
    unittest.main()
