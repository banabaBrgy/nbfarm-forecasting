import unittest
from datetime import datetime, timedelta

from src.service import run_forecast


class ForecastTests(unittest.TestCase):
    def test_perfect_line_legacy(self):
        start = datetime(2026, 8, 1)
        history = [
            {"date": (start + timedelta(days=i)).strftime("%Y-%m-%d"), "eggs": 1000 + i * 10}
            for i in range(7)
        ]
        result = run_forecast(
            history,
            predict_dates=["2026-08-08"],
            window=14,
            model="linear_regression",
        )
        self.assertIsNone(result["error"])
        self.assertEqual(result["predicted_eggs"], 1070)
        self.assertEqual(result["data"][0], 1070)

    def test_three_year_linear_regression_uses_same_calendar_day_trend(self):
        history = []
        for year, eggs in ((2023, 6000), (2024, 6600), (2025, 7200)):
            history.append(
                {
                    "date": f"{year}-10-05",
                    "eggs": eggs,
                    "small": 900,
                    "medium": 2100,
                    "large": 2700,
                    "extra_large": 900,
                    "cracked": 0,
                }
            )
            history.append(
                {
                    "date": f"{year}-10-06",
                    "eggs": eggs - 300,
                    "small": 900,
                    "medium": 2100,
                    "large": 2400,
                    "extra_large": 900,
                    "cracked": 0,
                }
            )

        result = run_forecast(
            history,
            predict_dates=["2026-10-05", "2026-10-06"],
            temperatures={"2026-10-05": 28, "2026-10-06": 28},
            model="three_year_linear_regression",
        )
        self.assertIsNone(result["error"])
        self.assertEqual(result["model"], "three_year_linear_regression")
        self.assertEqual(result["sample_days"], 3)
        self.assertEqual(result["predicted_eggs"], 7800)
        self.assertNotEqual(result["data"][0], result["data"][1])
        self.assertEqual(sum(result["sizes"].values()), result["predicted_eggs"])

    def test_archive_end_excludes_later_years(self):
        history = [
            {
                "date": "2023-01-02",
                "eggs": 6000,
                "small": 900,
                "medium": 2100,
                "large": 2400,
                "extra_large": 600,
                "cracked": 0,
            },
            {
                "date": "2024-01-02",
                "eggs": 6600,
                "small": 900,
                "medium": 2100,
                "large": 2700,
                "extra_large": 900,
                "cracked": 0,
            },
            {
                "date": "2025-01-02",
                "eggs": 7200,
                "small": 900,
                "medium": 2400,
                "large": 3000,
                "extra_large": 900,
                "cracked": 0,
            },
            {
                "date": "2026-01-02",
                "eggs": 3000,
                "small": 1500,
                "medium": 900,
                "large": 300,
                "extra_large": 300,
                "cracked": 0,
            },
        ]
        result = run_forecast(
            history,
            predict_dates=["2027-01-02"],
            temperatures={"2027-01-02": 28},
            model="three_year_linear_regression",
            archive_start="2023-01-01",
            archive_end="2025-12-31",
        )
        self.assertIsNone(result["error"])
        self.assertEqual(result["sample_days"], 3)
        # 2023–2025 trend only; 2026 seed row is clipped out → 8400 for 2027.
        self.assertEqual(result["predicted_eggs"], 8400)

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
        cool = run_forecast(
            history,
            predict_dates=["2026-10-05"],
            temperatures={"2026-10-05": 22},
            model="three_year_linear_regression",
        )
        hot = run_forecast(
            history,
            predict_dates=["2026-10-05"],
            temperatures={"2026-10-05": 34},
            model="three_year_linear_regression",
        )
        self.assertEqual(cool["predicted_eggs"], hot["predicted_eggs"])
        self.assertGreater(hot["sizes"]["small"], cool["sizes"]["small"])
        self.assertLess(hot["sizes"]["large"], cool["sizes"]["large"])

    def test_needs_two_points_legacy(self):
        result = run_forecast(
            [{"date": "2026-08-01", "eggs": 100}],
            window=14,
            model="linear_regression",
        )
        self.assertEqual(result["predicted_eggs"], 0)
        self.assertIsNotNone(result["error"])

    def test_no_negative_eggs_legacy(self):
        history = [
            {"date": "2026-08-01", "eggs": 50},
            {"date": "2026-08-02", "eggs": 40},
            {"date": "2026-08-03", "eggs": 30},
        ]
        result = run_forecast(
            history,
            predict_dates=["2026-09-01"],
            window=14,
            model="linear_regression",
        )
        self.assertGreaterEqual(result["data"][0], 0)


if __name__ == "__main__":
    unittest.main()
