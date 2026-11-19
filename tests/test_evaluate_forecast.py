import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from evaluate_forecast import holdout_split, load_daily_totals, render_report, score_holdout


class EvaluateForecastTests(unittest.TestCase):
    def test_perfect_line_scores_zero_error(self):
        start = datetime(2026, 1, 1)
        rows = [(start + timedelta(days=index), 1000 + index * 10) for index in range(10)]
        train, test = holdout_split(rows)
        self.assertEqual(len(train), 8)
        self.assertEqual(len(test), 2)
        metrics = score_holdout(train, test)
        self.assertEqual(metrics["mae"], 0)
        self.assertEqual(metrics["rmse"], 0)
        self.assertEqual(metrics["mape"], 0)
        self.assertEqual(metrics["r2"], 1)
        self.assertEqual(metrics["skipped_zero"], 0)

    def test_zero_egg_days_are_skipped_in_mape(self):
        start = datetime(2026, 1, 1)
        rows = [(start + timedelta(days=index), 100.0) for index in range(5)]
        rows.append((start + timedelta(days=5), 0.0))
        train, test = rows[:4], rows[4:]
        metrics = score_holdout(train, test)
        self.assertEqual(metrics["tested"], 1)
        self.assertEqual(metrics["skipped_zero"], 1)

    def test_report_matches_terminal_layout(self):
        start = datetime(2026, 1, 5)
        test = [(start, 10.0), (start + timedelta(days=1), 12.0)]
        report = render_report(
            test,
            1100,
            {"tested": 2, "mae": 198.735, "rmse": 232.416, "mape": 2.876, "r2": -0.22236, "skipped_zero": 0},
        )
        self.assertEqual(
            report,
            "\n".join(
                [
                    "Test period: 2026-01-05 to 2026-01-06 (2 days, trained on 1100)",
                    "MAE  : 198.74 eggs",
                    "RMSE : 232.42 eggs",
                    "MAPE : 2.88% (zero-egg days skipped: 0)",
                    "R2   : -0.2224",
                ]
            ),
        )

    def test_csv_sums_rows_for_the_same_date(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "production.csv"
            path.write_text(
                "date,eggs\n2026-01-01,100\n2026-01-01,50\n2026-01-02,200\n2026-01-03,300\n",
                encoding="utf-8",
            )
            rows = load_daily_totals(path)
        self.assertEqual(rows, [
            (datetime(2026, 1, 1), 150.0),
            (datetime(2026, 1, 2), 200.0),
            (datetime(2026, 1, 3), 300.0),
        ])


if __name__ == "__main__":
    unittest.main()
