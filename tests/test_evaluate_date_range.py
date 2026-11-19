import unittest
from datetime import datetime, timedelta

from evaluate_7day_forecast import parse_cli, select_test_window as select_7day
from evaluate_forecast import select_test_window as select_daily


def rows_from(start: str, days: int) -> list[tuple[datetime, float]]:
    first = datetime.strptime(start, "%Y-%m-%d")
    return [(first + timedelta(days=index), 7000.0) for index in range(days)]


class EvaluateDateRangeTests(unittest.TestCase):
    def test_7day_cli_accepts_start_and_end(self):
        args = parse_cli([
            "evaluate_7day_forecast.py",
            "production.csv",
            "--start",
            "2026-10-01",
            "--end",
            "2026-10-07",
        ])
        self.assertEqual(args.csv_path, "production.csv")
        self.assertEqual(args.start, "2026-10-01")
        self.assertEqual(args.end, "2026-10-07")

    def test_7day_window_uses_inclusive_dates(self):
        rows = rows_from("2026-09-01", 40)
        start = datetime(2026, 10, 1)
        end = datetime(2026, 10, 7)
        first, stop = select_7day(rows, start, end)
        self.assertEqual(rows[first][0], start)
        self.assertEqual(rows[stop - 1][0], end)
        self.assertEqual(stop - first, 7)

    def test_daily_window_requires_both_flags(self):
        rows = rows_from("2026-01-01", 20)
        with self.assertRaises(SystemExit):
            select_daily(rows, datetime(2026, 1, 10), None)

    def test_default_window_keeps_80_percent_split(self):
        rows = rows_from("2023-01-01", 100)
        first, stop = select_daily(rows, None, None)
        self.assertEqual(first, 80)
        self.assertEqual(stop, 100)


if __name__ == "__main__":
    unittest.main()
