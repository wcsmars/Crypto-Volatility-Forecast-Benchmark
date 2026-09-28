import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from crypto_volatility.data import load_dataset, validate_frame


class DataTests(unittest.TestCase):
    def frame(self, n=100):
        return pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=n),
                             "Close": 100 * np.exp(np.sin(np.arange(n)) / 10),
                             "Volume": np.arange(n), "Currency": "USD"})

    def test_formula_and_first_window(self):
        raw = self.frame()
        frame, _, _ = validate_frame(raw)
        returns = np.diff(np.log(raw.Close.to_numpy()))[:30]
        expected = np.sqrt(365 * np.mean((returns - returns.mean()) ** 2))
        self.assertTrue(frame.volatility_30.iloc[:30].isna().all())
        self.assertAlmostEqual(frame.volatility_30.iloc[30], expected, places=13)

    def test_missing_date_breaks_returns_and_windows(self):
        frame, audit, _ = validate_frame(self.frame().drop(index=40))
        self.assertEqual(audit["missing_days"], 1)
        self.assertTrue(frame.log_return.iloc[40:42].isna().all())
        self.assertTrue(frame.volatility_30.iloc[40:71].isna().all())
        self.assertTrue(np.isfinite(frame.volatility_30.iloc[71]))

    def test_invalid_close_is_not_forward_filled(self):
        raw = self.frame()
        raw.loc[40, "Close"] = 0
        frame, audit, _ = validate_frame(raw)
        self.assertEqual(audit["invalid_close_rows"], 1)
        self.assertTrue(frame.log_return.iloc[40:42].isna().all())

    def test_sorts_before_returns(self):
        forward, _, _ = validate_frame(self.frame())
        reverse, _, _ = validate_frame(self.frame().iloc[::-1])
        pd.testing.assert_frame_equal(forward, reverse)

    def test_reject_duplicate_dates(self):
        raw = self.frame()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_frame(pd.concat([raw, raw.iloc[:1]]))

    def test_reject_numeric_or_intraday_dates(self):
        raw = self.frame()
        raw["Date"] = np.arange(len(raw))
        with self.assertRaisesRegex(ValueError, "numeric timestamps"):
            validate_frame(raw)
        raw = self.frame()
        raw["Date"] += pd.Timedelta(hours=1)
        with self.assertRaisesRegex(ValueError, "midnight UTC"):
            validate_frame(raw)

    def test_iso_string_dates_are_accepted(self):
        raw = self.frame()
        raw["Date"] = raw["Date"].dt.strftime("%Y-%m-%d")
        frame, audit, _ = validate_frame(raw)
        self.assertEqual(len(frame), 100)
        self.assertEqual(frame.index[0], pd.Timestamp("2020-01-01"))
        self.assertEqual(audit["missing_days"], 0)

    def test_reduced_precision_dates_cannot_invent_calendar_days(self):
        for labels in (["2020", "2021"], ["2020-01", "2020-02"],
                       ["2020-01-01", "2020-02"]):
            with self.subTest(labels=labels):
                raw = self.frame(2)
                raw["Date"] = labels
                with self.assertRaisesRegex(ValueError, "explicit calendar day"):
                    validate_frame(raw)

    def test_full_iso_midnight_timestamps_and_basic_dates_are_accepted(self):
        for pattern in ("%Y-%m-%dT00:00:00Z", "%Y%m%d"):
            with self.subTest(pattern=pattern):
                raw = self.frame()
                raw["Date"] = raw["Date"].dt.strftime(pattern)
                frame, audit, _ = validate_frame(raw)
                self.assertEqual(frame.index[0], pd.Timestamp("2020-01-01"))
                self.assertEqual(audit["missing_days"], 0)

    def test_ambiguous_day_first_dates_are_rejected_not_permuted(self):
        raw = self.frame()
        raw["Date"] = raw["Date"].dt.strftime("%d/%m/%Y")
        with self.assertRaisesRegex(ValueError, "ISO 8601"):
            validate_frame(raw)
        raw = self.frame()
        labels = raw["Date"].dt.strftime("%Y-%m-%d").tolist()
        labels[6] = "07/01/2020"
        raw["Date"] = labels
        with self.assertRaisesRegex(ValueError, "ISO 8601"):
            validate_frame(raw)

    def test_inconsistent_candles_are_audited_without_dropping_close(self):
        raw = self.frame()
        raw["Open"] = raw["Close"]
        raw["High"] = raw["Close"] * 1.01
        raw["Low"] = raw["Close"] * 0.99
        raw.loc[5, "High"] = raw.loc[5, "Low"] * 0.5
        frame, audit, events = validate_frame(raw)
        self.assertEqual(audit["ohlc_invalid_rows"], 1)
        self.assertEqual([event["issue"] for event in events], ["inconsistent_ohlc"])
        self.assertEqual(events[0]["date"], pd.Timestamp("2020-01-06"))
        self.assertTrue(np.isfinite(frame.log_return.iloc[5]))

    def test_volume_counter_describes_source_rows_not_calendar_gaps(self):
        raw = self.frame().drop(index=40)
        raw.loc[3, "Volume"] = np.nan
        frame, audit, _ = validate_frame(raw)
        self.assertEqual(audit["missing_days"], 1)
        self.assertEqual(audit["missing_or_invalid_volume_rows"], 1)
        self.assertEqual(audit["zero_volume_rows"], 1)
        self.assertTrue(np.isnan(frame.Volume.loc["2020-02-10"]))

    def test_absent_optional_columns_are_recorded_not_counted_as_defects(self):
        _, audit, _ = validate_frame(self.frame().drop(columns=["Volume"]))
        self.assertEqual(audit["missing_columns"], "Open,High,Low,Volume")
        self.assertTrue(np.isnan(audit["ohlc_invalid_rows"]))
        self.assertTrue(np.isnan(audit["zero_volume_rows"]))
        self.assertTrue(np.isnan(audit["missing_or_invalid_volume_rows"]))
        _, audit, _ = validate_frame(self.frame().drop(columns=["Currency"]))
        self.assertEqual(audit["currency"], "not supplied")
        self.assertEqual(audit["missing_columns"], "Open,High,Low")

    def test_too_short_or_all_invalid_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "two source observations"):
            validate_frame(self.frame(1))
        raw = self.frame()
        raw["Close"] = 0
        with self.assertRaisesRegex(ValueError, "two valid positive closes"):
            validate_frame(raw)
        with self.assertRaisesRegex(ValueError, "Required columns"):
            validate_frame(self.frame().drop(columns=["Close"]))

    def test_case_insensitive_name_collision_is_rejected(self):
        class Entry:
            def __init__(self, name, content):
                self.name, self.stem, self.suffix, self._content = name, Path(name).stem, Path(name).suffix, content

            def is_file(self):
                return True

            def read_bytes(self):
                return self._content

        payload = self.frame().to_csv(index=False).encode()
        # The outcome must not depend on the directory order the filesystem returns.
        for names in (["coin.csv", "COIN.csv"], ["COIN.csv", "coin.csv"]):
            with self.subTest(order=names):
                entries = [Entry(name, payload) for name in names] + [Entry("notes.txt", b"ignored")]
                with tempfile.TemporaryDirectory() as directory, \
                        patch.object(Path, "iterdir", return_value=iter(entries)):
                    dataset = load_dataset(directory)
                self.assertEqual(list(dataset.frames), ["COIN"])
                self.assertEqual(sorted(dataset.hashes), ["COIN.csv", "coin.csv"])
                rejected = dataset.validation[dataset.validation["status"].eq("rejected")]
                self.assertEqual(rejected["file"].tolist(), ["coin.csv"])
                self.assertIn("collision", rejected["reason"].iloc[0])

    def test_loader_reads_basic_iso_dates_as_calendar_labels(self):
        # An all-digit YYYYMMDD column would otherwise be parsed as integers and rejected.
        with tempfile.TemporaryDirectory() as directory:
            raw = self.frame()
            raw["Date"] = raw["Date"].dt.strftime("%Y%m%d")
            raw.to_csv(Path(directory) / "coin.csv", index=False)
            dataset = load_dataset(Path(directory))
        self.assertEqual(dataset.validation["status"].tolist(), ["accepted"])
        self.assertEqual(dataset.frames["coin"].index[0], pd.Timestamp("2020-01-01"))
        self.assertEqual(len(dataset.frames["coin"]), 100)

    def test_volume_zero_preserved_negative_marked(self):
        raw = self.frame()
        raw.loc[1, "Volume"] = -1
        frame, audit, _ = validate_frame(raw)
        self.assertEqual(frame.Volume.iloc[0], 0)
        self.assertTrue(np.isnan(frame.Volume.iloc[1]))
        self.assertEqual(audit["zero_volume_rows"], 1)

    def test_currency_mismatch_rejected(self):
        raw = self.frame()
        raw.loc[0, "Currency"] = "EUR"
        with self.assertRaisesRegex(ValueError, "USD"):
            validate_frame(raw)

    def test_loading_records_rejections_and_leaderboard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.frame().to_csv(path / "valid.csv", index=False)
            pd.DataFrame({"Rank": [1]}).to_csv(path / "Current Crypto leaderboard.csv", index=False)
            pd.DataFrame({"unrelated": [1]}).to_csv(path / "bad.csv", index=False)
            dataset = load_dataset(path)
            self.assertEqual(list(dataset.frames), ["valid"])
            self.assertEqual(set(dataset.validation.status), {"accepted", "rejected", "skipped"})
            self.assertEqual(len(dataset.hashes), 3)


if __name__ == "__main__":
    unittest.main()
