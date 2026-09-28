"""Regression checks for reporting errors that could misrepresent the data."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crypto_volatility.report import (
    COLORS, _choose_panel, _macro_metrics, _small_multiples, _statistics, _volume_level_shifts,
)


def history(end, days, seed=0, blank_recent=0):
    """A validated-style frame ending on ``end`` with an optional run of missing recent closes."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(end=end, periods=days, freq="D", name="Date")
    frame = pd.DataFrame(index=index)
    frame["Close"] = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, days)))
    if blank_recent:
        frame.loc[frame.index[-blank_recent - 5:-5], "Close"] = np.nan
    frame["log_return"] = np.log(frame["Close"]).diff()
    frame["volatility_30"] = frame["log_return"].rolling(30).std(ddof=0) * np.sqrt(365)
    frame["Volume"] = 1000.0
    return frame


class ReportTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_panel_anchors_on_majors_and_names_every_exclusion(self):
        end = pd.Timestamp("2022-08-23")
        frames = {
            "bitcoin": history(end, 450, seed=1),
            "ETH": history(end, 450, seed=2),
            "solana": history(pd.Timestamp("2021-08-28"), 450, seed=3),
            "cardano": history(end, 200, seed=4),
            "Other Coin": history(end + pd.Timedelta(days=1), 500, seed=5),
        }
        assets, labels, start, panel_end, selection = _choose_panel(frames)
        self.assertEqual(assets, ["bitcoin", "ETH"])
        self.assertEqual(labels, {"bitcoin": "Bitcoin", "ETH": "Ethereum"})
        self.assertEqual(panel_end, end)
        self.assertEqual(start, end - pd.Timedelta(days=364))
        self.assertEqual(selection["path"], "fixed_list")
        self.assertEqual(set(selection["excluded"]), {"Solana", "Cardano"})
        self.assertIn("no valid close on the panel end date 2022-08-23 (last valid close 2021-08-28)",
                      selection["excluded"]["Solana"])
        self.assertIn("history starts", selection["excluded"]["Cardano"])

    def test_one_major_with_an_extra_day_does_not_evict_the_others(self):
        end = pd.Timestamp("2022-08-23")
        frames = {
            "bitcoin": history(end + pd.Timedelta(days=1), 451, seed=1),
            "ethereum": history(end, 450, seed=2),
            "litecoin": history(end, 450, seed=3),
        }
        assets, labels, start, panel_end, selection = _choose_panel(frames)
        self.assertEqual(assets, ["bitcoin", "ethereum", "litecoin"])
        self.assertEqual(panel_end, end)
        self.assertEqual(selection["excluded"], {})
        # The most common last-valid date wins; a tie resolves to the latest date,
        # and the day-short assets are then named with the reason.
        frames["ethereum"] = history(end + pd.Timedelta(days=1), 451, seed=2)
        frames["xrp"] = history(end, 450, seed=4)
        assets, labels, start, panel_end, selection = _choose_panel(frames)
        self.assertEqual(panel_end, end + pd.Timedelta(days=1))
        self.assertEqual(assets, ["bitcoin", "ethereum"])
        self.assertEqual(set(selection["excluded"]), {"XRP", "Litecoin"})
        self.assertIn("no valid close on the panel end date 2022-08-24", selection["excluded"]["XRP"])

    def test_fallback_never_lists_a_displayed_asset_as_excluded(self):
        end = pd.Timestamp("2022-08-23")
        frames = {"bitcoin": history(end, 200, seed=1), "Foo": history(end, 200, seed=2)}
        assets, labels, start, panel_end, selection = _choose_panel(frames)
        self.assertEqual(selection["path"], "completeness_fallback")
        self.assertEqual(sorted(assets), ["Foo", "bitcoin"])
        self.assertEqual(labels["bitcoin"], "Bitcoin")
        self.assertEqual(selection["excluded"], {})
        frames = {
            "bitcoin": history(end, 200, seed=1),
            "cardano": history(end - pd.Timedelta(days=2), 200, seed=2),
            "Zed": history(end + pd.Timedelta(days=5), 500, seed=3),
        }
        assets, labels, start, panel_end, selection = _choose_panel(frames)
        self.assertEqual(assets, ["Zed"])
        self.assertEqual(panel_end, end + pd.Timedelta(days=5))
        self.assertEqual(set(selection["excluded"]), {"Bitcoin", "Cardano"})
        for reason in selection["excluded"].values():
            self.assertIn("panel end date 2022-08-28", reason)

    def test_panel_falls_back_to_completeness_ranking(self):
        end = pd.Timestamp("2022-08-23")
        frames = {
            "aaa": history(end, 450, seed=1, blank_recent=50),
            "bbb": history(end, 450, seed=2),
            "ccc": history(end, 100, seed=3),
        }
        assets, labels, start, panel_end, selection = _choose_panel(frames)
        self.assertEqual(selection["path"], "completeness_fallback")
        self.assertEqual(assets, ["bbb", "aaa"])
        self.assertEqual(labels, {"bbb": "bbb", "aaa": "aaa"})
        self.assertEqual(panel_end, end)
        self.assertEqual(start, end - pd.Timedelta(days=364))
        self.assertEqual(selection["excluded"], {})
        self.assertEqual(_choose_panel({})[4]["path"], "none")

    def test_small_multiples_accepts_more_series_than_colours(self):
        index = pd.date_range("2022-01-01", periods=3)
        panel = pd.DataFrame({f"asset{i}": [1.0, 2.0, 3.0] for i in range(len(COLORS) + 2)}, index=index)
        with patch("crypto_volatility.report._finish") as finish:
            _small_multiples(panel, {key: key for key in panel}, "Title", "Value", Path("unused.png"))
        figure = finish.call_args.args[0]
        self.assertEqual(sum(axis.get_visible() for axis in figure.axes), len(COLORS) + 2)

    def test_shared_axes_include_later_asset_peak(self):
        panel = pd.DataFrame({"first": [1.0, 2.0], "later": [1.0, 250.0]},
                             index=pd.date_range("2022-01-01", periods=2))
        with patch("crypto_volatility.report._finish") as finish:
            _small_multiples(panel, {key: key for key in panel}, "Title", "Value", Path("unused.png"))
        figure = finish.call_args.args[0]
        for axis in figure.axes:
            self.assertEqual(axis.get_ylim()[0], 0)
            self.assertGreater(axis.get_ylim()[1], 250)
        self.assertEqual(figure.axes[0].get_ylim(), figure.axes[1].get_ylim())

    def test_volume_symlog_retains_zero_and_extreme(self):
        panel = pd.DataFrame({"asset": [0.0, 1.0, 50000.0]},
                             index=pd.date_range("2022-01-01", periods=3))
        with patch("crypto_volatility.report._finish") as finish:
            _small_multiples(panel, {"asset": "Asset"}, "Title", "Volume ratio", Path("unused.png"), symlog=True)
        axis = finish.call_args.args[0].axes[0]
        self.assertEqual(axis.get_yscale(), "symlog")
        np.testing.assert_array_equal(axis.lines[0].get_ydata(), [0, 1, 50000])
        self.assertGreater(axis.get_ylim()[1], 50000)

    def test_sustained_volume_jumps_are_flagged_but_brief_spikes_are_not(self):
        index = pd.date_range("2022-01-01", periods=40)
        ratios = np.ones(40)
        ratios[5:8] = 500.0      # a three-day spike: plausible trading activity
        ratios[20:30] = 4000.0   # ten days at a new level: a probable unit change
        ratios[25] = np.nan      # a missing day splits the run into 5 + 4 days, both too short
        self.assertEqual(_volume_level_shifts(pd.DataFrame({"coin": ratios, "flat": 1.0}, index=index)), [])
        ratios[25] = 4000.0
        shifts = _volume_level_shifts(pd.DataFrame({"coin": ratios, "flat": 1.0}, index=index))
        self.assertEqual([(run["asset"], run["start"], run["end"], run["days"], run["median_ratio"])
                          for run in shifts],
                         [("coin", pd.Timestamp("2022-01-21"), pd.Timestamp("2022-01-30"), 10, 4000.0)])

    def test_latest_valid_volatility_has_its_actual_date(self):
        frame = pd.DataFrame({"Close": [100, 101, np.nan],
                              "log_return": [np.nan, 0.01, np.nan],
                              "volatility_30": [0.5, 0.6, np.nan],
                              "Volume": [0, 100, np.nan]},
                             index=pd.date_range("2022-01-01", periods=3))
        row = _statistics({"asset": frame}).iloc[0]
        self.assertEqual(row["latest_30d_volatility_date"], "2022-01-02")
        self.assertEqual(row["latest_30d_annualized_volatility"], 0.6)
        self.assertEqual(row["reported_volume_zero_share_of_observed"], 0.5)

    def test_macro_coverage_includes_unscored_assets(self):
        metrics = pd.DataFrame([
            {"asset": asset, "model": model, "sample": "daily_origins",
             "n_requested": 100, "n_scored": 100 if asset == "scored" else 0,
             "mae": 0.1 if asset == "scored" else np.nan,
             "rmse": 0.2 if asset == "scored" else np.nan}
            for asset in ("scored", "unavailable")
            for model in ("Persistence", "Historical90", "Ridge")
        ])
        result = _macro_metrics(metrics)
        self.assertTrue(result["coverage"].eq(0.5).all())
        self.assertTrue(result["n_assets"].eq(1).all())
        self.assertTrue(result["n_assets_requested"].eq(2).all())


if __name__ == "__main__":
    unittest.main()
