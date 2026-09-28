"""Regression checks for forecast timing, gaps, failure coverage and scoring."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from crypto_volatility.model import (
    FEATURE_COLUMNS, MODELS, PREDICTION_COLUMNS, backtest_asset, build_features, score_predictions,
)


def daily_frame(n=750, seed=7):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.025, n)
    returns[0] = np.nan
    index = pd.date_range("2020-01-01", periods=n, freq="D", name="Date")
    frame = pd.DataFrame(index=index)
    frame["Close"] = 100 * np.exp(np.nan_to_num(returns).cumsum())
    frame["Volume"] = 1_000.0
    frame["log_return"] = returns
    frame["volatility_30"] = frame["log_return"].rolling(30).std(ddof=0) * np.sqrt(365)
    return frame


class FeaturesTests(unittest.TestCase):
    def test_forward_window_exactly_next_30_returns(self):
        frame = daily_frame(170)
        features = build_features(frame)
        origin = frame.index[100]
        expected = frame["log_return"].iloc[101:131].std(ddof=0) * np.sqrt(365)
        self.assertAlmostEqual(features.at[origin, "target_volatility"], expected, places=12)
        self.assertEqual(features.at[origin, "target_start"], frame.index[101])
        self.assertEqual(features.at[origin, "target_end"], frame.index[130])
        self.assertTrue(features["target_volatility"].iloc[-30:].isna().all())
        self.assertEqual(len(frame), len(features))

    def test_features_are_past_only_and_exclude_volume(self):
        frame = daily_frame(170)
        original = build_features(frame)
        changed = frame.copy()
        changed.loc[changed.index[101]:, "log_return"] += 2
        changed["Volume"] *= 1e12
        changed["volatility_30"] = changed["log_return"].rolling(30).std(ddof=0) * np.sqrt(365)
        perturbed = build_features(changed)
        pd.testing.assert_series_equal(
            original.loc[frame.index[100], list(FEATURE_COLUMNS)],
            perturbed.loc[frame.index[100], list(FEATURE_COLUMNS)],
        )

    def test_gaps_are_not_compressed_or_filled(self):
        frame = daily_frame(250)
        with self.assertRaisesRegex(ValueError, "complete daily calendar"):
            build_features(frame.drop(frame.index[100]))
        frame.loc[frame.index[100:102], "log_return"] = np.nan
        frame["volatility_30"] = frame["log_return"].rolling(30).std(ddof=0) * np.sqrt(365)
        features = build_features(frame)
        self.assertTrue(features["volatility_90"].iloc[100:191].isna().all())
        self.assertTrue(np.isfinite(features["volatility_90"].iloc[191]))

    def test_labels_come_from_returns_and_supplied_volatility_must_agree(self):
        frame = daily_frame(170)
        expected = frame["log_return"].iloc[101:131].std(ddof=0) * np.sqrt(365)
        without_column = build_features(frame.drop(columns="volatility_30"))
        self.assertAlmostEqual(without_column.at[frame.index[100], "target_volatility"], expected, places=12)
        inconsistent = frame.copy()
        inconsistent["volatility_30"] = inconsistent["log_return"].rolling(30).std(ddof=1) * np.sqrt(365)
        with self.assertRaisesRegex(ValueError, "volatility_30 disagrees"):
            build_features(inconsistent)

    def test_calendar_check_is_independent_of_datetime_resolution(self):
        frame = daily_frame(170)
        frame.index = frame.index.as_unit("s")
        features = build_features(frame)
        self.assertEqual(len(features), 170)
        with self.assertRaisesRegex(ValueError, "complete daily calendar"):
            build_features(frame.drop(frame.index[50]))


class BacktestTests(unittest.TestCase):
    def test_settings_must_be_positive_integers(self):
        frame = daily_frame(100)
        for bad in (0, -1, True, 1.5, "3"):
            for name in ("test_days", "min_train", "refit_every"):
                with self.subTest(setting=name, value=bad):
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        backtest_asset(frame, "Coin", **{name: bad})

    def test_history_shorter_than_one_window_is_reported(self):
        predictions, latest, status = backtest_asset(daily_frame(20), "Tiny")
        self.assertTrue(predictions.empty)
        self.assertEqual(list(predictions.columns), list(PREDICTION_COLUMNS))
        self.assertEqual(status["backtest_status"], "skipped")
        self.assertIn("Fewer than 31 calendar dates", status["errors"][0])
        self.assertEqual(len(latest), len(MODELS))
        self.assertTrue(latest["status"].eq("insufficient_training").all())

    def test_baseline_alignment_training_boundary_and_final_forecast(self):
        frame = daily_frame()
        predictions, latest, status = backtest_asset(frame, "Coin", test_days=45)
        self.assertEqual(status["backtest_status"], "ok")
        self.assertEqual(len(predictions), 45 * len(MODELS))
        self.assertTrue(predictions["status"].eq("ok").all())
        origin = frame.index[-31]
        persistence = predictions.loc[
            predictions["origin"].eq(origin) & predictions["model"].eq("Persistence")
        ].iloc[0]
        self.assertAlmostEqual(persistence.predicted, frame.at[origin, "volatility_30"])
        self.assertAlmostEqual(persistence.actual, frame["volatility_30"].iloc[-1])
        historical = predictions.loc[
            predictions["origin"].eq(origin) & predictions["model"].eq("Historical90")
        ].iloc[0]
        expected90 = frame["log_return"].loc[:origin].iloc[-90:].std(ddof=0) * np.sqrt(365)
        self.assertAlmostEqual(historical.predicted, expected90)
        ridge = predictions.loc[predictions["model"].eq("Ridge")]
        self.assertTrue(ridge["train_target_end"].le(ridge["origin"]).all())
        self.assertEqual(ridge["train_target_end"].nunique(), 2)
        self.assertTrue(ridge["n_train"].ge(250).all())
        self.assertTrue(latest["origin"].eq(frame.index[-1]).all())
        self.assertTrue(latest["target_end"].eq(frame.index[-1] + pd.Timedelta(days=30)).all())
        self.assertTrue(latest["actual"].isna().all())
        self.assertTrue(latest["status"].eq("ok").all())

    def test_future_changes_do_not_change_origin_predictions(self):
        frame = daily_frame()
        first, _, _ = backtest_asset(frame, "Coin", test_days=45)
        origin = first["origin"].min() + pd.Timedelta(days=30)
        modified = frame.copy()
        modified.loc[modified.index > origin, "log_return"] *= 12
        modified["volatility_30"] = modified["log_return"].rolling(30).std(ddof=0) * np.sqrt(365)
        second, _, _ = backtest_asset(modified, "Coin", test_days=45)
        one = first.loc[first["origin"].eq(origin)].set_index("model")
        two = second.loc[second["origin"].eq(origin)].set_index("model")
        np.testing.assert_allclose(one["predicted"], two["predicted"], rtol=0, atol=1e-12)
        self.assertFalse(np.allclose(one["actual"], two["actual"]))
        pd.testing.assert_series_equal(one["n_train"], two["n_train"])

    def test_min_train_threshold_is_inclusive_at_both_origins(self):
        frame = daily_frame()
        _, _, probe = backtest_asset(frame, "Coin", test_days=45)
        for count, name in (("n_train_initial", "backtest_status"), ("n_train_latest", "latest_status")):
            exact = probe[count]
            with self.subTest(count=count, min_train=exact):
                _, _, status = backtest_asset(frame, "Coin", test_days=45, min_train=exact)
                self.assertEqual(status[name], "ok")
            with self.subTest(count=count, min_train=exact + 1):
                predictions, latest, status = backtest_asset(frame, "Coin", test_days=45, min_train=exact + 1)
                self.assertNotEqual(status[name], "ok")
                rows = predictions if count == "n_train_initial" else latest
                self.assertTrue(rows["status"].eq("insufficient_training").all())

    def test_oversized_test_window_uses_every_available_origin(self):
        # Beyond pandas' Timedelta range (about 106,752 days) this must not overflow.
        frame = daily_frame(400)
        available, latest, _ = backtest_asset(frame, "Coin", test_days=400)
        for test_days in (10 ** 6, 10 ** 9):
            with self.subTest(test_days=test_days):
                predictions, latest_rows, _ = backtest_asset(frame, "Coin", test_days=test_days)
                pd.testing.assert_frame_equal(predictions, available)
                pd.testing.assert_frame_equal(latest_rows, latest)
        self.assertEqual(available["origin"].nunique(), 400 - 30)

    def test_short_history_is_explicitly_skipped(self):
        predictions, latest, status = backtest_asset(daily_frame(300), "Short")
        self.assertEqual(status["backtest_status"], "skipped")
        self.assertEqual(len(predictions), 180 * len(MODELS))
        self.assertEqual(len(latest), len(MODELS))
        self.assertTrue(predictions["status"].eq("insufficient_training").all())
        self.assertTrue(latest["status"].eq("insufficient_training").all())
        metrics = score_predictions(predictions)
        self.assertEqual(len(metrics), 2 * len(MODELS))
        self.assertTrue(metrics["n_scored"].eq(0).all())
        self.assertTrue(metrics["coverage"].eq(0).all())

    def test_gap_outputs_are_explicit_and_latest_is_not_stale(self):
        frame = daily_frame()
        frame.loc[frame.index[-10:], ["Close", "log_return", "volatility_30"]] = np.nan
        frame["volatility_30"] = frame["log_return"].rolling(30).std(ddof=0) * np.sqrt(365)
        predictions, latest, status = backtest_asset(frame, "Gap", test_days=45)
        self.assertEqual(status["backtest_status"], "partial")
        self.assertTrue(predictions["status"].eq("missing_target").any())
        missing = predictions.loc[predictions["status"].eq("missing_target")]
        self.assertTrue(np.isfinite(missing["predicted"]).all())
        self.assertTrue(latest["origin"].eq(frame.index[-1]).all())
        self.assertTrue(latest["status"].eq("missing_features").all())
        self.assertTrue(latest["predicted"].isna().all())

    def test_nonfinite_model_output_is_failure_not_score(self):
        with patch("crypto_volatility.model._ridge_prediction", return_value=(np.inf, False)):
            predictions, latest, _ = backtest_asset(daily_frame(), "Coin", test_days=3)
        ridge = predictions.loc[predictions["model"].eq("Ridge")]
        self.assertEqual(len(ridge), 3)
        self.assertTrue(ridge["status"].eq("model_error").all())
        self.assertTrue(ridge["predicted"].isna().all())
        metrics = score_predictions(predictions)
        self.assertEqual(len(metrics), 2 * len(MODELS))
        self.assertTrue(metrics["n_scored"].eq(0).all())
        self.assertTrue(latest.loc[latest["model"].eq("Ridge"), "status"].eq("model_error").all())

    def test_negative_transformed_output_is_clipped_and_recorded(self):
        class NegativeModel:
            def predict(self, values):
                return np.array([-0.2] * len(values))

        with patch("crypto_volatility.model._fit_ridge", return_value=NegativeModel()):
            predictions, latest, status = backtest_asset(daily_frame(), "Coin", test_days=3)
        ridge = predictions.loc[predictions["model"].eq("Ridge")]
        self.assertTrue(ridge["predicted"].eq(0).all())
        self.assertTrue(ridge["clipped"].all())
        self.assertEqual(status["ridge_clipped_backtest"], 3)
        self.assertEqual(status["ridge_clipped_latest"], 1)
        self.assertTrue(latest.loc[latest["model"].eq("Ridge"), "predicted"].eq(0).all())

    def test_refit_error_is_reported_with_baselines_preserved(self):
        with patch("crypto_volatility.model._fit_ridge", side_effect=ValueError("fit failed")):
            predictions, _, status = backtest_asset(daily_frame(), "Coin", test_days=3)
        self.assertEqual(len(predictions), 3 * len(MODELS))
        self.assertTrue(predictions.loc[predictions["model"].eq("Ridge"), "status"].eq("model_error").all())
        self.assertTrue(predictions.loc[~predictions["model"].eq("Ridge"), "status"].eq("ok").all())
        self.assertIn("fit failed", " ".join(status["errors"]))


class ScoringTests(unittest.TestCase):
    def test_finite_extreme_errors_have_finite_accurate_metrics(self):
        origins = pd.date_range("2020-01-01", periods=2, freq="D")
        for scale in (1e-200, 1e200, np.finfo(float).max):
            with self.subTest(scale=scale):
                rows = [{
                    "asset": "Coin", "model": model, "origin": origin,
                    "target_start": origin + pd.Timedelta(days=1),
                    "target_end": origin + pd.Timedelta(days=30),
                    "actual": 0.0, "predicted": fraction * scale, "status": "ok",
                } for origin, fraction in zip(origins, (0.5, 1.0)) for model in MODELS]
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    metrics = score_predictions(pd.DataFrame(rows))
                daily = metrics.loc[metrics["sample"].eq("daily_origins")]
                np.testing.assert_allclose(daily["mae"] / scale, 0.75, rtol=1e-14)
                np.testing.assert_allclose(daily["rmse"] / scale, np.sqrt(0.625), rtol=1e-14)

    def test_common_dates_and_nonoverlap_schedule_survive_missing_rows(self):
        predictions, _, _ = backtest_asset(daily_frame(), "Coin", test_days=61)
        first_origin = predictions["origin"].min()
        bad = predictions["origin"].eq(first_origin) & predictions["model"].eq("Ridge")
        predictions.loc[bad, "status"] = "model_error"
        predictions.loc[bad, "predicted"] = np.nan
        metrics = score_predictions(predictions)
        daily = metrics.loc[metrics["sample"].eq("daily_origins")]
        sparse = metrics.loc[metrics["sample"].eq("nonoverlap_30d")]
        self.assertTrue(daily["n_requested"].eq(61).all())
        self.assertTrue(daily["n_scored"].eq(60).all())
        self.assertTrue(sparse["n_requested"].eq(3).all())
        self.assertTrue(sparse["n_scored"].eq(2).all())
        baseline = predictions.loc[
            predictions["model"].eq("Persistence") & ~predictions["origin"].eq(first_origin)
        ]
        expected = np.mean(np.abs(baseline["predicted"] - baseline["actual"]))
        self.assertAlmostEqual(daily.loc[daily["model"].eq("Persistence"), "mae"].iloc[0], expected)

    def test_duplicate_predictions_rejected(self):
        predictions, _, _ = backtest_asset(daily_frame(), "Coin", test_days=2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            score_predictions(pd.concat([predictions, predictions.iloc[[0]]]))

    def test_corrupted_target_dates_rejected(self):
        predictions, _, _ = backtest_asset(daily_frame(), "Coin", test_days=2)
        for column in ("target_start", "target_end"):
            with self.subTest(column=column):
                corrupted = predictions.copy()
                corrupted.loc[corrupted.index[0], column] += pd.Timedelta(days=1)
                with self.assertRaisesRegex(ValueError, "target dates"):
                    score_predictions(corrupted)
        with self.assertRaisesRegex(ValueError, "required scoring columns"):
            score_predictions(predictions.drop(columns="target_end"))

    def test_actual_disagreement_rejected_even_for_failed_model(self):
        predictions, _, _ = backtest_asset(daily_frame(), "Coin", test_days=2)
        predictions.loc[predictions.index[0], "actual"] += 0.05
        predictions.loc[predictions.index[0], "status"] = "model_error"
        with self.assertRaisesRegex(ValueError, "disagree on the actual target"):
            score_predictions(predictions)

    def test_unknown_model_is_rejected(self):
        predictions, _, _ = backtest_asset(daily_frame(), "Coin", test_days=2)
        predictions.loc[predictions["model"].eq("Ridge"), "model"] = "Typo"
        with self.assertRaisesRegex(ValueError, "unknown model"):
            score_predictions(predictions)

    def test_missing_model_row_reduces_common_coverage(self):
        predictions, _, _ = backtest_asset(daily_frame(), "Coin", test_days=3)
        middle = predictions["origin"].min() + pd.Timedelta(days=1)
        removed = predictions["origin"].eq(middle) & predictions["model"].eq("Ridge")
        metrics = score_predictions(predictions.loc[~removed])
        daily = metrics.loc[metrics["sample"].eq("daily_origins")]
        self.assertTrue(daily["n_requested"].eq(3).all())
        self.assertTrue(daily["n_scored"].eq(2).all())
        self.assertTrue(daily["coverage"].eq(2 / 3).all())

    def test_malformed_prediction_tables_are_rejected(self):
        self.assertTrue(score_predictions(pd.DataFrame(columns=PREDICTION_COLUMNS)).empty)
        predictions, _, _ = backtest_asset(daily_frame(), "Coin", test_days=2)
        cases = [
            ("asset", None, "cannot be missing"),
            ("origin", predictions["origin"].iloc[0] + pd.Timedelta(hours=1), "midnight"),
            ("predicted", "text", "could not convert|Unable to parse"),
        ]
        for column, value, message in cases:
            with self.subTest(column=column):
                corrupted = predictions.copy()
                corrupted[column] = corrupted[column].astype(object)
                corrupted.loc[corrupted.index[0], column] = value
                with self.assertRaisesRegex(ValueError, message):
                    score_predictions(corrupted)

    def test_missing_entire_interior_origin_preserves_denominator_and_schedule(self):
        predictions, _, _ = backtest_asset(daily_frame(), "Coin", test_days=61)
        middle = predictions["origin"].min() + pd.Timedelta(days=30)
        metrics = score_predictions(predictions.loc[~predictions["origin"].eq(middle)])
        daily = metrics.loc[metrics["sample"].eq("daily_origins")]
        sparse = metrics.loc[metrics["sample"].eq("nonoverlap_30d")]
        self.assertTrue(daily["n_requested"].eq(61).all())
        self.assertTrue(daily["n_scored"].eq(60).all())
        self.assertTrue(daily["coverage"].eq(60 / 61).all())
        self.assertTrue(sparse["n_requested"].eq(3).all())
        self.assertTrue(sparse["n_scored"].eq(2).all())


if __name__ == "__main__":
    unittest.main()
