"""Guard the committed reference run against silently drifting from the code and README."""

import hashlib
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_volatility.data import load_dataset
from crypto_volatility.model import backtest_asset, score_predictions
from crypto_volatility.report import _macro_metrics, _volume_level_shifts
from scripts.verify_results import VerificationError, verify

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results" / "reference"
UNHASHED = {"run_metadata.json", "validation_summary.json", ".DS_Store"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output_hashes(directory: Path) -> dict:
    return {p.relative_to(directory).as_posix(): sha256(p)
            for p in sorted(directory.rglob("*")) if p.is_file() and p.name not in UNHASHED}


class ReferenceTests(unittest.TestCase):
    def test_reference_results_were_produced_by_the_current_code(self):
        metadata = json.loads((REFERENCE / "run_metadata.json").read_text())
        self.assertEqual(metadata["status"], "complete")
        current = {p.relative_to(ROOT).as_posix(): sha256(p)
                   for p in sorted((ROOT / "crypto_volatility").glob("*.py"))}
        self.assertEqual(
            metadata["code_hashes"], current,
            "crypto_volatility/ changed after results/reference was generated; rerun the reference analysis "
            "(a CRLF checkout also changes the hashes: .gitattributes pins LF line endings)",
        )

    def test_every_reference_artifact_is_the_one_the_run_wrote(self):
        # Needs no raw data: a stale or edited table, figure or report fails here.
        metadata = json.loads((REFERENCE / "run_metadata.json").read_text())
        present = output_hashes(REFERENCE)
        self.assertGreaterEqual(len(present), 22)
        for name, digest in present.items():
            self.assertEqual(metadata["output_hashes"].get(name), digest, f"{name} differs from the reference run")

    def test_reference_tables_follow_from_the_saved_predictions(self):
        metadata = json.loads((REFERENCE / "run_metadata.json").read_text())
        predictions = pd.read_csv(REFERENCE / "backtest_predictions.csv")
        metrics = pd.read_csv(REFERENCE / "backtest_metrics.csv")
        pd.testing.assert_frame_equal(score_predictions(predictions), metrics, rtol=1e-12, check_dtype=False)
        pd.testing.assert_frame_equal(_macro_metrics(metrics), pd.read_csv(REFERENCE / "macro_metrics.csv"),
                                      rtol=1e-12, check_dtype=False)
        self.assertEqual(metadata["successful_prediction_rows"], int(predictions["status"].eq("ok").sum()))
        self.assertEqual(metadata["scored_prediction_rows"],
                         int(metrics.loc[metrics["sample"].eq("daily_origins"), "n_scored"].sum()))

    def test_reference_run_used_the_manifest_snapshot_and_pinned_packages(self):
        metadata = json.loads((REFERENCE / "run_metadata.json").read_text())
        manifest = json.loads((ROOT / "data" / "source_manifest.json").read_text())
        self.assertTrue(metadata["matches_reference_source"])
        self.assertEqual(metadata["input_hashes"], manifest["files"])
        entries = [line.split("#")[0].strip() for line in (ROOT / "requirements.txt").read_text().splitlines()]
        # Every requirement must be an exact pin of a version the reference run recorded.
        pins = {name.strip(): (version.strip() if separator else None)
                for name, separator, version in (entry.partition("==") for entry in entries if entry)}
        self.assertEqual(pins, metadata["packages"], "requirements.txt pins differ from the reference run's packages")

    def test_reference_verification_summary_is_consistent(self):
        metadata = json.loads((REFERENCE / "run_metadata.json").read_text())
        summary = json.loads((REFERENCE / "validation_summary.json").read_text())
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["run_metadata_sha256"], sha256(REFERENCE / "run_metadata.json"))
        self.assertEqual(summary["scored_prediction_rows"], metadata["scored_prediction_rows"])
        self.assertEqual(summary["source_files_verified"], len(metadata["input_hashes"]))
        self.assertEqual(summary["assets_verified"], len(metadata["assets"]))
        self.assertEqual(summary["code_files_verified"], len(metadata["code_hashes"]))
        self.assertEqual(summary["output_files_recorded"], len(metadata["output_hashes"]))
        # Every reference Ridge forecast was compared with an independent refit.
        self.assertEqual(summary["ridge_forecasts_rounding_sensitive"], 0)

    def test_readme_quotes_the_reference_scores(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        macro = pd.read_csv(REFERENCE / "macro_metrics.csv")
        daily = macro[macro["sample"].eq("daily_origins")].set_index("model")
        for model in ("Persistence", "Historical90", "Ridge"):
            self.assertIn(f"{100 * daily.loc[model, 'macro_mae']:.2f}", readme, model)
        self.assertIn(f"**{int(daily.loc['Ridge', 'n_assets'])} assets**", readme)
        self.assertIn(f"**{int(daily.loc['Ridge', 'n_scored']):,} asset-origin observations per model**", readme)
        skipped = [row for row in json.loads((REFERENCE / "model_status.json").read_text())
                   if row["backtest_status"] == "skipped"]
        self.assertIn(f"; {len(skipped)} histories lack", readme)
        validation = pd.read_csv(REFERENCE / "data_validation.csv")
        accepted = validation[validation["status"].eq("accepted")]
        self.assertIn(f"**{len(accepted)} asset histories", readme)
        self.assertIn(f"{int(accepted['missing_days'].sum())} missing calendar days", readme)
        self.assertIn(f"{int(accepted['invalid_close_rows'].sum())} nonpositive closes", readme)
        self.assertIn(f"{int(accepted['source_rows'].sum()):,} source price rows", readme)

    def test_report_names_every_sustained_volume_level_shift(self):
        volume = pd.read_csv(REFERENCE / "common_window_normalized_reported_volume.csv", index_col="Date",
                             parse_dates=["Date"])
        report = (REFERENCE / "report.md").read_text(encoding="utf-8")
        shifts = _volume_level_shifts(volume)
        self.assertTrue(shifts)
        for run in shifts:
            self.assertIn(f"({run['start']:%Y-%m-%d} to {run['end']:%Y-%m-%d}, {run['days']} days,", report)
        self.assertIn("more likely reflect changes in the source's volume denomination", report)

    def test_reference_directory_holds_only_documented_outputs(self):
        present = {p.relative_to(REFERENCE).as_posix() for p in REFERENCE.rglob("*") if p.is_file()}
        documented = {
            "report.md", "run_metadata.json", "validation_summary.json", "data_validation.csv",
            "data_quality_events.csv", "summary_statistics.csv", "backtest_predictions.csv",
            "backtest_metrics.csv", "macro_metrics.csv", "next_forecasts.csv", "model_status.json",
            "common_window_statistics.csv", "common_window_indexed_close.csv", "common_window_log_returns.csv",
            "common_window_volatility_30.csv", "common_window_return_correlations.csv",
            "common_window_correlation_pair_counts.csv", "common_window_normalized_reported_volume.csv",
        } | {f"figures/{name}.png" for name in ("indexed_prices", "rolling_volatility", "return_distributions",
                                                "return_correlations", "normalized_volume", "forecast_comparison")}
        # The 17 MB history table is regenerated by every run and stays out of Git.
        self.assertEqual(present - {"validated_daily_history.csv"}, documented)
        report = (REFERENCE / "report.md").read_text(encoding="utf-8")
        for link in re.findall(r"\]\(([^)]+)\)", report):
            if not link.startswith("http"):
                self.assertTrue((REFERENCE / link).exists(), f"report.md links to a missing file: {link}")


class VerificationIntegrityTests(unittest.TestCase):
    """Corrupt saved artifacts that an incomplete verifier would accept."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.data = cls.root / "data"
        cls.data.mkdir()
        dates = pd.date_range("2020-01-01", periods=420)
        close = 100 * np.exp(np.cumsum(np.random.default_rng(7).normal(0, 0.03, len(dates))))
        raw = pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Close": close})
        raw.to_csv(cls.data / "coin.csv", index=False)
        # Load through the real loader so parsing matches a real run exactly.
        dataset = load_dataset(cls.data)
        predictions, latest, _ = backtest_asset(dataset.frames["coin"], "coin",
                                                test_days=20, min_train=50, refit_every=10)
        metrics = score_predictions(predictions)
        cls.output = cls.root / "original"
        cls.output.mkdir()
        dataset.validation.to_csv(cls.output / "data_validation.csv", index=False)
        predictions.to_csv(cls.output / "backtest_predictions.csv", index=False)
        latest.to_csv(cls.output / "next_forecasts.csv", index=False)
        metrics.to_csv(cls.output / "backtest_metrics.csv", index=False)
        _macro_metrics(metrics).to_csv(cls.output / "macro_metrics.csv", index=False)
        metadata = {
            "status": "complete", "assets": ["coin"], "data_directory": str(cls.data),
            "configuration": {"test_days": 20, "min_train": 50, "refit_every": 10, "ridge_alpha": 1.0,
                              "window": 30, "forecast_horizon_days": 30, "ddof": 0, "days_per_year": 365,
                              "assets_filter": None},
            "input_hashes": {"coin.csv": sha256(cls.data / "coin.csv")},
            "code_hashes": {p.relative_to(ROOT).as_posix(): sha256(p)
                            for p in (ROOT / "crypto_volatility").glob("*.py")},
            "successful_prediction_rows": int(predictions.status.eq("ok").sum()),
            "scored_prediction_rows": int(metrics[metrics["sample"].eq("daily_origins")].n_scored.sum()),
            "output_hashes": output_hashes(cls.output),
        }
        (cls.output / "run_metadata.json").write_text(json.dumps(metadata))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.copy = Path(tempfile.mkdtemp(dir=self.root)) / "copy"
        shutil.copytree(self.output, self.copy)

    def mutate(self, name, change):
        path = self.copy / name
        rows = pd.read_csv(path)
        change(rows).to_csv(path, index=False)

    def rewrite_metadata(self, change):
        path = self.copy / "run_metadata.json"
        metadata = json.loads((self.output / "run_metadata.json").read_text())
        change(metadata)
        path.write_text(json.dumps(metadata))

    def write_metrics_and_macro(self, metrics):
        """Save per-asset rows with a macro table rebuilt from them, so only per-asset checks can fire."""
        metrics.to_csv(self.copy / "backtest_metrics.csv", index=False)
        _macro_metrics(metrics).to_csv(self.copy / "macro_metrics.csv", index=False)

    def test_every_origin_and_forecast_is_independently_checked(self):
        summary = verify(self.copy)
        self.assertEqual(summary["source_windows_independently_recalculated"], 20)
        self.assertEqual(summary["backtest_baseline_forecasts_recalculated"], 40)
        self.assertEqual(summary["backtest_ridge_forecasts_refitted"], 20)
        self.assertEqual(summary["latest_historical90_forecasts_recalculated"], 1)
        self.assertEqual(summary["latest_ridge_forecasts_refitted"], 1)
        self.assertLess(summary["max_absolute_ridge_difference"], 1e-12)
        self.assertEqual(summary["assets_verified"], 1)
        self.assertEqual(summary["output_files_recorded"], 5)
        self.assertEqual(summary["ridge_forecasts_rounding_sensitive"], 0)

    def test_unsampled_actual_and_each_models_actual_are_checked(self):
        for model in ("Persistence", "Historical90", "Ridge"):
            with self.subTest(model=model):
                rows = pd.read_csv(self.output / "backtest_predictions.csv")
                index = rows.index[rows.model.eq(model)][2]
                rows.loc[index, "actual"] += 0.1
                rows.to_csv(self.copy / "backtest_predictions.csv", index=False)
                with self.assertRaisesRegex(VerificationError, "differs from recalculated"):
                    verify(self.copy)

    def test_missing_boundary_forecast_is_rejected(self):
        self.mutate("backtest_predictions.csv", lambda rows: rows.iloc[3:])
        with self.assertRaisesRegex(VerificationError, "requested source-date schedule"):
            verify(self.copy)

    def test_incomplete_latest_forecasts_are_rejected(self):
        self.mutate("next_forecasts.csv", lambda rows: rows.iloc[1:])
        with self.assertRaisesRegex(VerificationError, "latest rows must contain every model"):
            verify(self.copy)

    def test_duplicate_latest_forecasts_are_rejected(self):
        self.mutate("next_forecasts.csv", lambda rows: pd.concat([rows, rows.iloc[[0]]]))
        with self.assertRaisesRegex(VerificationError, "duplicate asset/model/origin"):
            verify(self.copy)

    def test_latest_training_leakage_is_rejected(self):
        def change(rows):
            rows.loc[rows.model.eq("Ridge"), "train_target_end"] = rows.target_end
            return rows
        self.mutate("next_forecasts.csv", change)
        with self.assertRaisesRegex(VerificationError, "label ending after"):
            verify(self.copy)

    def test_training_count_and_refit_cutoff_are_rebuilt_from_source(self):
        for file in ("backtest_predictions.csv", "next_forecasts.csv"):
            for field in ("n_train", "train_target_end"):
                with self.subTest(file=file, field=field):
                    rows = pd.read_csv(self.output / file)
                    index = rows.index[rows.model.eq("Ridge")][0]
                    if field == "n_train":
                        rows.loc[index, field] += 1
                    else:
                        rows.loc[index, field] = "2020-01-01"
                    rows.to_csv(self.copy / file, index=False)
                    with self.assertRaisesRegex(VerificationError, "training count or cutoff differs"):
                        verify(self.copy)
                    (self.copy / file).write_bytes((self.output / file).read_bytes())

    def test_false_missing_features_status_is_rejected(self):
        rows = pd.read_csv(self.copy / "next_forecasts.csv")
        rows.loc[0, ["status", "predicted"]] = ["missing_features", np.nan]
        rows.to_csv(self.copy / "next_forecasts.csv", index=False)
        with self.assertRaisesRegex(VerificationError, "status disagrees with available source windows"):
            verify(self.copy)

    def test_false_insufficient_training_status_is_rejected(self):
        def change(rows):
            rows["status"] = "insufficient_training"
            rows[["predicted", "train_target_end"]] = np.nan
            rows["n_train"] = 0
            return rows
        self.mutate("next_forecasts.csv", change)
        with self.assertRaisesRegex(VerificationError, "status disagrees with available source windows"):
            verify(self.copy)

    def test_latest_target_start_is_checked(self):
        self.mutate("next_forecasts.csv", lambda rows: rows.assign(target_start=rows.origin))
        with self.assertRaisesRegex(VerificationError, "target_start"):
            verify(self.copy)

    def test_latest_ridge_value_must_be_finite(self):
        def change(rows):
            rows.loc[rows.model.eq("Ridge"), "predicted"] = np.inf
            return rows
        self.mutate("next_forecasts.csv", change)
        with self.assertRaisesRegex(VerificationError, "invalid prediction"):
            verify(self.copy)

    def test_latest_forecasts_cannot_carry_actuals(self):
        self.mutate("next_forecasts.csv", lambda rows: rows.assign(actual=0.5))
        with self.assertRaisesRegex(VerificationError, "latest forecasts cannot have observed targets"):
            verify(self.copy)

    def test_all_baselines_are_recalculated(self):
        for file, model in (("backtest_predictions.csv", "Persistence"),
                            ("backtest_predictions.csv", "Historical90"),
                            ("next_forecasts.csv", "Historical90")):
            with self.subTest(file=file, model=model):
                rows = pd.read_csv(self.output / file)
                rows.loc[rows.model.eq(model), "predicted"] += 0.1
                rows.to_csv(self.copy / file, index=False)
                with self.assertRaisesRegex(VerificationError, "forecast differs"):
                    verify(self.copy)
                (self.copy / file).write_bytes((self.output / file).read_bytes())

    def test_every_ridge_forecast_is_refitted_from_source(self):
        for file in ("backtest_predictions.csv", "next_forecasts.csv"):
            for change in (1e-6, 0.5):
                with self.subTest(file=file, change=change):
                    rows = pd.read_csv(self.output / file)
                    index = rows.index[rows.model.eq("Ridge") & rows.status.eq("ok")][-1]
                    rows.loc[index, "predicted"] *= 1 + change
                    rows.to_csv(self.copy / file, index=False)
                    with self.assertRaisesRegex(VerificationError, "Ridge forecast differs from an independent refit"):
                        verify(self.copy)
                    (self.copy / file).write_bytes((self.output / file).read_bytes())

    def test_clipping_flag_must_match_the_refit(self):
        rows = pd.read_csv(self.output / "backtest_predictions.csv")
        rows.loc[rows.index[rows.model.eq("Ridge")][0], "clipped"] = True
        rows.to_csv(self.copy / "backtest_predictions.csv", index=False)
        with self.assertRaisesRegex(VerificationError, "Ridge forecast differs from an independent refit"):
            verify(self.copy)
        rows = pd.read_csv(self.output / "backtest_predictions.csv")
        rows.loc[rows.index[rows.model.eq("Persistence")][0], "clipped"] = True
        rows.to_csv(self.copy / "backtest_predictions.csv", index=False)
        with self.assertRaisesRegex(VerificationError, "only an available Ridge forecast can be clipped"):
            verify(self.copy)

    def test_ridge_failure_requires_a_nonfinite_refit(self):
        # Relabelling a poor forecast as a model error would remove it from every score.
        for file in ("backtest_predictions.csv", "next_forecasts.csv"):
            with self.subTest(file=file):
                rows = pd.read_csv(self.output / file)
                index = rows.index[rows.model.eq("Ridge") & rows.status.eq("ok")][0]
                rows.loc[index, ["status", "predicted"]] = ["model_error", np.nan]
                rows.to_csv(self.copy / file, index=False)
                with self.assertRaisesRegex(VerificationError, "model_error although an independent refit is finite"):
                    verify(self.copy)
                (self.copy / file).write_bytes((self.output / file).read_bytes())

    def test_status_cannot_hide_an_available_forecast(self):
        for status in ("model_error", "missing_target", "missing_features"):
            with self.subTest(status=status):
                rows = pd.read_csv(self.output / "backtest_predictions.csv")
                rows.loc[0, "status"] = status
                rows.to_csv(self.copy / "backtest_predictions.csv", index=False)
                with self.assertRaisesRegex(VerificationError, "unavailable forecast|missing_target status"):
                    verify(self.copy)

    def test_hidden_actual_is_rejected(self):
        rows = pd.read_csv(self.output / "backtest_predictions.csv")
        hidden = rows.origin.eq(rows.origin.iloc[5])
        rows.loc[hidden, ["actual", "status"]] = [np.nan, "missing_target"]
        rows.to_csv(self.copy / "backtest_predictions.csv", index=False)
        self.write_metrics_and_macro(score_predictions(pd.read_csv(self.copy / "backtest_predictions.csv")))
        with self.assertRaisesRegex(VerificationError, "saved actual nan differs from recalculated"):
            verify(self.copy)

    def test_metric_model_valid_count_is_checked(self):
        self.mutate("backtest_metrics.csv", lambda rows: rows.assign(n_model_valid=0))
        with self.assertRaisesRegex(VerificationError, "n_model_valid differs"):
            verify(self.copy)

    def test_per_asset_scores_and_counts_are_recalculated(self):
        for column, change, message in (("mae", 1e-6, "MAE differs"), ("rmse", 1e-6, "RMSE differs"),
                                        ("coverage", 1e-6, "coverage differs"),
                                        ("n_scored", 1, "n_scored differs"), ("n_requested", 1, "n_requested differs")):
            with self.subTest(column=column):
                metrics = pd.read_csv(self.output / "backtest_metrics.csv")
                ridge = metrics.model.eq("Ridge")
                metrics.loc[ridge, column] = metrics.loc[ridge, column] + change
                self.write_metrics_and_macro(metrics)
                with self.assertRaisesRegex(VerificationError, f"Ridge/daily_origins: {message}"):
                    verify(self.copy)

    def test_fabricated_asset_rows_are_rejected(self):
        metrics = pd.read_csv(self.output / "backtest_metrics.csv")
        ghost = metrics.assign(asset="ghost")
        ghost.loc[ghost.model.eq("Ridge"), ["mae", "rmse"]] = 0.0
        self.write_metrics_and_macro(pd.concat([metrics, ghost]))
        with self.assertRaisesRegex(VerificationError, "rows differ from the requested assets"):
            verify(self.copy)

    def test_macro_scores_counts_and_row_set_are_checked(self):
        for column, change, message in (("n_assets_requested", 1, "n_assets_requested differs"),
                                        ("n_requested", 1, "n_requested differs"), ("n_assets", 1, "n_assets differs"),
                                        ("n_scored", 1, "n_scored differs"), ("coverage", 1e-6, "coverage differs"),
                                        ("macro_mae", 1e-6, "macro MAE differs"),
                                        ("macro_rmse", 1e-6, "macro RMSE differs")):
            with self.subTest(column=column):
                rows = pd.read_csv(self.output / "macro_metrics.csv")
                rows.loc[0, column] += change
                rows.to_csv(self.copy / "macro_metrics.csv", index=False)
                with self.assertRaisesRegex(VerificationError, message):
                    verify(self.copy)
        self.mutate("macro_metrics.csv", lambda rows: rows.iloc[1:])
        with self.assertRaisesRegex(VerificationError, "differ from the per-asset table"):
            verify(self.copy)

    def test_extra_metrics_and_duplicate_macro_rows_are_rejected(self):
        for file in ("backtest_metrics.csv", "macro_metrics.csv"):
            with self.subTest(file=file):
                self.mutate(file, lambda rows: pd.concat([rows, rows.iloc[[0]]]))
                with self.assertRaises(VerificationError):
                    verify(self.copy)
                (self.copy / file).write_bytes((self.output / file).read_bytes())

    def test_metadata_row_counts_are_checked(self):
        for field in ("successful_prediction_rows", "scored_prediction_rows"):
            with self.subTest(field=field):
                self.rewrite_metadata(lambda metadata: metadata.update({field: metadata[field] + 7}))
                with self.assertRaisesRegex(VerificationError, f"{field} disagrees"):
                    verify(self.copy)

    def test_incomplete_run_is_rejected(self):
        for status in ("running", "failed", "interrupted"):
            with self.subTest(status=status):
                self.rewrite_metadata(lambda metadata: metadata.update(status=status))
                with self.assertRaisesRegex(VerificationError, "status is not 'complete'"):
                    verify(self.copy)

    def test_changed_source_or_code_bytes_are_rejected(self):
        self.rewrite_metadata(lambda metadata: metadata["code_hashes"].update({"crypto_volatility/model.py": "0" * 64}))
        with self.assertRaisesRegex(VerificationError, "Code changed since the run"):
            verify(self.copy)
        self.rewrite_metadata(lambda metadata: metadata["code_hashes"].pop("crypto_volatility/model.py"))
        with self.assertRaisesRegex(VerificationError, "code hash set"):
            verify(self.copy)
        data = Path(tempfile.mkdtemp(dir=self.root))
        pd.read_csv(self.data / "coin.csv").assign(Volume=1.0).to_csv(data / "coin.csv", index=False)
        with self.assertRaisesRegex(VerificationError, "Source file changed since the run"):
            verify(self.output, data)

    def test_unsupported_configuration_is_rejected(self):
        for field, value in (("window", 20), ("forecast_horizon_days", 10), ("ddof", 1),
                             ("days_per_year", 252), ("ridge_alpha", 10.0)):
            with self.subTest(field=field):
                self.rewrite_metadata(lambda metadata: metadata["configuration"].update({field: value}))
                with self.assertRaisesRegex(VerificationError, f"Unsupported run configuration: {field}"):
                    verify(self.copy)

    def test_assets_must_be_every_accepted_history_or_the_requested_subset(self):
        # A consistent subset of assets would otherwise change every aggregate score unnoticed.
        self.mutate("data_validation.csv", lambda rows: rows.assign(status="rejected"))
        with self.assertRaisesRegex(VerificationError, "differ from the accepted histories"):
            verify(self.copy)
        (self.copy / "data_validation.csv").write_bytes((self.output / "data_validation.csv").read_bytes())
        self.mutate("data_validation.csv", lambda rows: rows.iloc[0:0])
        with self.assertRaisesRegex(VerificationError, "each of the run's input files exactly once"):
            verify(self.copy)
        (self.copy / "data_validation.csv").write_bytes((self.output / "data_validation.csv").read_bytes())
        self.rewrite_metadata(lambda metadata: metadata["configuration"].update(assets_filter=["other"]))
        with self.assertRaisesRegex(VerificationError, "differ from the run's assets_filter"):
            verify(self.copy)
        self.rewrite_metadata(lambda metadata: metadata["configuration"].update(assets_filter=["COIN"]))
        self.assertEqual(verify(self.copy)["status"], "passed")

    def test_validation_table_rules_are_each_enforced(self):
        original = (self.output / "data_validation.csv").read_bytes()
        for change, message in ((lambda rows: pd.concat([rows, rows.iloc[[0]]]), "each of the run's input files exactly once"),
                                (lambda rows: rows.assign(status="skipped"), "skips a price history"),
                                (lambda rows: rows.assign(status="bogus"), "unknown status")):
            with self.subTest(message=message):
                (self.copy / "data_validation.csv").write_bytes(original)
                self.mutate("data_validation.csv", change)
                with self.assertRaisesRegex(VerificationError, message):
                    verify(self.copy)
        # A requested asset must still be an accepted history.
        (self.copy / "data_validation.csv").write_bytes(original)
        self.mutate("data_validation.csv", lambda rows: rows.assign(status="rejected"))
        self.rewrite_metadata(lambda metadata: metadata["configuration"].update(assets_filter=["coin"]))
        with self.assertRaisesRegex(VerificationError, "differ from the run's assets_filter"):
            verify(self.copy)

    def test_recorded_artifacts_cannot_be_missing(self):
        self.rewrite_metadata(lambda metadata: metadata["output_hashes"].update({"figures/extra.png": "0" * 64}))
        with self.assertRaisesRegex(VerificationError, "Recorded artifacts are missing: figures/extra.png"):
            verify(self.copy)
        # Only the regenerated history table may be left out of a published copy.
        self.rewrite_metadata(lambda metadata: metadata["output_hashes"].update(
            {"validated_daily_history.csv": "0" * 64}))
        self.assertEqual(verify(self.copy)["status"], "passed")

    def test_every_artifact_must_match_the_recorded_run(self):
        # Reordered rows pass every recalculation but are not the bytes the run wrote.
        self.mutate("backtest_predictions.csv", lambda rows: rows.iloc[::-1])
        with self.assertRaisesRegex(VerificationError, "Artifact changed since the run: backtest_predictions.csv"):
            verify(self.copy)
        (self.copy / "backtest_predictions.csv").write_bytes((self.output / "backtest_predictions.csv").read_bytes())
        (self.copy / "notes.txt").write_text("added later")
        with self.assertRaisesRegex(VerificationError, "not recorded in run_metadata.json: notes.txt"):
            verify(self.copy)
        (self.copy / "notes.txt").unlink()
        # Only the top-level metadata is exempt, not a nested file with the same name.
        (self.copy / "nested").mkdir()
        (self.copy / "nested" / "run_metadata.json").write_text("{}")
        with self.assertRaisesRegex(VerificationError, "not recorded in run_metadata.json: nested/run_metadata.json"):
            verify(self.copy)


if __name__ == "__main__":
    unittest.main()
