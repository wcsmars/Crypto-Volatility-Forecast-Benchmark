"""End-to-end runs of the command line entry point without a download.

The reference run needs the downloaded dataset; these tests build a small
self-contained synthetic data directory instead, run ``python -m crypto_volatility``
through ``main()``, and check the outputs with the independent verifier. A
second run on the committed ``data/sample`` must reproduce its reference rows.
"""

import contextlib
import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

from crypto_volatility.__main__ import main
from scripts import verify_results
from scripts.verify_results import VerificationError, verify


def write_asset(path: Path, seed: int, days: int = 420, drop_day: int = None, zero_close_day: int = None) -> None:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=days, freq="D")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.03, days)))
    frame = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Open": close * (1 + rng.normal(0, 0.005, days)),
        "High": close * (1 + np.abs(rng.normal(0, 0.01, days))),
        "Low": close * (1 - np.abs(rng.normal(0, 0.01, days))),
        "Close": close,
        "Volume": rng.integers(1_000, 100_000, days),
        "Currency": "USD",
    })
    if zero_close_day is not None:
        frame.loc[zero_close_day, "Close"] = 0
    if drop_day is not None:
        frame = frame.drop(index=drop_day)
    frame.to_csv(path, index=False)


def run_main(argv) -> int:
    """Run the CLI without its progress output cluttering the test log."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return main(argv)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.data = root / "raw"
        cls.data.mkdir()
        write_asset(cls.data / "alpha.csv", seed=1)
        write_asset(cls.data / "beta.csv", seed=2, drop_day=200)
        write_asset(cls.data / "Gamma Coin.csv", seed=3, zero_close_day=150)
        pd.DataFrame({"Rank": [1], "Name": ["alpha"]}).to_csv(cls.data / "Current Crypto leaderboard.csv", index=False)
        pd.DataFrame({"unrelated": [1, 2]}).to_csv(cls.data / "broken.csv", index=False)
        cls.output = root / "run"
        cls.exit_code = run_main([
            "--data", str(cls.data), "--output", str(cls.output),
            "--test-days", "20", "--min-train", "100", "--refit-every", "10",
        ])

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_run_completes_and_writes_every_documented_output(self):
        self.assertEqual(self.exit_code, 0)
        expected = [
            "report.md", "run_metadata.json", "data_validation.csv", "data_quality_events.csv",
            "validated_daily_history.csv", "summary_statistics.csv", "backtest_predictions.csv",
            "backtest_metrics.csv", "macro_metrics.csv", "next_forecasts.csv", "model_status.json",
            "common_window_statistics.csv", "common_window_indexed_close.csv", "common_window_log_returns.csv",
            "common_window_volatility_30.csv", "common_window_return_correlations.csv",
            "common_window_correlation_pair_counts.csv", "common_window_normalized_reported_volume.csv",
        ]
        for name in expected:
            self.assertTrue((self.output / name).is_file(), name)
        for name in ["indexed_prices", "rolling_volatility", "return_distributions",
                     "return_correlations", "normalized_volume", "forecast_comparison"]:
            figure = self.output / "figures" / f"{name}.png"
            self.assertTrue(figure.is_file(), name)
            self.assertEqual(figure.read_bytes()[:8], b"\x89PNG\r\n\x1a\n", name)
        self.assertFalse((self.output / "error.txt").exists())

    def test_metadata_records_configuration_and_provenance(self):
        metadata = json.loads((self.output / "run_metadata.json").read_text())
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(metadata["configuration"]["test_days"], 20)
        self.assertEqual(metadata["configuration"]["min_train"], 100)
        self.assertEqual(metadata["configuration"]["refit_every"], 10)
        self.assertFalse(metadata["matches_reference_source"])
        self.assertEqual(sorted(metadata["assets"]), ["Gamma Coin", "alpha", "beta"])
        self.assertEqual(len(metadata["input_hashes"]), 5)
        self.assertEqual(set(metadata["code_hashes"]), {
            "crypto_volatility/__init__.py", "crypto_volatility/__main__.py", "crypto_volatility/data.py",
            "crypto_volatility/model.py", "crypto_volatility/report.py",
        })
        self.assertEqual(metadata["configuration"]["ridge_alpha"], 1.0)
        artifacts = {p.relative_to(self.output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in self.output.rglob("*") if p.is_file() and p.name != "run_metadata.json"}
        self.assertEqual(metadata["output_hashes"], artifacts)

    def test_validation_reports_each_input_disposition(self):
        validation = pd.read_csv(self.output / "data_validation.csv").set_index("file")
        self.assertEqual(validation.loc["alpha.csv", "status"], "accepted")
        self.assertEqual(validation.loc["beta.csv", "missing_days"], 1)
        self.assertEqual(validation.loc["Gamma Coin.csv", "invalid_close_rows"], 1)
        self.assertEqual(validation.loc["Current Crypto leaderboard.csv", "status"], "skipped")
        self.assertEqual(validation.loc["broken.csv", "status"], "rejected")
        events = pd.read_csv(self.output / "data_quality_events.csv")
        self.assertIn("missing_calendar_day", set(events["issue"]))
        self.assertIn("invalid_close", set(events["issue"]))

    def test_forecasts_are_scored_for_every_asset(self):
        metrics = pd.read_csv(self.output / "backtest_metrics.csv")
        daily = metrics[metrics["sample"].eq("daily_origins")]
        self.assertEqual(sorted(daily["asset"].unique()), ["Gamma Coin", "alpha", "beta"])
        self.assertTrue(daily["n_requested"].eq(20).all())
        self.assertTrue(daily["n_scored"].gt(0).all())
        self.assertTrue(np.isfinite(daily["mae"]).all())
        macro = pd.read_csv(self.output / "macro_metrics.csv")
        self.assertEqual(sorted(macro["model"].unique()), ["Historical90", "Persistence", "Ridge"])
        self.assertTrue(macro["n_assets"].eq(3).all())
        latest = pd.read_csv(self.output / "next_forecasts.csv")
        self.assertTrue(latest["status"].eq("ok").all())
        self.assertTrue(latest["actual"].isna().all())

    def test_report_describes_the_synthetic_inputs(self):
        report = (self.output / "report.md").read_text(encoding="utf-8")
        self.assertIn("## Descriptive analysis", report)
        self.assertIn("## Forecast", report)
        self.assertIn("**3 asset histories**", report)
        self.assertIn("**1 missing calendar days**", report)
        self.assertIn("**1 nonpositive or invalid close rows**", report)
        self.assertIn("### Comparable recent window", report)
        self.assertIn("| daily_origins | Ridge | 3 / 3 |", report)
        self.assertIn("![", report)

    def test_independent_verifier_accepts_the_run(self):
        summary = verify(self.output, self.data)
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["source_files_verified"], 5)
        self.assertGreater(summary["source_windows_independently_recalculated"], 0)
        self.assertLess(summary["max_absolute_target_difference"], 1e-9)
        self.assertEqual(summary["metric_rows_independently_recalculated"], 3 * 2 * 3)
        self.assertEqual(summary["macro_rows_independently_recalculated"], 2 * 3)
        self.assertEqual(summary["latest_persistence_forecasts_recalculated"], 3)
        self.assertEqual(summary["assets_verified"], 3)
        self.assertEqual(summary["source_files_rejected"], 1)
        self.assertEqual(summary["latest_ridge_forecasts_refitted"], 3)
        self.assertGreater(summary["backtest_ridge_forecasts_refitted"], 0)
        self.assertLess(summary["max_absolute_ridge_difference"], 1e-9)
        # Every artifact of the run, including the report and figures, is hash-checked.
        self.assertEqual(summary["output_files_recorded"], 23)
        self.assertEqual(summary["ridge_forecasts_rounding_sensitive"], 0)
        # The recorded data directory lets the verifier run without --data.
        metadata = json.loads((self.output / "run_metadata.json").read_text())
        self.assertEqual(Path(metadata["data_directory"]), self.data.resolve())
        self.assertEqual(verify(self.output)["status"], "passed")

    def copy_run(self, directory):
        copy = Path(directory) / "run"
        shutil.copytree(self.output, copy)
        return copy

    def test_verifier_rejects_a_tampered_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            copy = self.copy_run(directory)
            predictions = pd.read_csv(copy / "backtest_predictions.csv")
            first_ok = predictions.index[predictions["status"].eq("ok")][0]
            predictions.loc[first_ok, "actual"] += 0.05
            predictions.to_csv(copy / "backtest_predictions.csv", index=False)
            with self.assertRaisesRegex(VerificationError, "differs from recalculated"):
                verify(copy, self.data)

    def test_verifier_rejects_tampered_aggregates_and_latest_forecasts(self):
        with tempfile.TemporaryDirectory() as directory:
            copy = self.copy_run(directory)
            macro = pd.read_csv(copy / "macro_metrics.csv")
            macro.loc[0, "macro_mae"] *= 2
            macro.to_csv(copy / "macro_metrics.csv", index=False)
            with self.assertRaisesRegex(VerificationError, "macro MAE differs"):
                verify(copy, self.data)
        with tempfile.TemporaryDirectory() as directory:
            copy = self.copy_run(directory)
            latest = pd.read_csv(copy / "next_forecasts.csv")
            row = latest.index[latest["model"].eq("Persistence")][0]
            latest.loc[row, "predicted"] += 0.1
            latest.to_csv(copy / "next_forecasts.csv", index=False)
            with self.assertRaisesRegex(VerificationError, "latest persistence forecast differs"):
                verify(copy, self.data)

    def test_verifier_command_line_reports_outcome_and_exit_code(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = verify_results.main([str(self.output), "--data", str(self.data)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "passed")
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = verify_results.main([str(self.output / "does-not-exist")])
        self.assertEqual(code, 1)
        self.assertIn("Verification failed", stderr.getvalue())

    def test_run_without_any_scorable_asset_still_completes_and_verifies(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unscorable"
            code = run_main(["--data", str(self.data), "--output", str(output),
                             "--test-days", "5", "--min-train", "100000"])
            self.assertEqual(code, 0)
            metrics = pd.read_csv(output / "backtest_metrics.csv")
            self.assertTrue(metrics["n_scored"].eq(0).all())
            self.assertEqual((output / "macro_metrics.csv").read_text().strip(), "")
            report = (output / "report.md").read_text(encoding="utf-8")
            self.assertIn("No assets have scored common forecasts", report)
            summary = verify(output, self.data)
            self.assertEqual(summary["status"], "passed")
            self.assertEqual(summary["macro_rows_independently_recalculated"], 0)

    def test_verifier_rejects_a_changed_source_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "raw"
            data.mkdir()
            for file in self.data.iterdir():
                (data / file.name).write_bytes(file.read_bytes())
            (data / "extra.csv").write_text("Date,Close\n")
            with self.assertRaisesRegex(VerificationError, "unexpected \\['extra.csv'\\]"):
                verify(self.output, data)

    def test_verifier_rejects_a_changed_report_or_figure(self):
        for name in ("report.md", "figures/indexed_prices.png"):
            with self.subTest(artifact=name), tempfile.TemporaryDirectory() as directory:
                copy = self.copy_run(directory)
                (copy / name).write_bytes((self.output / name).read_bytes() + b"edited")
                with self.assertRaisesRegex(VerificationError, "Artifact changed since the run"):
                    verify(copy, self.data)

    def test_interrupted_run_is_recorded_as_interrupted(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("crypto_volatility.__main__.backtest_asset", side_effect=KeyboardInterrupt):
            output = Path(directory) / "interrupted"
            code = run_main(["--data", str(self.data), "--output", str(output)])
            self.assertEqual(code, 130)
            metadata = json.loads((output / "run_metadata.json").read_text())
            self.assertEqual(metadata["status"], "interrupted")
            self.assertTrue((output / "error.txt").is_file())

    def test_edge_case_histories_verify_independently(self):
        # A text asset name that pandas would read as missing, a constant price
        # (every Ridge input constant), and a gap inside the test window, which
        # leaves Ridge rows without features. An oversized test window means
        # every available origin and must verify too.
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "raw"
            data.mkdir()
            write_asset(data / "NA.csv", seed=4)
            write_asset(data / "gappy.csv", seed=5, drop_day=380)
            flat = pd.read_csv(data / "NA.csv").assign(Open=1.0, High=1.0, Low=1.0, Close=1.0)
            flat.to_csv(data / "flat.csv", index=False)
            for test_days in ("20", "1000000000"):
                with self.subTest(test_days=test_days):
                    output = Path(directory) / f"run-{test_days}"
                    code = run_main(["--data", str(data), "--output", str(output), "--test-days", test_days,
                                     "--min-train", "50", "--refit-every", "10"])
                    self.assertEqual(code, 0)
                    predictions = pd.read_csv(output / "backtest_predictions.csv", keep_default_na=False,
                                              na_values=[""])
                    self.assertEqual(sorted(predictions["asset"].unique()), ["NA", "flat", "gappy"])
                    summary = verify(output, data)
                    self.assertEqual(summary["status"], "passed")
                    self.assertEqual(summary["assets_verified"], 3)
                    if test_days == "20":
                        ridge = predictions[predictions["model"].eq("Ridge")].groupby("asset")["status"]
                        self.assertIn("missing_features", set(ridge.get_group("gappy")))
                        self.assertTrue(ridge.get_group("flat").eq("ok").all())
                        self.assertEqual(summary["backtest_ridge_forecasts_refitted"], 20 + 20 + 10)
                    else:
                        # Every available origin starts at the first date, before any label matures.
                        self.assertTrue(predictions["status"].eq("insufficient_training").all())

    def test_existing_output_directory_is_refused(self):
        with self.assertRaises(SystemExit):
            run_main(["--data", str(self.data), "--output", str(self.output)])
        self.assertFalse((self.output / "error.txt").exists())

    def test_unknown_asset_fails_visibly(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "subset"
            code = run_main(["--data", str(self.data), "--output", str(output), "--assets", "alpha", "missing-coin"])
            self.assertEqual(code, 1)
            metadata = json.loads((output / "run_metadata.json").read_text())
            self.assertEqual(metadata["status"], "failed")
            self.assertIn("missing-coin", metadata["error"])
            self.assertTrue((output / "error.txt").is_file())

    def test_asset_subset_is_case_insensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "subset"
            code = run_main(["--data", str(self.data), "--output", str(output), "--assets", "ALPHA", "gamma coin",
                         "--test-days", "5", "--min-train", "50"])
            self.assertEqual(code, 0)
            metadata = json.loads((output / "run_metadata.json").read_text())
            self.assertEqual(metadata["assets"], ["alpha", "Gamma Coin"])
            metrics = pd.read_csv(output / "backtest_metrics.csv")
            self.assertEqual(sorted(metrics["asset"].unique()), ["Gamma Coin", "alpha"])


class SampleDataTests(unittest.TestCase):
    """The committed sample is unmodified source data that reproduces its reference rows."""

    ROOT = Path(__file__).resolve().parents[1]
    SAMPLE = ROOT / "data" / "sample"
    ASSETS = ["BNB", "bitcoin", "cardano", "dogecoin", "ethereum", "litecoin", "solana", "xrp"]

    def test_sample_files_match_the_source_manifest(self):
        manifest = json.loads((self.ROOT / "data" / "source_manifest.json").read_text())
        sample = sorted(self.SAMPLE.glob("*.csv"))
        self.assertEqual(sorted(path.stem for path in sample), self.ASSETS)
        for path in sample:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, manifest["files"][path.name], path.name)

    def test_sample_run_reproduces_the_reference_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sample"
            self.assertEqual(run_main(["--data", str(self.SAMPLE), "--output", str(output)]), 0)
            for name in ("backtest_predictions.csv", "backtest_metrics.csv", "next_forecasts.csv"):
                sample = pd.read_csv(output / name)
                reference = pd.read_csv(self.ROOT / "results" / "reference" / name)
                reference = reference[reference["asset"].isin(self.ASSETS)].reset_index(drop=True)
                # Per-asset models are independent, so a subset run must match the full run.
                pd.testing.assert_frame_equal(sample, reference, rtol=1e-9, atol=1e-12, obj=name)


if __name__ == "__main__":
    unittest.main()
