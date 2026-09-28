"""Run the descriptive analysis and forecast evaluation: python -m crypto_volatility --help."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DAYS_PER_YEAR, WINDOW, load_dataset
from .model import HORIZON, RIDGE_ALPHA, backtest_asset, score_predictions


ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://www.kaggle.com/datasets/kaushiksuresh147/top-10-cryptocurrencies-historical-dataset"
RUNTIME_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib")


def json_value(value):
    """Convert pandas/numpy values into JSON-compatible Python objects."""
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset, np.ndarray)):
        return [json_value(v) for v in value]
    if value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    # ``default=str`` keeps the failure path from raising on an unexpected type.
    path.write_text(json.dumps(json_value(value), indent=2, allow_nan=False, default=str) + "\n")


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _display_path(path: Path) -> str:
    """Record repository-relative paths where possible; absolute paths otherwise."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m crypto_volatility",
                                     description="Summary statistics, charts and forward 30-day volatility forecasts.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "raw", help="Directory of source CSV files")
    parser.add_argument("--output", type=Path,
                        help="New result directory (default results/run-<UTC timestamp>); existing directories are refused")
    parser.add_argument("--assets", nargs="+", help="Optional exact file stems, case insensitive; quote multiword names")
    parser.add_argument("--test-days", type=int, default=180, help="Calendar forecast origins per asset (default 180)")
    parser.add_argument("--min-train", type=int, default=250, help="Minimum matured training examples (default 250)")
    parser.add_argument("--refit-every", type=int, default=30, help="Refit interval in calendar days (default 30)")
    args = parser.parse_args(argv)
    for key in ("test_days", "min_train", "refit_every"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if not args.data.is_dir():
        parser.error("Data directory missing. Run python scripts/download_data.py first.")
    started = datetime.now(timezone.utc)
    if args.output is None:
        args.output = ROOT / "results" / started.strftime("run-%Y%m%dT%H%M%SZ")
    if args.output.exists():
        parser.error("Output directory exists; choose a new --output to preserve previous results")
    try:
        args.output.mkdir(parents=True)
    except OSError as exc:
        parser.error(f"Cannot create output directory {args.output}: {exc}")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    metadata = {"status": "running", "created_at": started}
    try:
        metadata.update({
            "source_url": SOURCE_URL,
            "data_directory": _display_path(args.data),
            "python": platform.python_version(),
            "packages": {name: _package_version(name) for name in RUNTIME_PACKAGES},
            "configuration": {"test_days": args.test_days, "min_train": args.min_train,
                              "refit_every": args.refit_every, "ridge_alpha": RIDGE_ALPHA,
                              "window": WINDOW, "forecast_horizon_days": HORIZON,
                              "ddof": 0, "days_per_year": DAYS_PER_YEAR,
                              "assets_filter": args.assets},
            "code_hashes": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted((ROOT / "crypto_volatility").glob("*.py"))},
            "interpretation": "Retrospective historical backtest; not a live market forecast",
        })
        write_json(args.output / "run_metadata.json", metadata)
        dataset = load_dataset(args.data)
        dataset.validation.to_csv(args.output / "data_validation.csv", index=False)
        dataset.events.to_csv(args.output / "data_quality_events.csv", index=False)
        metadata["input_hashes"] = dataset.hashes
        manifest_path = ROOT / "data" / "source_manifest.json"
        metadata["matches_reference_source"] = (
            dataset.hashes == json.loads(manifest_path.read_text())["files"] if manifest_path.exists() else False)
        frames = dataset.frames
        if not frames:
            raise ValueError("No usable asset histories; inspect data_validation.csv")
        if args.assets:
            lookup = {name.casefold(): name for name in frames}
            missing = set(name.casefold() for name in args.assets) - set(lookup)
            if missing:
                raise ValueError(f"Unknown or rejected assets: {', '.join(sorted(missing))}")
            frames = {lookup[name.casefold()]: frames[lookup[name.casefold()]] for name in args.assets}
        metadata["assets"] = list(frames)
        metadata["source_date_range"] = [min(frame.index.min() for frame in frames.values()),
                                         max(frame.index.max() for frame in frames.values())]
        write_json(args.output / "run_metadata.json", metadata)
        histories = pd.concat(frames, names=["asset", "Date"]).reset_index()
        histories.to_csv(args.output / "validated_daily_history.csv", index=False)
        predictions, next_rows, statuses = [], [], []
        for i, (asset, frame) in enumerate(frames.items(), 1):
            result, latest, status = backtest_asset(frame, asset, test_days=args.test_days,
                                                  min_train=args.min_train, refit_every=args.refit_every)
            predictions.append(result)
            next_rows.append(latest)
            statuses.append(status)
            print(f"[{i}/{len(frames)}] {asset}: {status.get('status', status.get('backtest_status', 'processed'))}", flush=True)
        predictions = pd.concat(predictions, ignore_index=True)
        next_rows = pd.concat(next_rows, ignore_index=True)
        metrics = score_predictions(predictions)
        predictions.to_csv(args.output / "backtest_predictions.csv", index=False)
        next_rows.to_csv(args.output / "next_forecasts.csv", index=False)
        metrics.to_csv(args.output / "backtest_metrics.csv", index=False)
        write_json(args.output / "model_status.json", statuses)
        from .report import write_report
        write_report(frames, dataset.validation, predictions, metrics, next_rows, args.output)
        metadata["status"] = "complete"
        metadata["completed_at"] = datetime.now(timezone.utc)
        metadata["successful_prediction_rows"] = int(predictions.status.eq("ok").sum())
        metadata["scored_prediction_rows"] = int(
            metrics.loc[metrics["sample"].eq("daily_origins"), "n_scored"].sum())
        # Bind every artifact's bytes to this run, so a stale or edited copy of
        # a table, figure or report cannot sit beside this metadata unnoticed.
        metadata["output_hashes"] = {
            p.relative_to(args.output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.output.rglob("*")) if p.is_file() and p.name != "run_metadata.json"}
        write_json(args.output / "run_metadata.json", metadata)
        print(f"Report written to {args.output / 'report.md'}")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        # An interrupted run must not keep claiming status "running".
        interrupted = isinstance(exc, KeyboardInterrupt)
        metadata["status"] = "interrupted" if interrupted else "failed"
        metadata["error"] = "KeyboardInterrupt" if interrupted else f"{type(exc).__name__}: {exc}"
        write_json(args.output / "run_metadata.json", metadata)
        (args.output / "error.txt").write_text(traceback.format_exc())
        print(f"Analysis {metadata['status']}: {metadata['error']}; details in {args.output / 'error.txt'}",
              file=sys.stderr)
        return 130 if interrupted else 1


if __name__ == "__main__":
    raise SystemExit(main())
