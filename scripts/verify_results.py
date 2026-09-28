#!/usr/bin/env python3
"""Independently verify saved real-data labels, forecasts, losses, provenance and dates.

Every check raises :class:`VerificationError` with a message, so the script
reports failures even when Python is run with ``-O`` (which strips ``assert``).
The recalculations deliberately avoid the package's own feature and model
code: labels, trailing volatility and the nine Ridge inputs are rebuilt from
the raw Close columns with NumPy, every available Ridge forecast is refitted
from the ridge normal equations, and aggregate scores are rebuilt from the
per-asset tables. The recorded assets must be exactly the accepted (or
requested) source histories, and every saved artifact must still match the
hash recorded when the run finished.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTION = "Independently verify saved real-data labels, forecasts, losses, provenance and dates."
MODELS = {"Persistence", "Historical90", "Ridge"}
STATUSES = {"ok", "insufficient_training", "missing_features", "missing_target", "model_error"}
HORIZON = 30
RIDGE_ALPHA = 1.0
FEATURE_WINDOWS = (7, 30, 90)
LEADERBOARD = "current crypto leaderboard.csv"
DATE_COLUMNS = ["origin", "target_start", "target_end", "train_target_end"]
# Independent recalculations agree to about 1e-13; this still rejects any material edit.
TOLERANCE = {"rtol": 1e-8, "atol": 1e-10}
# Written after the run's own hashes, so never recorded; Finder metadata is ignored anywhere.
UNHASHED = {"run_metadata.json", "validation_summary.json"}
# The only recorded artifact a published copy leaves out (a 17 MB regenerated table).
OPTIONAL_ARTIFACTS = {"validated_daily_history.csv"}


class VerificationError(Exception):
    """A saved result disagrees with an independent recalculation or its provenance."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_directory(metadata: dict, data: Path | None) -> Path:
    if data is not None:
        return Path(data)
    recorded = metadata.get("data_directory")
    if recorded:
        path = Path(recorded)
        return path if path.is_absolute() else ROOT / path
    return ROOT / "data" / "raw"


def _display_path(path: Path) -> str:
    """Repository-relative when possible; otherwise the resolved absolute path."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _source_close(data: Path, asset: str, input_hashes: dict) -> pd.Series:
    """Load one asset's Close series by the file name recorded for the run.

    Dates are parsed exactly as the loader parses them (text, ISO 8601, UTC,
    then made naive), so a vendor file with ``T00:00:00Z`` labels verifies too.
    """
    names = [name for name in input_hashes if Path(name).stem == asset]
    _check(len(names) == 1, f"{asset}: expected exactly one recorded source file, found {len(names)}")
    raw = pd.read_csv(data / names[0], dtype={"Date": "string"})
    close = pd.to_numeric(raw["Close"], errors="coerce")
    close.index = pd.DatetimeIndex(pd.to_datetime(raw["Date"], format="ISO8601", utc=True).dt.tz_localize(None))
    return close.sort_index()


def _read_output(path: Path) -> pd.DataFrame:
    """Read a pipeline table. It writes missing values as empty fields, so text such
    as an asset named "NA" or "None" must stay text rather than become missing."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def _read_table(path: Path) -> pd.DataFrame:
    """Read a CSV that the pipeline may legitimately have written empty."""
    try:
        return _read_output(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _raw_volatility(close: pd.Series, window: int) -> pd.Series:
    """Recalculate each complete return window with NumPy, without pandas rolling."""
    calendar = pd.date_range(close.index.min(), close.index.max(), freq="D")
    values = close.reindex(calendar).to_numpy(dtype=float)
    result = np.full(len(calendar), np.nan)
    if len(values) > window:
        windows = np.lib.stride_tricks.sliding_window_view(values, window + 1)
        valid = (np.isfinite(windows) & (windows > 0)).all(axis=1)
        returns = np.diff(np.log(windows[valid]), axis=1)
        result[window + np.flatnonzero(valid)] = np.sqrt(365 * np.mean(
            (returns - returns.mean(axis=1, keepdims=True)) ** 2, axis=1))
    return pd.Series(result, index=calendar)


def _raw_ridge_inputs(close: pd.Series) -> np.ndarray:
    """Rebuild the nine past-only Ridge inputs from complete daily return windows.

    Columns are annualized volatility, mean absolute return and mean return over
    7, 30 and 90 days, one row per calendar day; a window that needs any missing
    or invalid price stays missing.
    """
    calendar = pd.date_range(close.index.min(), close.index.max(), freq="D")
    prices = close.reindex(calendar).to_numpy(dtype=float)
    prices = np.where(np.isfinite(prices) & (prices > 0), prices, np.nan)
    returns = np.full(len(prices), np.nan)
    returns[1:] = np.diff(np.log(prices))
    columns = {}
    for window in FEATURE_WINDOWS:
        values = np.full((3, len(returns)), np.nan)
        if len(returns) >= window:
            windows = np.lib.stride_tricks.sliding_window_view(returns, window)
            valid = np.isfinite(windows).all(axis=1)
            complete, rows = windows[valid], window - 1 + np.flatnonzero(valid)
            values[0, rows] = np.sqrt(365 * np.mean((complete - complete.mean(axis=1, keepdims=True)) ** 2, axis=1))
            values[1, rows] = np.abs(complete).mean(axis=1)
            values[2, rows] = complete.mean(axis=1)
        columns[window] = values
    return np.column_stack([columns[window][stat] for stat in range(3) for window in FEATURE_WINDOWS])


def _ridge_fit(inputs: np.ndarray, target: np.ndarray):
    """Standardize the inputs and solve the ridge normal equations for log1p(target).

    Inputs are scaled by their population standard deviation, except that a
    numerically constant input keeps unit scale (the standard-scaler convention).
    The last element flags a fit that depends on rounding noise: an input that is
    constant in exact arithmetic (e.g. a mean return of a price series that
    repeats exactly) can carry different ~1e-19 noise under any other summation
    order, and standardizing that noise makes the forecast irreproducible.
    """
    count = len(inputs)
    mean, variance = inputs.mean(axis=0), inputs.var(axis=0)
    eps = np.finfo(float).eps
    constant = variance <= count * eps * variance + (count * mean * eps) ** 2
    # Each column's natural size: its window's largest mean absolute return,
    # annualized for the volatility columns (column order as in _raw_ridge_inputs).
    size = np.tile(np.abs(inputs[:, 3:6]).max(axis=0), 3) * np.repeat([np.sqrt(365), 1.0, 1.0], 3)
    rounding_sensitive = bool((~constant & (np.sqrt(variance) <= 1e-9 * size)).any())
    scale = np.where(constant, 1.0, np.sqrt(variance))
    standardized = (inputs - mean) / scale
    offset = standardized.mean(axis=0)
    centered = standardized - offset
    response = np.log1p(target)
    coefficients = np.linalg.solve(centered.T @ centered + RIDGE_ALPHA * np.eye(inputs.shape[1]),
                                   centered.T @ (response - response.mean()))
    return mean, scale, coefficients, response.mean() - offset @ coefficients, rounding_sensitive


def _ridge_forecast(fit, inputs: np.ndarray) -> float:
    """Unclipped volatility forecast; non-finite when the fitted model extrapolates too far."""
    mean, scale, coefficients, intercept, _ = fit
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.expm1(((inputs - mean) / scale) @ coefficients + intercept))


def _forecast_table(path: Path, assets: set, min_train: int, latest: bool = False) -> pd.DataFrame:
    rows = _read_output(path)
    required = set(DATE_COLUMNS) | {"asset", "model", "status", "actual", "predicted", "n_train", "clipped"}
    _check(required <= set(rows), f"{path.name}: missing forecast columns")
    for column in DATE_COLUMNS:
        rows[column] = pd.to_datetime(rows[column], errors="raise")
    for column in ("actual", "predicted", "n_train"):
        rows[column] = pd.to_numeric(rows[column], errors="raise")
    clipped = rows["clipped"].astype(str).str.lower()
    _check(clipped.isin({"true", "false"}).all(), f"{path.name}: clipped must be true or false")
    rows["clipped"] = clipped.eq("true")
    _check(rows.asset.isin(assets).all(), f"{path.name}: unknown or missing asset")
    _check(rows.model.isin(MODELS).all(), f"{path.name}: unknown or missing model")
    _check(rows.status.isin(STATUSES).all(), f"{path.name}: unknown or missing status")
    _check(not rows.duplicated(["asset", "model", "origin"]).any(),
           f"{path.name}: duplicate asset/model/origin rows")
    _check(rows.origin.notna().all() and rows.origin.eq(rows.origin.dt.normalize()).all(),
           f"{path.name}: origins must be nonmissing daily dates")
    _check(rows.target_start.eq(rows.origin + pd.Timedelta(days=1)).all(),
           f"{path.name}: target_start is not origin + 1 day for every row")
    _check(rows.target_end.eq(rows.origin + pd.Timedelta(days=HORIZON)).all(),
           f"{path.name}: target_end is not origin + 30 days for every row")
    fitted = rows[rows.train_target_end.notna()]
    _check(fitted.train_target_end.le(fitted.origin).all(),
           f"{path.name}: a fit used a label ending after its forecast origin")
    _check(np.isfinite(rows.n_train).all() and rows.n_train.ge(0).all()
           and rows.n_train.mod(1).eq(0).all(), f"{path.name}: invalid training counts")
    available = rows.status.isin({"ok", "missing_target"})
    _check(rows.loc[~available, "predicted"].isna().all(),
           f"{path.name}: an unavailable forecast has a prediction")
    _check(rows.loc[rows.status.eq("missing_target"), "actual"].isna().all(),
           f"{path.name}: missing_target status has an observed actual")
    predicted = rows.loc[available, "predicted"]
    _check((np.isfinite(predicted) & predicted.ge(0)).all(),
           f"{path.name}: a successful forecast has an invalid prediction")
    _check(not rows.loc[~(available & rows.model.eq("Ridge")), "clipped"].any(),
           f"{path.name}: only an available Ridge forecast can be clipped")
    ridge = rows[available & rows.model.eq("Ridge")]
    _check(ridge.train_target_end.notna().all() and ridge.n_train.ge(min_train).all(),
           f"{path.name}: an available Ridge forecast lacks sufficient dated training")
    baselines = rows[~rows.model.eq("Ridge")]
    _check(baselines.train_target_end.isna().all() and baselines.n_train.eq(0).all(),
           f"{path.name}: a baseline unexpectedly records fitted training")
    if latest:
        _check(rows.actual.isna().all(), f"{path.name}: latest forecasts cannot have observed targets")
        _check(not rows.status.eq("missing_target").any(), f"{path.name}: invalid latest forecast status")
    else:
        actual = rows.loc[rows.status.eq("ok"), "actual"]
        _check((np.isfinite(actual) & actual.ge(0)).all(),
               f"{path.name}: a successful row has an invalid actual")
    return rows


def verify(output: Path, data: Path | None = None) -> dict:
    output = Path(output)
    metadata = json.loads((output / "run_metadata.json").read_text())
    _check(isinstance(metadata, dict), "run_metadata.json must be a JSON object")
    _check(metadata.get("status") == "complete", "Run did not finish: run_metadata.json status is not 'complete'")
    data = _source_directory(metadata, data)
    _check(data.is_dir(), f"Source directory not found: {data}; run python scripts/download_data.py or pass --data")
    input_hashes = metadata["input_hashes"]
    _check(isinstance(input_hashes, dict) and bool(input_hashes), "Recorded input hashes must be a nonempty object")
    present = {p.name for p in data.iterdir() if p.is_file() and p.suffix.casefold() == ".csv"}
    _check(present == set(input_hashes),
           "Source directory CSV set differs from the run's inputs: "
           f"unexpected {sorted(present - set(input_hashes))}, missing {sorted(set(input_hashes) - present)}")
    for name, digest in input_hashes.items():
        _check(_sha256(data / name) == digest, f"Source file changed since the run: {name}")
    current_code = {p.relative_to(ROOT).as_posix() for p in (ROOT / "crypto_volatility").glob("*.py")}
    _check(isinstance(metadata["code_hashes"], dict), "Recorded code hashes must be an object")
    _check(set(metadata["code_hashes"]) == current_code, "Recorded code hash set differs from the current package")
    for name, digest in metadata["code_hashes"].items():
        _check(_sha256(ROOT / name) == digest, f"Code changed since the run: {name}")
    config = metadata["configuration"]
    _check(isinstance(config, dict), "Recorded configuration must be an object")
    for field in ("test_days", "min_train", "refit_every"):
        _check(type(config[field]) is int and config[field] > 0, f"Invalid run configuration: {field}")
    for field, expected in {"window": 30, "forecast_horizon_days": 30, "ddof": 0, "days_per_year": 365,
                            "ridge_alpha": RIDGE_ALPHA}.items():
        _check(config[field] == expected, f"Unsupported run configuration: {field}")
    assets = metadata["assets"]
    _check(isinstance(assets, list) and len(assets) > 0 and all(isinstance(asset, str) for asset in assets)
           and len(set(assets)) == len(assets),
           "Recorded assets must be nonempty and unique")

    # The analyzed assets must be every accepted history, or exactly the requested
    # subset, so a run cannot silently drop histories that would change the scores.
    validation = pd.read_csv(output / "data_validation.csv", dtype=str, keep_default_na=False)
    _check({"file", "asset", "status"} <= set(validation) and not validation.file.duplicated().any()
           and set(validation.file) == set(input_hashes),
           "data_validation.csv must record each of the run's input files exactly once")
    _check(validation.status.isin({"accepted", "rejected", "skipped"}).all()
           and validation.loc[validation.status.eq("skipped"), "file"].str.casefold().eq(LEADERBOARD).all(),
           "data_validation.csv has an unknown status or skips a price history")
    accepted = set(validation.loc[validation.status.eq("accepted"), "asset"])
    requested_assets = config.get("assets_filter")
    if requested_assets is None:
        _check(set(assets) == accepted, "Recorded assets differ from the accepted histories in data_validation.csv")
    else:
        _check(isinstance(requested_assets, list) and set(assets) <= accepted
               and {asset.casefold() for asset in assets} == {str(name).casefold() for name in requested_assets},
               "Recorded assets differ from the run's assets_filter")

    predictions = _forecast_table(output / "backtest_predictions.csv", set(assets), config["min_train"])
    latest = _forecast_table(output / "next_forecasts.csv", set(assets), config["min_train"], latest=True)

    # Check every saved target, including unavailable forecasts, against raw
    # closes. A sample of dates cannot catch corruption elsewhere in a table.
    checked = 0
    largest_difference = largest_ridge_difference = 0.0
    baseline_checks = latest_checks = latest_historical_checks = training_checks = 0
    ridge_checks = {"backtest": 0, "latest": 0}
    ridge_rounding_sensitive = 0
    schedules = {}
    for asset in assets:
        close = _source_close(data, asset, input_hashes)
        _check(close.index.is_unique and close.index.notna().all(), f"{asset}: invalid source dates")
        trailing = {window: _raw_volatility(close, window) for window in (HORIZON, 90)}
        calendar = trailing[HORIZON].index
        target = trailing[HORIZON].reindex(calendar + pd.Timedelta(days=HORIZON)).to_numpy()
        usable = np.isfinite(trailing[90].to_numpy()) & np.isfinite(target)
        eligible = calendar[usable] + pd.Timedelta(days=HORIZON)
        # Training origins in label order, matching the cutoffs in `eligible`.
        training_origins = np.flatnonzero(usable)
        inputs = _raw_ridge_inputs(close)
        fits = {}
        # Clamping to the calendar length is exact and cannot overflow pandas' Timedelta.
        span = min(config["test_days"], (close.index.max() - close.index.min()).days + 1)
        first = max(close.index.min(), close.index.max() - pd.Timedelta(days=HORIZON + span - 1))
        schedules[asset] = pd.date_range(first, close.index.max() - pd.Timedelta(days=HORIZON), freq="D")
        rows = predictions[predictions.asset.eq(asset)]
        expected = {(origin, model) for origin in schedules[asset] for model in MODELS}
        _check(set(zip(rows.origin, rows.model)) == expected,
               f"{asset}: backtest rows differ from the requested source-date schedule")
        next_rows = latest[latest.asset.eq(asset)]
        _check(set(zip(next_rows.origin, next_rows.model)) == {(close.index.max(), model) for model in MODELS},
               f"{asset}: latest rows must contain every model at the last source date")
        direct = trailing[HORIZON].reindex(pd.DatetimeIndex(rows.target_end)).to_numpy()
        equal = np.isclose(direct, rows.actual.to_numpy(), equal_nan=True, **TOLERANCE)
        if not equal.all():
            row = rows.iloc[np.flatnonzero(~equal)[0]]
            raise VerificationError(f"{asset} {row.origin.date()}: saved actual {row.actual} "
                                    f"differs from recalculated {direct[np.flatnonzero(~equal)[0]]}")
        finite = np.isfinite(direct)
        if finite.any():
            largest_difference = max(largest_difference, float(np.max(np.abs(direct[finite] - rows.actual.to_numpy()[finite]))))
        checked += rows.loc[finite, "origin"].nunique()
        for label, table in (("backtest", rows), ("latest", next_rows)):
            if table.empty:
                continue
            initial_origin = schedules[asset][0] if label == "backtest" else close.index.max()
            insufficient = eligible.searchsorted(initial_origin, side="right") < config["min_train"]
            feature_valid = np.isfinite(trailing[90].reindex(pd.DatetimeIndex(table.origin)).to_numpy())
            if insufficient:
                _check(table.status.eq("insufficient_training").all() and table.n_train.eq(0).all()
                       and table.train_target_end.isna().all(),
                       f"{asset}: {label} insufficient-training status disagrees with raw source windows")
            else:
                _check(table.loc[~feature_valid, "status"].eq("missing_features").all(),
                       f"{asset}: {label} missing-feature status disagrees with raw source windows")
                available = table.loc[feature_valid]
                expected_status = np.where(available.actual.notna() | (label == "latest"), "ok", "missing_target")
                correct = available.status.eq(expected_status) | (available.model.eq("Ridge") & available.status.eq("model_error"))
                _check(correct.all(), f"{asset}: {label} forecast status disagrees with available source windows")
                ridge = table[table.model.eq("Ridge")]
                for row in ridge.itertuples():
                    cutoff = (initial_origin + pd.Timedelta(days=((row.origin - initial_origin).days
                              // config["refit_every"]) * config["refit_every"]))
                    count = eligible.searchsorted(cutoff, side="right")
                    _check(row.n_train == count and row.train_target_end == eligible[count - 1],
                           f"{asset} {row.origin.date()}: {label} Ridge training count or cutoff differs from matured source windows")
                    training_checks += 1
                    if row.status == "missing_features":
                        continue
                    # Refit on exactly the matured labels available at this refit date.
                    if cutoff not in fits:
                        positions = training_origins[:count]
                        fits[cutoff] = _ridge_fit(inputs[positions], target[positions])
                    raw = _ridge_forecast(fits[cutoff], inputs[(row.origin - calendar[0]).days])
                    if fits[cutoff][4]:
                        # Reported, not compared: no independent calculation can reproduce noise.
                        ridge_rounding_sensitive += 1
                        continue
                    if row.status == "model_error":
                        _check(not np.isfinite(raw),
                               f"{asset} {row.origin.date()}: {label} Ridge model_error although an independent refit is finite")
                        continue
                    forecast = max(0.0, raw)
                    _check(np.isfinite(raw) and np.isclose(row.predicted, forecast, **TOLERANCE)
                           and (row.clipped == (raw < 0) or abs(raw) <= TOLERANCE["atol"]),
                           f"{asset} {row.origin.date()}: {label} Ridge forecast differs from an independent refit on raw closes")
                    largest_ridge_difference = max(largest_ridge_difference, abs(row.predicted - forecast))
                    ridge_checks[label] += 1
            for model, window in (("Persistence", HORIZON), ("Historical90", 90)):
                available = table[table.model.eq(model) & table.status.isin({"ok", "missing_target"})]
                expected = trailing[window].reindex(pd.DatetimeIndex(available.origin)).to_numpy()
                _check(np.isclose(expected, available.predicted, **TOLERANCE).all(),
                       f"{asset}: {label} {model.lower()} forecast differs from trailing volatility of raw closes")
                if label == "backtest":
                    baseline_checks += len(available)
                elif model == "Persistence":
                    latest_checks += len(available)
                else:
                    latest_historical_checks += len(available)

    # Independent matched-date losses, including a fixed nonoverlap schedule.
    metrics = _read_output(output / "backtest_metrics.csv")
    metric_keys = ["asset", "model", "sample"]
    _check(set(metric_keys + ["n_requested", "n_model_valid", "n_scored", "coverage", "mae", "rmse"]) <= set(metrics),
           "backtest_metrics.csv: missing metric columns")
    expected_metrics = {(asset, model, sample) for asset, schedule in schedules.items() if len(schedule)
                        for model in MODELS for sample in ("daily_origins", "nonoverlap_30d")}
    _check(not metrics.duplicated(metric_keys).any() and set(map(tuple, metrics[metric_keys].to_numpy())) == expected_metrics,
           "backtest_metrics.csv rows differ from the requested assets/models/samples")
    metric_checks = 0
    for asset, rows in predictions.groupby("asset"):
        complete = rows[rows.status == "ok"].groupby("origin").model.nunique()
        common = complete.index[complete == len(MODELS)]
        calendar = pd.date_range(rows.origin.min(), rows.origin.max(), freq="D")
        for sample, requested in [("daily_origins", calendar), ("nonoverlap_30d", calendar[::HORIZON])]:
            scored_dates = common.intersection(requested)
            for model in MODELS:
                label = f"{asset}/{model}/{sample}"
                selected = metrics[(metrics.asset == asset) & (metrics.model == model) & (metrics["sample"] == sample)]
                _check(len(selected) == 1, f"{label}: expected exactly one metric row")
                observed = selected.iloc[0]
                values = rows[(rows.model == model) & rows.origin.isin(scored_dates)]
                error = values.predicted.to_numpy() - values.actual.to_numpy()
                _check(observed.n_requested == len(requested), f"{label}: n_requested differs")
                _check(observed.n_scored == len(error), f"{label}: n_scored differs")
                _check(observed.n_model_valid == len(rows[rows.model.eq(model) & rows.status.eq("ok")
                                                            & rows.origin.isin(requested)]), f"{label}: n_model_valid differs")
                _check(np.isclose(observed.coverage, len(error) / len(requested), **TOLERANCE), f"{label}: coverage differs")
                if len(error):
                    # Sum bounded ratios and use hypot's rescaled norm so
                    # finite extreme errors do not overflow when squared.
                    scale = float(np.max(np.abs(error)))
                    mae = scale * (math.fsum(np.abs(error) / scale) / len(error)) if scale else 0.0
                    rmse = np.hypot.reduce(error / np.sqrt(len(error)))
                    _check(np.isclose(observed.mae, mae, **TOLERANCE), f"{label}: MAE differs")
                    _check(np.isclose(observed.rmse, rmse, **TOLERANCE), f"{label}: RMSE differs")
                else:
                    _check(pd.isna(observed.mae) and pd.isna(observed.rmse), f"{label}: unscored sample has a loss")
                metric_checks += 1

    # Aggregate scores are equal-weight means over assets scored for all models.
    # The table is legitimately empty when no asset is scored for every model.
    macro = _read_table(output / "macro_metrics.csv")
    _check(not len(macro) or {"sample", "model", "n_assets", "n_assets_requested", "n_requested", "n_scored",
                             "coverage", "macro_mae", "macro_rmse"} <= set(macro),
           "macro_metrics.csv: missing metric columns")
    expected_rows = set()
    for sample in metrics["sample"].unique() if len(metrics) else []:
        subset = metrics[metrics["sample"] == sample]
        valid = subset[subset.n_scored.gt(0) & subset.mae.notna() & subset.rmse.notna()]
        if valid.groupby("asset").model.nunique().eq(len(MODELS)).any():
            expected_rows |= {(sample, model) for model in MODELS}
    observed_rows = set(zip(macro["sample"], macro["model"])) if len(macro) else set()
    _check(not len(macro) or not macro.duplicated(["sample", "model"]).any(), "macro_metrics.csv contains duplicate rows")
    _check(observed_rows == expected_rows,
           f"macro_metrics.csv rows {sorted(observed_rows)} differ from the per-asset table {sorted(expected_rows)}")
    macro_checks = 0
    for row in macro.itertuples():
        subset = metrics[metrics["sample"] == row.sample]
        valid = subset[subset.n_scored.gt(0) & subset.mae.notna() & subset.rmse.notna()]
        counts = valid.groupby("asset").model.nunique()
        group = valid[valid.asset.isin(counts.index[counts == len(MODELS)]) & valid.model.eq(row.model)]
        requested = subset[subset.model.eq(row.model)]
        label = f"macro {row.sample}/{row.model}"
        _check(row.n_assets == len(group), f"{label}: n_assets differs")
        _check(row.n_assets_requested == requested.asset.nunique(), f"{label}: n_assets_requested differs")
        _check(row.n_requested == int(requested.n_requested.sum()), f"{label}: n_requested differs")
        _check(row.n_scored == int(group.n_scored.sum()), f"{label}: n_scored differs")
        _check(np.isclose(row.coverage, group.n_scored.sum() / requested.n_requested.sum(), **TOLERANCE),
               f"{label}: coverage differs")
        _check(np.isclose(row.macro_mae, group.mae.mean(), **TOLERANCE), f"{label}: macro MAE differs")
        _check(np.isclose(row.macro_rmse, group.rmse.mean(), **TOLERANCE), f"{label}: macro RMSE differs")
        macro_checks += 1

    daily = metrics[metrics["sample"] == "daily_origins"]
    _check(metadata["successful_prediction_rows"] == int(predictions.status.eq("ok").sum()),
           "run_metadata.json successful_prediction_rows disagrees with backtest_predictions.csv")
    _check(metadata["scored_prediction_rows"] == int(daily.n_scored.sum()),
           "run_metadata.json scored_prediction_rows disagrees with backtest_metrics.csv")

    # Last, bind every artifact (report, figures and the tables checked above) to
    # the bytes recorded by the run. Only the regenerated history table may be absent.
    outputs = metadata["output_hashes"]
    _check(isinstance(outputs, dict) and all(isinstance(digest, str) for digest in outputs.values()),
           "Recorded output hashes must be an object")
    artifacts = sorted(name for name in (p.relative_to(output).as_posix() for p in output.rglob("*")
                                         if p.is_file() and p.name != ".DS_Store")
                       if name not in UNHASHED)
    for name in artifacts:
        _check(name in outputs, f"Artifact is not recorded in run_metadata.json: {name}")
        _check(_sha256(output / name) == outputs[name], f"Artifact changed since the run: {name}")
    missing = sorted(set(outputs) - set(artifacts) - OPTIONAL_ARTIFACTS)
    _check(not missing, f"Recorded artifacts are missing: {', '.join(missing)}")
    return {
        "status": "passed",
        "source_directory": _display_path(data),
        "source_files_verified": len(input_hashes),
        "source_files_rejected": int(validation.status.eq("rejected").sum()),
        "assets_verified": len(assets),
        "code_files_verified": len(metadata["code_hashes"]),
        "output_files_recorded": len(outputs),
        "run_metadata_sha256": _sha256(output / "run_metadata.json"),
        "source_windows_independently_recalculated": checked,
        "max_absolute_target_difference": largest_difference,
        "backtest_baseline_forecasts_recalculated": baseline_checks,
        "backtest_ridge_forecasts_refitted": ridge_checks["backtest"],
        "max_absolute_ridge_difference": largest_ridge_difference,
        "ridge_training_rows_independently_validated": training_checks,
        "metric_rows_independently_recalculated": metric_checks,
        "macro_rows_independently_recalculated": macro_checks,
        "latest_persistence_forecasts_recalculated": latest_checks,
        "latest_historical90_forecasts_recalculated": latest_historical_checks,
        "latest_ridge_forecasts_refitted": ridge_checks["latest"],
        "ridge_forecasts_rounding_sensitive": ridge_rounding_sensitive,
        "all_forecast_dates_and_training_cutoffs_validated": True,
        "scored_prediction_rows": int(daily.n_scored.sum()),
        "forecast_origin_observations_per_model": int(daily[daily.model == "Ridge"].n_scored.sum()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("output", type=Path, help="Result directory written by python -m crypto_volatility")
    parser.add_argument("--data", type=Path, default=None,
                        help="Source CSV directory (default: the data_directory recorded in run_metadata.json)")
    args = parser.parse_args(argv)
    try:
        summary = verify(args.output, args.data)
    except (VerificationError, OSError, KeyError, ValueError, TypeError, np.linalg.LinAlgError) as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
