"""Forecast the next 30 calendar days' annualized daily-return volatility.

Forecasts are issued after the origin day's close. A training label is available
only when its entire 30-day return window has ended. Fixed model settings and
calendar-based refits avoid fitting preprocessing or selecting models on test
outcomes. Daily evaluation windows overlap; a separate 30-day sample is provided.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HORIZON = 30
RIDGE_ALPHA = 1.0
ANNUALIZATION = np.sqrt(365.0)
MODELS = ("Persistence", "Historical90", "Ridge")
FEATURE_COLUMNS = tuple(
    "{}_{}".format(stat, window)
    for stat in ("volatility", "mean_abs_return", "mean_return")
    for window in (7, 30, 90)
)
PREDICTION_COLUMNS = (
    "asset", "model", "origin", "target_start", "target_end", "actual",
    "predicted", "train_target_end", "n_train", "status", "error", "clipped",
)
METRIC_COLUMNS = (
    "asset", "model", "sample", "n_requested", "n_model_valid", "n_scored",
    "coverage", "mae", "rmse",
)


def _validate_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Input must have a DatetimeIndex of daily observations.")
    if frame.index.hasnans or not frame.index.is_unique:
        raise ValueError("Input dates must be valid and unique.")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Input dates must be sorted in ascending order.")
    if not frame.index.equals(frame.index.normalize()):
        raise ValueError("Input dates must be midnight daily dates.")
    # Compare day steps rather than index objects so that the check does not
    # depend on the datetime64 resolution the caller happened to store.
    if len(frame) > 1 and not (np.diff(frame.index.to_numpy()) == np.timedelta64(1, "D")).all():
        raise ValueError("Reindex to a complete daily calendar; do not drop missing dates.")
    required = {"Close", "log_return"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Missing input columns: {}".format(", ".join(sorted(missing))))


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return past-only features and explicitly dated next-30-day labels.

    Missing or invalid returns remain missing: a rolling window needs every
    daily observation. Volume is excluded because its units vary by asset and
    are not established by the supplied dataset. Unlabeled latest rows are kept.
    Features and labels are both derived from ``log_return`` so that they can
    never use different volatility conventions; a supplied ``volatility_30``
    column must agree with that recomputation.
    """
    _validate_frame(frame)
    returns = pd.to_numeric(frame["log_return"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    features = pd.DataFrame(index=frame.index)
    for window in (7, 30, 90):
        features["volatility_{}".format(window)] = (
            returns.rolling(window, min_periods=window).std(ddof=0) * ANNUALIZATION
        )
        features["mean_abs_return_{}".format(window)] = returns.abs().rolling(
            window, min_periods=window
        ).mean()
        features["mean_return_{}".format(window)] = returns.rolling(
            window, min_periods=window
        ).mean()
    features = features.loc[:, list(FEATURE_COLUMNS)]
    trailing = returns.rolling(HORIZON, min_periods=HORIZON).std(ddof=0) * ANNUALIZATION
    if "volatility_30" in frame:
        supplied = pd.to_numeric(frame["volatility_30"], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if not (
            supplied.isna().equals(trailing.isna())
            and np.allclose(supplied.dropna(), trailing.dropna(), rtol=1e-9, atol=1e-12)
        ):
            raise ValueError(
                "volatility_30 disagrees with the 30-day population volatility of log_return."
            )
    # On a contiguous daily index this ends exactly 30 calendar days later and
    # contains r[t+1], ..., r[t+30], rather than any return at the origin.
    features["target_volatility"] = trailing.shift(-HORIZON)
    features["target_start"] = features.index + pd.Timedelta(days=1)
    features["target_end"] = features.index + pd.Timedelta(days=HORIZON)
    return features


def _eligible_training(features: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    numeric = features.loc[:, list(FEATURE_COLUMNS) + ["target_volatility"]]
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    eligible = (
        finite
        & features["target_volatility"].ge(0)
        & features["target_end"].le(origin)
    )
    return features.loc[eligible]


def _fit_ridge(training: pd.DataFrame):
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(
        training.loc[:, list(FEATURE_COLUMNS)],
        np.log1p(training["target_volatility"].to_numpy(dtype=float)),
    )
    return model


def _ridge_prediction(model, row: pd.DataFrame) -> Tuple[float, bool]:
    with np.errstate(over="ignore", invalid="ignore"):
        raw = float(np.expm1(model.predict(row.loc[:, list(FEATURE_COLUMNS)])[0]))
    if not np.isfinite(raw):
        raise ValueError("Ridge produced a non-finite volatility forecast.")
    return max(0.0, raw), raw < 0.0


def _row(asset, model, origin, actual=np.nan, predicted=np.nan,
         train_target_end=pd.NaT, n_train=0, status="ok", error="", clipped=False):
    return {
        "asset": asset, "model": model, "origin": origin,
        "target_start": origin + pd.Timedelta(days=1),
        "target_end": origin + pd.Timedelta(days=HORIZON),
        "actual": actual, "predicted": predicted,
        "train_target_end": train_target_end, "n_train": int(n_train),
        "status": status, "error": error, "clipped": bool(clipped),
    }


def _forecast_rows(asset, origin, features, model, training, fit_error="", latest=False):
    feature_row = features.loc[[origin]]
    actual = float(feature_row["target_volatility"].iloc[0])
    finite_features = np.isfinite(
        feature_row.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    ).all()
    rows = []
    for model_name in MODELS:
        train_end = pd.NaT
        n_train = 0
        if model_name == "Ridge" and training is not None:
            n_train = len(training)
            train_end = training["target_end"].max()
        if not finite_features:
            rows.append(_row(
                asset, model_name, origin, actual, train_target_end=train_end,
                n_train=n_train, status="missing_features",
                error="A complete past 90-day return history is required; gaps are not filled.",
            ))
            continue
        try:
            clipped = False
            if model_name == "Persistence":
                predicted = float(feature_row["volatility_30"].iloc[0])
            elif model_name == "Historical90":
                predicted = float(feature_row["volatility_90"].iloc[0])
            else:
                if model is None:
                    raise ValueError(fit_error or "Ridge fitting did not produce a model.")
                predicted, clipped = _ridge_prediction(model, feature_row)
            if not np.isfinite(predicted) or predicted < 0:
                raise ValueError("Forecast must be finite and nonnegative.")
            target_valid = np.isfinite(actual) and actual >= 0
            status = "ok" if latest or target_valid else "missing_target"
            rows.append(_row(
                asset, model_name, origin, actual, predicted, train_end, n_train,
                status=status,
                error="" if status == "ok" else "The next 30 days contain missing or invalid returns.",
                clipped=clipped,
            ))
        except (ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            rows.append(_row(
                asset, model_name, origin, actual, train_target_end=train_end,
                n_train=n_train, status="model_error", error=str(exc),
            ))
    return rows


def backtest_asset(
    frame: pd.DataFrame, asset: str, test_days: int = 180,
    min_train: int = 250, refit_every: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Run a fixed expanding-window backtest and forecast at the final data date.

    The test window is fixed before filtering gaps: up to ``test_days`` calendar
    origins ending 30 days before the final input date. The whole asset backtest
    is skipped if its first origin has fewer than ``min_train`` matured labels.
    Ridge is refitted on a fixed calendar schedule; scaling uses that training
    subset only. Latest forecasts fit all labels matured at the final input date.
    """
    for name, value in (("test_days", test_days), ("min_train", min_train),
                        ("refit_every", refit_every)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError("{} must be a positive integer.".format(name))
    features = build_features(frame)
    predictions: List[Dict] = []
    latest_rows: List[Dict] = []
    status = {
        "asset": asset, "status": "skipped", "backtest_status": "skipped",
        "latest_status": "unavailable", "n_origins_requested": 0,
        "n_train_initial": 0, "n_train_latest": 0,
        "n_forecasts_ok": 0, "n_forecasts_requested": 0,
        "ridge_clipped_backtest": 0, "ridge_clipped_latest": 0, "errors": [],
    }
    if features.empty:
        status["errors"].append("No dated observations available.")
        return (pd.DataFrame(columns=PREDICTION_COLUMNS),
                pd.DataFrame(columns=PREDICTION_COLUMNS), status)

    final_origin = features.index[-1]
    last_test_origin = final_origin - pd.Timedelta(days=HORIZON)
    # Clamp before building a Timedelta: an oversized test_days means every
    # available origin and must not overflow pandas' Timedelta range.
    available_days = (last_test_origin - features.index[0]).days + 1
    first_test_origin = last_test_origin - pd.Timedelta(
        days=min(test_days, available_days) - 1
    )
    origins = features.index[
        (features.index >= first_test_origin) & (features.index <= last_test_origin)
    ]
    status["n_origins_requested"] = len(origins)
    status["n_forecasts_requested"] = len(origins) * len(MODELS)
    if len(origins):
        initial_training = _eligible_training(features, origins[0])
        status["n_train_initial"] = len(initial_training)
        if len(initial_training) < min_train:
            error = "First test origin has {} matured training labels; {} required.".format(
                len(initial_training), min_train
            )
            status["errors"].append(error)
            for origin in origins:
                for model_name in MODELS:
                    predictions.append(_row(
                        asset, model_name, origin,
                        float(features.at[origin, "target_volatility"]),
                        status="insufficient_training", error=error,
                    ))
        else:
            status["backtest_status"] = "ok"
            model = None
            training = None
            fit_error = ""
            for origin in origins:
                # This schedule does not move when either features or future
                # targets are missing. All available labels end at or before t.
                if (origin - origins[0]).days % refit_every == 0:
                    training = _eligible_training(features, origin)
                    try:
                        model = _fit_ridge(training)
                        fit_error = ""
                    except (ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                        model = None
                        fit_error = str(exc)
                        status["errors"].append("{}: {}".format(origin.date(), fit_error))
                predictions.extend(_forecast_rows(
                    asset, origin, features, model, training, fit_error=fit_error
                ))
    else:
        status["errors"].append("Fewer than 31 calendar dates; no full future window available.")

    latest_training = _eligible_training(features, final_origin)
    status["n_train_latest"] = len(latest_training)
    if len(latest_training) < min_train:
        error = "Latest origin has {} matured training labels; {} required.".format(
            len(latest_training), min_train
        )
        latest_rows = [_row(
            asset, model_name, final_origin, status="insufficient_training", error=error
        ) for model_name in MODELS]
    else:
        model = None
        fit_error = ""
        try:
            model = _fit_ridge(latest_training)
        except (ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            fit_error = str(exc)
            status["errors"].append("Latest fit: " + fit_error)
        latest_rows = _forecast_rows(
            asset, final_origin, features, model, latest_training,
            fit_error=fit_error, latest=True,
        )

    prediction_frame = pd.DataFrame(predictions, columns=PREDICTION_COLUMNS)
    latest_frame = pd.DataFrame(latest_rows, columns=PREDICTION_COLUMNS)
    n_ok = int(prediction_frame["status"].eq("ok").sum())
    status["n_forecasts_ok"] = n_ok
    if status["backtest_status"] == "ok" and n_ok != len(prediction_frame):
        status["backtest_status"] = "partial"
    status["status"] = status["backtest_status"]
    status["latest_status"] = (
        "ok" if latest_frame["status"].eq("ok").all() else "unavailable"
    )
    status["ridge_clipped_backtest"] = int(prediction_frame["clipped"].sum())
    status["ridge_clipped_latest"] = int(latest_frame["clipped"].sum())
    return prediction_frame, latest_frame, status


def score_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Score all models on identical successful dates, retaining coverage.

    MAE and RMSE are expressed as annualized volatility in decimal units. The
    nonoverlapping sample starts at the first requested calendar origin and
    keeps every 30th day even when a scheduled observation is unscorable. These
    are descriptive errors, not significance tests or model-selection evidence.

    Each asset's first and last recorded origins define its requested interval;
    every intervening calendar day remains in the coverage denominator even if
    all model rows for that day are missing. Missing boundary rows cannot be
    recovered from a prediction table alone, so callers must preserve the first
    and last requested origins, including explicit failure rows.
    """
    if predictions.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)
    required = {
        "asset", "model", "origin", "target_start", "target_end",
        "actual", "predicted", "status",
    }
    if not required.issubset(predictions.columns):
        raise ValueError("Prediction table is missing required scoring columns.")
    predictions = predictions.copy()
    if predictions[["asset", "model"]].isna().any().any():
        raise ValueError("Prediction asset and model identifiers cannot be missing.")
    if not predictions["model"].isin(MODELS).all():
        raise ValueError("Prediction table contains unknown model names.")
    for column in ("origin", "target_start", "target_end"):
        predictions[column] = pd.to_datetime(predictions[column], errors="raise")
        if predictions[column].isna().any():
            raise ValueError("Prediction dates cannot be missing.")
    if not predictions["origin"].eq(predictions["origin"].dt.normalize()).all():
        raise ValueError("Forecast origins must be midnight daily dates.")
    if not (
        predictions["target_start"].eq(predictions["origin"] + pd.Timedelta(days=1)).all()
        and predictions["target_end"].eq(
            predictions["origin"] + pd.Timedelta(days=HORIZON)
        ).all()
    ):
        raise ValueError("Forecast target dates must be origin + 1 through origin + 30 days.")
    if predictions.duplicated(["asset", "model", "origin"]).any():
        raise ValueError("Prediction table contains duplicate asset/model/origin rows.")
    for column in ("actual", "predicted"):
        predictions[column] = pd.to_numeric(predictions[column], errors="raise")
    finite_actual = predictions.loc[np.isfinite(predictions["actual"])]
    if finite_actual.groupby(["asset", "origin"])["actual"].nunique().gt(1).any():
        raise ValueError("Models disagree on the actual target for an asset and origin.")
    results = []
    for asset, asset_rows in predictions.groupby("asset", sort=True):
        origins = pd.date_range(
            asset_rows["origin"].min(), asset_rows["origin"].max(), freq="D"
        )
        valid = (
            asset_rows["status"].eq("ok")
            & np.isfinite(asset_rows["actual"])
            & np.isfinite(asset_rows["predicted"])
            & asset_rows["actual"].ge(0)
            & asset_rows["predicted"].ge(0)
        )
        successful = asset_rows.loc[valid & asset_rows["model"].isin(MODELS)]
        complete = successful.groupby("origin").agg(
            models=("model", "nunique"), actual_values=("actual", "nunique")
        )
        common_origins = complete.index[
            complete["models"].eq(len(MODELS)) & complete["actual_values"].eq(1)
        ]
        for sample, requested in (
            ("daily_origins", origins),
            ("nonoverlap_30d", origins[(origins - origins[0]).days % HORIZON == 0]),
        ):
            shared = common_origins.intersection(requested)
            for model_name in MODELS:
                model_rows = successful.loc[successful["model"].eq(model_name)]
                scored = model_rows.loc[model_rows["origin"].isin(shared)]
                errors = (scored["predicted"] - scored["actual"]).to_numpy(dtype=float)
                mae = rmse = np.nan
                if len(errors):
                    scale = float(np.max(np.abs(errors)))
                    # A finite forecast can still overflow when errors are
                    # summed or squared; scaling also avoids tiny-error underflow.
                    normalized = errors / scale if scale else errors
                    mae = float(np.mean(np.abs(normalized)) * scale)
                    rmse = float(np.sqrt(np.mean(normalized ** 2)) * scale)
                results.append({
                    "asset": asset, "model": model_name, "sample": sample,
                    "n_requested": len(requested),
                    "n_model_valid": int(model_rows["origin"].isin(requested).sum()),
                    "n_scored": len(scored),
                    "coverage": len(scored) / len(requested) if len(requested) else 0.0,
                    "mae": mae,
                    "rmse": rmse,
                })
    return pd.DataFrame(results, columns=METRIC_COLUMNS)
