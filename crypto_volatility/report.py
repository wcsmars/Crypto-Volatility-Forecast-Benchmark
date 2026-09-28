"""Reproducible descriptive statistics, figures, and analysis report.

All return and volatility values stay in decimal units in machine-readable
tables. Percentage formatting is applied only to figures and Markdown.
"""

from pathlib import Path
import json
import math
import re
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from .model import MODELS


COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#332288", "#777777"]
PANEL_SIZE = len(COLORS)
MAJORS = [
    ("Bitcoin", ("bitcoin", "btc")),
    ("Ethereum", ("ethereum", "eth")),
    ("BNB", ("bnb", "binancecoin")),
    ("XRP", ("xrp", "ripple")),
    ("Cardano", ("cardano", "ada")),
    ("Solana", ("solana", "sol")),
    ("Dogecoin", ("dogecoin", "doge")),
    ("Litecoin", ("litecoin", "ltc")),
]
MAJOR_NAMES = ", ".join(label for label, _ in MAJORS[:-1]) + f", and {MAJORS[-1][0]}"


def _finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return _finite(frame[name])


def _date(value: object) -> str:
    return "n/a" if pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d")


def _pct(value: float, digits: int = 1) -> str:
    return "n/a" if pd.isna(value) else f"{value:.{digits}%}"


def _number(value: float, digits: int = 0) -> str:
    return "n/a" if pd.isna(value) else f"{value:,.{digits}f}"


def _table(headers: List[str], rows: List[List[object]]) -> str:
    """Write Markdown without adding a tabulate runtime dependency."""
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join([
        "| " + " | ".join(map(cell, headers)) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(map(cell, row)) + " |" for row in rows],
    ])


def _drawdown(close: pd.Series) -> float:
    """Observed-close drawdown: leave gaps missing and never compound filled returns."""
    valid = close.where(close > 0)
    return float((valid / valid.cummax() - 1).min())


def _statistics(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for asset, frame in frames.items():
        close = _column(frame, "Close").where(lambda values: values > 0)
        returns = _column(frame, "log_return")
        volatility = _column(frame, "volatility_30")
        volume = _column(frame, "Volume")
        volume = volume.where(volume >= 0)
        latest = volatility.last_valid_index()
        rows.append({
            "asset": asset,
            "start": _date(frame.index.min()),
            "end": _date(frame.index.max()),
            "calendar_days": len(frame),
            "valid_close_days": int(close.notna().sum()),
            "valid_return_days": int(returns.notna().sum()),
            "close_min": close.min(),
            "close_median": close.median(),
            "close_mean": close.mean(),
            "close_max": close.max(),
            "close_std_ddof0": close.std(ddof=0),
            "daily_log_return_mean": returns.mean(),
            "daily_log_return_std_ddof0": returns.std(ddof=0),
            "daily_log_return_skew": returns.skew(),
            "daily_log_return_excess_kurtosis": returns.kurt(),
            "annualized_full_history_volatility": returns.std(ddof=0) * np.sqrt(365),
            "median_30d_annualized_volatility": volatility.median(),
            "latest_30d_annualized_volatility": volatility.loc[latest] if latest is not None else np.nan,
            "latest_30d_volatility_date": _date(latest),
            "max_observed_close_drawdown": _drawdown(close),
            "reported_volume_median_unverified_units": volume.median(),
            "reported_volume_zero_days": int(volume.eq(0).sum()),
            "reported_volume_zero_share_of_observed": volume.eq(0).sum() / volume.notna().sum() if volume.notna().any() else np.nan,
            "reported_volume_missing_share_of_calendar": volume.isna().mean(),
        })
    return pd.DataFrame(rows)


def _panel_end(last_valid: Dict[str, pd.Timestamp]) -> pd.Timestamp:
    """The last-valid-close date shared by the most histories; ties go to the latest date."""
    counts = pd.Series(list(last_valid.values())).value_counts()
    return pd.Timestamp(max(counts.index[counts.eq(counts.max())]))


def _has_close(close: pd.Series, date: pd.Timestamp) -> bool:
    value = close.get(date)
    return value is not None and pd.notna(value) and value > 0


def _exclusion_reason(close: pd.Series, end: pd.Timestamp, floor: pd.Timestamp):
    """Why a history cannot join a panel ending on ``end``, or None when it qualifies."""
    if not _has_close(close, end):
        return f"no valid close on the panel end date {_date(end)} (last valid close {_date(close.last_valid_index())})"
    if close.index.min() > floor:
        return f"history starts {_date(close.index.min())}, after the one-year floor {_date(floor)}"
    return None


def _choose_panel(frames: Dict[str, pd.DataFrame]) -> Tuple[List[str], Dict[str, str], pd.Timestamp, pd.Timestamp, dict]:
    """Select the illustrative panel and record how the selection was made.

    The panel ends on the last-valid-close date shared by the most fixed-list
    major assets, so neither an unrelated history nor one major with an extra
    day can evict the others; a longer history simply stops at the panel end.
    Each listed asset that is present but not displayed is named with its
    reason. When no major qualifies, up to ``PANEL_SIZE`` histories are ranked
    by recent close-data completeness instead.
    """
    closes = {asset: _column(frame, "Close").where(lambda series: series > 0) for asset, frame in frames.items()}
    last_valid = {asset: close.last_valid_index() for asset, close in closes.items()}
    eligible = {asset: pd.Timestamp(end) for asset, end in last_valid.items() if end is not None}
    selection = {"path": "none", "excluded": {}}
    if not eligible:
        return [], {}, pd.NaT, pd.NaT, selection
    normalized = {asset: re.sub(r"[^a-z0-9]", "", asset.lower()) for asset in frames}
    majors = []
    for label, aliases in MAJORS:
        matches = [asset for asset in frames if asset in eligible and normalized[asset] in aliases]
        if matches:
            majors.append((matches[0], label))
    major_labels = dict(majors)
    chosen, labels = [], {}
    if majors:
        end = _panel_end({asset: eligible[asset] for asset, _ in majors})
        floor = end - pd.Timedelta(days=364)
        for asset, label in majors:
            reason = _exclusion_reason(closes[asset], end, floor)
            if reason:
                selection["excluded"][label] = reason
            else:
                chosen.append(asset)
                labels[asset] = label
        selection["path"] = "fixed_list"
    # Nonstandard/small input directories still produce an informative report.
    # Completeness determines the fallback; file-system/alphabetic order does not.
    if not chosen:
        end = _panel_end(eligible)
        floor = end - pd.Timedelta(days=364)
        qualifying = [asset for asset in eligible if _exclusion_reason(closes[asset], end, floor) is None]
        available = qualifying or [asset for asset in eligible if _has_close(closes[asset], end)]
        chosen = sorted(
            available,
            key=lambda asset: int(closes[asset].loc[floor:end].gt(0).sum()),
            reverse=True,
        )[:PANEL_SIZE]
        labels = {asset: major_labels.get(asset, asset) for asset in chosen}
        floor = max([floor] + [frames[asset].index.min() for asset in chosen])
        selection["path"] = "completeness_fallback"
        # Reasons are restated against the fallback panel so that no displayed
        # asset is also listed as excluded, and the quoted dates match the heading.
        selection["excluded"] = {
            label: _exclusion_reason(closes[asset], end, floor)
            or f"not among the {PANEL_SIZE} most complete recent histories"
            for asset, label in majors if asset not in chosen
        }
    if not chosen:
        return [], {}, pd.NaT, pd.NaT, selection
    prices = pd.concat({asset: _column(frames[asset], "Close") for asset in chosen}, axis=1).loc[floor:end]
    shared = prices.gt(0).all(axis=1) & prices.notna().all(axis=1)
    if not shared.any():
        return [], {}, pd.NaT, pd.NaT, selection
    start = prices.index[shared][0]
    return chosen, labels, pd.Timestamp(start), end, selection


def _finish(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _date_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _small_multiples(panel: pd.DataFrame, labels: Dict[str, str], title: str, ylabel: str, path: Path, percent: bool = False, symlog: bool = False) -> None:
    count = len(panel.columns)
    rows = math.ceil(count / 2)
    fig, axes = plt.subplots(rows, min(count, 2), figsize=(12, 2.65 * rows + 0.5), squeeze=False, sharex=True, sharey=True, layout="constrained")
    for index, (asset, ax) in enumerate(zip(panel.columns, axes.flat)):
        ax.plot(panel.index, panel[asset], color=COLORS[index % len(COLORS)], linewidth=1.25)
        ax.set_title(labels[asset], loc="left", fontsize=11)
        ax.set_ylabel(ylabel)
        if percent:
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        if symlog:
            ax.set_yscale("symlog", linthresh=1, linscale=1, base=10)
        _date_axis(ax)
        ax.tick_params(axis="x", labelbottom=True)
    values = panel.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    maximum = float(finite.max()) if len(finite) else 0
    # Set the shared limits only after every series has been plotted. Setting
    # bottom=0 in the loop disables autoscaling and clips later, larger series.
    axes.flat[0].set_ylim(0, 1.08 * maximum if maximum > 0 else 1)
    for ax in list(axes.flat)[count:]:
        ax.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=16, fontweight="bold")
    _finish(fig, path)


def _volume_level_shifts(normalized: pd.DataFrame, threshold: float = 30.0, min_days: int = 7) -> List[dict]:
    """Runs of at least ``min_days`` consecutive dates above ``threshold`` times an asset's own median.

    Jumps of this size that persist for weeks more likely reflect a change in the
    source's volume denomination than trading activity; they are reported, not corrected.
    """
    runs = []
    for asset in normalized:
        high = normalized[asset].gt(threshold)
        for _, days in normalized[asset][high].groupby(high.ne(high.shift()).cumsum()[high]):
            if len(days) >= min_days:
                runs.append({"asset": asset, "start": days.index[0], "end": days.index[-1],
                             "days": len(days), "median_ratio": float(days.median())})
    return runs


def _panel_outputs(frames: Dict[str, pd.DataFrame], output: Path) -> dict:
    assets, labels, start, end, selection = _choose_panel(frames)
    info = {"assets": assets, "labels": labels, "start": start, "end": end, "selection": selection}
    if not assets:
        return info
    dates = pd.date_range(start, end, freq="D", name="Date")
    panels = {
        name: pd.concat({asset: _column(frames[asset], name).reindex(dates) for asset in assets}, axis=1)
        for name in ["Close", "log_return", "volatility_30", "Volume"]
    }
    prices = panels["Close"].where(panels["Close"] > 0)
    prices = prices.where(prices.notna().all(axis=1), axis=0)
    normalized = prices.divide(prices.iloc[0]).multiply(100)
    returns = panels["log_return"].dropna(how="any")
    volatility = panels["volatility_30"].where(panels["volatility_30"].notna().all(axis=1), axis=0)
    volume = panels["Volume"].where(panels["Volume"] >= 0)
    volume = volume.where(volume.notna().all(axis=1), axis=0)
    volume_medians = volume.median()
    volume_normalized = volume.divide(volume_medians.where(volume_medians > 0))
    for name, panel in [
        ("common_window_indexed_close", normalized),
        ("common_window_log_returns", returns),
        ("common_window_volatility_30", volatility),
        ("common_window_normalized_reported_volume", volume_normalized),
    ]:
        panel.to_csv(output / f"{name}.csv", index_label="Date")
    correlation = returns.corr(min_periods=30)
    present = returns.notna().astype("int64")
    pair_counts = present.T.dot(present)
    correlation.to_csv(output / "common_window_return_correlations.csv", index_label="asset")
    pair_counts.to_csv(output / "common_window_correlation_pair_counts.csv", index_label="asset")
    common_stats = pd.DataFrame([
        {
            "asset": asset, "start": _date(start), "end": _date(end),
            "common_return_days": len(returns),
            "daily_log_return_mean": returns[asset].mean(),
            "annualized_common_window_volatility": returns[asset].std(ddof=0) * np.sqrt(365),
            "median_30d_annualized_volatility": volatility[asset].median(),
            "common_rolling_volatility_days": int(volatility[asset].notna().sum()),
            "close_price_change": prices[asset].iloc[-1] / prices[asset].iloc[0] - 1,
            "max_observed_close_drawdown": _drawdown(prices[asset]),
            "reported_volume_median_unverified_units": volume_medians[asset],
            "common_volume_days": int(volume[asset].notna().sum()),
        }
        for asset in assets
    ])
    common_stats.to_csv(output / "common_window_statistics.csv", index=False)
    info.update({"returns": returns, "correlation": correlation, "statistics": common_stats, "volume": volume_normalized, "rolling_days": int(volatility.notna().all(axis=1).sum())})
    figures = output / "figures"
    span = f"{_date(start)} to {_date(end)}"
    _small_multiples(normalized, labels, f"Close prices on a common calendar window\n{span}; each asset starts at 100", "Close index", figures / "indexed_prices.png")
    _small_multiples(volatility, labels, f"Trailing 30-day annualized volatility\n{span}; shared valid dates and common vertical scale", "Annualized volatility", figures / "rolling_volatility.png", percent=True)
    if volume_normalized.notna().any().any():
        _small_multiples(volume_normalized, labels, f"Reported volume relative to each asset's median\n{span}; linear scale 0–1, logarithmic above 1", "Volume / own median", figures / "normalized_volume.png", symlog=True)
    if len(returns):
        fig, ax = plt.subplots(figsize=(11, 5.7), layout="constrained")
        plot = ax.boxplot([returns[asset].to_numpy() for asset in assets], patch_artist=True, tick_labels=[labels[asset] for asset in assets], showfliers=True, flierprops={"markersize": 2.5, "alpha": 0.45}, medianprops={"color": "#222222", "linewidth": 1.5})
        for index, patch in enumerate(plot["boxes"]):
            patch.set_facecolor(COLORS[index % len(COLORS)])
            patch.set_alpha(0.6)
        ax.axhline(0, color="#777777", linewidth=0.7)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_ylabel("Daily log return")
        ax.set_title(f"Daily returns on {len(returns):,} shared valid dates\n{span}; dots retain tail observations", loc="left", pad=12)
        _finish(fig, figures / "return_distributions.png")
    if len(returns) >= 30:
        fig, ax = plt.subplots(figsize=(8.6, 7.2), layout="constrained")
        matrix = ax.imshow(correlation.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(assets)), [labels[asset] for asset in assets], rotation=35, ha="right")
        ax.set_yticks(range(len(assets)), [labels[asset] for asset in assets])
        ax.grid(False)
        for row in range(len(assets)):
            for col in range(len(assets)):
                value = correlation.iloc[row, col]
                ax.text(col, row, "n/a" if pd.isna(value) else f"{value:.2f}", ha="center", va="center", color="white" if pd.notna(value) and abs(value) > 0.65 else "#222222", fontsize=10)
        fig.colorbar(matrix, ax=ax, shrink=0.8, label="Pearson correlation of daily log returns")
        ax.set_title(f"Return correlations use identical observation dates\n{span}; {len(returns):,} paired days per cell", loc="left", pad=14)
        _finish(fig, figures / "return_correlations.png")
    return info


def _macro_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    metrics = metrics.copy()
    if "sample" not in metrics:
        metrics["sample"] = "daily_origins"
    rows = []
    for sample, subset in metrics.groupby("sample", sort=False):
        valid = subset[subset["n_scored"].gt(0) & subset["mae"].notna() & subset["rmse"].notna()]
        counts = valid.groupby("asset")["model"].nunique()
        shared = counts.index[counts.eq(len(MODELS))]
        for model in MODELS:
            requested = subset[subset["model"].eq(model)]
            group = valid[valid["asset"].isin(shared) & valid["model"].eq(model)]
            if group.empty:
                continue
            rows.append({
                "sample": sample, "model": model, "n_assets": len(group),
                "n_assets_requested": requested["asset"].nunique(),
                "n_requested": int(requested["n_requested"].sum()),
                "n_scored": int(group["n_scored"].sum()),
                "coverage": group["n_scored"].sum() / requested["n_requested"].sum() if requested["n_requested"].sum() else np.nan,
                "macro_mae": group["mae"].mean(), "macro_rmse": group["rmse"].mean(),
            })
    return pd.DataFrame(rows)


def _forecast_plot(predictions: pd.DataFrame, output: Path, preferred_assets: List[str],
                   labels: Dict[str, str]) -> str:
    if predictions.empty:
        return ""
    valid = predictions.copy()
    if "sample" in valid:
        valid = valid[valid["sample"].eq("daily_origins")]
    valid = valid[valid["actual"].notna() & valid["predicted"].notna()]
    if valid.empty:
        return ""
    candidates = [asset for asset in preferred_assets if asset in set(valid["asset"])]
    asset = candidates[0] if candidates else valid.groupby("asset")["origin"].nunique().idxmax()
    selected = valid[valid["asset"].eq(asset)].copy()
    selected["origin"] = pd.to_datetime(selected["origin"])
    pivot = selected.pivot_table(index="origin", columns="model", values="predicted", aggfunc="first").reindex(columns=list(MODELS)).dropna(how="any")
    if pivot.empty:
        return ""
    actual = selected.groupby("origin")["actual"].first().reindex(pivot.index)
    fig, ax = plt.subplots(figsize=(12, 5.2), layout="constrained")
    ax.plot(actual.index, actual, color="#222222", linewidth=2.2, label="Realized following 30 days")
    for model, color, style in zip(MODELS, COLORS, ["--", ":", "-."]):
        ax.plot(pivot.index, pivot[model], color=color, linestyle=style, linewidth=1.5, label=model)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Annualized 30-day volatility")
    ax.set_xlabel("Forecast origin (target is the following 30 calendar days)")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_title(f"{labels.get(asset, asset)}: forecasts and subsequently realized volatility\n{len(pivot):,} common daily origins; adjacent targets overlap", loc="left", pad=14)
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    _date_axis(ax)
    _finish(fig, output / "figures" / "forecast_comparison.png")
    return asset


def _metadata(output: Path) -> dict:
    path = output / "run_metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return {}


def write_report(
    frames: Dict[str, pd.DataFrame],
    validation: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    next_forecasts: pd.DataFrame,
    output: Path,
) -> None:
    """Write data-backed EDA and an honestly scoped forward-volatility evaluation."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    if not frames:
        raise ValueError("Cannot report without a validated asset history.")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 13,
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.2, "grid.linewidth": 0.6,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    summary = _statistics(frames)
    summary.to_csv(output / "summary_statistics.csv", index=False)
    panel = _panel_outputs(frames, output)
    macro = _macro_metrics(metrics)
    macro.to_csv(output / "macro_metrics.csv", index=False)
    forecast_asset = _forecast_plot(predictions, output, panel["assets"], panel.get("labels", {}))
    metadata = _metadata(output)
    earliest = min(frame.index.min() for frame in frames.values())
    latest = max(frame.index.max() for frame in frames.values())
    selected_validation = validation[validation["asset"].isin(frames)] if "asset" in validation else validation
    source_rows = int(pd.to_numeric(selected_validation.get("source_rows", pd.Series(dtype=float)), errors="coerce").sum())
    missing_days = int(pd.to_numeric(selected_validation.get("missing_days", pd.Series(dtype=float)), errors="coerce").sum())
    invalid_close = int(pd.to_numeric(selected_validation.get("invalid_close_rows", pd.Series(dtype=float)), errors="coerce").sum())
    currencies = selected_validation.get("currency", pd.Series(dtype=str)).dropna()
    verified_usd = len(currencies) == len(frames) and currencies.eq("USD").all()
    currency_text = "Prices are denominated in USD according to the source currency fields." if verified_usd else "Currency is not consistently supplied in these custom inputs; price units must be checked against their source. The source dataset's original price columns use USD."
    config = metadata.get("configuration", {})
    test_days = config.get("test_days", 180)
    min_train = config.get("min_train", 250)
    refit_every = config.get("refit_every", 30)
    alpha = config.get("ridge_alpha", 1)
    observed_volume = sum(int(_column(frame, "Volume").ge(0).sum()) for frame in frames.values())
    zeros = int(summary["reported_volume_zero_days"].sum())
    stale = summary[pd.to_datetime(summary["end"]) < pd.Timestamp(latest)]
    lines = [
        "# Cryptocurrency prices and 30-day volatility",
        "",
        "## Data and scope",
        "",
        "Source: [Kaggle — Top 100 Cryptocurrencies Historical Dataset](https://www.kaggle.com/datasets/kaushiksuresh147/top-10-cryptocurrencies-historical-dataset). "
        "Results below are computed from the local downloaded histories. The dataset title does not determine the actual number of assets in this snapshot.",
        "",
        f"The validated input contains **{len(frames):,} asset histories** and **{source_rows:,} source rows**, spanning **{_date(earliest)} to {_date(latest)}**. "
        f"There are **{int(summary['valid_close_days'].sum()):,} positive, finite close observations** and **{int(summary['valid_return_days'].sum()):,} usable one-day log returns**. "
        f"{currency_text} Dates are source daily dates; no intraday time alignment or exchange consolidation is inferred.",
        "",
        f"Validation found **{missing_days:,} missing calendar days** and **{invalid_close:,} nonpositive or invalid close rows**. "
        "Missing dates are inserted as missing observations; invalid prices become missing. Prices and returns are never forward filled. "
        "A return is valid only when both consecutive calendar-day closes are valid, and a 30-day estimate requires 30 valid daily returns (31 consecutive daily prices). "
        f"**{len(stale):,} histories end before {_date(latest)}**. Different start/end dates make full-history statistics unsuitable for like-for-like asset rankings. "
        "See [data_validation.csv](data_validation.csv) for file-level findings.",
        "",
        "## Descriptive analysis — Summary statistics and visualization",
        "",
        "[summary_statistics.csv](summary_statistics.csv) contains every asset's sample size and dates, close-price minimum/median/mean/maximum and population standard deviation, "
        "daily log-return mean and population standard deviation, sample skewness and excess kurtosis, full-history annualized volatility, median/latest trailing 30-day volatility, "
        "observed-close maximum drawdown, and reported-volume quality statistics. All return, drawdown, and volatility columns use decimal units: 0.50 means 50%. "
        "Skewness and excess kurtosis use pandas' bias-adjusted sample estimates; standard deviations use ddof=0, as in the 30-day volatility formula.",
        "",
        "Full-history annualized volatility is √365 times the standard deviation of all valid daily returns in that asset's history. "
        "It is a separate statistic from the trailing 30-day volatility used in the forecast section. "
        "Maximum drawdown is the largest observed close-price fall from a preceding observed peak; missing dates remain missing and may conceal a deeper intervening drawdown.",
        "",
        f"Reported volume is zero on **{zeros:,} of {observed_volume:,} observed nonnegative volume days ({_pct(zeros / observed_volume if observed_volume else np.nan)})**. "
        "Zeros are retained, not silently converted to missing. Their economic meaning is unverified. "
        "The source's Volume denomination is not established, so absolute reported volumes are not compared across assets or described as USD turnover. "
        "The volume chart divides each asset by its own median on shared valid dates; an asset with a zero median cannot be normalized.",
        "",
    ]
    if panel["assets"]:
        common = panel["statistics"].set_index("asset")
        labels = panel["labels"]
        assets = panel["assets"]
        selection = panel.get("selection", {})
        excluded = selection.get("excluded", {})
        if selection.get("path") == "fixed_list":
            how_chosen = (
                f"The illustrative panel uses the fixed list {MAJOR_NAMES}, keeping each listed asset that is present, "
                f"has a full recent year of history, and has an observed close on the panel end date {_date(panel['end'])}. "
            )
            how_chosen += (
                "Excluded from that list: " + "; ".join(f"{label} ({reason})" for label, reason in excluded.items()) + ". "
                if excluded else "Every listed asset present in the input qualified. "
            )
        else:
            how_chosen = (
                f"No fixed-list major asset ({MAJOR_NAMES}) has both a full recent year of history and a valid close on the "
                f"shared end date, so the panel falls back to up to {PANEL_SIZE} histories ranked by recent close-data completeness, "
                f"all with a valid close on the panel end date {_date(panel['end'])}. "
            )
            if excluded:
                how_chosen += "Listed assets present but not displayed: " + "; ".join(f"{label} ({reason})" for label, reason in excluded.items()) + ". "
        lines += [
            f"### Comparable recent window: {_date(panel['start'])} to {_date(panel['end'])}",
            "",
            how_chosen
            + "Prices start at 100 on the first date with a valid close for every selected asset. Daily gaps remain visible. "
            f"Returns are compared on **{len(panel['returns']):,} identical valid dates**, and trailing volatility on **{panel['rolling_days']:,} identical valid dates**. "
            "Volume normalization also uses shared valid dates. Complete panel data and correlation observation counts are saved alongside the figures.",
            "",
            _table(
                ["Asset", "Close-price change", "Annualized window volatility", "Median 30-day volatility", "Observed max drawdown"],
                [[labels[asset], _pct(common.loc[asset, "close_price_change"]), _pct(common.loc[asset, "annualized_common_window_volatility"]), _pct(common.loc[asset, "median_30d_annualized_volatility"]), _pct(common.loc[asset, "max_observed_close_drawdown"])] for asset in assets],
            ),
            "",
        ]
        risk = common["annualized_common_window_volatility"].dropna()
        change = common["close_price_change"].dropna()
        if len(risk) >= 2:
            high, low = risk.idxmax(), risk.idxmin()
            lines += [f"On these common dates, **{labels[high]}** has the highest full-window annualized return volatility among the displayed assets ({_pct(risk[high])}); **{labels[low]}** has the lowest ({_pct(risk[low])}). This comparison describes realized variability within this fixed sample.", ""]
        if len(change):
            best, worst = change.idxmax(), change.idxmin()
            lines += [f"Close-price changes range from **{_pct(change[worst])} for {labels[worst]}** to **{_pct(change[best])} for {labels[best]}** across the same endpoints. Price levels are normalized because unit prices alone are not comparable measures of asset performance.", ""]
        corr = panel["correlation"]
        pairs = [(corr.iloc[i, j], corr.index[i], corr.columns[j]) for i in range(len(corr)) for j in range(i + 1, len(corr)) if pd.notna(corr.iloc[i, j])]
        if pairs:
            strongest = max(pairs, key=lambda item: item[0])
            weakest = min(pairs, key=lambda item: item[0])
            lines += [f"Pairwise daily-return correlations range from **{weakest[0]:.2f}** ({labels[weakest[1]]}/{labels[weakest[2]]}) to **{strongest[0]:.2f}** ({labels[strongest[1]]}/{labels[strongest[2]]}). These are return correlations on shared dates, not correlations between trending price levels; they establish no causal relationship.", ""]
        volume_max = panel["volume"].max().dropna()
        if not volume_max.empty:
            asset = volume_max.idxmax()
            date = panel["volume"][asset].idxmax()
            text = f"The largest reported-volume/own-median ratio in this panel is **{volume_max[asset]:,.1f}× for {labels[asset]} on {_date(date)}**. "
            shifts = _volume_level_shifts(panel["volume"])
            if shifts:
                text += ("Reported volume stays above 30× the asset's own median for at least 7 consecutive days in "
                         + "; ".join(f"{labels[run['asset']]} ({_date(run['start'])} to {_date(run['end'])}, {run['days']} days, median {run['median_ratio']:,.0f}×)"
                                     for run in shifts)
                         + ". Sustained jumps of this size more likely reflect changes in the source's volume denomination than trading activity; they are not corrected, so these ratios are not evidence of volume surges. ")
            lines += [text + "Such abrupt scale differences need source verification before economic interpretation. Normalization does not establish that volume units are stable over time. "
                      "The volume figure uses a common scale that is linear from 0 to 1 and logarithmic above 1, preserving zeros and extreme observations.", ""]
        lines += [
            "![Normalized close prices on common calendar dates](figures/indexed_prices.png)", "",
            "![Trailing 30-day annualized volatility using identical dates and a shared vertical scale](figures/rolling_volatility.png)", "",
        ]
        for filename, alt in [
            ("return_distributions.png", "Daily log-return distributions, with tail observations retained"),
            ("return_correlations.png", "Daily log-return correlations on identical observation dates"),
            ("normalized_volume.png", "Reported volume divided by each asset's own median; source units unverified"),
        ]:
            if (output / "figures" / filename).exists():
                lines.extend([f"![{alt}](figures/{filename})", ""])
        lines += ["Data alternatives: [common-window statistics](common_window_statistics.csv), [indexed prices](common_window_indexed_close.csv), [daily returns](common_window_log_returns.csv), [rolling volatility](common_window_volatility_30.csv), [correlations](common_window_return_correlations.csv), [pair counts](common_window_correlation_pair_counts.csv), and [normalized reported volume](common_window_normalized_reported_volume.csv).", ""]
    else:
        lines += ["No common valid recent price panel is available for these inputs; see the per-asset statistics instead.", ""]
    lines += [
        "## Forecast — The following 30 days' annualized volatility",
        "",
        "For daily closes Sₜ, define rₜ = log(Sₜ / Sₜ₋₁). For a window W of 30 daily returns:",
        "",
        "```text",
        "mean_return(W) = sum(r in W) / 30",
        "sigma(W) = sqrt(365) * sqrt(sum((r - mean_return(W))**2 for r in W) / 30)",
        "trailing_volatility_at_t = sigma(r[t-29], ..., r[t])",
        "prediction_target_at_t = sigma(r[t+1], ..., r[t+30])",
        "```",
        "",
        "The model predicts the volatility of the next **30 calendar days** after each daily close. The target is a forward realized measure, "
        "not the already-known trailing volatility. The annualization uses 365 days and the population denominator N=30, as in the formula above. "
        "A target crossing any missing/invalid daily return is unavailable and cannot be scored.",
        "",
        "### Model and validation design",
        "",
        "Three models are evaluated separately for each asset: **Persistence** predicts the observed trailing 30-day annualized volatility; "
        "**Historical90** predicts the trailing 90-day annualized return volatility; **Ridge** uses regularized linear regression with historical features available at the forecast origin. "
        "Its nine features are trailing volatility, mean absolute return, and mean return, each over 7, 30, and 90 days. "
        "It fits log(1 + forward volatility), converts predictions back with exp(prediction) − 1, and clips negative results to zero (clipping is recorded). "
        "Volume is excluded from the predictive features because its units are unverified. "
        f"This run uses **alpha={alpha:g}**, **up to {test_days} requested daily test origins per asset**, **at least {min_train} matured training labels**, "
        f"and a refit every **{refit_every} calendar days**. No penalty is selected by looking at the evaluation errors.",
        "",
        "Evaluation is chronological with expanding training histories. At each fit, a training example is eligible only when its entire forward 30-day label has ended on or before the forecast origin. "
        "This purges unresolved labels at the training boundary. Scaling is fitted on the eligible training rows; the fitted scaler/model is then held fixed until the next scheduled refit. "
        "Training cutoffs and row counts are saved with predictions. Forecasts from all three models are scored on common asset/origin dates; missing or failed forecasts reduce coverage rather than being assigned invented values. "
        "Coverage records the number of scored versus requested origins.",
        "",
        "Adjacent daily targets share 29 of 30 returns, so daily error observations are strongly dependent. "
        "A separate 30-day-spaced sample reduces direct target-window overlap; it still does not guarantee independent observations or remove market-regime dependence. "
        "Both sets of scores are descriptive held-out historical comparisons. They are not a significance test, proof of a persistent forecasting advantage, or a trading-profit evaluation.",
        "",
        "### Historical evaluation results",
        "",
        "MAE and RMSE below are expressed in **annualized volatility percentage points**. The macro scores average per-asset errors equally over assets with scores for all three models. "
        "The macro RMSE is the mean of per-asset RMSEs, not the square root of a pooled mean squared error. "
        "Coverage uses all requested asset/origin forecasts, including assets excluded from the error averages because they lack enough training history or valid forecasts. "
        "Sample counts count asset/origin observations and are not independent observations.",
        "",
    ]
    if not predictions.empty:
        realized = predictions[predictions["actual"].notna()]
        if not realized.empty:
            lines += [f"Recorded historical forecast origins span **{_date(pd.to_datetime(realized['origin']).min())} to {_date(pd.to_datetime(realized['origin']).max())}** across assets; individual histories can have different evaluation endpoints.", ""]
    if not macro.empty:
        lines += [_table(
            ["Sample", "Model", "Scored / requested assets", "Scored / requested origins", "Coverage", "Macro MAE (pp)", "Macro RMSE (pp)"],
            [[row["sample"], row["model"], f"{int(row['n_assets'])} / {int(row['n_assets_requested'])}", f"{int(row['n_scored']):,} / {int(row['n_requested']):,}", _pct(row["coverage"]), f"{100 * row['macro_mae']:.2f}", f"{100 * row['macro_rmse']:.2f}"] for _, row in macro.iterrows()],
        ), ""]
    else:
        lines += ["No assets have scored common forecasts for all three models. The prediction status/error fields explain unavailable estimates; no synthetic evaluation results are substituted.", ""]
    if not metrics.empty and panel["assets"]:
        selected = metrics[metrics["asset"].isin(panel["assets"])]
        if "sample" in selected:
            selected = selected[selected["sample"].eq("daily_origins")]
        if not selected.empty:
            rows = []
            for asset in panel["assets"]:
                group = selected[selected["asset"].eq(asset)].set_index("model")
                if group.empty:
                    continue
                for model in MODELS:
                    if model not in group.index:
                        continue
                    row = group.loc[model]
                    rows.append([panel["labels"][asset], model, _number(row["n_scored"]), _pct(row["coverage"]), _number(100 * row["mae"], 2), _number(100 * row["rmse"], 2)])
            lines += ["Daily-origin results for the illustrative assets:", "", _table(["Asset", "Model", "Scored origins", "Coverage", "MAE (pp)", "RMSE (pp)"], rows), ""]
    lines += ["Complete results: [per-asset metrics](backtest_metrics.csv), [macro metrics](macro_metrics.csv), and [dated predictions and fit cutoffs](backtest_predictions.csv). "
              "A lower historical score in this table does not establish future superiority; retain both simple baselines when interpreting the fitted model.", ""]
    if forecast_asset:
        lines += ["![Out-of-sample forecasts and subsequent 30-day realized volatility for one representative asset](figures/forecast_comparison.png)", ""]
    lines += ["### Forecasts after each asset's last source date", "",
              "[next_forecasts.csv](next_forecasts.csv) records predictions for the next 30 days after each asset's last source date. "
              "If the final close or its required trailing history is invalid, the forecast is unavailable rather than moved to an earlier date. "
              "These are **historical as-of projections**, because this dataset ends in the past; they are not forecasts from today's prices. "
              "Different assets may have different origins. Actual future volatility is left missing when it is not present in the input snapshot.", ""]
    if not next_forecasts.empty:
        selected = next_forecasts[next_forecasts["model"].eq("Ridge")]
        if panel["assets"]:
            selected = selected[selected["asset"].isin(panel["assets"])]
        else:
            selected = selected.head(8)
        if not selected.empty:
            lines += [_table(["Asset", "Origin", "Forward window", "Ridge annualized volatility", "Status"], [
                [panel.get("labels", {}).get(row["asset"], row["asset"]), _date(row["origin"]), f"{_date(row['target_start'])}–{_date(row['target_end'])}", _pct(row["predicted"]), row.get("status", "")]
                for _, row in selected.iterrows()
            ]), ""]
    lines += [
        "## Reproduction and limitations",
        "",
        "Use the commands in the repository README to validate the input and reproduce this report. "
        "[run_metadata.json](run_metadata.json) records the run settings and source metadata.",
        "",
        "This is a single historical snapshot with changing asset coverage, missing dates, invalid price rows, unverified volume denominations, and some stale histories. "
        "Asset inclusion can create survivorship/selection bias. Close-to-close returns omit intraday variation, and a √365 conversion is an annualization convention rather than a guarantee of future annual risk. "
        "The forecast is a point estimate without calibrated prediction intervals. No result here establishes causation, a profitable strategy, or performance on a new prospective dataset.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
