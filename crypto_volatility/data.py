"""Validate source records without inventing prices or multi-day daily returns."""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW = 30
DAYS_PER_YEAR = 365
LEADERBOARD = "current crypto leaderboard.csv"
PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


@dataclass
class Dataset:
    frames: dict
    validation: pd.DataFrame
    events: pd.DataFrame
    hashes: dict


def annualized_volatility(returns: pd.Series) -> pd.Series:
    """Exactly 30 calendar-daily returns, population denominator N (ddof=0)."""
    return returns.rolling(WINDOW, min_periods=WINDOW).std(ddof=0) * np.sqrt(DAYS_PER_YEAR)


def validate_frame(raw: pd.DataFrame, asset: str = "asset"):
    """Return calendar-reindexed data, audit counters, and row-level quality events.

    Malformed, ambiguous (non-ISO 8601), or duplicate dates and non-USD
    currency reject a file. Invalid prices become missing, remain on their
    calendar dates, and invalidate every return and rolling window that
    requires them. Observed zero volume is preserved. Naive dates are vendor
    calendar-day labels, not verified exchange timestamps. Absent optional
    columns are recorded, and counters that need them are left missing.
    """
    if not {"Date", "Close"}.issubset(raw.columns):
        raise ValueError("Required columns: Date, Close")
    if len(raw) < 2:
        raise ValueError("At least two source observations are required")
    if pd.api.types.is_numeric_dtype(raw["Date"]):
        raise ValueError("Dates must be calendar labels, not numeric timestamps")
    # ISO 8601 permits reduced precision, but a year or month alone does not
    # identify a daily observation. Pandas otherwise invents the first day.
    explicit_day = raw["Date"].astype("string").str.match(
        r"^(?:\d{4}-\d{2}-\d{2}|\d{8})(?:$|[T ])"
    )
    if not explicit_day.fillna(False).all():
        raise ValueError(
            "Dates must be unambiguous ISO 8601 labels with an explicit calendar day (YYYY-MM-DD)"
        )
    # ISO 8601 only: day-first or mixed layouts such as 07/01/2020 would parse
    # element by element under format="mixed" and silently permute the calendar.
    try:
        dates = pd.to_datetime(raw["Date"], format="ISO8601", errors="raise", utc=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Dates must be unambiguous ISO 8601 labels (YYYY-MM-DD): {exc}") from None
    if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Dates must be nonmissing midnight UTC daily labels")
    dates = pd.DatetimeIndex(dates).tz_localize(None)
    if dates.duplicated().any():
        raise ValueError("Duplicate dates: resolve against source; no automatic deduplication")
    if "Currency" in raw:
        currency = raw["Currency"].astype("string").str.strip().str.upper()
        if currency.isna().any() or not currency.eq("USD").all():
            raise ValueError("Only consistently labeled USD input is supported")
        currency_label = "USD"
    else:
        currency_label = "not supplied"
    clean = pd.DataFrame(index=dates.rename("Date"))
    events = []
    absent = [column for column in PRICE_COLUMNS if column not in raw]
    for column in PRICE_COLUMNS:
        if column in absent:
            clean[column] = np.nan
            continue
        values = pd.to_numeric(raw[column], errors="coerce").to_numpy(dtype=float)
        invalid = ~np.isfinite(values) | (values < 0 if column == "Volume" else values <= 0)
        for date in dates[invalid]:
            events.append({"asset": asset, "date": date, "issue": f"invalid_{column.lower()}"})
        clean[column] = np.where(invalid, np.nan, values)
    invalid_close = int(clean.Close.isna().sum())
    if clean.Close.notna().sum() < 2:
        raise ValueError("Fewer than two valid positive closes")
    # Source-row counters are taken before calendar gaps are inserted, so they
    # describe the vendor file rather than duplicating missing_days.
    has_volume = "Volume" not in absent
    invalid_volume = int(clean.Volume.isna().sum()) if has_volume else np.nan
    zero_volume = int(clean.Volume.eq(0).sum()) if has_volume else np.nan
    # Candle inconsistencies are reported independently: a usable positive close
    # is not discarded solely because a separate OHLC field is suspect.
    has_candles = not {"Open", "High", "Low"} & set(absent)
    candle_valid = clean[["Open", "High", "Low", "Close"]].notna().all(axis=1)
    inconsistent = candle_valid & (
        (clean.High < clean[["Open", "Close", "Low"]].max(axis=1))
        | (clean.Low > clean[["Open", "Close", "High"]].min(axis=1))
    )
    for date in clean.index[inconsistent]:
        events.append({"asset": asset, "date": date, "issue": "inconsistent_ohlc"})
    ohlc_invalid = int((~candle_valid | inconsistent).sum()) if has_candles else np.nan
    clean = clean.sort_index()
    calendar = pd.date_range(clean.index.min(), clean.index.max(), freq="D", name="Date")
    missing = calendar.difference(clean.index)
    for date in missing:
        events.append({"asset": asset, "date": date, "issue": "missing_calendar_day"})
    source_rows = len(clean)
    clean["observed"] = True
    clean = clean.reindex(calendar)
    clean["observed"] = clean["observed"].eq(True)
    clean["log_return"] = np.log(clean.Close).diff()
    clean["volatility_30"] = annualized_volatility(clean.log_return)
    record = {
        "asset": asset, "status": "accepted", "source_rows": source_rows,
        "calendar_rows": len(clean), "start": calendar.min(), "end": calendar.max(),
        "missing_days": len(missing), "invalid_close_rows": invalid_close,
        "valid_daily_returns": int(clean.log_return.notna().sum()),
        "valid_30day_windows": int(clean.volatility_30.notna().sum()),
        "zero_volume_rows": zero_volume,
        "missing_or_invalid_volume_rows": invalid_volume,
        "ohlc_invalid_rows": ohlc_invalid,
        "missing_columns": ",".join(absent),
        "currency": currency_label,
        "reason": "Invalid values and calendar gaps retained as missing; no filling",
    }
    return clean, record, events


def load_dataset(path: Path) -> Dataset:
    """Load all direct CSV files; explicitly exclude leaderboard metadata."""
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"Data directory does not exist: {path}")
    frames, records, events, hashes = {}, [], [], {}
    seen = set()
    # The exact name breaks case-insensitive ties, so which of two colliding
    # files is modeled never depends on the filesystem's directory order.
    for file in sorted(path.iterdir(), key=lambda p: (p.name.casefold(), p.name)):
        if not file.is_file() or file.suffix.casefold() != ".csv":
            continue
        contents = file.read_bytes()
        hashes[file.name] = hashlib.sha256(contents).hexdigest()
        if file.name.casefold() == LEADERBOARD:
            records.append({"file": file.name, "asset": file.stem, "status": "skipped",
                            "reason": "Leaderboard is a static snapshot, not daily prices"})
            continue
        try:
            if file.stem.casefold() in seen:
                raise ValueError("Case-insensitive asset-name collision")
            seen.add(file.stem.casefold())
            # Date labels stay text: an all-digit basic ISO date (YYYYMMDD) would
            # otherwise be inferred as an integer and rejected as a numeric timestamp.
            raw = pd.read_csv(io.BytesIO(contents), dtype={"Date": "string"})
            clean, record, row_events = validate_frame(raw, file.stem)
            record["file"] = file.name
            frames[file.stem] = clean
            records.append(record)
            events.extend(row_events)
        except (ValueError, TypeError, KeyError, pd.errors.ParserError) as exc:
            records.append({"file": file.name, "asset": file.stem, "status": "rejected",
                            "reason": f"{type(exc).__name__}: {exc}"})
    return Dataset(frames, pd.DataFrame(records),
                   pd.DataFrame(events, columns=["asset", "date", "issue"]), hashes)
