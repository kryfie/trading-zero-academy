from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_mtf_frames(
    data_dir: str | Path,
    symbols: list[str],
    timeframes: list[str],
) -> dict[str, dict[str, pd.DataFrame]]:
    data_dir = Path(data_dir)
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol in symbols:
        out[symbol] = {}
        for tf in timeframes:
            p = data_dir / f"{symbol}_{tf}.parquet"
            if not p.exists():
                raise FileNotFoundError(f"Missing {p}. Run scripts/download_mtf_data.py first.")
            df = pd.read_parquet(p).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
            if len(df) < 100:
                raise ValueError(f"Not enough {tf} data for {symbol}: {len(df)} rows")
            out[symbol][tf] = df
    return out


def _slice(df: pd.DataFrame, start_exclusive: int | None = None, end_inclusive: int | None = None):
    mask = pd.Series(True, index=df.index)
    if start_exclusive is not None:
        mask &= df["ts"] > int(start_exclusive)
    if end_inclusive is not None:
        mask &= df["ts"] <= int(end_inclusive)
    return df[mask].reset_index(drop=True)


def build_mtf_partitions(
    frames: dict[str, dict[str, pd.DataFrame]],
    legacy_manifest: dict,
    decision_bar: str,
    live_shadow_days: int,
    train_maturation_days: int,
    min_rolling_validation_bars: int,
):
    """Apply Student #1's immutable date boundaries to every timeframe.

    This makes the comparison fair: M5 baseline and Market Students use the same
    historical eras for TRAIN / validation / FROZEN FINAL.
    """
    if train_maturation_days <= live_shadow_days:
        raise ValueError("train_maturation_days must be greater than live_shadow_days")

    day_ms = 86_400_000
    train, fixed_val, rolling_val, final, live = {}, {}, {}, {}, {}
    rolling_ready = True

    for sym, tfmap in frames.items():
        cuts = legacy_manifest["symbols"][sym]
        train_end = int(cuts["train_end_ts"])
        validation_end = int(cuts["validation_end_ts"])
        bootstrap_end = int(cuts["bootstrap_end_ts"])
        latest_ts = int(tfmap[decision_bar]["ts"].max())
        live_cutoff = latest_ts - int(live_shadow_days) * day_ms
        maturity_cutoff = latest_ts - int(train_maturation_days) * day_ms

        train[sym], fixed_val[sym], rolling_val[sym], final[sym], live[sym] = {}, {}, {}, {}, {}
        for tf, df in tfmap.items():
            base_train = _slice(df, end_inclusive=train_end)
            matured = df[(df["ts"] > bootstrap_end) & (df["ts"] <= maturity_cutoff)].reset_index(drop=True)
            train[sym][tf] = (
                pd.concat([base_train, matured], ignore_index=True)
                .sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
            )
            fixed_val[sym][tf] = df[(df["ts"] > train_end) & (df["ts"] <= validation_end)].reset_index(drop=True)
            final[sym][tf] = df[(df["ts"] > validation_end) & (df["ts"] <= bootstrap_end)].reset_index(drop=True)
            rolling_val[sym][tf] = df[
                (df["ts"] > max(bootstrap_end, maturity_cutoff)) & (df["ts"] <= live_cutoff)
            ].reset_index(drop=True)
            live[sym][tf] = df[df["ts"] > live_cutoff].reset_index(drop=True)

        # Readiness is defined on the decision clock, matching the M5 baseline.
        if len(rolling_val[sym][decision_bar]) < int(min_rolling_validation_bars):
            rolling_ready = False
        for bucket in (train[sym], fixed_val[sym], final[sym]):
            if any(df.empty for df in bucket.values()):
                raise ValueError(f"MTF partition produced an empty core partition for {sym}")

    validation_source = "ROLLING_VALIDATION" if rolling_ready else "FROZEN_BOOTSTRAP_VALIDATION"
    validation = rolling_val if rolling_ready else fixed_val
    return train, validation, final, live, validation_source


def mtf_partition_summary(parts, decision_bar: str):
    out = {}
    for sym, tfmap in parts.items():
        out[sym] = {}
        for tf, df in tfmap.items():
            out[sym][tf] = {
                "rows": int(len(df)),
                "start_ts": None if df.empty else int(df["ts"].min()),
                "end_ts": None if df.empty else int(df["ts"].max()),
            }
        out[sym]["decision_clock"] = decision_bar
    return out
