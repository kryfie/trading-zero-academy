from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


SPLIT_MANIFEST_NAME = "split_manifest.json"
MANIFEST_VERSION = 2


def load_frames(data_dir: str | Path, symbols: list[str], bar: str) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    frames = {}
    for symbol in symbols:
        p = data_dir / f"{symbol}_{bar}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run scripts/download_data.py first.")
        df = pd.read_parquet(p).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
        if len(df) < 500:
            raise ValueError(f"Not enough data for {symbol}: {len(df)} rows")
        frames[symbol] = df
    return frames


def create_split_manifest(
    frames: dict[str, pd.DataFrame],
    train_fraction: float,
    validation_fraction: float,
    path: str | Path,
) -> dict:
    """Create immutable boundaries for the original TRAIN/VALIDATION/FINAL world.

    FINAL is bounded by bootstrap_end_ts. New bars arriving after Academy launch can
    therefore never silently become part of the frozen final exam.
    """
    manifest = {"version": MANIFEST_VERSION, "symbols": {}}
    for sym, df in frames.items():
        n = len(df)
        a = int(n * train_fraction)
        b = int(n * (train_fraction + validation_fraction))
        if a < 1 or b <= a or b >= n:
            raise ValueError(f"Invalid split for {sym}: n={n}, a={a}, b={b}")
        manifest["symbols"][sym] = {
            "bootstrap_start_ts": int(df.iloc[0]["ts"]),
            "train_end_ts": int(df.iloc[a - 1]["ts"]),
            "validation_end_ts": int(df.iloc[b - 1]["ts"]),
            "bootstrap_end_ts": int(df.iloc[-1]["ts"]),
            "created_from_rows": int(n),
        }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _upgrade_v1_manifest(
    manifest: dict,
    frames: dict[str, pd.DataFrame],
    train_fraction: float,
    validation_fraction: float,
) -> dict:
    """Upgrade v0.2.1 manifest without moving its already-frozen final holdout.

    v1 stored train_end/validation_end and original row count, but not bootstrap_end.
    The original FINAL rows are the first `final_rows` rows after validation_end.
    Newer bars were only appended, so this reconstructs the original final boundary.
    """
    upgraded = {"version": MANIFEST_VERSION, "symbols": {}}
    for sym, old in manifest.get("symbols", {}).items():
        if sym not in frames:
            raise KeyError(f"{sym} missing while upgrading split manifest")
        df = frames[sym].sort_values("ts").reset_index(drop=True)
        n0 = int(old["created_from_rows"])
        b0 = int(n0 * (train_fraction + validation_fraction))
        final_rows = max(1, n0 - b0)
        validation_end = int(old["validation_end_ts"])
        after_validation = df[df["ts"] > validation_end]
        if len(after_validation) < final_rows:
            raise ValueError(
                f"Cannot safely reconstruct frozen final for {sym}: "
                f"need {final_rows} rows after validation boundary, have {len(after_validation)}"
            )
        bootstrap_end = int(after_validation.iloc[final_rows - 1]["ts"])
        bootstrap_rows = df[df["ts"] <= bootstrap_end]
        upgraded["symbols"][sym] = {
            "bootstrap_start_ts": int(bootstrap_rows.iloc[0]["ts"]),
            "train_end_ts": int(old["train_end_ts"]),
            "validation_end_ts": validation_end,
            "bootstrap_end_ts": bootstrap_end,
            "created_from_rows": n0,
        }
    return upgraded


def ensure_split_manifest(
    frames: dict[str, pd.DataFrame],
    train_fraction: float,
    validation_fraction: float,
    path: str | Path,
) -> dict:
    path = Path(path)
    if not path.exists():
        return create_split_manifest(frames, train_fraction, validation_fraction, path)

    manifest = json.loads(path.read_text(encoding="utf-8"))
    if int(manifest.get("version", 1)) < MANIFEST_VERSION:
        manifest = _upgrade_v1_manifest(manifest, frames, train_fraction, validation_fraction)
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _concat_parts(*parts: pd.DataFrame) -> pd.DataFrame:
    keep = [p for p in parts if p is not None and not p.empty]
    if not keep:
        return pd.DataFrame()
    return (
        pd.concat(keep, ignore_index=True)
        .sort_values("ts")
        .drop_duplicates("ts", keep="last")
        .reset_index(drop=True)
    )


def build_academy_partitions(
    frames: dict[str, pd.DataFrame],
    manifest: dict,
    live_shadow_days: int,
    train_maturation_days: int,
    min_rolling_validation_bars: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame], str]:
    """Build leakage-safe dynamic partitions.

    Original bootstrap:
      - first 70%: permanent base TRAIN
      - next 15%: frozen fallback VALIDATION
      - final 15%: FROZEN FINAL TEST forever

    Data created after bootstrap_end_ts:
      - newest live_shadow_days: LIVE SHADOW
      - older than live_shadow but younger than train_maturation_days: rolling VALIDATION
      - age >= train_maturation_days: added to TRAIN

    The frozen final interval is never released to training.
    """
    if train_maturation_days <= live_shadow_days:
        raise ValueError("train_maturation_days must be greater than live_shadow_days")

    day_ms = 86_400_000
    train: dict[str, pd.DataFrame] = {}
    fixed_val: dict[str, pd.DataFrame] = {}
    rolling_val: dict[str, pd.DataFrame] = {}
    final: dict[str, pd.DataFrame] = {}
    live: dict[str, pd.DataFrame] = {}

    rolling_ready = True
    for sym, df in frames.items():
        cuts = manifest.get("symbols", {}).get(sym)
        if cuts is None:
            raise KeyError(f"{sym} missing from split manifest")

        train_end = int(cuts["train_end_ts"])
        validation_end = int(cuts["validation_end_ts"])
        bootstrap_end = int(cuts["bootstrap_end_ts"])
        latest_ts = int(df["ts"].max())

        live_cutoff = latest_ts - int(live_shadow_days) * day_ms
        maturity_cutoff = latest_ts - int(train_maturation_days) * day_ms

        base_train = df[df["ts"] <= train_end]
        fixed_val[sym] = df[(df["ts"] > train_end) & (df["ts"] <= validation_end)].reset_index(drop=True)
        final[sym] = df[(df["ts"] > validation_end) & (df["ts"] <= bootstrap_end)].reset_index(drop=True)

        post = df[df["ts"] > bootstrap_end]
        matured = post[post["ts"] <= maturity_cutoff]
        rv = post[(post["ts"] > maturity_cutoff) & (post["ts"] <= live_cutoff)].reset_index(drop=True)
        lv = post[post["ts"] > live_cutoff].reset_index(drop=True)

        train[sym] = _concat_parts(base_train, matured)
        rolling_val[sym] = rv
        live[sym] = lv

        if len(rv) < int(min_rolling_validation_bars):
            rolling_ready = False
        if min(len(train[sym]), len(fixed_val[sym]), len(final[sym])) == 0:
            raise ValueError(f"Academy partition produced an empty core partition for {sym}")

    validation_source = "ROLLING_VALIDATION" if rolling_ready else "FROZEN_BOOTSTRAP_VALIDATION"
    validation = rolling_val if rolling_ready else fixed_val
    return train, validation, final, live, validation_source


def split_frames(
    frames: dict[str, pd.DataFrame],
    train_fraction: float,
    validation_fraction: float,
    manifest_path: str | Path | None = None,
):
    """Backward-compatible static splitter used by older callers/tests.

    With a v2 manifest, FINAL is bounded at bootstrap_end and therefore remains frozen.
    """
    train, val, test = {}, {}, {}
    manifest = None
    if manifest_path is not None and Path(manifest_path).exists():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    for sym, df in frames.items():
        if manifest is None:
            n = len(df)
            a = int(n * train_fraction)
            b = int(n * (train_fraction + validation_fraction))
            train[sym] = df.iloc[:a].reset_index(drop=True)
            val[sym] = df.iloc[a:b].reset_index(drop=True)
            test[sym] = df.iloc[b:].reset_index(drop=True)
            continue

        if sym not in manifest.get("symbols", {}):
            raise KeyError(f"{sym} missing from split manifest")
        cuts = manifest["symbols"][sym]
        train_end = int(cuts["train_end_ts"])
        validation_end = int(cuts["validation_end_ts"])
        bootstrap_end = int(cuts.get("bootstrap_end_ts", df["ts"].max()))
        train[sym] = df[df["ts"] <= train_end].reset_index(drop=True)
        val[sym] = df[(df["ts"] > train_end) & (df["ts"] <= validation_end)].reset_index(drop=True)
        test[sym] = df[(df["ts"] > validation_end) & (df["ts"] <= bootstrap_end)].reset_index(drop=True)

        if min(len(train[sym]), len(val[sym]), len(test[sym])) == 0:
            raise ValueError(f"Frozen split produced an empty partition for {sym}")

    return train, val, test


def partition_summary(parts: dict[str, pd.DataFrame]) -> dict:
    out = {}
    for sym, df in parts.items():
        if df.empty:
            out[sym] = {"rows": 0, "start_ts": None, "end_ts": None}
        else:
            out[sym] = {
                "rows": int(len(df)),
                "start_ts": int(df["ts"].min()),
                "end_ts": int(df["ts"].max()),
            }
    return out
