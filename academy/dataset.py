from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


SPLIT_MANIFEST_NAME = "split_manifest.json"


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
    """Freeze time boundaries once so later refreshes cannot move the final holdout into training."""
    manifest = {"version": 1, "symbols": {}}
    for sym, df in frames.items():
        n = len(df)
        a = int(n * train_fraction)
        b = int(n * (train_fraction + validation_fraction))
        if a < 1 or b <= a or b >= n:
            raise ValueError(f"Invalid split for {sym}: n={n}, a={a}, b={b}")
        manifest["symbols"][sym] = {
            "train_end_ts": int(df.iloc[a - 1]["ts"]),
            "validation_end_ts": int(df.iloc[b - 1]["ts"]),
            "created_from_rows": int(n),
        }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def ensure_split_manifest(
    frames: dict[str, pd.DataFrame],
    train_fraction: float,
    validation_fraction: float,
    path: str | Path,
) -> dict:
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return create_split_manifest(frames, train_fraction, validation_fraction, path)


def split_frames(
    frames: dict[str, pd.DataFrame],
    train_fraction: float,
    validation_fraction: float,
    manifest_path: str | Path | None = None,
):
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
        train[sym] = df[df["ts"] <= train_end].reset_index(drop=True)
        val[sym] = df[(df["ts"] > train_end) & (df["ts"] <= validation_end)].reset_index(drop=True)
        test[sym] = df[df["ts"] > validation_end].reset_index(drop=True)

        if min(len(train[sym]), len(val[sym]), len(test[sym])) == 0:
            raise ValueError(f"Frozen split produced an empty partition for {sym}")

    return train, val, test
