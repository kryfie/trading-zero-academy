from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_frames(data_dir: str | Path, symbols: list[str], bar: str) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    frames = {}
    for symbol in symbols:
        p = data_dir / f"{symbol}_{bar}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run scripts/download_data.py first.")
        df = pd.read_parquet(p).sort_values("ts").reset_index(drop=True)
        if len(df) < 500:
            raise ValueError(f"Not enough data for {symbol}: {len(df)} rows")
        frames[symbol] = df
    return frames


def split_frames(frames: dict[str, pd.DataFrame], train_fraction: float, validation_fraction: float):
    train, val, test = {}, {}, {}
    for sym, df in frames.items():
        n = len(df)
        a = int(n * train_fraction)
        b = int(n * (train_fraction + validation_fraction))
        train[sym] = df.iloc[:a].reset_index(drop=True)
        val[sym] = df.iloc[a:b].reset_index(drop=True)
        test[sym] = df.iloc[b:].reset_index(drop=True)
    return train, val, test
