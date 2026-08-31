from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def load_partition_reference(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing immutable partition reference: {path}")
    ref = json.loads(path.read_text(encoding="utf-8"))
    required = {"train", "validation", "frozen_final"}
    for sym, spec in ref.get("symbols", {}).items():
        if not required.issubset(spec):
            raise ValueError(f"Incomplete partition reference for {sym}")
    return ref


def load_mtf_frames(data_dir: str | Path, symbols: list[str], timeframes: list[str]) -> dict[str, dict[str, pd.DataFrame]]:
    data_dir = Path(data_dir)
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for sym in symbols:
        out[sym] = {}
        for tf in timeframes:
            p = data_dir / f"{sym}_{tf}.parquet"
            if not p.exists():
                raise FileNotFoundError(f"Missing {p}. Run scripts/prepare_frozen_world.py first.")
            df = pd.read_parquet(p).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
            if df.empty:
                raise ValueError(f"Empty market frame: {sym} {tf}")
            out[sym][tf] = df
    return out


def build_frozen_partitions(
    frames: dict[str, dict[str, pd.DataFrame]],
    reference: dict,
    timeframes: list[str],
) -> tuple[dict, dict, dict]:
    """Recreate the exact Generation-1 TRAIN / VALIDATION / FROZEN FINAL eras.

    Generation 2 intentionally does not mature new bars into TRAIN and does not
    switch to rolling validation. The only experimental variable is the action space.
    """
    train, validation, final = {}, {}, {}
    for sym, tfmap in frames.items():
        spec = reference["symbols"][sym]
        train[sym], validation[sym], final[sym] = {}, {}, {}
        for tf in timeframes:
            df = tfmap[tf]
            tr = spec["train"][tf]
            va = spec["validation"][tf]
            fi = spec["frozen_final"][tf]

            train[sym][tf] = df[(df["ts"] >= int(tr["start_ts"])) & (df["ts"] <= int(tr["end_ts"]))].reset_index(drop=True)
            validation[sym][tf] = df[(df["ts"] >= int(va["start_ts"])) & (df["ts"] <= int(va["end_ts"]))].reset_index(drop=True)
            final[sym][tf] = df[(df["ts"] >= int(fi["start_ts"])) & (df["ts"] <= int(fi["end_ts"]))].reset_index(drop=True)

            for name, bucket, expected in (
                ("train", train[sym][tf], tr),
                ("validation", validation[sym][tf], va),
                ("frozen_final", final[sym][tf], fi),
            ):
                if bucket.empty:
                    raise ValueError(f"{sym} {tf} {name} is empty")
                # Exact boundaries are required; row counts are checked too.
                actual = {
                    "rows": int(len(bucket)),
                    "start_ts": int(bucket["ts"].min()),
                    "end_ts": int(bucket["ts"].max()),
                }
                expected_core = {k: int(expected[k]) for k in ("rows", "start_ts", "end_ts")}
                if actual != expected_core:
                    raise ValueError(
                        f"Generation-2 world differs from Generation-1 reference for "
                        f"{sym} {tf} {name}: expected={expected_core}, actual={actual}"
                    )
    return train, validation, final


def partition_summary(parts: dict[str, dict[str, pd.DataFrame]], decision_bar: str) -> dict:
    out = {}
    for sym, tfmap in parts.items():
        out[sym] = {}
        for tf, df in tfmap.items():
            out[sym][tf] = {
                "rows": int(len(df)),
                "start_ts": int(df["ts"].min()),
                "end_ts": int(df["ts"].max()),
            }
        out[sym]["decision_clock"] = decision_bar
    return out
