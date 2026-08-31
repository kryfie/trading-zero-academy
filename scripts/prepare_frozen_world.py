from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pandas as pd

from academy.config import load_config, ROOT
from academy.mtf_data import resample_market
from academy.okx_data import OKXPublicClient, merge_market_and_funding
from academy.splits import load_partition_reference, load_mtf_frames, build_frozen_partitions

cfg = load_config()
m = cfg["market"]
timeframes = list(m["multi_timeframes"])
ref_path = ROOT / m["frozen_partition_reference"]
reference = load_partition_reference(ref_path)
out_dir = ROOT / "data" / "mtf"
out_dir.mkdir(parents=True, exist_ok=True)
client = OKXPublicClient(pause_seconds=0.11)
now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)

summary = {}
for sym in m["symbols"]:
    spec = reference["symbols"][sym]
    start_ts = int(spec.get("source_start_ts", spec["world_start_ts"]))
    end_ts = int(spec.get("source_end_ts", spec["world_end_ts"]))
    m1_path = out_dir / f"{sym}_1m.parquet"
    source = None
    source_mode = None

    if m1_path.exists():
        try:
            cached = pd.read_parquet(m1_path).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
            if not cached.empty and int(cached["ts"].min()) <= start_ts and int(cached["ts"].max()) >= end_ts:
                source = cached
                source_mode = "restored_gen1_or_gen2_cache"
        except Exception:
            source = None

    if source is None:
        # Backfill enough history to cover the immutable Generation-1 interval.
        days = max(2, int(math.ceil((now_ms - start_ts) / 86_400_000)) + 2)
        print(f"{sym}: frozen cache coverage missing; backfilling {days} days from OKX")
        candles = client.candles(sym, bar="1m", days=days, limit=int(m.get("candle_limit_per_request", 300)))
        funding = client.funding_history(sym, days=days, limit=100)
        source = merge_market_and_funding(candles, funding)
        source["symbol"] = sym
        source_mode = "historical_backfill"

    # Keep only the minimal source tape needed to recreate all Gen1 candles.
    # source_end_ts extends beyond the M1 FINAL boundary only so a final H1/H4
    # candle whose *open timestamp* is inside FINAL can be completed correctly.
    frozen = source[(source["ts"] >= start_ts) & (source["ts"] <= end_ts)].copy()
    frozen = frozen.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    if frozen.empty or int(frozen["ts"].min()) != start_ts or int(frozen["ts"].max()) != end_ts:
        raise SystemExit(
            f"{sym}: cannot recreate exact Gen1 M1 world. "
            f"expected {start_ts}..{end_ts}, got "
            f"{None if frozen.empty else int(frozen['ts'].min())}.."
            f"{None if frozen.empty else int(frozen['ts'].max())}"
        )
    frozen["symbol"] = sym
    if "funding_rate" not in frozen.columns:
        frozen["funding_rate"] = 0.0
    frozen["funding_rate"] = frozen["funding_rate"].fillna(0.0)

    for tf in timeframes:
        derived = resample_market(frozen, tf, sym)
        derived.to_parquet(out_dir / f"{sym}_{tf}.parquet", index=False)

    summary[sym] = {
        "source_mode": source_mode,
        "m1_rows": int(len(frozen)),
        "start_ts": int(frozen["ts"].min()),
        "end_ts": int(frozen["ts"].max()),
    }

# Hard verification: every TRAIN/VALIDATION/FINAL row count and boundary must
# match the Generation-1 Cohort #129 reference before any Gen2 student can train.
frames = load_mtf_frames(out_dir, m["symbols"], timeframes)
build_frozen_partitions(frames, reference, timeframes)

print("GENERATION 2 FROZEN WORLD VERIFIED AGAINST GENERATION 1")
print(json.dumps(summary, indent=2))
