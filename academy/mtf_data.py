from __future__ import annotations

from pathlib import Path
from typing import Iterable
import math

import pandas as pd

from .okx_data import OKXPublicClient, merge_market_and_funding, bar_to_milliseconds

M1_MS = 60_000


def _merge(existing: pd.DataFrame, newer: pd.DataFrame, symbol: str) -> pd.DataFrame:
    parts = [x for x in (existing, newer) if x is not None and not x.empty]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False)
    out = out.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    out["symbol"] = symbol
    if "funding_rate" not in out.columns:
        out["funding_rate"] = 0.0
    out["funding_rate"] = out["funding_rate"].fillna(0.0)
    return out


def _gap_candles(client: OKXPublicClient, symbol: str, last_cached_ts: int, limit: int = 300) -> pd.DataFrame:
    """Fetch only the missing 1m bridge from now backwards to the cached archive."""
    cursor = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    rows = []
    seen = set()
    while cursor > last_cached_ts + M1_MS:
        data = client._get(
            "/api/v5/market/history-candles",
            {"instId": symbol, "bar": "1m", "after": str(cursor), "limit": str(limit)},
        )
        if not data:
            break
        oldest = cursor
        for row in data:
            ts = int(row[0])
            if ts not in seen:
                seen.add(ts)
                rows.append(row)
            oldest = min(oldest, ts)
        if oldest >= cursor:
            break
        if oldest <= last_cached_ts + M1_MS:
            break
        cursor = oldest - 1

    # Reuse the public parser semantics without importing a private symbol:
    if not rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    schema = ["ts", "open", "high", "low", "close", "volume", "vol_ccy", "vol_quote", "confirm"]
    width = max(len(r) for r in rows)
    df = pd.DataFrame(rows, columns=schema[:width])
    if "confirm" in df.columns:
        df = df[df["confirm"].astype(str) == "1"]
    for c in ["open", "high", "low", "close", "volume", "ts"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = df["ts"].astype("int64")
    return (
        df[["ts", "open", "high", "low", "close", "volume"]]
        .sort_values("ts")
        .drop_duplicates("ts", keep="last")
        .reset_index(drop=True)
    )


def refresh_m1_symbol(
    symbol: str,
    days: int,
    out_dir: str | Path,
    candle_limit: int = 300,
    client: OKXPublicClient | None = None,
) -> dict:
    """Bootstrap M1 once, then bridge only the missing gap on later runs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_1m.parquet"
    client = client or OKXPublicClient(pause_seconds=0.11)

    if not path.exists():
        candles = client.candles(symbol, bar="1m", days=days, limit=candle_limit)
        funding = client.funding_history(symbol, days=days, limit=100)
        merged = merge_market_and_funding(candles, funding)
        merged["symbol"] = symbol
        merged.to_parquet(path, index=False)
        return {"symbol": symbol, "mode": "full_m1_bootstrap", "bars": len(merged), "new_bars": len(merged)}

    cached = pd.read_parquet(path).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    if cached.empty:
        path.unlink(missing_ok=True)
        return refresh_m1_symbol(symbol, days, out_dir, candle_limit, client)

    last_ts = int(cached["ts"].max())
    recent = client.latest_candles(symbol, bar="1m", limit=300)
    first_recent = int(recent["ts"].min())

    if first_recent <= last_ts + M1_MS:
        candles = recent
    else:
        candles = _gap_candles(client, symbol, last_ts, limit=candle_limit)

    # Funding is sparse. Pull enough history to cover the gap plus a safety day.
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    gap_days = max(2, int(math.ceil((now_ms - last_ts) / 86_400_000)) + 1)
    funding = client.funding_history(symbol, days=min(days, gap_days), limit=100)
    newer = merge_market_and_funding(candles, funding)
    newer["symbol"] = symbol

    combined = _merge(cached, newer, symbol)
    combined.to_parquet(path, index=False)
    return {
        "symbol": symbol,
        "mode": "incremental_m1",
        "bars": int(len(combined)),
        "new_bars": int((combined["ts"] > last_ts).sum()),
    }


def resample_market(df: pd.DataFrame, bar: str, symbol: str) -> pd.DataFrame:
    """Build higher raw OHLCV scales from the same M1 tape.

    Funding events are summed into the target candle, preserving the cash-flow event.
    """
    if bar == "1m":
        out = df.copy()
        out["symbol"] = symbol
        return out.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    rule_map = {"5m": "5min", "15m": "15min", "1H": "1h", "4H": "4h"}
    if bar not in rule_map:
        raise ValueError(f"Unsupported Market Student timeframe: {bar}")

    x = df.copy()
    x["dt"] = pd.to_datetime(x["ts"], unit="ms", utc=True)
    x = x.set_index("dt").sort_index()

    grouped = x.resample(rule_map[bar], label="left", closed="left", origin="epoch")
    agg = grouped.agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "funding_rate": "sum",
    })
    # Only fully closed/complete higher-timeframe candles are admitted.
    # This removes the partial bucket at the beginning/end of the M1 archive and
    # also protects against a missing M1 candle silently creating a fake full bar.
    expected = {"5m": 5, "15m": 15, "1H": 60, "4H": 240}[bar]
    counts = grouped["close"].count()
    agg = agg[counts == expected]
    agg = agg.dropna(subset=["open", "high", "low", "close"])
    agg["ts"] = (agg.index.astype("int64") // 1_000_000).astype("int64")
    agg["symbol"] = symbol
    return agg.reset_index(drop=True)[
        ["ts", "open", "high", "low", "close", "volume", "funding_rate", "symbol"]
    ]


def refresh_multitimeframe_universe(
    symbols: Iterable[str],
    timeframes: list[str],
    days: int,
    out_dir: str | Path,
    candle_limit: int = 300,
) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = OKXPublicClient(pause_seconds=0.11)
    results = []

    for symbol in symbols:
        r = refresh_m1_symbol(symbol, days, out_dir, candle_limit=candle_limit, client=client)
        source = pd.read_parquet(out_dir / f"{symbol}_1m.parquet")
        for tf in timeframes:
            derived = resample_market(source, tf, symbol)
            derived.to_parquet(out_dir / f"{symbol}_{tf}.parquet", index=False)
        r["timeframes"] = list(timeframes)
        results.append(r)
        print(f"{symbol}: {r['mode']}, +{r['new_bars']} M1 bars, {r['bars']:,} M1 bars total")
    return results
