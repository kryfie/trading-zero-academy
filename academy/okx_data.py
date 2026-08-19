from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://www.okx.com"


def bar_to_milliseconds(bar: str) -> int:
    """Convert the OKX bar string used by the Academy to milliseconds."""
    value = bar.strip()
    if not value:
        raise ValueError("bar cannot be empty")
    unit = value[-1]
    number = int(value[:-1])
    if unit == "m":
        return number * 60_000
    if unit in {"H", "h"}:
        return number * 3_600_000
    if unit in {"D", "d"}:
        return number * 86_400_000
    raise ValueError(f"Unsupported bar size: {bar}")


def _parse_candles(rows: list[list[str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    width = max(len(r) for r in rows)
    schema = ["ts", "open", "high", "low", "close", "volume", "vol_ccy", "vol_quote", "confirm"]
    cols = schema[:width]
    df = pd.DataFrame(rows, columns=cols)

    # Never train on a still-forming candle. OKX uses confirm=1 for a completed candle.
    if "confirm" in df.columns:
        df = df[df["confirm"].astype(str) == "1"]

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = df["ts"].astype("int64")
    return (
        df[["ts", "open", "high", "low", "close", "volume"]]
        .sort_values("ts")
        .drop_duplicates("ts", keep="last")
        .reset_index(drop=True)
    )


@dataclass
class OKXPublicClient:
    timeout: int = 20
    pause_seconds: float = 0.12

    def _get(self, path: str, params: dict) -> list:
        r = requests.get(BASE_URL + path, params=params, timeout=self.timeout)
        r.raise_for_status()
        payload = r.json()
        if str(payload.get("code", "0")) != "0":
            raise RuntimeError(f"OKX error {payload.get('code')}: {payload.get('msg')}")
        time.sleep(self.pause_seconds)
        return payload.get("data", [])

    def candles(self, inst_id: str, bar: str, days: int, limit: int = 100) -> pd.DataFrame:
        """Full bootstrap/backfill. This is intentionally used only when no usable cache exists."""
        end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        start_ms = int((pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).timestamp() * 1000)
        rows: list[list[str]] = []
        cursor = end_ms
        seen = set()
        while cursor > start_ms:
            data = self._get(
                "/api/v5/market/history-candles",
                {"instId": inst_id, "bar": bar, "after": str(cursor), "limit": str(limit)},
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
            cursor = oldest - 1
            if oldest <= start_ms:
                break

        df = _parse_candles(rows)
        if df.empty:
            raise RuntimeError(f"No candle data returned for {inst_id}")
        return df[df["ts"] >= start_ms].reset_index(drop=True)

    def latest_candles(self, inst_id: str, bar: str, limit: int = 300) -> pd.DataFrame:
        """Fetch only the newest candles for fast incremental refreshes."""
        data = self._get(
            "/api/v5/market/candles",
            {"instId": inst_id, "bar": bar, "limit": str(limit)},
        )
        df = _parse_candles(data)
        if df.empty:
            raise RuntimeError(f"No recent candle data returned for {inst_id}")
        return df

    def funding_history(self, inst_id: str, days: int, limit: int = 100) -> pd.DataFrame:
        end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        start_ms = int((pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).timestamp() * 1000)
        rows: list[dict] = []
        cursor = end_ms
        seen = set()
        while cursor > start_ms:
            data = self._get(
                "/api/v5/public/funding-rate-history",
                {"instId": inst_id, "after": str(cursor), "limit": str(limit)},
            )
            if not data:
                break
            oldest = cursor
            for row in data:
                ts = int(row["fundingTime"])
                if ts not in seen:
                    seen.add(ts)
                    rows.append(row)
                oldest = min(oldest, ts)
            if oldest >= cursor:
                break
            cursor = oldest - 1
            if oldest <= start_ms:
                break

        return _parse_funding(rows, start_ms=start_ms)

    def latest_funding_history(self, inst_id: str, limit: int = 100) -> pd.DataFrame:
        data = self._get(
            "/api/v5/public/funding-rate-history",
            {"instId": inst_id, "limit": str(limit)},
        )
        return _parse_funding(data)


def _parse_funding(rows: list[dict], start_ms: int | None = None) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["funding_time", "funding_rate"])
    df = pd.DataFrame(rows)
    if "fundingTime" not in df.columns or "fundingRate" not in df.columns:
        return pd.DataFrame(columns=["funding_time", "funding_rate"])
    df["funding_time"] = pd.to_numeric(df["fundingTime"], errors="coerce")
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["funding_time"])
    df["funding_time"] = df["funding_time"].astype("int64")
    if start_ms is not None:
        df = df[df["funding_time"] >= start_ms]
    return (
        df[["funding_time", "funding_rate"]]
        .sort_values("funding_time")
        .drop_duplicates("funding_time", keep="last")
        .reset_index(drop=True)
    )


def merge_market_and_funding(candles: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    out = candles.copy().sort_values("ts").reset_index(drop=True)
    out["funding_rate"] = 0.0
    if not funding.empty and not out.empty:
        funding_map = dict(zip(funding["funding_time"].astype(int), funding["funding_rate"].astype(float)))
        # Funding is charged at discrete settlement timestamps. Put the event on the first bar at/after settlement.
        cts = out["ts"].to_numpy()
        for fts, rate in funding_map.items():
            idx = np.searchsorted(cts, fts, side="left")
            if idx < len(out):
                out.loc[out.index[idx], "funding_rate"] += rate
    return out


def _full_symbol_download(
    client: OKXPublicClient,
    symbol: str,
    bar: str,
    days: int,
    path: Path,
    candle_limit: int,
) -> dict:
    candles = client.candles(symbol, bar=bar, days=days, limit=candle_limit)
    funding = client.funding_history(symbol, days=days)
    merged = merge_market_and_funding(candles, funding)
    merged["symbol"] = symbol

    # Never throw away older Academy history. A recovery/backfill only repairs or
    # extends the world; it does not replace the accumulated historical archive.
    old_len = 0
    if path.exists():
        try:
            cached = pd.read_parquet(path).sort_values("ts").drop_duplicates("ts", keep="last")
            old_len = len(cached)
            merged = pd.concat([cached, merged], ignore_index=True, sort=False)
            merged = merged.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
            merged["symbol"] = symbol
        except Exception:
            pass

    merged.to_parquet(path, index=False)
    return {
        "symbol": symbol,
        "mode": "full",
        "bars": int(len(merged)),
        "new_bars": int(max(0, len(merged) - old_len)),
        "funding_events": int((merged["funding_rate"] != 0).sum()),
    }


def refresh_symbol(
    symbol: str,
    bar: str,
    days: int,
    out_dir: str | Path,
    client: OKXPublicClient | None = None,
    candle_limit: int = 100,
    recent_limit: int = 300,
) -> dict:
    """
    Fast path: merge the latest OKX candles/funding into the cached parquet.
    Safe path: if the cache is missing/stale enough that recent candles do not overlap it,
    perform a full historical backfill instead of leaving a hole.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_{bar}.parquet"
    client = client or OKXPublicClient()

    if not path.exists():
        return _full_symbol_download(client, symbol, bar, days, path, candle_limit)

    try:
        cached = pd.read_parquet(path).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    except Exception:
        return _full_symbol_download(client, symbol, bar, days, path, candle_limit)

    if cached.empty or "ts" not in cached.columns:
        return _full_symbol_download(client, symbol, bar, days, path, candle_limit)

    latest = client.latest_candles(symbol, bar=bar, limit=recent_limit)
    bar_ms = bar_to_milliseconds(bar)
    last_cached_ts = int(cached["ts"].max())
    first_recent_ts = int(latest["ts"].min())

    # An overlap proves that the recent window bridges the cached world to "now".
    # Without overlap we intentionally choose correctness over speed and backfill again.
    if first_recent_ts > last_cached_ts + bar_ms:
        return _full_symbol_download(client, symbol, bar, days, path, candle_limit)

    funding = client.latest_funding_history(symbol, limit=100)
    recent = merge_market_and_funding(latest, funding)
    recent["symbol"] = symbol

    old_len = len(cached)
    combined = pd.concat([cached, recent], ignore_index=True, sort=False)
    combined = combined.sort_values("ts").drop_duplicates("ts", keep="last")

    # Keep the entire accumulated Academy history. Old training data must not
    # disappear just because the live market moves forward.
    combined = combined.reset_index(drop=True)
    combined["symbol"] = symbol
    combined.to_parquet(path, index=False)

    new_bars = int((combined["ts"] > last_cached_ts).sum())
    gap_minutes = max(0.0, (int(latest["ts"].max()) - last_cached_ts) / 60_000.0)
    return {
        "symbol": symbol,
        "mode": "incremental",
        "bars": int(len(combined)),
        "new_bars": new_bars,
        "cached_bars_before": int(old_len),
        "gap_minutes": round(gap_minutes, 1),
        "funding_events": int((combined["funding_rate"] != 0).sum()),
    }


def refresh_universe(
    symbols: Iterable[str],
    bar: str,
    days: int,
    out_dir: str | Path,
    candle_limit: int = 100,
) -> list[dict]:
    client = OKXPublicClient()
    results = []
    for symbol in symbols:
        result = refresh_symbol(
            symbol,
            bar=bar,
            days=days,
            out_dir=out_dir,
            client=client,
            candle_limit=candle_limit,
        )
        results.append(result)
        if result["mode"] == "incremental":
            print(
                f"{symbol}: incremental refresh, +{result['new_bars']} bars, "
                f"{result['bars']:,} cached bars total, gap={result['gap_minutes']}m"
            )
        else:
            print(
                f"{symbol}: full bootstrap/backfill, {result['bars']:,} bars, "
                f"{result['funding_events']} funding events"
            )
    return results


def download_universe(symbols: Iterable[str], bar: str, days: int, out_dir: str | Path) -> None:
    """Backward-compatible name used by older callers."""
    refresh_universe(symbols, bar, days, out_dir)
