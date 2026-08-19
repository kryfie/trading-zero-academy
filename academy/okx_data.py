from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://www.okx.com"


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

        if not rows:
            raise RuntimeError(f"No candle data returned for {inst_id}")
        # OKX candle schema: ts,o,h,l,c,vol,volCcy,volCcyQuote,confirm
        width = max(len(r) for r in rows)
        cols = ["ts", "open", "high", "low", "close", "volume", "vol_ccy", "vol_quote", "confirm"][:width]
        df = pd.DataFrame(rows, columns=cols)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("int64")
        df = df[df["ts"] >= start_ms].sort_values("ts").drop_duplicates("ts")
        return df[["ts", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

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

        if not rows:
            return pd.DataFrame(columns=["funding_time", "funding_rate"])
        df = pd.DataFrame(rows)
        df["funding_time"] = pd.to_numeric(df["fundingTime"], errors="coerce").astype("int64")
        df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce").fillna(0.0)
        return df[df["funding_time"] >= start_ms][["funding_time", "funding_rate"]].sort_values("funding_time").reset_index(drop=True)


def merge_market_and_funding(candles: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    out = candles.copy().sort_values("ts")
    out["funding_rate"] = 0.0
    if not funding.empty:
        funding_map = dict(zip(funding["funding_time"].astype(int), funding["funding_rate"].astype(float)))
        # Funding is charged at discrete settlement timestamps. Put the event on the first bar at/after settlement.
        cts = out["ts"].to_numpy()
        for fts, rate in funding_map.items():
            idx = np.searchsorted(cts, fts, side="left")
            if idx < len(out):
                out.loc[out.index[idx], "funding_rate"] += rate
    return out


def download_universe(symbols: Iterable[str], bar: str, days: int, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = OKXPublicClient()
    for symbol in symbols:
        candles = client.candles(symbol, bar=bar, days=days)
        funding = client.funding_history(symbol, days=days)
        merged = merge_market_and_funding(candles, funding)
        merged["symbol"] = symbol
        merged.to_parquet(out_dir / f"{symbol}_{bar}.parquet", index=False)
        print(f"{symbol}: {len(merged):,} bars, {int((merged.funding_rate != 0).sum())} funding events")
