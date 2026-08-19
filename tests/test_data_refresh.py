import pandas as pd

from academy.okx_data import merge_market_and_funding, bar_to_milliseconds
from academy.dataset import create_split_manifest, split_frames


def frame(start=0, n=1000, step=300000):
    ts = [start + i * step for i in range(n)]
    return pd.DataFrame({
        "ts": ts,
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.0] * n,
        "volume": [10.0] * n,
        "funding_rate": [0.0] * n,
        "symbol": ["TEST"] * n,
    })


def test_bar_to_milliseconds():
    assert bar_to_milliseconds("5m") == 300000
    assert bar_to_milliseconds("1H") == 3600000


def test_funding_maps_to_first_bar_at_or_after_event():
    candles = frame(n=5).drop(columns=["funding_rate", "symbol"])
    funding = pd.DataFrame({"funding_time": [450000], "funding_rate": [0.001]})
    merged = merge_market_and_funding(candles, funding)
    assert merged.loc[2, "funding_rate"] == 0.001


def test_frozen_split_does_not_move_when_new_rows_arrive(tmp_path):
    original = {"TEST": frame(n=1000)}
    manifest = tmp_path / "split_manifest.json"
    create_split_manifest(original, 0.70, 0.15, manifest)
    train1, val1, test1 = split_frames(original, 0.70, 0.15, manifest)

    extended = {"TEST": frame(n=1100)}
    train2, val2, test2 = split_frames(extended, 0.70, 0.15, manifest)

    assert len(train2["TEST"]) == len(train1["TEST"])
    assert len(val2["TEST"]) == len(val1["TEST"])
    assert len(test2["TEST"]) == len(test1["TEST"]) + 100
