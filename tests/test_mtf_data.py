import numpy as np
import pandas as pd

from academy.mtf_data import resample_market


def make_m1(n=120, start=1_700_000_000_000):
    ts = start + np.arange(n) * 60_000
    close = 100 + np.arange(n) * 0.001
    return pd.DataFrame({
        "ts": ts,
        "open": close - 0.01,
        "high": close + 0.02,
        "low": close - 0.02,
        "close": close,
        "volume": np.ones(n),
        "funding_rate": np.zeros(n),
        "symbol": "TEST",
    })


def test_resample_only_complete_buckets_and_preserves_funding():
    m1 = make_m1()
    m1.loc[10, "funding_rate"] = 0.001
    m5 = resample_market(m1, "5m", "TEST")
    assert len(m5) >= 20
    assert abs(float(m5["funding_rate"].sum()) - 0.001) < 1e-12
