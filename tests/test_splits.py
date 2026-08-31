import pandas as pd

from academy.splits import build_frozen_partitions


def frame(ts_values):
    return pd.DataFrame({
        "ts": ts_values,
        "open": [1.0] * len(ts_values),
        "high": [1.0] * len(ts_values),
        "low": [1.0] * len(ts_values),
        "close": [1.0] * len(ts_values),
        "volume": [1.0] * len(ts_values),
        "funding_rate": [0.0] * len(ts_values),
    })


def test_frozen_partitions_match_reference_exactly():
    frames = {"X": {"5m": frame([1, 2, 3, 4, 5, 6])}}
    ref = {
        "symbols": {
            "X": {
                "train": {"5m": {"rows": 2, "start_ts": 1, "end_ts": 2}},
                "validation": {"5m": {"rows": 2, "start_ts": 3, "end_ts": 4}},
                "frozen_final": {"5m": {"rows": 2, "start_ts": 5, "end_ts": 6}},
            }
        }
    }
    tr, va, fi = build_frozen_partitions(frames, ref, ["5m"])
    assert tr["X"]["5m"]["ts"].tolist() == [1, 2]
    assert va["X"]["5m"]["ts"].tolist() == [3, 4]
    assert fi["X"]["5m"]["ts"].tolist() == [5, 6]
