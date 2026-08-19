import json
import pandas as pd

from academy.okx_data import merge_market_and_funding, bar_to_milliseconds
from academy.dataset import (
    create_split_manifest,
    ensure_split_manifest,
    split_frames,
    build_academy_partitions,
)

DAY = 86_400_000
BAR = 300_000


def frame(start=0, n=1000, step=BAR):
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


def test_frozen_final_does_not_absorb_new_rows(tmp_path):
    original = {"TEST": frame(n=1000)}
    manifest = tmp_path / "split_manifest.json"
    create_split_manifest(original, 0.70, 0.15, manifest)
    _, _, test1 = split_frames(original, 0.70, 0.15, manifest)

    extended = {"TEST": frame(n=1100)}
    _, _, test2 = split_frames(extended, 0.70, 0.15, manifest)
    assert len(test2["TEST"]) == len(test1["TEST"])


def test_v1_manifest_upgrade_reconstructs_original_bootstrap_end(tmp_path):
    original = frame(n=1000)
    a, b = 700, 850
    v1 = {
        "version": 1,
        "symbols": {
            "TEST": {
                "train_end_ts": int(original.iloc[a - 1]["ts"]),
                "validation_end_ts": int(original.iloc[b - 1]["ts"]),
                "created_from_rows": 1000,
            }
        },
    }
    p = tmp_path / "split_manifest.json"
    p.write_text(json.dumps(v1), encoding="utf-8")
    extended = {"TEST": frame(n=1050)}
    upgraded = ensure_split_manifest(extended, 0.70, 0.15, p)
    assert upgraded["version"] == 2
    assert upgraded["symbols"]["TEST"]["bootstrap_end_ts"] == int(original.iloc[-1]["ts"])


def test_conveyor_belt_never_trains_live_or_validation(tmp_path):
    # Bootstrap 60 days, then 40 new post-launch days at daily resolution for simple math.
    n0 = 60
    original = {"TEST": frame(start=0, n=n0, step=DAY)}
    p = tmp_path / "split_manifest.json"
    manifest = create_split_manifest(original, 0.70, 0.15, p)
    extended = {"TEST": frame(start=0, n=100, step=DAY)}

    train, val, final, live, source = build_academy_partitions(
        extended,
        manifest,
        live_shadow_days=7,
        train_maturation_days=30,
        min_rolling_validation_bars=1,
    )
    bootstrap_end = manifest["symbols"]["TEST"]["bootstrap_end_ts"]
    latest = int(extended["TEST"]["ts"].max())
    maturity_cutoff = latest - 30 * DAY
    live_cutoff = latest - 7 * DAY

    # Post-launch matured rows are trainable, but original FINAL is never trainable.
    assert not ((train["TEST"]["ts"] > manifest["symbols"]["TEST"]["validation_end_ts"]) &
                (train["TEST"]["ts"] <= bootstrap_end)).any()
    assert (train["TEST"]["ts"] <= maturity_cutoff).all()
    assert ((val["TEST"]["ts"] > maturity_cutoff) & (val["TEST"]["ts"] <= live_cutoff)).all()
    assert (live["TEST"]["ts"] > live_cutoff).all()
    assert (final["TEST"]["ts"] <= bootstrap_end).all()
    assert source == "ROLLING_VALIDATION"
