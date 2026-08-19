import json
import time

from academy.config import load_config, ROOT
from academy.dataset import load_frames, ensure_split_manifest, SPLIT_MANIFEST_NAME
from academy.okx_data import refresh_universe

cfg = load_config()
m = cfg["market"]
e = cfg["evaluation"]
data_dir = ROOT / "data" / "processed"
data_dir.mkdir(parents=True, exist_ok=True)
manifest_path = data_dir / SPLIT_MANIFEST_NAME
expected = [data_dir / f"{s}_{m['bar']}.parquet" for s in m["symbols"]]

# IMPORTANT: create/upgrade the immutable bootstrap boundaries BEFORE adding
# today's new candles. This preserves the original frozen FINAL interval.
if all(p.exists() for p in expected):
    cached_frames = load_frames(data_dir, m["symbols"], m["bar"])
    manifest = ensure_split_manifest(
        cached_frames,
        float(e["train_fraction"]),
        float(e["validation_fraction"]),
        manifest_path,
    )
    print(f"Split manifest ready (version {manifest.get('version')}).")

started = time.time()
results = refresh_universe(
    m["symbols"],
    m["bar"],
    int(m["history_days"]),
    data_dir,
    candle_limit=int(m.get("candle_limit_per_request", 100)),
)

# Fresh install: bootstrap first, then freeze the original world once.
if not manifest_path.exists():
    frames = load_frames(data_dir, m["symbols"], m["bar"])
    manifest = ensure_split_manifest(
        frames,
        float(e["train_fraction"]),
        float(e["validation_fraction"]),
        manifest_path,
    )
    print(f"Created frozen bootstrap boundaries (version {manifest.get('version')}).")

summary = {
    "elapsed_seconds": round(time.time() - started, 2),
    "symbols": results,
    "split_manifest": str(manifest_path.relative_to(ROOT)),
}
print(json.dumps(summary, indent=2))
