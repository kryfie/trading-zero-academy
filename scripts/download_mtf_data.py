import json
import time

from academy.config import load_config, ROOT
from academy.mtf_data import refresh_multitimeframe_universe

cfg = load_config()
m = cfg["market"]
out_dir = ROOT / "data" / "mtf"
started = time.time()

results = refresh_multitimeframe_universe(
    m["symbols"],
    list(m["multi_timeframes"]),
    int(m["history_days"]),
    out_dir,
    candle_limit=int(m.get("candle_limit_per_request", 300)),
)

print(json.dumps({
    "elapsed_seconds": round(time.time() - started, 2),
    "world": "RAW_MTF",
    "decision_bar": m["decision_bar"],
    "timeframes": m["multi_timeframes"],
    "symbols": results,
}, indent=2))
