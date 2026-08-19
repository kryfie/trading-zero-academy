from pathlib import Path
import json
from academy.config import load_config, ROOT

cfg = load_config()
p = ROOT / cfg["runtime"]["status_path"]
if not p.exists():
    print("NO STATUS YET — run training first")
else:
    s = json.loads(p.read_text(encoding="utf-8"))
    v = s["validation"]
    print("TRADING ZERO ACADEMY")
    print(f"STATUS: {s['status']}")
    print(f"Median validation return: {v['median_return_pct']:.2f}%")
    print(f"Median max DD: {v['median_max_drawdown_pct']:.2f}%")
    print(f"Profitable episodes: {v['profitable_episode_ratio']*100:.1f}%")
    print(f"Profit factor: {v['profit_factor']:.2f}")
    print("Final exam: LOCKED")
