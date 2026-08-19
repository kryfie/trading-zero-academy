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
    print(f"Student: #{s.get('student', 1)}")
    print(f"Total timesteps: {s.get('total_timesteps', 0):,}")
    print(f"Validation source: {s.get('validation_source', 'unknown')}")
    print(f"Median validation return: {v['median_return_pct']:.2f}%")
    print(f"Median max DD: {v['median_max_drawdown_pct']:.2f}%")
    print(f"Profitable episodes: {v['profitable_episode_ratio']*100:.1f}%")
    print(f"Profit factor: {v['profit_factor']:.2f}")
    live = s.get("live_shadow")
    if isinstance(live, dict) and "median_return_pct" in live:
        print(f"LIVE SHADOW median return: {live['median_return_pct']:.2f}% (observation only)")
    else:
        print("LIVE SHADOW: collecting / not enough data yet")
    print("Final exam: LOCKED")
