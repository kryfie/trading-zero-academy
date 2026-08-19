from academy.config import load_config, ROOT
from academy.okx_data import download_universe

cfg = load_config()
m = cfg["market"]
download_universe(m["symbols"], m["bar"], int(m["history_days"]), ROOT / "data" / "processed")
