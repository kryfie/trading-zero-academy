# Trading Zero Academy v0.4.0 — Market Students

## Purpose

Keep Student #1 unchanged as the original M5-only baseline.

Students #2 and above start from zero and receive raw market views from:

- M1
- M5
- M15
- H1
- H4

No RSI, MA, MACD, Heikin Ashi, trend labels, support/resistance, RR, SL or TP are provided.

M5 is only the environment clock: the agent may update its position every five minutes.
All timeframe candles are visible only after they are fully closed, preventing look-ahead.

## World rules kept unchanged

- OKX perpetual futures
- max leverage x10
- taker fee 0.08% per turnover fill
- historical funding
- modeled slippage
- same symbols
- same frozen historical FINAL interval
- same candidate gate
- FINAL TEST remains manual

## Why Student #1 is not converted

Its PPO input shape was trained on M5-only observations. Giving that same network a new
multi-timeframe observation space would invalidate the checkpoint. Therefore Student #1
remains a clean baseline and Students #2+ are the first Market Students.

## Operational note

The first v0.4 run must bootstrap ~180 days of M1 data for five symbols. This is much larger
than the old M5 world and can take materially longer. It is cached afterward; future rounds
only bridge the missing M1 gap and regenerate higher-timeframe parquet files locally.

Students #2+ train 4M steps per cohort round (checkpoint each 500k, validation every 2M).
Student #1 can still train 10M per round.
