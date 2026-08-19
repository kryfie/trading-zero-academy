# Trading Zero Academy v0.2.2

An AlphaZero-inspired autonomous learning laboratory for OKX perpetual futures.

## Student rules

The student starts without human trading indicators or strategy rules. It observes raw market/portfolio state, chooses exposure and leverage up to x10, and learns only from consequences inside the simulator.

World assumptions currently include:

- OKX perpetual swaps
- M5 bars
- BTC, ETH, SOL, XRP and SUI USDT swaps
- max leverage x10
- taker fee 0.08% per turnover fill
- historical funding events
- deterministic volatility-aware slippage
- no hand-authored RSI/MA/Heikin-Ashi/RR/SL/TP strategy

## Daily Academy schedule

GitHub Actions runs once per day at `00:15 UTC`.

One run contains four autonomous training blocks:

```text
refresh OKX world
  -> 500k steps -> checkpoint
  -> 500k steps -> checkpoint
  -> 500k steps -> checkpoint
  -> 500k steps -> checkpoint
  -> validation
  -> LIVE SHADOW observation
  -> status
```

Stable-Baselines PPO completes rollouts in fixed chunks, so the actual number of steps can be slightly above 2,000,000 per day.

## Data conveyor belt

The Academy preserves the original bootstrap split and never moves the original FINAL TEST.

New market data created after the Academy bootstrap moves automatically with age:

```text
0–7 days old      LIVE SHADOW          never trains
7–30 days old     rolling validation   never trains
30+ days old      TRAIN eligible
```

Until enough post-launch data exists for rolling validation, the Academy uses the original frozen bootstrap validation set. This transition happens automatically.

The original FINAL TEST remains permanently isolated even after it becomes old. It is only consumed through the separate Final Exam workflow after deliberate unlock.

## Memory

GitHub Actions restores and saves:

- `models/latest.zip` — Student #1 PPO checkpoint
- `models/training_state.json` — run/RNG progression
- `data/processed` — accumulated OKX market history and immutable split manifest

Therefore scheduled runs continue the same student rather than restarting it.

## Status artifact

Every run uploads `reports/status.json` and `reports/progress.json`. Status includes total timesteps, validation source/results, LIVE SHADOW result when enough data exists, and the timestamp/row ranges of TRAIN, validation, LIVE SHADOW and FROZEN FINAL.

## Final exam

The Final Exam is locked by default. Do not repeatedly run it to tune the learner. The frozen final holdout is deliberately protected from both training and routine model selection.
