# Trading Zero Academy v0.1

An autonomous reinforcement-learning laboratory inspired by the **principle** behind AlphaGo Zero: the learner is not given a handcrafted trading strategy. It receives market state, legal actions, transaction costs and consequences.

## World rules

- Exchange universe: OKX perpetual swaps (`*-USDT-SWAP`)
- Timeframe: 5m
- Maximum leverage: x10
- Fee assumption: **0.08% per executed turnover** (`0.0008`)
- Historical OKX funding is charged at funding events
- Slippage is non-zero and volatility-aware
- No RSI, MA, Heikin Ashi, RR, fixed SL or TP is supplied to the learner
- Learner chooses short / flat / long and exposure up to x10
- Training data and evaluation data are chronological and separate

## Architecture

`OKX public data -> market simulator -> PPO learner -> checkpoint -> validation evaluator`

The routine evaluator sees **validation**, not the final holdout. The final 15% chronological holdout is locked behind `reports/FINAL_TEST_UNLOCK` and has a separate manual workflow. This prevents the Academy from silently optimizing against its own final exam.

## One-time GitHub setup

1. Create a new GitHub repository, e.g. `trading-zero-academy`.
2. Upload all files from this package preserving folders.
3. Open **Actions -> Trading Zero Academy -> Run workflow** once.
4. After that, GitHub schedules a run every day at 02:17 UTC.
5. Each run restores the latest model from GitHub Actions cache, refreshes OKX data, continues learning, validates, saves the new checkpoint and uploads `status.json` as an artifact.

No OKX API key is needed for historical public market/funding data in v0.1. There is deliberately **no live-order code** in this release.

## When has it learned something?

The learner itself does not decide that. The evaluator marks `MASTER_CANDIDATE` only when the validation distribution passes all configured gates, currently:

- median return >= +2%
- median max drawdown <= 20%
- at least 55% of validation episodes profitable
- profit factor >= 1.05
- minimum episode count met

These are *candidate gates*, not proof of edge. When a model becomes a serious candidate, run the separate **Trading Zero Final Exam** workflow once. Do not repeatedly use final-test results to tune the learner.

## Local commands (optional)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python scripts/download_data.py
python scripts/train.py
python scripts/status.py
```

## Important v0.1 limitations

This is an academy prototype, not a production execution simulator. It includes historical funding, taker fees and a slippage model, but does not yet reconstruct order-book depth, partial fills, latency, maintenance-margin tiers, exchange liquidation engine behavior or changing historical fee tiers. Those belong in the next realism layer before demo/live evaluation.

## Safety boundary

Do not connect this learner directly to live capital. The intended progression is:

`TRAIN -> VALIDATION -> locked FINAL HOLDOUT -> OKX DEMO/PAPER -> tiny controlled live experiment`


## Automatic training schedule

Version 0.2 runs the autonomous learner four times per day via GitHub Actions:

- 00:15 UTC
- 06:15 UTC
- 12:15 UTC
- 18:15 UTC

Each run restores the newest `models/latest.zip` checkpoint from the prior Academy cache, continues PPO learning for `total_timesteps_per_run`, evaluates the frozen checkpoint on validation data, saves a new checkpoint, and exits. A run does **not** create a fresh student unless no prior checkpoint exists.

With the default `500000` timesteps per run, the scheduled target is up to 2,000,000 additional environment timesteps per day, subject to GitHub Actions runtime limits and successful completion of each run.

The final holdout remains separate and is never used by the scheduled learning workflow.
