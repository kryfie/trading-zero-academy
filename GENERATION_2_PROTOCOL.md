# GENERATION 2 PROTOCOL — FROZEN BEFORE TRAINING

## Hypothesis

The continuous target-position representation used by Generation 1 causes or enables
pathological micro-rebalancing. A discrete target-position action space may let the
same PPO+MLP learner retain useful market behavior while reducing unnecessary
turnover.

This is the **only intended experimental change**.

## Independent variable

Action space.

### Gen1 RAW_MTF
Two continuous outputs:
- side
- leverage

### Gen2
One categorical output with 21 target states:

`SHORT x10 ... SHORT x1 | FLAT | LONG x1 ... LONG x10`

The state is a **target position**, not an instruction to place a fresh trade.
Repeating the current target means HOLD.

## Controlled variables

Frozen:
- PPO hyperparameters
- MLP architecture 256x256
- observation features
- timeframe windows
- market symbols
- market eras
- decision clock
- reward
- fee
- slippage
- funding
- leverage ceiling
- episode length
- validation episode count
- candidate thresholds
- candidate streak
- 500M ceiling

## Success gate

A validation checkpoint is a candidate only if all remain true:
- episodes >= 20
- median net return >= +2%
- median max drawdown <= 20%
- profitable episode ratio >= 55%
- profit factor >= 1.05

Three consecutive candidate validations freeze `master_candidate.zip`.

Only that frozen checkpoint may take FINAL.

## Interpretation rule

Lower turnover alone is **not** success.
Positive gross market PnL alone is **not** success.
A good-looking single episode is **not** success.

The target is profitable **net** out-of-sample behavior after modeled fee, slippage
and funding.

## No mid-run intervention

During the 0 -> 500M experiment do not:
- change reward
- change actions
- change fees/slippage
- change PPO
- add indicators
- add Internet/news
- add trading rules
- alter validation gates
- extend the ceiling

Observations may be recorded for the post-mortem only.
