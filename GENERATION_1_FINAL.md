# Trading Zero Academy — Generation 1 Final Verdict

**Status:** CLOSED / FAILED EXPERIMENT  
**Training ceiling:** 500,000,000 timesteps per student  
**Cohort:** 8 students  
**Final cohort report:** #129  
**Final exam:** NOT RUN (no student qualified)

## Hypothesis tested

Can PPO + MLP, trained without hand-coded trading indicators or discretionary trading rules, discover a profitable OKX perpetual-futures policy from raw market observations under realistic trading costs?

## Result

Generation 1 did **not** discover a profitable out-of-sample policy.

- 0 MASTER_CANDIDATE students
- candidate streak = 0 for all students
- profitable validation episode ratio = 0.0 for all preserved best checkpoints in Cohort #129
- no preserved Cohort #129 best checkpoint had positive median validation return after costs
- FINAL TEST remained locked, as designed

## Cohort #129 best-validation checkpoints

| Student | Mode | Total steps | Best step | Best median return | Best PF | Best median DD | Profitable episodes |
|---|---|---:|---:|---:|---:|---:|---:|
| 005 | RAW_MTF | 500,002,816 | 18,022,400 | -58.77% | 0.498 | 59.03% | 0% |
| 008 | RAW_MTF | 500,002,816 | 184,082,432 | -63.55% | 0.497 | 63.93% | 0% |
| 004 | RAW_MTF | 500,002,816 | 8,003,584 | -76.19% | 0.313 | 76.26% | 0% |
| 003 | RAW_MTF | 500,002,816 | 262,131,712 | -76.72% | 0.357 | 76.80% | 0% |
| 006 | RAW_MTF | 500,002,816 | 222,113,792 | -80.07% | 0.774 | 82.19% | 0% |
| 002 | RAW_MTF | 500,002,816 | 42,033,152 | -83.53% | 0.327 | 83.55% | 0% |
| 001 | M5_ONLY_BASELINE | 500,006,912 | 127,066,112 | -85.35% | 0.275 | 85.41% | 0% |
| 007 | RAW_MTF | 500,002,816 | 12,005,376 | -85.80% | 0.321 | 85.80% | 0% |

## Student #1 legacy best

The true historical best checkpoint of Student #1 predates the v0.4 Cohort leaderboard and must be preserved separately:

- total timesteps: **13,205,504**
- median validation return: **-8.7909%**
- profit factor: **0.92417**
- median max drawdown: **11.3184%**
- profitable episode ratio: **0%**
- best validation episode: **-1.0377%**

This was the best historical checkpoint found by Generation 1, but it was still unprofitable and therefore did not qualify for FINAL TEST.

## Main observation for the next experiment

Across multiple independent students, later policies frequently developed very high turnover. Trading fees and slippage became a dominant source of losses. Some checkpoints showed positive gross market PnL before costs, but none converted it into profitable net out-of-sample performance.

This observation motivates a **single-variable Generation 2 experiment focused on action representation/execution**, not an open-ended series of reward tweaks or added indicators.

## Closure rule

Generation 1 is frozen. Do not extend it beyond 500M, retune it, lower costs, unlock FINAL TEST, or reinterpret the gate after seeing the result.
