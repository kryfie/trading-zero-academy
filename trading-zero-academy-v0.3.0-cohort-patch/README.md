# Trading Zero Academy

AlphaZero-inspired autonomous trading research lab for OKX perpetual futures.

Current mode: **v0.3.0 Cohort + Policy Autopsy**.

## Fixed world rules

- OKX perpetual futures (`SWAP`)
- M5 market data
- maximum leverage x10
- taker fee assumption 0.08% per executed turnover
- historical funding events
- deterministic volatility-aware slippage model
- no RSI, MA, Heikin Ashi, fixed RR, SL/TP, or hand-authored entry strategy

The learner sees market/portfolio state and learns a policy from reward. Human analysis may inspect behavior, but audit results are never fed back into the learner.

## Cohort mode

`.github/workflows/cohort.yml` manages a class of independent students.

Default first cohort:

- Student #1 continues from the completed legacy Marathon checkpoint.
- Students #2–#8 start from zero.
- each student gets an independent seed and private checkpoint/cache namespace;
- every student receives the same market-world snapshot;
- checkpoint about every 500k requested timesteps;
- validation about every 2M requested timesteps;
- BEST checkpoint retained separately from LATEST;
- first 3x consecutive MASTER_CANDIDATE checkpoint is frozen for a future Final Exam;
- Policy Autopsy compares BEST vs LATEST on identical validation episodes;
- one combined cohort artifact contains the leaderboard and all student reports.

See `COHORT_MODE.md` for migration and operating details.

## Data isolation

The original FINAL TEST interval is frozen forever and never enters training, rolling validation, leaderboard, or Policy Autopsy.

New post-launch data passes through:

`LIVE SHADOW -> ROLLING VALIDATION -> TRAIN after maturation`

The Final Exam is manual and student-specific. It refuses to run unless that student already has a frozen 3x `MASTER_CANDIDATE` checkpoint.

## Important

Do not start Cohort mode while the legacy Student #1 Marathon is still chaining. Let the current Marathon reach its target and stop first, then apply v0.3.0 and run **Trading Zero Academy — Cohort**.
