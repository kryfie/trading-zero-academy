# Trading Zero Academy — Generation 2

Generation 2 is a new, frozen experiment. It does **not** continue Generation 1 weights.

## Research question

Can PPO + MLP learn profitable raw multi-timeframe trading when the continuous
position-control output from Generation 1 is replaced by a discrete target-position
action space?

Generation 1 showed a repeated pathology: later policies often increased turnover,
fees and slippage dramatically. Generation 2 changes only the action representation.

## What changed

Generation 1 RAW_MTF action:
- continuous side output
- continuous leverage output
- tiny output drift could resize the position and generate turnover

Generation 2 action:
- one categorical target position from `-10, -9, ..., 0, ..., +9, +10`
- `0` = FLAT
- negative = SHORT at that leverage
- positive = LONG at that leverage
- repeating the same target is a true HOLD and creates zero resize turnover

Example:
- current `LONG x4`, next action `LONG x4` -> HOLD, no fee/slippage from resizing
- current `LONG x4`, next action `LONG x5` -> turnover 1x
- current `LONG x4`, next action `SHORT x4` -> turnover 8x

## What did NOT change

- PPO
- MLP (`MlpPolicy`)
- raw M1/M5/M15/H1/H4 observations
- M5 decision clock
- five OKX perpetual markets
- max leverage x10
- taker fee 0.08%
- slippage model
- funding
- reward
- validation gates
- 30 validation episodes
- 3 consecutive MASTER_CANDIDATE passes
- FINAL TEST remains untouched unless a candidate qualifies
- 500M hard ceiling per student

## Market data

Generation 2 reuses the exact TRAIN / VALIDATION / FROZEN FINAL eras recorded at
Generation 1 Cohort #129.

`data/manifests/generation1_partition_reference.json`

Before training starts, the workflow hard-verifies row counts and timestamp
boundaries for every symbol and every timeframe. If the world differs, training stops.

No new bars mature into TRAIN during Generation 2. No rolling validation is used.
This keeps the action-space comparison clean.

## Students

Eight new students start from zero.

Gen2 #001..#007 intentionally reuse the random-seed identities of Gen1 RAW_MTF
#002..#008. Gen2 #008 uses the next unused seed. This gives seven seed-matched
comparisons.

No Generation 1 model weights are loaded.

## Run protocol

First manual run:
1. Actions -> `Trading Zero Academy — Generation 2`
2. `Run workflow`
3. Set `run_mode = start`
4. Leave:
   - students = 8
   - target = 500000000
   - auto_continue = true

Every automatically queued run uses `run_mode = resume`.

Safety: if a resume run cannot restore a student's Gen2 checkpoint, it fails rather
than silently recreating that student from zero.

## Stop rule

500M per student is a hard ceiling.

If no MASTER_CANDIDATE exists at 500M:
`GENERATION 2 = FAILED`.

Do not extend to 750M/1B. The predetermined next experiment is Generation 3
(sequence memory), not another Gen2 tweak.

## FINAL TEST

Run `Trading Zero Final Exam — Generation 2` only for a student that has already
frozen a 3x `MASTER_CANDIDATE`.

The script itself refuses to run FINAL without that checkpoint.
