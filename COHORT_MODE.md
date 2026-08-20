# Trading Zero Academy v0.3.0 — Cohort + Policy Autopsy

## Purpose

One Academy workflow manages multiple fully independent RL students.

- Student #1 is migrated from the completed legacy Marathon and keeps its weights, optimizer state, best validation checkpoint, and learning history.
- Students #2+ start from zero with independent deterministic seeds.
- Students do **not** copy each other.
- All students receive the exact same market-world snapshot in a cohort round.
- Each student owns a separate cache, model, training state, history, BEST checkpoint, and MASTER_CANDIDATE checkpoint.
- FINAL TEST is never used by training, validation, leaderboard, or Policy Autopsy.

## Default cohort

The workflow defaults to 8 students and a 500,000,000 total-timestep target per student.
Student #1 continues from its existing total; Students #2–#8 begin at zero.
Each cohort round requests up to 10M new timesteps per unfinished student, with:

- checkpoint every ~500k requested timesteps,
- validation every 4 blocks (~2M requested timesteps),
- BEST validation checkpoint retained forever,
- first 3-consecutive-pass MASTER_CANDIDATE frozen separately,
- automatic next cohort round while anyone is below target.

SB3 rollout boundaries can make exact timestep counts slightly higher than requested numbers.

## Policy Autopsy

After each student round, BEST and LATEST are replayed deterministically on the **same validation episodes**. The audit records behavioral evidence such as:

- % time FLAT / LONG / SHORT,
- average active leverage,
- entries, exits, reversals, leverage rebalances,
- turnover proxy,
- average holding time,
- fees, modeled slippage, funding paid/received,
- gross market PnL before explicit costs,
- return and drawdown.

This lets us ask *what behavior was different when a student was better* without feeding the answer back into the learner.

## FINAL TEST

A student can create `master_candidate.zip` only after the pre-defined candidate gate passes 3 consecutive validations. The Final Exam workflow requires a Student ID and refuses to run if that frozen candidate does not exist.

The exam uses `master_candidate.zip`, not LATEST or a checkpoint selected after seeing FINAL results.

## Migration rule for Student #1

Apply v0.3.0 only after the current Student #1 Marathon reaches its target and stops chaining. On the first Cohort run, the workflow restores the final `academy-marathon-*` cache and migrates Student #1 into `students/001/`.

If Student #1 cannot be restored, Cohort training fails instead of silently creating a new Student #1.
