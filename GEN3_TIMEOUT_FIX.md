# Generation 3 timeout fix

## What happened
The original Gen3 workflow had:
- `timeout-minutes: 330` (5h30)
- `steps_per_round: 4,000,000`

RecurrentPPO/LSTM is much slower than Gen2 MLP. The student jobs therefore reached the GitHub job timeout before the round could finish cleanly.

## Fix
This patch changes infrastructure only:
- `steps_per_round`: 4,000,000 -> 2,000,000
- keeps the scientific target at 1,000,000,000 steps/student
- keeps PPO/LSTM architecture, reward, market, actions, costs and validation gates unchanged
- adds a 270-minute soft runtime guard
- persists state after each block and after each validation
- exits cleanly before GitHub's 330-minute hard timeout so cache/artifacts can be written

A normal 2M round is exactly four 500k blocks, matching the existing validation cadence.

## Recovery procedure
1. Disable `Trading Zero Academy — Generation 3`.
2. Cancel the currently running old-code Gen3 run.
3. Copy these patch files into the repository root and overwrite the matching Gen3 files.
4. Commit + push.
5. Re-enable the Gen3 workflow.
6. Run it ONCE manually with:
   - run_mode: `resume`
   - student_count: `8`
   - target_total_timesteps: `1000000000`
   - auto_continue: `true`
7. Do not use `start`. Existing students should resume from the latest Gen3 cache.

If any student reports `checkpoint was not restored in RESUME mode`, stop and inspect that student's cache before doing anything else.
