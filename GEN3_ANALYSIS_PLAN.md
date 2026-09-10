# Generation 3 — frozen analysis plan

This plan is fixed before the first Gen3 training run. It is for analysis only and must not feed back into training.

## Checkpoints to review

Use cohort snapshots near 100M, 250M, 500M, 750M and 1B steps/student when available. Intermediate ZIPs may be inspected, but do not alter the protocol.

## Primary comparison: Gen2 vs Gen3 through 500M

For each seed-matched student #001..#008 compare at or near the same training budgets:
- validation median and mean net return,
- profit factor, drawdown and profitable-episode ratio,
- gross market PnL before explicit costs,
- HOLD / repeated-target rate,
- position-change rate, turnover, entries, reversals and leverage,
- fee, slippage and funding burden.

The cohort-level comparison must report median/mean across all eight students, not only the best student.

## Gen3-only continuation: 500M–1B

Analyze whether additional recurrent training produces:
- continued OOS improvement,
- plateau,
- policy drift / degradation,
- lower or higher turnover and explicit costs,
- persistent rather than one-off improvements.

Do not attribute 500M–1B gains purely to LSTM vs MLP because Gen2 has no matched training window beyond 500M.

## Optimization telemetry

For every student track trajectories of:
- entropy estimate,
- approximate KL,
- clip fraction,
- policy-gradient loss,
- value loss and total loss,
- explained variance,
- gradient norm,
- parameter norm and relative parameter delta,
- actor/critic LSTM norms,
- throughput.

Telemetry is diagnostic evidence. It is not a reason to retune learning rate, entropy, architecture or other frozen parameters mid-run.

## Checkpoint stability / policy drift

For each student compare:
- latest policy,
- best-validation policy,
- top-5 validation registry.

Record whether later training preserves, improves or loses previously discovered behavior. This is especially important because Gen2 frequently found a better checkpoint around 200–300M and then degraded by 500M.

## Success

The only formal success route remains the frozen candidate gate and 3 consecutive passes, followed by the manual one-shot FINAL exam. Descriptive improvements below the gate are not called success.

## Failure

If all eight students reach 1B with no valid MASTER_CANDIDATE, Generation 3 is failed. Do not extend the same experiment after seeing the result.
