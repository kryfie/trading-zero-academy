# Trading Zero Academy Constitution — v0.3.0

1. **No strategy hints.** Students are not given human indicators, named setups, fixed RR, SL/TP logic, or discretionary entry rules.
2. **World rules are external laws.** OKX SWAP market, max x10 leverage, configured fees, funding, slippage, and simulator accounting cannot be altered by a student.
3. **Independent students.** Every student has its own seed, weights, optimizer state, checkpoint, and history. Students do not copy or exchange policies.
4. **Same exam conditions.** In a cohort round, every student receives the same market-world snapshot and the same validation protocol.
5. **No FINAL leakage.** Frozen FINAL data is never used for training, rolling validation, leaderboard ranking, Policy Autopsy, or hyperparameter decisions.
6. **Checkpoint history matters.** LATEST is not assumed to be BEST. The best validation checkpoint is retained even if later learning collapses.
7. **MASTER_CANDIDATE is pre-defined.** A candidate requires the configured gate to pass on 3 consecutive validations. The first such checkpoint is frozen separately.
8. **FINAL is manual.** No workflow may automatically consume FINAL. The exam is allowed only for a student with a frozen 3x MASTER_CANDIDATE checkpoint.
9. **Policy Autopsy is observational.** It may explain differences in behavior (flat time, leverage, turnover, fees, slippage, funding, gross PnL) but its findings are not fed back into learning.
10. **Student #1 continuity is protected.** Migration from legacy Marathon must restore its existing model; failure stops the workflow instead of silently creating a new Student #1.
11. **New students truly start at zero.** Student #2+ may be created only when their own cache does not yet exist.
12. **No manual rescue.** A losing student is not repaired by adding indicators or changing its policy based on observed validation performance. Experimental changes require a new version/experiment, not silent intervention in an existing student.
