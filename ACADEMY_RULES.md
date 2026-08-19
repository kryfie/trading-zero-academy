# Trading Zero Academy — Constitution v0.2.3 Marathon

1. The learner receives no human trading strategy, indicators, RR, SL/TP rules or directional hints.
2. The market world is OKX perpetual futures, max leverage x10, taker fee 0.08%, historical funding and modeled slippage.
3. Reward/world/learner rules are not changed merely because performance is poor.
4. Student #1 always continues from its checkpoint; a new GitHub run must not silently restart learning.
5. Marathon training uses 500k requested-step blocks and saves the PPO checkpoint after every block.
6. Routine validation is performed every 4 blocks (~2M requested steps) and appended to `learning_history.csv`.
7. Validation never directly alters learner weights/reward. It is only an evaluator/model-selection signal.
8. The original FINAL TEST is permanently frozen and never enters training or routine marathon evaluation.
9. Post-launch data follows the automatic conveyor belt: 0–7d LIVE SHADOW; 7–30d rolling validation; 30+d TRAIN eligible.
10. Historical market data accumulates; old training history is not deleted just because time passes.
11. A chained GitHub marathon run handles at most ~10M requested steps, then persists Academy memory and queues the next run.
12. Default marathon target is 123M TOTAL Student #1 timesteps, not +123M from the moment Marathon Mode starts.
13. If the existing MASTER_CANDIDATE gate passes 3 routine validations consecutively, marathon training stops early and reports `graduation_ready`. FINAL TEST remains locked and manual.
14. `best_validation.zip` is an archive only. Student #1 continues learning from `latest.zip`; the best-validation archive is never injected back into training.
15. Live execution is not part of this version. FINAL TEST and later OKX Demo remain separate graduation stages.
