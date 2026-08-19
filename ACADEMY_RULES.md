# Trading Zero Academy — Constitution v0.2.2

1. The learner receives no human trading strategy, indicators, RR, SL/TP rules or directional hints.
2. The market world is OKX perpetual futures, max leverage x10, taker fee 0.08%, historical funding and modeled slippage.
3. Reward/world/learner rules are not changed merely because performance is poor.
4. Student #1 continues from its checkpoint; a new GitHub run must not silently restart learning.
5. One Academy day runs once per day and contains four autonomous 500k-step training blocks. A checkpoint is saved after every block.
6. The original FINAL TEST is permanently frozen and never enters training.
7. Post-launch data follows an automatic conveyor belt:
   - age 0–7 days: LIVE SHADOW — observation only;
   - age 7–30 days: rolling validation — evaluation only;
   - age 30+ days: eligible for TRAIN.
8. LIVE SHADOW and validation results do not directly alter learner weights and are not fed back as reward.
9. Historical market data is accumulated; old training history is not deleted just because time passes.
10. Live execution is not part of this version. A candidate must pass independent evaluation before OKX Demo/live is considered.
