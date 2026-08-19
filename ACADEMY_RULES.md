# Trading Zero Academy v0.2

# Academy Constitution

1. The learner may change its neural-network weights freely.
2. A human does not add trading indicators, entry rules, RR rules, SL rules or TP rules during learning.
3. The learner cannot change exchange costs, leverage ceiling, data split or evaluator rules.
4. Training receives only the TRAIN partition.
5. Routine scoring receives only VALIDATION.
6. FINAL TEST is not fed back into training.
7. A profitable training curve is not evidence of mastery.
8. A validation pass creates a MASTER CANDIDATE, not a live trader.
9. Demo/paper forward performance is required before any live-capital stage.
10. Every realism cost discovered before live deployment is added to the simulator rather than ignored.

## Data integrity rule (v0.2.1)

TRAIN / VALIDATION / FINAL TEST timestamp boundaries are frozen once in `data/processed/split_manifest.json`. Routine market refreshes may append new candles, but they must never move previously held-out data into TRAIN. Infrastructure may be optimized for speed; learner strategy inputs and reward rules must not be changed merely because results are poor.
