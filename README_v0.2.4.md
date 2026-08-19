# Trading Zero Academy v0.2.4 — Marathon artifact safety patch

This patch changes only marathon artifact packaging. It does **not** change PPO, reward, market inputs, leverage, fees, funding, action space, validation criteria, or Student #1 learning behavior.

Each marathon artifact now carries:

- `models/best_validation.zip` — the best validation checkpoint found anywhere in the marathon so far.
- `models/best_validation.json` — the metrics and timestep of that best checkpoint.
- `models/latest.zip` — the newest Student #1 checkpoint at the end of the current marathon leg.
- `reports/learning_history.csv` — cumulative validation history.
- `reports/status.json` and `reports/progress.json` — current status/progress.
- `data/processed/split_manifest.json` and `data/processed/training_state.json` — partition/state metadata when present.

Therefore, if Student #1 peaks at e.g. 55M steps and later deteriorates, the final artifact still contains the 55M `best_validation.zip` as long as no later checkpoint legitimately beats it under the same validation-selection rule.
