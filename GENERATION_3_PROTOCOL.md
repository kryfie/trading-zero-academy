# Trading Zero Academy — Generation 3 frozen protocol

**Experiment ID:** `GEN3_SEQUENCE_MEMORY_V1`  
**Status:** pre-registered before first training run

## Research question

Can PPO with recurrent sequence memory learn better net out-of-sample trading behavior than Generation 2's PPO+MLP when the market world, observations, action space, costs, reward and validation gates are held fixed?

## Primary hypothesis

Generation 2 proved that a true discrete HOLD action alone does not produce a profitable policy. Generation 3 adds recurrent state across decisions so the policy can retain temporal context beyond a single decision call and distinguish persistence from transient noise.

## The one strategic change

Generation 2:
- Stable-Baselines3 `PPO`
- `MlpPolicy`

Generation 3:
- SB3-Contrib `RecurrentPPO`
- `MlpLstmPolicy`
- actor LSTM hidden size: 256
- critic LSTM enabled, separate from actor
- one LSTM layer

No Generation 2 weights are loaded. Every Gen3 student starts from zero.

## Frozen controls inherited from Generation 2

Generation 3 keeps unchanged:
- raw M1/M5/M15/H1/H4 market observations and their windows,
- M5 decision clock,
- BTC/ETH/SOL/XRP/SUI OKX perpetual universe,
- exact frozen TRAIN / VALIDATION / FINAL partition reference,
- discrete signed target leverage `-10..+10`, including true HOLD when the target is repeated,
- max leverage x10,
- starting equity 100,
- taker fee 0.08%,
- slippage model,
- funding treatment,
- reward function,
- 2016-bar episode length,
- PPO learning rate, rollout size, minibatch size, gamma, GAE lambda, entropy coefficient and clip range,
- validation gates,
- 30 validation episodes,
- 3 consecutive candidate passes before freezing a MASTER_CANDIDATE,
- FINAL isolation.

For reproducibility, PPO defaults that were implicit in Gen2 are explicit in Gen3: `n_epochs=10`, `vf_coef=0.5`, `max_grad_norm=0.5`.

## Students and seeds

Exactly eight new students start from zero. Gen3 #001..#008 reuse the same seed identities as Gen2 #001..#008 to create seed-matched architecture comparisons. This does **not** load or inherit Gen2 weights.

## Training budget

- Hard ceiling: **1,000,000,000 timesteps per student**.
- **Primary matched Gen2-vs-Gen3 architecture comparison window: 0–500M steps.** Both generations have the same seed identities and the same frozen world over this window; the intended architectural difference is MLP vs recurrent LSTM state.
- **500M–1B is a pre-registered Gen3 continuation window.** It tests whether the recurrent learner benefits from additional compute, but it must not be described as a pure architecture comparison with Gen2 because Gen2 stopped at 500M.
- Round budget: 4,000,000 timesteps.
- Training block: 500,000 timesteps.
- Validation: every 4 blocks (about every 2M steps) and at round end.
- Auto-continuation is allowed until the ceiling is reached.

The 1B ceiling is frozen before training. Do not extend it after seeing results. Reaching 500M is an analysis checkpoint only, not permission to tune or stop based on intermediate performance.

## Stop / success rule

A policy becomes `MASTER_CANDIDATE` only after three consecutive frozen-validation passes using the unchanged gates. FINAL remains manual and one-shot.

If no MASTER_CANDIDATE exists when all students reach the 1B ceiling, **GENERATION 3 = FAILED**.

## Diagnostic telemetry (non-strategic instrumentation)

Telemetry is observational only and does not change actions or rewards. Each training block records:
- policy entropy estimate,
- policy-gradient loss,
- value loss,
- total loss,
- approximate KL,
- clip fraction,
- explained variance,
- learning rate,
- update count,
- policy parameter norm,
- last-minibatch gradient norm,
- parameter change L2 and relative change during the block,
- actor/critic LSTM parameter norms,
- elapsed time and throughput.

Trading behavior continues to be audited with net/gross return, HOLD, position changes, leverage, turnover, entries, exits, reversals, rebalances, holding time, fees, slippage and funding.

## Checkpoint registry

In addition to `latest` and the single `best_validation`, each student keeps the top 5 validation checkpoints under the same frozen ranking rule. This is diagnostic preservation only; it does not feed old policies back into training.

The active learner always continues from `latest`. Top checkpoints are never used to steer, reset or tune training.

## No mid-run intervention

After the first Gen3 training run starts, do not change:
- recurrent architecture,
- PPO hyperparameters,
- observation/action space,
- fees/slippage/funding,
- reward,
- market partitions,
- validation gates,
- seeds/student count,
- 1B ceiling.

Interesting findings are logged for the next experiment, not patched into Gen3.

## Strategy knowledge is deliberately NOT part of Gen3

Existing internet strategies, hand-built indicators, strategy signals, behavior cloning and strategy priors are excluded from this experiment. They are reserved for the separately pre-registered Strategy-Augmented experiment so we can distinguish the effect of sequence memory from the effect of prior trading knowledge.
