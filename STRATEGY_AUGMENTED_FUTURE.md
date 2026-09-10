# Future experiment — Strategy-Augmented Academy

This experiment is **not Generation 3** and must not be enabled during `GEN3_SEQUENCE_MEMORY_V1`.

## Research question

Does curated prior trading knowledge improve learning beyond recurrent RL from raw market observations?

## Why separate it from Gen3

If LSTM memory and strategy knowledge are added together, a better result cannot be attributed to either change. Gen3 therefore tests sequence memory alone. Strategy augmentation is a separate experiment after Gen3 has a frozen result.

## Candidate strategy families

A future eight-student cohort can use one frozen strategy family per student:
1. trend following,
2. pullback / continuation,
3. breakout,
4. mean reversion,
5. momentum,
6. volatility expansion / compression,
7. support-resistance / market structure,
8. multi-timeframe swing.

## Preferred first implementation

The cleanest first strategy-augmented test is to convert each strategy into **additional observation features/signals**, not hard trading rules. The RL agent remains free to ignore them, combine them, delay entries, hold positions or trade against them.

Possible later variants:
- behavioral cloning / pretraining from a frozen rule policy,
- strategy-conditioned policy embeddings,
- one strategy prior per student followed by RL fine-tuning.

These variants should be separate experiments, not mixed at once.

## Internet research protocol

Internet research happens **before training**, not live inside training.

For every strategy:
- collect multiple reputable descriptions,
- save source URLs/titles/publication dates,
- write a neutral strategy definition,
- formalize only information available without looking at validation/final outcomes,
- freeze the specification and source bundle before the first training run,
- never use examples selected because they worked on the Academy's validation/final period.

This avoids an unrepeatable agent that changes as the internet changes and reduces look-ahead/data-leakage risk.

## Future strategy-spec schema

Each frozen strategy specification should contain:
- `strategy_id`
- `family`
- `version`
- `research_cutoff_utc`
- `sources[]`
- `concepts[]`
- `formal_features[]`
- `forbidden_future_information`
- `notes`

Do not prepare or tune these features using Gen3 validation or FINAL results.
