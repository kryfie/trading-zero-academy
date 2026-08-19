# Trading Zero Academy v0.2.3 — Marathon Mode

An AlphaZero-inspired autonomous learning laboratory for OKX perpetual futures.

## Marathon

Student #1 keeps the same strategy-free learning rules from v0.2.2, but training no longer waits for the next day.

A single manual start launches an automatic chain toward **123,000,000 total timesteps** (the target can be changed in the Run workflow form).

Each chained GitHub run trains at most ~10M requested steps and then saves the full Academy memory before automatically dispatching the next run.

Inside every run:

```text
+500k requested steps -> local model checkpoint
+500k -> checkpoint
+500k -> checkpoint
+500k -> checkpoint
                 -> VALIDATION (~2M cadence)
repeat until this ~10M leg ends
                 -> save GitHub cache
                 -> upload status/history
                 -> automatically queue next leg
```

Stable-Baselines PPO completes fixed rollout chunks, so actual timestep counts can be slightly above the requested values. The marathon stops when the total counter reaches/exceeds the requested total target.

## Automatic graduation stop

Routine validation still uses the existing candidate gate. The marathon keeps a consecutive `MASTER_CANDIDATE` streak. If the gate is passed **3 validations in a row**, the marathon stops early to preserve compute and reports `graduation_ready: true`.

This **does not** unlock or run the FINAL TEST. The frozen final holdout remains manual and locked.

## Progress visibility

`reports/learning_history.csv` receives one row every validation (~2M requested training steps), including:

- total timesteps
- median/mean validation return
- median max drawdown
- profitable episode ratio
- profit factor
- candidate status/streak

Every chained run uploads this file together with `status.json` as a GitHub Artifact.

The Academy also preserves `models/best_validation.zip` and metadata for the best validation checkpoint seen so far. This archive is observation/model-selection only; its score is never fed back into the learner's reward or weights.

## Data policy — unchanged

- original FINAL TEST: frozen forever; never trains
- post-launch 0–7d: LIVE SHADOW
- post-launch 7–30d: rolling validation
- post-launch 30+d: eligible for TRAIN
- OKX data refresh remains incremental

## Student world — unchanged

- OKX perpetual swaps
- M5
- BTC / ETH / SOL / XRP / SUI USDT swaps
- leverage max x10
- taker fee 0.08%
- historical funding
- modeled slippage
- no human RSI / MA / Heikin-Ashi / RR / SL / TP strategy

## How to start

GitHub -> Actions -> **Trading Zero Academy — Marathon** -> Run workflow.

Leave `target_total_timesteps = 123000000` and click Run workflow once. Subsequent legs are queued automatically. Do not manually run multiple marathon chains in parallel.
