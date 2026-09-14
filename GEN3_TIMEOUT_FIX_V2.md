# Gen3 timeout fix v2

## Why v1 still timed out
The v1 soft-runtime check happened only **between 500,000-step training blocks**.
For the slowest recurrent students (#001 and #004), one `model.learn(500k)` call can itself run longer than GitHub's 5h30 hard timeout. Therefore Python never got back to the soft-runtime check.

The screenshots from run #16 confirm this pattern:
- 6 students completed successfully,
- #001 and #004 hit `maximum execution time of 5h30m0s`.

## v2 fix
This patch does NOT change the Gen3 experiment.

It changes only workflow/runtime behavior:

1. Adds a Stable-Baselines `BaseCallback` inside `model.learn`.
2. The callback checks wall-clock time while PPO is actively training.
3. At 210 minutes it returns `False`, which makes `model.learn` exit cleanly.
4. The script then saves the current model/state so GitHub can save the cache.
5. GitHub's hard timeout remains 330 minutes, leaving ~2 hours of safety margin.
6. Validation cadence is now cumulative-progress based (~2M training steps) and persists across workflow runs.
7. A short infrastructure run does NOT create an extra validation, so the frozen candidate-streak semantics are preserved.

## Recovery
1. Disable `Trading Zero Academy — Generation 3`.
2. Cancel the currently running old-code run.
3. Copy the two patch files into the repo root, overwriting:
   - `.github/workflows/generation3.yml`
   - `scripts/gen3_cohort_train.py`
4. Commit + push.
5. Re-enable workflow.
6. Run once manually:
   - `run_mode = resume`
   - `student_count = 8`
   - `target_total_timesteps = 1000000000`
   - `auto_continue = true`
7. Do NOT use `start`.

Expected behavior:
- Faster students may complete the full 2M round.
- Slow students can stop around the 210-minute wall-clock mark, save cleanly, and continue in the next automatic resume run.
- No student should reach GitHub's 5h30 timeout anymore.
