# Install and start Generation 3

This patch is additive. It intentionally does **not** overwrite Generation 2 code, `config.yaml`, its workflows or student caches.

## 1. Add the patch to the repository

Extract/copy the patch contents into the root of `trading-zero-academy` while preserving directories, then commit and push.

New important files include:
- `config_generation3.yaml`
- `requirements_generation3.txt`
- `academy/gen3_*.py`, `academy/recurrent_utils.py`, `academy/checkpoint_registry.py`
- `scripts/gen3_*.py`
- `.github/workflows/generation3.yml`
- `.github/workflows/generation3-final-exam.yml`
- the Gen3 protocol, frozen analysis plan, pre-launch checklist and tests

Do not delete Gen2 files.

## 2. Verify the first GitHub Actions run

Go to **Actions → Trading Zero Academy — Generation 3 → Run workflow**.

For the first and only fresh start use:

```text
run_mode = start
student_count = 8
target_total_timesteps = 1000000000
auto_continue = true
```

The first run executes Gen3 unit tests and hard-verifies the frozen market world.

## 3. After the first run

Do not manually start another copy. Successful runs automatically dispatch the next run in `resume` mode.

A healthy chain should have one active/pending Generation 3 run at a time. The concurrency group prevents simultaneous Gen3 cohort runs, but a manually queued duplicate is still unnecessary.

## 4. Artifacts to download for analysis

The cohort artifact is named:

`academy-generation3-cohort-<run-number>`

It contains, for each student:
- learning history,
- training telemetry,
- policy autopsy,
- status/progress/summary,
- best validation metadata,
- checkpoint registry metadata.

You can send any cohort ZIP here for intermediate diagnostics. Do not change the protocol based on those diagnostics.

## 5. FINAL exam

Only if a student freezes a `MASTER_CANDIDATE`, use:

**Actions → Trading Zero Final Exam — Generation 3**

Never run FINAL for ordinary students. The script itself also refuses to proceed without the frozen candidate checkpoint.

## 6. Frozen rule

Once the first `start` run begins, do not edit Gen3 architecture, hyperparameters, gates, market world or 1B target. Record ideas for the next experiment instead.
