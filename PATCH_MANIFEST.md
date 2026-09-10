# Generation 3 patch manifest

## Add these files

- `.github/workflows/generation3.yml`
- `.github/workflows/generation3-final-exam.yml`
- `academy/checkpoint_registry.py`
- `academy/gen3_env.py`
- `academy/gen3_evaluate.py`
- `academy/gen3_paths.py`
- `academy/gen3_policy_audit.py`
- `academy/gen3_telemetry.py`
- `academy/recurrent_utils.py`
- `scripts/gen3_build_leaderboard.py`
- `scripts/gen3_cohort_decision.py`
- `scripts/gen3_cohort_train.py`
- `scripts/gen3_final_exam.py`
- `scripts/gen3_policy_autopsy.py`
- `scripts/gen3_prepare_frozen_world.py`
- `scripts/gen3_student_summary.py`
- `tests/test_checkpoint_registry.py`
- `tests/test_gen3_protocol.py`
- `tests/test_recurrent_utils.py`
- `config_generation3.yaml`
- `requirements_generation3.txt`
- `GENERATION_3_PROTOCOL.md`
- `INSTALL_GENERATION3.md`
- `STRATEGY_AUGMENTED_FUTURE.md`
- `GEN3_ANALYSIS_PLAN.md`
- `GEN3_PRELAUNCH_CHECKLIST.md`
- `baseline/GEN2_FINAL_500M.csv`
- `baseline/GEN2_FINAL_500M.json`

## Existing files intentionally untouched

- `config.yaml`
- `.github/workflows/generation2.yml`
- `.github/workflows/generation2-final-exam.yml`
- `academy/gen2_env.py`
- existing Gen2 scripts and tests

This separation lets Gen2 remain a frozen historical control.
