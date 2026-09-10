# Generation 3 — pre-launch checklist

Complete this once before the first `start` run.

- [ ] Generation 2 is finished and no Gen2 training workflow is intentionally running.
- [ ] The Gen3 patch is copied into the repository root with directory structure preserved.
- [ ] Existing `config.yaml`, Gen2 workflows, Gen2 scripts and Gen2 student data were not deleted or overwritten.
- [ ] `requirements_generation3.txt` is present.
- [ ] `config_generation3.yaml` says `GEN3_SEQUENCE_MEMORY_V1`, 8 students by workflow guard, and 1B target.
- [ ] `GENERATION_3_PROTOCOL.md` and `GEN3_ANALYSIS_PLAN.md` are committed before training.
- [ ] GitHub Actions shows `Trading Zero Academy — Generation 3`.
- [ ] First run is launched exactly once with `start / 8 / 1000000000 / true`.
- [ ] The first `prepare-world` job passes the Gen3 protocol/unit tests.
- [ ] All eight student jobs create new Gen3 students from zero.
- [ ] The cohort artifact contains telemetry, learning history, summaries and registry metadata.
- [ ] After a healthy first run, no manual duplicate runs are started; auto-continuation uses `resume`.

If the first run fails because of a code/dependency/infrastructure error before meaningful training, fix the implementation and restart the pre-registered experiment from zero. Do not use validation performance from a partial implementation run to tune the protocol.
