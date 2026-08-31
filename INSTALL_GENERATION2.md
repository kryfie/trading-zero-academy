# INSTALL GENERATION 2

Generation 1 is already archived. Its Git history remains in the repository and the
188 MB final archive should be kept outside GitHub.

The active repository can now be cleaned aggressively.

## Recommended cleanup

Keep the hidden `.git` directory.

Delete the old active Generation-1 files/folders, then copy the contents of the
Generation-2 package into the repository root.

It is safe to remove the old:
- `.github/workflows/*`
- `academy/*`
- `scripts/*`
- `tests/*`
- old root README/version/repair files
- old `config.yaml`
- old `cohort.yml`
- old `students` working files
- old local `data` working files

Do not add the 188 MB Generation-1 model archive to the repository.

The Generation-2 package already contains:
- the only two active workflows
- clean Gen2 Python code
- frozen Gen1 partition reference
- `.gitignore`
- empty `data/mtf` and `students` placeholders
- protocol documentation

After replacement:
1. Commit: `Generation 2 - discrete action experiment`
2. Push
3. Open Actions
4. Run `Trading Zero Academy — Generation 2`
5. On the FIRST run only select `run_mode = start`
6. Keep 8 students, 500000000 target, auto_continue=true

If the first run reaches all 8 `train-students` jobs, Generation 2 is live.
