# Trading Zero Academy v0.4.2 repair

This release repairs the malformed cumulative v0.4 package.

Fixes:
- `academy/cohort.py` and all v0.3 cohort files live at repository root where Python expects them.
- first v0.4 run restores the final Marathon cache explicitly;
- pytest only collects the real root `tests/` directory;
- legacy nested patch folders are ignored/removed;
- Final Exam supports both the M5 baseline and RAW_MTF Market Students;
- legacy Marathon workflow remains removed.

No Student #1 checkpoint is modified by applying this repair.
