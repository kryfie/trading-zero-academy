@echo off
setlocal
echo Cleaning obsolete Generation 1 files. New Generation 2 files are not touched.

del /q ".github\workflows\archive_generation1.yml" 2>nul
del /q ".github\workflows\cohort.yml" 2>nul
del /q ".github\workflows\final-exam.yml" 2>nul

del /q "ACADEMY_RULES.md" 2>nul
del /q "CLOSE_GENERATION_1.md" 2>nul
del /q "COHORT_MODE.md" 2>nul
del /q "GENERATION_1_FINAL.md" 2>nul
del /q "MARKET_STUDENTS_v0.4.md" 2>nul
del /q "README_v0.2.4.md" 2>nul
del /q "README_v0.2.5.md" 2>nul
del /q "REPAIR_v0.4.2.md" 2>nul
del /q "apply_v0.4.2_cleanup.bat" 2>nul
del /q "cohort.yml" 2>nul

rmdir /s /q "data\processed" 2>nul
rmdir /s /q "data\raw" 2>nul
rmdir /s /q "models" 2>nul

del /q "academy\dataset.py" 2>nul
del /q "academy\env.py" 2>nul
del /q "academy\marathon.py" 2>nul
del /q "academy\mtf_dataset.py" 2>nul

del /q "scripts\download_data.py" 2>nul
del /q "scripts\download_mtf_data.py" 2>nul
del /q "scripts\marathon_decision.py" 2>nul
del /q "scripts\marathon_train.py" 2>nul
del /q "scripts\status.py" 2>nul
del /q "scripts\train.py" 2>nul

del /q "tests\test_cohort.py" 2>nul
del /q "tests\test_data_refresh.py" 2>nul
del /q "tests\test_env.py" 2>nul
del /q "tests\test_marathon.py" 2>nul
del /q "tests\test_mtf.py" 2>nul

echo.
echo Generation 1 active files cleaned.
echo Keep the hidden .git directory and your external 188 MB Generation 1 archive.
pause
