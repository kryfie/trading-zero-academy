@echo off
setlocal

if not exist ".git" (
  echo ERROR: Uruchom ten plik z glownego folderu repo trading-zero-academy.
  pause
  exit /b 1
)

if exist "trading-zero-academy-v0.3.0-cohort-patch" (
  rmdir /s /q "trading-zero-academy-v0.3.0-cohort-patch"
  echo OK: usunieto zagniezdzony bledny folder v0.3 patch.
)

if exist ".pytest_cache" rmdir /s /q ".pytest_cache"
for /d /r %%D in (__pycache__) do @if exist "%%D" rmdir /s /q "%%D"

if exist ".github\workflows\academy.yml" (
  del ".github\workflows\academy.yml"
  echo OK: usunieto legacy Marathon workflow.
)

echo.
echo Repair cleanup complete.
echo Wroc do GitHub Desktop, Commit i Push.
echo.
pause
