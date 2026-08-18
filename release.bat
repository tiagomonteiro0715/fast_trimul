@echo off
REM ============================================================================
REM  Release fast_trimul: bump version, push commit + tag to GitHub.
REM
REM  The pushed tag (e.g. v2.4.31) triggers .github/workflows/publish.yml, which
REM  builds and publishes to PyPI via Trusted Publishing -- so there is NO PyPI
REM  token here. This only touches git.
REM
REM  Fill in the two settings below, then run from the repo root:  release.bat
REM ============================================================================

REM ---- EDIT THESE TWO EVERY RELEASE -----------------------------------------
REM  BUMP: major | minor | patch      (patch: 2.4.30 -> 2.4.31)
set BUMP=patch

REM  DESCRIPTION: one-line summary (goes into the commit message)
set DESCRIPTION=Updating README file for minor changes
REM ---------------------------------------------------------------------------

set BRANCH=main

where uv  >nul 2>nul || (echo ERROR: uv not found.  Install: https://docs.astral.sh/uv/ & exit /b 1)
where git >nul 2>nul || (echo ERROR: git not found. & exit /b 1)
if "%DESCRIPTION%"=="" (echo ERROR: set DESCRIPTION at the top of this file first. & exit /b 1)

echo [1/5] Syncing with GitHub (branch %BRANCH%)...
git checkout %BRANCH% || goto :error
git pull --rebase --autostash origin %BRANCH% || goto :error

echo [2/5] Bumping version: %BUMP%
uv version --bump %BUMP% --frozen || goto :error
for /f "delims=" %%v in ('uv version --short') do set VER=%%v
echo        new version: %VER%

echo [3/5] Committing...
git add -A || goto :error
git commit -m "Release v%VER%: %DESCRIPTION%" || goto :error

echo [4/5] Tagging v%VER% + pushing...
git tag v%VER%
git push origin %BRANCH% || goto :error
git push origin v%VER% || goto :error

echo.
echo Done. Pushing tag v%VER% started the Publish workflow.
echo   Watch it:  https://github.com/tiagomonteiro0715/fast_trimul/actions
echo   PyPI:      https://pypi.org/project/fast_trimul/  (appears when the run finishes)
goto :eof

:error
echo.
echo RELEASE FAILED - see the error above.
exit /b 1
