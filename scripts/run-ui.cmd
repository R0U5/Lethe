@echo off
REM Launch the lethe web UI on this project's own .venv interpreter.
REM
REM `abliterate ui` runs on whichever Python is first on PATH. A bare
REM `pip install -e .` also installs an `abliterate` launcher into a global
REM interpreter, and that older copy wins -- so the UI silently serves with a
REM transformers that may not know the model you are loading. Naming the
REM interpreter here means the answer is never in doubt.
REM
REM Usage:
REM   scripts\run-ui.cmd
REM   scripts\run-ui.cmd --port 8000 --no-browser
REM
setlocal
set "HERE=%~dp0"

pushd "%HERE%.."

if not exist ".venv\Scripts\python.exe" (
    echo No .venv yet - creating it first ^(one time, downloads torch^)...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%setup-env.ps1"
    if errorlevel 1 goto setupfailed
)

".venv\Scripts\python.exe" -m abliterate.cli ui %*
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%

:setupfailed
echo.
echo Environment setup failed - see the messages above.
popd
exit /b 1
