@echo off
setlocal EnableExtensions
REM Install mgrcli on Windows from this checkout.
REM Creates %USERPROFILE%\.mgrcli\venv (override with MGRCLI_HOME). No admin required.
REM Rust: https://rustup.rs if cargo is not on PATH.
REM Python: 3.10+ (3.11-3.13 recommended).

cd /d "%~dp0\.."
if not exist "pyproject.toml" (
  echo error: run this from a clone of mgr_cli_py. pyproject.toml was not found in %CD%.
  exit /b 1
)
if not exist "mgr_src\mgrcli_rust\Cargo.toml" (
  echo error: mgr_src\mgrcli_rust\Cargo.toml is missing. This is not a complete checkout.
  exit /b 1
)

where cargo >nul 2>&1
if errorlevel 1 (
  echo error: cargo is not on PATH.
  echo Install Rust with rustup, then open a new terminal and re-run:
  echo   winget install Rustlang.Rustup
  echo   or https://rustup.rs
  exit /b 1
)

set "PY="
if defined MGRCLI_PYTHON (
  set "PY=%MGRCLI_PYTHON%"
) else (
  where py >nul 2>&1 && set "PY=py -3"
)
if not defined PY (
  where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo error: Python 3.10+ was not found. Install it from https://www.python.org/downloads/ and re-run.
  exit /b 1
)

if defined MGRCLI_HOME (
  set "PREFIX=%MGRCLI_HOME%"
) else (
  set "PREFIX=%USERPROFILE%\.mgrcli"
)
set "VENV=%PREFIX%\venv"

echo Detected Windows
echo Using prefix %PREFIX%
echo Creating virtualenv
if exist "%VENV%\Scripts\python.exe" (
  echo Reusing %VENV%
) else (
  where uv >nul 2>&1
  if not errorlevel 1 (
    uv venv "%VENV%"
  ) else (
    %PY% -m venv "%VENV%"
  )
)
if not exist "%VENV%\Scripts\python.exe" (
  echo error: virtualenv was not created at %VENV%.
  exit /b 1
)

echo Installing mgrcli. The Rust compile often takes 5-15 minutes the first time.
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"%VENV%\Scripts\python.exe" -m pip install "%CD%"
if errorlevel 1 (
  echo error: pip install failed. Confirm Rust is on PATH and Python is 3.10 or newer.
  exit /b 1
)

echo Checking the install
"%VENV%\Scripts\mgrcli.exe" --version
if errorlevel 1 exit /b 1
"%VENV%\Scripts\python.exe" -c "import mgrcli, mgr, mgrcli_rust; print('import ok', mgrcli.__version__)"
if errorlevel 1 exit /b 1

echo.
echo mgrcli is installed.
echo Activate it with:
echo   %VENV%\Scripts\activate.bat
echo Or call:
echo   %VENV%\Scripts\mgrcli.exe --version
if exist "cifs\ZIF-8-RASPA.cif" (
  echo Sample:
  echo   %VENV%\Scripts\mgrcli.exe process cifs\ZIF-8-RASPA.cif -o %PREFIX%\verify_out
)
exit /b 0
