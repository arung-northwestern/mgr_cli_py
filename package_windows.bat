@echo off
REM Package MCLI for Windows distribution
REM Creates a distributable ZIP package with everything included

setlocal enabledelayedexpansion

REM Generate date-based version: YY-MM-DD-#
set DATE_STR=%date:~12,2%-%date:~4,2%-%date:~7,2%
set BUILD_NUM=1

REM Check for existing builds with today's date
if exist build_package_windows\mgrui-Windows-%DATE_STR%-*.zip (
    REM Get the highest build number for today
    for /f "delims=" %%i in ('dir /b build_package_windows\mgrui-Windows-%DATE_STR%-*.zip 2^>nul') do (
        set LATEST=%%i
    )
    if defined LATEST (
        for /f "tokens=4 delims=-" %%i in ("%LATEST%") do (
            set /a BUILD_NUM=%%i+1
        )
    )
)

set VERSION=%DATE_STR%-%BUILD_NUM%
set BUILD_DIR=build_package_windows
set PACKAGE_NAME=MCLI

echo 📦 Packaging MCLI for Windows...
echo 📅 Version: %VERSION%
echo.

REM Clean and create build directory
if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
mkdir "%BUILD_DIR%\%PACKAGE_NAME%"

echo 📋 Setting up package structure...

REM Save current directory
set ORIGINAL_DIR=%CD%

REM Copy icon if it exists
if exist "icon.png" (
    echo    Copying app icon...
    copy "icon.png" "%BUILD_DIR%\%PACKAGE_NAME%\mgrui.png" >nul
)

REM Create Python environment using UV
echo 🏗️  Creating Python environment...

REM Check if UV is installed
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo    Installing UV...
    pip install uv
)

set PYTHON_VERSION=3.11

echo    Installing Python runtime %PYTHON_VERSION%...
uv python install %PYTHON_VERSION%
for /f "delims=" %%i in ('uv python find %PYTHON_VERSION%') do set PYTHON_BIN=%%i
if not defined PYTHON_BIN (
    echo    Unable to locate Python runtime for %PYTHON_VERSION%
    exit /b 1
)

REM Create venv in the package directory using copies for relocatability
cd /d "%BUILD_DIR%\%PACKAGE_NAME%"
"%PYTHON_BIN%" -m venv --copies mgrui_env
set VENV_PY=%BUILD_DIR%\%PACKAGE_NAME%\mgrui_env\Scripts\python.exe

REM Install with release optimizations
echo    ⚡  Installing dependencies...
set SETUPTOOLS_RUST_RELEASE=1
set RUSTFLAGS=-C target-cpu=native -C opt-level=3

REM Go back to original repo for installation
cd /d "%ORIGINAL_DIR%"
"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install . --force-reinstall --no-cache-dir

REM Rewrite pyvenv.cfg to avoid builder-absolute paths
set PYVENV_CFG=%BUILD_DIR%\%PACKAGE_NAME%\mgrui_env\pyvenv.cfg
powershell -Command "$path = '%PYVENV_CFG%'; if (Test-Path $path) { $lines = Get-Content -Raw $path; $lines = $lines -split ""`n"" | ForEach-Object { if ($_ -like 'home =*') { 'home = .' } else { $_ } }; [IO.File]::WriteAllLines($path, $lines) }"

REM Copy source files and cifs into the package directory
echo    Copying source files and sample data...

REM Copy python_cli
if exist "python_cli" (
    xcopy /E /I /Y python_cli "%BUILD_DIR%\%PACKAGE_NAME%\python_cli" >nul
)

REM Copy mgr_src
if exist "mgr_src" (
    xcopy /E /I /Y mgr_src "%BUILD_DIR%\%PACKAGE_NAME%\mgr_src" >nul
)

REM Copy .streamlit config
if exist ".streamlit" (
    xcopy /E /I /Y .streamlit "%BUILD_DIR%\%PACKAGE_NAME%\.streamlit" >nul
)

REM Copy cifs folder if it exists
if exist "cifs" (
    echo    Copying cifs folder...
    xcopy /E /I /Y cifs "%BUILD_DIR%\%PACKAGE_NAME%\cifs" >nul
)

REM Create launcher script
echo    Creating launcher script...

(
echo @echo off
echo REM MCLI Launcher for Windows
echo.
echo REM Get the directory where this script is located
echo set SCRIPT_DIR=%~dp0
echo.
echo REM Try to find mgrui_env
echo if exist "!SCRIPT_DIR!mgrui_env" (
echo     set MCLI_ENV=!SCRIPT_DIR!mgrui_env
echo     set APP_DIR=!SCRIPT_DIR!
echo ^) else (
echo     echo Error: MCLI environment not found. Please reinstall MCLI.
echo     exit /b 1
echo ^)
echo.
echo REM Change to the app directory
echo cd /d "!APP_DIR!"
echo.
echo REM Activate the virtual environment
echo if exist "!MCLI_ENV!\Scripts\activate.bat" (
echo     call "!MCLI_ENV!\Scripts\activate.bat"
echo ^) else (
echo     echo Error: MCLI environment not found. Please reinstall MCLI.
echo     exit /b 1
echo ^)
echo.
echo REM Set environment
echo set PYTHONPATH=!CD!;!PYTHONPATH!
echo set MCLI_SAMPLE_CIF_DIR=!CD!\cifs
echo.
echo REM Launch Streamlit
echo streamlit run python_cli\mgrui\interactive_app.py
) > "%BUILD_DIR%\%PACKAGE_NAME%\mgrui-gui.bat"

REM Go back to original directory
cd /d "%ORIGINAL_DIR%"

echo 📝 Creating README...

(
echo MCLI - Molecular Geometry Analysis Tool
echo ========================================
echo.
echo QUICK START
echo -----------
echo.
echo 1. Extract the ZIP to your desired location
echo 2. Double-click "mgrui-gui.bat"
echo 3. A browser window will open automatically
echo 4. That's it!
echo.
echo WHAT'S INCLUDED
echo --------------
echo ✓ Complete Python environment
echo ✓ All dependencies ^(numpy, scipy, streamlit, etc.^)
echo ✓ MCLI application
echo ✓ Sample CIF files ^(in cifs/ folder^)
echo.
echo SYSTEM REQUIREMENTS
echo -------------------
echo - Windows 10 or later
echo - 500 MB free disk space
echo - 4 GB RAM recommended
echo.
echo TROUBLESHOOTING
echo ---------------
echo Q: Windows SmartScreen prevents running
echo A: Click "More info" then "Run anyway"
echo.
echo Q: App won't start
echo A: Make sure you're running Windows 10 or later
echo.
echo Q: Where can I find help?
echo A: Visit https://github.com/arung-northwestern/mgr_cli_py
echo.
echo UNINSTALLING
echo -----------
echo To remove MCLI:
echo 1. Delete the MCLI folder
echo.
echo That's it! Everything is self-contained.
) > "%BUILD_DIR%\%PACKAGE_NAME%\README.txt"

echo 📦 Creating ZIP package...

cd /d "%BUILD_DIR%"

REM Create ZIP using PowerShell
powershell -Command "Compress-Archive -Path '%PACKAGE_NAME%' -DestinationPath 'mgrui-Windows-%VERSION%.zip' -Force"

cd /d "%ORIGINAL_DIR%"

echo.
echo ✅ Package created: %BUILD_DIR%\mgrui-Windows-%VERSION%.zip
echo.
echo 📊 Package size:
for %%A in ("%BUILD_DIR%\mgrui-Windows-%VERSION%.zip") do echo %%~zA bytes
echo.
echo 🚀 Ready to distribute!
echo.
echo 📝 To test locally:
echo    1. Extract %BUILD_DIR%\mgrui-Windows-%VERSION%.zip
echo    2. cd %PACKAGE_NAME%
echo    3. mgrui-gui.bat
echo.
echo 📋 To upload to GitHub:
echo    1. Go to: https://github.com/arung-northwestern/mgr_cli_py/releases/new
echo    2. Upload: %BUILD_DIR%\mgrui-Windows-%VERSION%.zip
echo.

endlocal
