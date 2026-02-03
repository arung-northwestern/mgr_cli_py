@echo off
REM MCLI Installation Script for Windows
REM Installs MCLI with UV package manager for optimal performance

echo 🔬 MCLI - Molecular Geometry Analysis Tool Installation
echo ======================================================
echo.

REM Check if UV is installed
uv --version >nul 2>&1
if errorlevel 1 (
    echo 📦 Installing UV (fast package manager)...
    pip install uv
    if errorlevel 1 (
        echo ❌ Failed to install UV. Please install pip first and try again.
        pause
        exit /b 1
    )
) else (
    echo ✅ UV already installed
)

REM Create environment and install
echo 🏗️  Creating environment and installing dependencies...
uv venv mgrcli_env
if errorlevel 1 (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)

call mgrcli_env\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)

REM Install with release optimizations
echo ⚡ Building with release optimizations...
set SETUPTOOLS_RUST_RELEASE=1
set RUSTFLAGS=-C target-cpu=native -C opt-level=3 -C lto=true -C codegen-units=1

uv pip install -e . --force-reinstall --no-cache-dir
if errorlevel 1 (
    echo ❌ Failed to install MCLI
    pause
    exit /b 1
)

REM Create launcher scripts
echo 🚀 Creating command-line launchers...

REM CLI launcher
echo @echo off > mgrcli.bat
echo call "%%~dp0mgrcli_env\Scripts\activate.bat" >> mgrcli.bat
echo python "%%~dp0python_cli\mgrcli\cli.py" %%* >> mgrcli.bat

REM GUI launcher
echo @echo off > mgrcli-gui.bat
echo call "%%~dp0mgrcli_env\Scripts\activate.bat" >> mgrcli-gui.bat
echo streamlit run "%%~dp0python_cli\mgrcli\interactive_app.py" >> mgrcli-gui.bat

REM Create desktop shortcuts using PowerShell
echo 🖥️  Creating desktop shortcuts...
powershell -Command ^
"$WshShell = New-Object -comObject WScript.Shell; ^
$Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\MCLI CLI.lnk'); ^
$Shortcut.TargetPath = '%CD%\mgrcli.bat'; ^
$Shortcut.WorkingDirectory = '%CD%'; ^
$Shortcut.Save()"

powershell -Command ^
"$WshShell = New-Object -comObject WScript.Shell; ^
$Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\MCLI GUI.lnk'); ^
$Shortcut.TargetPath = '%CD%\mgrcli-gui.bat'; ^
$Shortcut.WorkingDirectory = '%CD%'; ^
$Shortcut.Save()"

echo.
echo ✅ Installation completed successfully!
echo.
echo 📋 Usage:
echo   mgrcli.bat --help                    # Show CLI help
echo   mgrcli.bat process structure.cif     # Process a CIF file
echo   mgrcli-gui.bat                       # Launch web interface
echo.
echo 🧪 Test with sample file:
echo   mgrcli.bat process cifs\ZIF-8-RASPA.cif
echo.
echo 📚 Documentation: https://github.com/arung-northwestern/mgr_cli_py
echo.
echo 💡 Tip: Desktop shortcuts created for easy access!
echo.
pause