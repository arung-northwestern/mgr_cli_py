#!/bin/bash
# MCLI Installation Script for Linux/macOS
# Installs MCLI with UV package manager for optimal performance

set -e

echo "🔬 MCLI - Molecular Geometry Analysis Tool Installation"
echo "======================================================"
echo ""

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing UV (fast package manager)..."
    pip install uv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install UV. Please install pip first and try again."
        exit 1
    fi
else
    echo "✅ UV already installed"
fi

# Create environment and install
echo "🏗️  Creating environment and installing dependencies..."
uv venv mcli_env
if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

source mcli_env/bin/activate
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Install with release optimizations
echo "⚡ Building with release optimizations..."
export SETUPTOOLS_RUST_RELEASE=1
export RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=true -C codegen-units=1"

uv pip install -e . --force-reinstall --no-cache-dir
if [ $? -ne 0 ]; then
    echo "❌ Failed to install MCLI"
    exit 1
fi

# Create launcher scripts
echo "🚀 Creating command-line launchers..."

# CLI launcher
cat > mcli << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source mcli_env/bin/activate
python python_cli/mcli/cli.py "$@"
EOF
chmod +x mcli

# GUI launcher
cat > mcli-gui << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source mcli_env/bin/activate
streamlit run python_cli/mcli/interactive_app.py
EOF
chmod +x mcli

# Create desktop entry for Linux (if possible)
if command -v desktop-file-install &> /dev/null && [ -d "$HOME/.local/share/applications" ]; then
    echo "🖥️  Creating desktop shortcuts..."
    CURRENT_DIR=$(pwd)
    cat > ~/.local/share/applications/mcli.desktop << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=MCLI CLI
Comment=Molecular Geometry Analysis CLI
Exec=$CURRENT_DIR/mcli
Icon=applications-utilities
Terminal=true
Categories=Science;
EOF

    cat > ~/.local/share/applications/mcli-gui.desktop << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=MCLI GUI
Comment=Molecular Geometry Analysis Web Interface
Exec=$CURRENT_DIR/mcli-gui
Icon=applications-utilities
Terminal=false
Categories=Science;
EOF
fi

echo ""
echo "✅ Installation completed successfully!"
echo ""
echo "📋 Usage:"
echo "  ./mcli --help                    # Show CLI help"
echo "  ./mcli process structure.cif     # Process a CIF file"
echo "  ./mcli-gui                       # Launch web interface"
echo ""
echo "🧪 Test with sample file:"
echo "  ./mcli process cifs/ZIF-8-RASPA.cif"
echo ""
echo "📚 Documentation: https://github.com/arung-northwestern/mgr_cli_py"
echo ""
echo "💡 Tip: Add $(pwd) to your PATH to use 'mcli' and 'mcli-gui' from anywhere"