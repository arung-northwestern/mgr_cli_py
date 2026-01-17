#!/bin/bash
# Package MGRUI for Linux distribution
# Creates a distributable tar.gz package with everything included

set -e

# Generate date-based version: YY-MM-DD-#
get_version() {
    local date_str=$(date +%y-%m-%d)
    local build_num=1

    # Check for existing builds with today's date
    if ls build_package/mgrui-linux-${date_str}-*.tar.gz 2>/dev/null | head -1; then
        # Get the highest build number for today
        local latest=$(ls build_package/mgrui-linux-${date_str}-*.tar.gz 2>/dev/null | sed "s/.*${date_str}-//" | sed 's/\.tar\.gz$//' | sort -n | tail -1)
        if [ -n "$latest" ]; then
            build_num=$((latest + 1))
        fi
    fi

    echo "${date_str}-${build_num}"
}

# Parse command line arguments
ARCH=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --arch)
            ARCH="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--arch ARCH]"
            echo "  --arch ARCH  Target architecture (x86_64, arm64, aarch64)"
            echo "               If not specified, auto-detected via uname -m"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Detect current architecture if not specified
if [ -z "$ARCH" ]; then
    ARCH=$(uname -m)
fi

case "$ARCH" in
    x86_64) ARCH_TAG="x86_64" ;;
    aarch64|arm64) ARCH_TAG="arm64" ;;
    *) ARCH_TAG="$ARCH" ;;
esac

VERSION=$(get_version)
BUILD_DIR="build_package"
PACKAGE_NAME="mgrui"

echo "📦 Packaging MGRUI for Linux..."
echo "📅 Version: $VERSION"
echo "🏗️  Architecture: $ARCH_TAG"
echo ""

# Clean and create build directory
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/$PACKAGE_NAME"

echo "📋 Setting up package structure..."

# Save current directory
ORIGINAL_DIR="$(pwd)"

# Check if icon.png exists and convert to .png for Linux
if [ -f "icon.png" ]; then
    echo "   Copying app icon..."
    cp icon.png "$BUILD_DIR/$PACKAGE_NAME/mgrui.png"
fi

# Create Python environment using UV
echo "🏗️  Creating Python environment..."

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "   Installing UV..."
    pip install uv
fi

PYTHON_VERSION="3.11"

echo "   Installing Python runtime $PYTHON_VERSION..."
uv python install "$PYTHON_VERSION"
PYTHON_BIN=$(uv python find "$PYTHON_VERSION")
if [ -z "$PYTHON_BIN" ]; then
    echo "❌ Unable to locate Python runtime for $PYTHON_VERSION"
    exit 1
fi

# Create venv in the package directory using copies for relocatability
cd "$BUILD_DIR/$PACKAGE_NAME"
"$PYTHON_BIN" -m venv --copies mgrui_env
VENV_PY="$BUILD_DIR/$PACKAGE_NAME/mgrui_env/bin/python"

# Install with release optimizations
echo "   ⚡  Installing dependencies..."
export SETUPTOOLS_RUST_RELEASE=1
export RUSTFLAGS="-C target-cpu=native -C opt-level=3"

# Go back to original repo for installation
cd "$ORIGINAL_DIR"
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install . --force-reinstall --no-cache-dir

# Rewrite pyvenv.cfg to avoid builder-absolute paths
export PYVENV_CFG="$BUILD_DIR/$PACKAGE_NAME/mgrui_env/pyvenv.cfg"
"$VENV_PY" - <<'PY'
import os
from pathlib import Path

cfg_path = Path(os.environ["PYVENV_CFG"])
if cfg_path.exists():
    lines = cfg_path.read_text().splitlines()
    new_lines = []
    for line in lines:
        if line.startswith("home ="):
            new_lines.append("home = .")
        else:
            new_lines.append(line)
    cfg_path.write_text("\n".join(new_lines) + "\n")
PY

# Copy source files and cifs into the package directory
echo "   Copying source files and sample data..."

# Copy python_cli
if [ -d "python_cli" ]; then
    cp -R python_cli "$BUILD_DIR/$PACKAGE_NAME/"
fi

# Copy mgr_src
if [ -d "mgr_src" ]; then
    cp -R mgr_src "$BUILD_DIR/$PACKAGE_NAME/"
fi

# Copy .streamlit config
if [ -d ".streamlit" ]; then
    cp -R .streamlit "$BUILD_DIR/$PACKAGE_NAME/"
fi

# Copy cifs folder if it exists
if [ -d "cifs" ]; then
    echo "   Copying cifs folder..."
    cp -R cifs "$BUILD_DIR/$PACKAGE_NAME/"
fi

# Create launcher script
echo "   Creating launcher script..."
cat > "$BUILD_DIR/$PACKAGE_NAME/mgrui" << 'EOF'
#!/bin/bash
# MGRUI Launcher for Linux

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Try to find mgrui_env
if [ -d "$SCRIPT_DIR/mgrui_env" ]; then
    MGRUI_ENV="$SCRIPT_DIR/mgrui_env"
    APP_DIR="$SCRIPT_DIR"
else
    echo "Error: MGRUI environment not found. Please reinstall MGRUI."
    exit 1
fi

# Change to the app directory
cd "$APP_DIR"

# Activate the virtual environment
if [ -d "$MGRUI_ENV" ]; then
    source "$MGRUI_ENV/bin/activate"
else
    echo "Error: MGRUI environment not found. Please reinstall MGRUI."
    exit 1
fi

# Set environment
export PYTHONPATH="$(pwd)/python_cli:$(pwd):$PYTHONPATH"
export MGRUI_SAMPLE_CIF_DIR="$(pwd)/cifs"

# Check if streamlit is already running
if lsof -i :8501 -sTCP:LISTEN -t > /dev/null 2>&1; then
    echo "MGRUI is already running!"
    echo "Opening browser..."
    # Open the existing server in browser
    xdg-open http://localhost:8501 2>/dev/null || sensible-browser http://localhost:8501 2>/dev/null || echo "Please open http://localhost:8501 in your browser"
else
    echo "Starting MGRUI server..."
    # Launch Streamlit in background
    streamlit run python_cli/mgrui/interactive_app.py &
    STREAMLIT_PID=$!

    # Wait a moment for server to start, then open browser
    sleep 2
    xdg-open http://localhost:8501 2>/dev/null || sensible-browser http://localhost:8501 2>/dev/null || echo "Please open http://localhost:8501 in your browser"

    echo "MGRUI server started (PID: $STREAMLIT_PID)"
    echo "To stop the server, press Ctrl+C or run:"
    echo "  pkill -f 'streamlit run'"
fi
EOF

chmod +x "$BUILD_DIR/$PACKAGE_NAME/mgrui"

# Create quit script
cat > "$BUILD_DIR/$PACKAGE_NAME/quit-mgrui.sh" << 'EOF'
#!/bin/bash
# Quit MGRUI - Stop the Streamlit server

echo "Stopping MGRUI server..."
STREAMLIT_PID=$(pgrep -f "streamlit run" | head -1)
if [ -n "$STREAMLIT_PID" ]; then
    kill -9 "$STREAMLIT_PID"
    echo "MGRUI server stopped (PID: $STREAMLIT_PID)"
else
    echo "MGRUI server is not running."
fi
EOF

chmod +x "$BUILD_DIR/$PACKAGE_NAME/quit-mgrui.sh"

# Create .desktop file for app launcher
if [ -f "$BUILD_DIR/$PACKAGE_NAME/mgrui.png" ]; then
    cat > "$BUILD_DIR/$PACKAGE_NAME/mgrui.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=MGRUI
Comment=Molecular Geometry Analysis
Exec=\$PWD/mgrui
Icon=\$PWD/mgrui.png
Terminal=true
Categories=Science;Chemistry;Physics;
EOF
fi

# Go back to original directory
cd "$ORIGINAL_DIR"

# ============================================================================
# CLEANUP: Remove unnecessary files to reduce bundle size
# ============================================================================
echo "   🧹 Cleaning up unnecessary files to reduce bundle size..."

# Remove Python cache files (~65 MB)
echo "   Removing Python cache files..."
find "$BUILD_DIR/$PACKAGE_NAME/mgrui_env" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
find "$BUILD_DIR/$PACKAGE_NAME/mgrui_env" -type f -name '*.pyc' -delete 2>/dev/null || true

# Remove test directories (~21 MB)
echo "   Removing test directories..."
find "$BUILD_DIR/$PACKAGE_NAME/mgrui_env" -type d -name 'tests' -exec rm -rf {} + 2>/dev/null || true
find "$BUILD_DIR/$PACKAGE_NAME/mgrui_env" -type d -name 'test' -exec rm -rf {} + 2>/dev/null || true
rm -rf "$BUILD_DIR/$PACKAGE_NAME/python_cli/tests" 2>/dev/null || true

# Remove Jupyter data (~18 MB)
echo "   Removing Jupyter data..."
rm -rf "$BUILD_DIR/$PACKAGE_NAME/mgrui_env/share/jupyter" 2>/dev/null || true

# NOTE: Do NOT remove .dist-info directories - packages need their metadata to work!

# Remove Rust build artifacts (~197 MB)
echo "   Removing Rust build artifacts..."
rm -rf "$BUILD_DIR/$PACKAGE_NAME/mgr_src/mgrui_rust/target" 2>/dev/null || true

# Remove pre-generated HTML files (~87 MB)
echo "   Removing pre-generated HTML files..."
rm -f "$BUILD_DIR/$PACKAGE_NAME/cifs"/*.html 2>/dev/null || true
rm -f "$BUILD_DIR/$PACKAGE_NAME/mgr_src/mgrui_rust/interactive_plot.html" 2>/dev/null || true

# NOTE: Do NOT remove pyarrow - Streamlit requires it for st.dataframe()

echo "   ✅ Cleanup complete!"

echo "📝 Creating README..."

cat > "$BUILD_DIR/$PACKAGE_NAME/README.txt" << 'EOF'
MGRUI - Molecular Geometry Analysis Tool
========================================

QUICK START
-----------

1. Extract the archive:
   tar -xzf mgrui-linux-*.tar.gz
   cd mgrui

2. Run MGRUI:
   ./mgrui

3. A browser window will open automatically at http://localhost:8501

4. To stop the server:
   ./quit-mgrui.sh
   or press Ctrl+C in the terminal

ALTERNATIVE: Install to Applications
-------------------------------------
1. Copy the MGRUI folder to /opt/MGRUI:
   sudo cp -R mgrui /opt/MGRUI

2. Create a symlink in /usr/local/bin:
   sudo ln -s /opt/MGRUI/mgrui /usr/local/bin/mgrui

3. Copy the desktop file:
   cp mgrui.desktop ~/.local/share/applications/

4. Launch from your application menu

WHAT'S INCLUDED
--------------
✓ Complete Python environment (built-in)
✓ All dependencies (numpy, scipy, streamlit, etc.)
✓ MGRUI application
✓ Sample CIF files

SYSTEM REQUIREMENTS
-------------------
- Linux (x86_64 or ARM64)
- Python 3.11+ (will be installed if needed)
- 500 MB free disk space
- 4 GB RAM recommended

DEPENDENCIES REQUIRED
---------------------
The package includes most dependencies, but you need:
- A C compiler (gcc/clang) for building Python packages
- Rust toolchain (for mgrui_rust backend)

To install Rust:
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

TROUBLESHOOTING
---------------
Q: "./mgrui: Permission denied"
A: Run: chmod +x mgrui

Q: Port 8501 already in use
A: MGRUI will automatically try ports 8502, 8503, etc.

Q: Browser doesn't open
A: Manually open http://localhost:8501 in your browser

Q: Where can I find help?
A: Visit https://github.com/arung-northwestern/mgr_cli_py

UNINSTALLING
-----------
To remove MGRUI:
1. Delete the extracted directory
2. If installed to /opt: sudo rm -rf /opt/MGRUI
3. Remove symlink: sudo rm /usr/local/bin/mgrui
4. Remove desktop file: rm ~/.local/share/applications/mgrui.desktop

That's it! Everything is self-contained.
EOF

echo "📦 Creating tar.gz package..."

cd "$BUILD_DIR"
tar czf "mgrui-linux-${ARCH_TAG}-$VERSION.tar.gz" mgrui/

cd - > /dev/null

echo ""
echo "✅ Package created: $BUILD_DIR/mgrui-linux-${ARCH_TAG}-$VERSION.tar.gz"
echo ""
echo "📊 Package size:"
du -sh "$BUILD_DIR/mgrui-linux-${ARCH_TAG}-$VERSION.tar.gz"
echo ""
echo "🚀 Ready to distribute!"
echo ""
echo "📝 To test locally:"
echo "   cd $BUILD_DIR"
echo "   tar xzf mgrui-linux-${ARCH_TAG}-$VERSION.tar.gz"
echo "   cd mgrui"
echo "   ./mgrui"
echo ""
echo "📋 To upload to GitHub:"
echo "   1. Go to: https://github.com/arung-northwestern/mgr_cli_py/releases/new"
echo "   2. Upload: $BUILD_DIR/mgrui-linux-${ARCH_TAG}-$VERSION.tar.gz"
echo ""
