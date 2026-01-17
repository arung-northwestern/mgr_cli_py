#!/bin/bash
# Package MGRUI for distribution
# Creates a distributable package for the current architecture
# For universal binary support, build separately on arm64 and x86_64 machines

set -e

# Generate date-based version: YY-MM-DD-#
get_version() {
    local date_str=$(date +%y-%m-%d)
    local build_num=1

    # Check for existing builds with today's date (pattern: mgrui-macOS-*-YY-MM-DD-#.dmg)
    if ls build_package/mgrui-macOS-*-*.dmg 2>/dev/null | grep -q "${date_str}"; then
        # Get the highest build number for today
        local latest=$(ls build_package/mgrui-macOS-*-*.dmg 2>/dev/null | grep "${date_str}" | sed "s/.*${date_str}-//" | sed 's/\.dmg$//' | sort -n | tail -1)
        if [ -n "$latest" ]; then
            build_num=$((latest + 1))
        fi
    fi

    echo "${date_str}-${build_num}"
}

VERSION=$(get_version)
BUILD_DIR="build_package"
PACKAGE_NAME="mgrui"

# Detect current architecture
ARCH=$(uname -m)
case "$ARCH" in
    arm64) ARCH_TAG="arm64" ;;
    x86_64) ARCH_TAG="x86_64" ;;
    *) ARCH_TAG="$ARCH" ;;
esac

echo "📦 Packaging MGRUI for distribution..."
echo "📅 Version: $VERSION"
echo "🏗️  Architecture: $ARCH ($ARCH_TAG)"
echo ""

# Clean and create build directory
# IMPORTANT: Only remove the package build dir, not the DMG files!
rm -rf "$BUILD_DIR/$PACKAGE_NAME"
mkdir -p "$BUILD_DIR/$PACKAGE_NAME"

echo "📋 Setting up package structure..."

# Save current directory
ORIGINAL_DIR="$(pwd)"

# Generate macOS icon from icon.png if it exists
if [ -f "icon.png" ]; then
    echo "   Generating app icon..."
    ICONSET_DIR="$(mktemp -d)/mgrui.iconset"
    mkdir -p "$ICONSET_DIR"

    # Generate all required sizes for macOS icon
    sips -z 16 16 icon.png --out "$ICONSET_DIR/icon_16x16.png" 2>/dev/null || true
    sips -z 32 32 icon.png --out "$ICONSET_DIR/icon_16x16@2x.png" 2>/dev/null || true
    sips -z 32 32 icon.png --out "$ICONSET_DIR/icon_32x32.png" 2>/dev/null || true
    sips -z 64 64 icon.png --out "$ICONSET_DIR/icon_32x32@2x.png" 2>/dev/null || true
    sips -z 128 128 icon.png --out "$ICONSET_DIR/icon_128x128.png" 2>/dev/null || true
    sips -z 256 256 icon.png --out "$ICONSET_DIR/icon_128x128@2x.png" 2>/dev/null || true
    sips -z 256 256 icon.png --out "$ICONSET_DIR/icon_256x256.png" 2>/dev/null || true
    sips -z 512 512 icon.png --out "$ICONSET_DIR/icon_256x256@2x.png" 2>/dev/null || true
    sips -z 512 512 icon.png --out "$ICONSET_DIR/icon_512x512.png" 2>/dev/null || true
    sips -z 1024 1024 icon.png --out "$ICONSET_DIR/icon_512x512@2x.png" 2>/dev/null || true

    # Create .icns file
    iconutil -c icns "$ICONSET_DIR" -o /tmp/mgrui.icns 2>/dev/null || true

    # Copy to app bundle
    if [ -f "/tmp/mgrui.icns" ]; then
        mkdir -p install/mgrui.app/Contents/Resources
        cp /tmp/mgrui.icns install/mgrui.app/Contents/Resources/
        echo "   Icon generated successfully!"
    fi
fi

# Source app bundle (create a minimal template if missing)
APP_TEMPLATE="install/mgrui.app"
if [ ! -d "$APP_TEMPLATE" ]; then
    echo "   Creating minimal mgrui.app template..."
    TMP_APP_DIR="$(mktemp -d)/mgrui.app"
    mkdir -p "$TMP_APP_DIR/Contents/MacOS" "$TMP_APP_DIR/Contents/Resources"

    cat > "$TMP_APP_DIR/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key><string>en</string>
    <key>CFBundleExecutable</key><string>mgrui</string>
    <key>CFBundleIdentifier</key><string>org.mgrui.app</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundleName</key><string>mgrui</string>
    <key>CFBundleDisplayName</key><string>mgrui</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>1.0</string>
    <key>CFBundleVersion</key><string>1.0</string>
    <key>LSMinimumSystemVersion</key><string>11.0</string>
</dict>
</plist>
EOF

    cat > "$TMP_APP_DIR/Contents/MacOS/mgrui" <<'EOF'
#!/bin/bash
# Minimal MGRUI launcher

MACOS_DIR="$(dirname "$0")"
APP_DIR="$MACOS_DIR"
MGRUI_ENV="$MACOS_DIR/mgrui_env"

LOG_FILE="$MACOS_DIR/mgrui_launcher.log"
exec > "$LOG_FILE" 2>&1

echo "=== MGRUI Launcher $(date) ==="

if [ ! -d "$MGRUI_ENV" ]; then
    echo "ERROR: mgrui_env not found at $MGRUI_ENV"
    exit 1
fi

PYTHON_BIN="$MGRUI_ENV/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: Python interpreter not found at $PYTHON_BIN"
    exit 1
fi

cd "$APP_DIR"
export PYTHONPATH="$(pwd)/python_cli:$(pwd):$PYTHONPATH"
export MGRUI_SAMPLE_CIF_DIR="$(pwd)/cifs"

if lsof -i :8501 -sTCP:LISTEN -t > /dev/null 2>&1; then
    open http://localhost:8501
else
    "$PYTHON_BIN" -m streamlit run python_cli/mgrui/interactive_app.py &
    sleep 3
    open http://localhost:8501
fi
EOF
    chmod +x "$TMP_APP_DIR/Contents/MacOS/mgrui"
    APP_TEMPLATE="$TMP_APP_DIR"
fi

echo "   Copying mgrui.app..."
cp -R "$APP_TEMPLATE" "$BUILD_DIR/$PACKAGE_NAME/"

# Update Info.plist with current version
echo "   Updating version in Info.plist..."
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VERSION" "$BUILD_DIR/$PACKAGE_NAME/mgrui.app/Contents/Info.plist" 2>/dev/null || true
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $VERSION" "$BUILD_DIR/$PACKAGE_NAME/mgrui.app/Contents/Info.plist" 2>/dev/null || true

# Create Python environment inside the app bundle
echo "🏗️  Creating Python environment inside mgrui.app..."

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "   Installing UV..."
    pip install uv
fi

PYTHON_VERSION="3.11"

# Install standalone Python runtime and build a copy-based venv (no symlinks)
echo "   Installing Python runtime $PYTHON_VERSION..."
uv python install "$PYTHON_VERSION"
PYTHON_BIN=$(uv python find "$PYTHON_VERSION")
if [ -z "$PYTHON_BIN" ]; then
    echo "❌ Unable to locate Python runtime for $PYTHON_VERSION"
    exit 1
fi

# Create venv inside the app bundle using copies for relocatability
cd "$BUILD_DIR/$PACKAGE_NAME/mgrui.app/Contents/MacOS"
"$PYTHON_BIN" -m venv --copies mgrui_env
VENV_PY="$BUILD_DIR/$PACKAGE_NAME/mgrui.app/Contents/MacOS/mgrui_env/bin/python"

# Install with release optimizations (same as install.sh)
echo "   ⚡  Installing dependencies..."
export SETUPTOOLS_RUST_RELEASE=1
export RUSTFLAGS="-C target-cpu=native -C opt-level=3"

# Go back to original repo for installation
cd "$ORIGINAL_DIR"
# IMPORTANT: Do NOT use -e (editable install) - it creates symlinks to source dir
# We need a regular install so all packages are copied into the app bundle
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install . --force-reinstall --no-cache-dir

# Rewrite pyvenv.cfg to avoid builder-absolute paths
export PYVENV_CFG="$BUILD_DIR/$PACKAGE_NAME/mgrui.app/Contents/MacOS/mgrui_env/pyvenv.cfg"
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

# Sanity check: ensure Streamlit is bundled (using the venv python directly for relocatability)
"$VENV_PY" -m streamlit --version >/dev/null

# Copy source files and cifs into the app bundle
echo "   Copying source files and sample data..."

# Copy python_cli
if [ -d "python_cli" ]; then
    cp -R python_cli "$BUILD_DIR/$PACKAGE_NAME/mgrui.app/Contents/MacOS/"
fi

# Copy mgr_src
if [ -d "mgr_src" ]; then
    cp -R mgr_src "$BUILD_DIR/$PACKAGE_NAME/mgrui.app/Contents/MacOS/"
fi

# Copy .streamlit config
if [ -d ".streamlit" ]; then
    cp -R .streamlit "$BUILD_DIR/$PACKAGE_NAME/mgrui.app/Contents/MacOS/"
fi

# Copy cifs folder if it exists
if [ -d "cifs" ]; then
    echo "   Copying cifs folder..."
    cp -R cifs "$BUILD_DIR/$PACKAGE_NAME/mgrui.app/Contents/MacOS/"
fi

# Go back to original directory
cd "$ORIGINAL_DIR"

# ============================================================================
# CLEANUP: Remove unnecessary files to reduce bundle size
# ============================================================================
echo "   🧹 Cleaning up unnecessary files to reduce bundle size..."

# Remove Python cache files (~65 MB)
echo "   Removing Python cache files..."
find "$BUILD_DIR/$PACKAGE_NAME/mgrui.app/Contents/MacOS/mgrui_env" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
find "$BUILD_DIR/$PACKAGE_NAME/mgrui.app/Contents/MacOS/mgrui_env" -type f -name '*.pyc' -delete 2>/dev/null || true

# Remove test directories (~21 MB)
echo "   Removing test directories..."
find "$BUILD_DIR/$PACKAGE_NAME/mgrui.app/Contents/MacOS/mgrui_env" -type d -name 'tests' -exec rm -rf {} + 2>/dev/null || true
find "$BUILD_DIR/$PACKAGE_NAME/mgrui.app/Contents/MacOS/mgrui_env" -type d -name 'test' -exec rm -rf {} + 2>/dev/null || true
rm -rf "$BUILD_DIR/$PACKAGE_NAME/mgrui.app/Contents/MacOS/python_cli/tests" 2>/dev/null || true

# Remove Jupyter data (~18 MB)
echo "   Removing Jupyter data..."
rm -rf "$BUILD_DIR/$PACKAGE_NAME/mgrui.app/Contents/MacOS/mgrui_env/share/jupyter" 2>/dev/null || true

# NOTE: Do NOT remove .dist-info directories - packages need their metadata to work!

# Remove Rust build artifacts (~197 MB)
echo "   Removing Rust build artifacts..."
rm -rf "$BUILD_DIR/$PACKAGE_NAME/mgrui.app/Contents/MacOS/mgr_src/mgrui_rust/target" 2>/dev/null || true

# Remove pre-generated HTML files (~87 MB)
echo "   Removing pre-generated HTML files..."
rm -f "$BUILD_DIR/$PACKAGE_NAME/mgrui.app/Contents/MacOS/cifs"/*.html 2>/dev/null || true
rm -f "$BUILD_DIR/$PACKAGE_NAME/mgrui.app/Contents/MacOS/mgr_src/mgrui_rust/interactive_plot.html" 2>/dev/null || true

# NOTE: Do NOT remove pyarrow - Streamlit requires it for st.dataframe()

echo "   ✅ Cleanup complete!"

echo "📝 Creating README..."

cat > "$BUILD_DIR/$PACKAGE_NAME/README.txt" << 'EOF'
MGRUI - Molecular Geometry Analysis Tool

========================================

QUICK START
-----------

1. Drag "mgrui.app" to your Applications folder
2. Double-click mgrui.app to launch
3. A browser window will open automatically
4. That's it!


⚠️  FIRST TIME OPENING? IMPORTANT ⚠️
------------------------------------
If you see "can't be opened because it is from an unidentified developer":

**Option 1 (Easiest):** Right-click mgrui.app → Open → Click "Open"

**Option 2 (If that doesn't work):** Open Terminal and run:
   xattr -cr /Applications/mgrui.app

This is a macOS security feature, not a bug with MGRUI! After the first open,
the app will work normally.


WHAT'S INCLUDED
--------------
✓ Complete Python environment (built into the app)
✓ All dependencies (numpy, scipy, streamlit, etc.)
✓ MGRUI application
✓ Sample CIF files (built into the app)

SYSTEM REQUIREMENTS
-------------------
- macOS 10.13 (High Sierra) or later
- 500 MB free disk space
- 4 GB RAM recommended

TROUBLESHOOTING
---------------
Q: App won't start (no browser opens)
A: Check the log file at:
   mgrui.app → Show Package Contents → Contents → MacOS → mgrui_launcher.log

Q: Where can I find help?
A: Visit https://github.com/arung-northwestern/mgr_cli_py

UNINSTALLING
-----------
To remove MGRUI:
1. Drag mgrui.app to Trash
2. Empty Trash

That's it! Everything is self-contained.
EOF

echo "📦 Creating DMG..."

# Create DMG - just mgrui.app and README.txt at the root
cd "$BUILD_DIR/$PACKAGE_NAME"
hdiutil create -volname "mgrui" \
    -srcfolder . \
    -ov -format UDZO \
    "../mgrui-macOS-${ARCH_TAG}-$VERSION.dmg"

cd - > /dev/null

echo ""
echo "✅ Package created: $BUILD_DIR/mgrui-macOS-${ARCH_TAG}-$VERSION.dmg"
echo ""
echo "📊 Package size:"
du -sh "$BUILD_DIR/mgrui-macOS-${ARCH_TAG}-$VERSION.dmg"
echo ""
echo "🚀 Ready to distribute!"
echo ""
echo "📝 To test locally:"
echo "   open $BUILD_DIR/mgrui-macOS-${ARCH_TAG}-$VERSION.dmg"
echo ""
echo "📋 To upload to GitHub:"
echo "   1. Go to: https://github.com/arung-northwestern/mgr_cli_py/releases/new"
echo "   2. Upload: $BUILD_DIR/mgrui-macOS-${ARCH_TAG}-$VERSION.dmg"
echo ""
