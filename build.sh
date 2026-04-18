#!/bin/bash
set -e

echo "=== Building Parallel Video Transcoder ==="

# Check for Rust
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust/Cargo not found. Please install from https://rustup.rs/"
    exit 1
fi

# Check for FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "Warning: FFmpeg not found. Install FFmpeg for development:"
    echo "  macOS:        brew install ffmpeg"
    echo "  RHEL/CentOS:  sudo dnf install ffmpeg-free-devel (enable RPM Fusion for full FFmpeg)"
    echo "  Fedora:       sudo dnf install ffmpeg-free-devel"
    echo "  Ubuntu/Debian: sudo apt-get install ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev"
    exit 1
fi

# Detect platform and architecture
OS="$(uname -s)"
ARCH="$(uname -m)"
echo "Platform: ${OS} ${ARCH}"

echo "Step 1: Building Rust binaries..."
cargo build --release

echo "Step 2: Creating bin/ directory..."
mkdir -p bin/

echo "Step 3: Copying binaries..."
cp target/release/transcoder-coordinator bin/
cp target/release/transcoder-worker bin/
if [ -f target/release/transcoder-node ]; then
    cp target/release/transcoder-node bin/
fi

# Validate desktop launcher shell script
if [ -f bin/transcoder-desktop ]; then
    echo "  Validating bin/transcoder-desktop syntax..."
    bash -n bin/transcoder-desktop
else
    echo "  Warning: bin/transcoder-desktop not found"
fi

# Make binaries executable
chmod +x bin/*

echo "Step 4: Bundling FFmpeg libraries..."
mkdir -p lib/

FFMPEG_LIBS="libavcodec libavformat libavutil libavfilter libavdevice libswscale libswresample"

# Find FFmpeg library directory
find_ffmpeg_libs() {
    # Try pkg-config first (most reliable across all distros)
    if command -v pkg-config &> /dev/null; then
        local libdir
        libdir=$(pkg-config --variable=libdir libavcodec 2>/dev/null) || true
        if [ -n "$libdir" ] && [ -d "$libdir" ]; then
            echo "$libdir"
            return 0
        fi
    fi

    # Platform-specific fallbacks
    case "$OS" in
        Darwin)
            # Apple Silicon Homebrew
            if [ -d "/opt/homebrew/lib" ]; then
                echo "/opt/homebrew/lib"
                return 0
            fi
            # Intel Homebrew
            if [ -d "/usr/local/lib" ]; then
                echo "/usr/local/lib"
                return 0
            fi
            # MacPorts
            if [ -d "/opt/local/lib" ]; then
                echo "/opt/local/lib"
                return 0
            fi
            ;;
        Linux)
            # RHEL/CentOS/Fedora/Rocky (64-bit)
            if [ -d "/usr/lib64" ] && ls /usr/lib64/libavcodec.so* &>/dev/null; then
                echo "/usr/lib64"
                return 0
            fi
            # Debian/Ubuntu x86_64
            if [ -d "/usr/lib/x86_64-linux-gnu" ] && ls /usr/lib/x86_64-linux-gnu/libavcodec.so* &>/dev/null; then
                echo "/usr/lib/x86_64-linux-gnu"
                return 0
            fi
            # Debian/Ubuntu aarch64
            if [ -d "/usr/lib/aarch64-linux-gnu" ] && ls /usr/lib/aarch64-linux-gnu/libavcodec.so* &>/dev/null; then
                echo "/usr/lib/aarch64-linux-gnu"
                return 0
            fi
            # Generic /usr/lib
            if ls /usr/lib/libavcodec.so* &>/dev/null; then
                echo "/usr/lib"
                return 0
            fi
            # RPM Fusion / third-party (RHEL variants)
            if [ -d "/usr/local/lib64" ] && ls /usr/local/lib64/libavcodec.so* &>/dev/null; then
                echo "/usr/local/lib64"
                return 0
            fi
            if [ -d "/usr/local/lib" ] && ls /usr/local/lib/libavcodec.so* &>/dev/null; then
                echo "/usr/local/lib"
                return 0
            fi
            ;;
    esac
    return 1
}

FFMPEG_LIBDIR=$(find_ffmpeg_libs) || true

if [ -n "$FFMPEG_LIBDIR" ]; then
    echo "Found FFmpeg libraries in: $FFMPEG_LIBDIR"
    case "$OS" in
        Darwin)
            for lib in $FFMPEG_LIBS; do
                cp "$FFMPEG_LIBDIR"/${lib}*.dylib lib/ 2>/dev/null || true
            done
            ;;
        Linux)
            for lib in $FFMPEG_LIBS; do
                cp "$FFMPEG_LIBDIR"/${lib}.so* lib/ 2>/dev/null || true
            done
            ;;
    esac
else
    echo "Warning: Could not find FFmpeg libraries to bundle."
    echo "  The binaries will use system-installed FFmpeg at runtime."
    echo ""
    echo "  Install FFmpeg development libraries:"
    echo "    RHEL/Rocky 8+:  sudo dnf install --enablerepo=powertools ffmpeg-free-devel"
    echo "    RHEL/Rocky 9+:  sudo dnf install --enablerepo=crb ffmpeg-free-devel"
    echo "    Fedora:         sudo dnf install ffmpeg-free-devel"
    echo "    CentOS Stream:  sudo dnf install --enablerepo=crb ffmpeg-free-devel"
    echo "    Ubuntu/Debian:  sudo apt-get install libavcodec-dev libavformat-dev libavutil-dev"
    echo "    macOS:          brew install ffmpeg"
    echo ""
    echo "  For full FFmpeg (non-free codecs like x264/x265), enable RPM Fusion:"
    echo "    sudo dnf install https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-\$(rpm -E %rhel).noarch.rpm"
    echo "    sudo dnf install ffmpeg-devel"
fi

echo ""
echo "Build complete!"
echo ""
echo "Binaries:"
ls -lh bin/
echo ""
echo "Libraries:"
ls -lh lib/ 2>/dev/null || echo "No libraries bundled (will use system FFmpeg)"
echo ""
echo "Next steps:"
echo "  1. Test: ./bin/transcoder-coordinator --help"
echo "  2. Cluster: ./bin/transcoder-node --help"
echo "  3. Desktop: ./bin/transcoder-desktop"
echo "  4. Package: ./package.sh"
