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
    echo "  macOS: brew install ffmpeg"
    echo "  Linux: sudo apt-get install ffmpeg libavcodec-dev libavformat-dev libavutil-dev"
    exit 1
fi

echo "Step 1: Building Rust binaries..."
cargo build --release

echo "Step 2: Creating bin/ directory..."
mkdir -p bin/

echo "Step 3: Copying binaries..."
cp target/release/transcoder-coordinator bin/
cp target/release/transcoder-worker bin/

# Make binaries executable
chmod +x bin/transcoder-coordinator
chmod +x bin/transcoder-worker

echo "Step 4: Bundling FFmpeg libraries..."
mkdir -p lib/

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS - copying FFmpeg dylibs..."
    if [ -d "/opt/homebrew/lib" ]; then
        cp /opt/homebrew/lib/libav*.dylib lib/ 2>/dev/null || true
        cp /opt/homebrew/lib/libswscale*.dylib lib/ 2>/dev/null || true
        cp /opt/homebrew/lib/libswresample*.dylib lib/ 2>/dev/null || true
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux - copying FFmpeg shared objects..."
    cp /usr/lib/x86_64-linux-gnu/libav*.so* lib/ 2>/dev/null || true
    cp /usr/lib/x86_64-linux-gnu/libswscale*.so* lib/ 2>/dev/null || true
    cp /usr/lib/x86_64-linux-gnu/libswresample*.so* lib/ 2>/dev/null || true
fi

echo ""
echo "✅ Build complete!"
echo ""
echo "Binaries:"
ls -lh bin/
echo ""
echo "Libraries:"
ls -lh lib/ 2>/dev/null || echo "No libraries bundled (may need FFmpeg installed on target system)"
echo ""
echo "Next steps:"
echo "  1. Test: ./bin/transcoder-coordinator --help"
echo "  2. Package: ./package.sh"
