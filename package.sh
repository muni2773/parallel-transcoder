#!/bin/bash
set -e

echo "=== Packaging Parallel Video Transcoder ==="

echo "Step 1: Installing Node.js dependencies..."
npm install --production

echo "Step 2: Building Rust binaries..."
./build.sh

echo ""
echo "Build and packaging complete!"
echo ""
echo "To start the web server:"
echo "  npm run web"
echo ""
echo "To run the CLI directly:"
echo "  ./bin/transcoder-coordinator --help"
