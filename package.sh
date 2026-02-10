#!/bin/bash
set -e

echo "=== Packaging Parallel Video Transcoder MCPB ==="

# Check for mcpb CLI
if ! command -v mcpb &> /dev/null; then
    echo "Error: mcpb CLI not found."
    echo "Install with: npm install -g @anthropic-ai/mcpb"
    exit 1
fi

echo "Step 1: Installing Node.js dependencies..."
npm install --production

echo "Step 2: Building Rust binaries..."
./build.sh

echo "Step 3: Validating manifest.json..."
mcpb validate manifest.json

echo "Step 4: Creating MCPB bundle..."
mcpb pack . parallel-transcoder.mcpb

echo ""
echo "✅ Package complete!"
echo ""
ls -lh parallel-transcoder.mcpb
echo ""
echo "To install in Claude Desktop:"
echo "  open parallel-transcoder.mcpb"
echo ""
echo "To test the MCP server directly:"
echo "  node server/index.js"
