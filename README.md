# Parallel Video Transcoder MCPB

A massively parallel video transcoding MCP Bundle with intelligent look-ahead optimization.

## Overview

This MCPB bundle enables users to transcode videos significantly faster by:

- **Parallel Processing**: Splitting video into segments and processing them simultaneously across multiple workers
- **Look-Ahead Optimization**: Analyzing future frames to avoid unnecessary compute (scene change detection, complexity analysis)
- **Rust + FFmpeg**: Using high-performance Rust with FFmpeg bindings for optimal transcoding

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   MCP Server (Node.js)                   │
│  - Receives video transcoding requests from Claude       │
│  - Coordinates Rust transcoding engine                   │
│  - Returns progress and results via MCP protocol         │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│            Coordinator (Rust Binary)                     │
│  - Pre-analyzes video (GOPs, keyframes, scenes)         │
│  - Splits video into segments at keyframe boundaries    │
│  - Spawns worker processes                              │
│  - Collects and reassembles results                     │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              Worker Processes (Rust)                     │
│  - Decode assigned video segment                        │
│  - Look-ahead frame analysis                            │
│  - Transcode with optimized parameters                  │
│  - Return encoded segment + metadata                    │
└─────────────────────────────────────────────────────────┘
```

## Features

- ✅ Keyframe-aligned segmentation for clean parallel processing
- ✅ Scene change detection with look-ahead analysis
- ✅ Complexity-based bitrate allocation
- ✅ HLS output format for seamless reassembly
- ✅ MCP protocol integration for Claude Desktop
- ✅ Cross-platform support (macOS, Windows, Linux)

## Performance

Expected performance gains:
- **4-8x speedup** on 8-core machines
- Minimal quality loss (< 0.5 dB PSNR difference)
- Optimized for multi-core CPUs

## Project Structure

```
parallel-transcoder/
├── coordinator/          # Rust coordinator binary
│   ├── src/
│   │   ├── main.rs      # Main coordinator logic
│   │   ├── analyzer.rs  # Video analysis module
│   │   └── segmenter.rs # Segmentation logic
│   └── Cargo.toml
├── worker/              # Rust worker binary
│   ├── src/
│   │   ├── main.rs      # Worker process logic
│   │   └── lookahead.rs # Look-ahead optimization
│   └── Cargo.toml
├── server/              # Node.js MCP server
│   └── index.js         # MCP protocol implementation
├── docs/                # Documentation
│   ├── PLAN.md          # Detailed implementation plan
│   └── RESEARCH.md      # Research findings
├── manifest.json        # MCPB manifest
├── package.json         # Node.js dependencies
└── Cargo.toml          # Rust workspace manifest
```

## Development Setup

### Prerequisites

- Rust (latest stable)
- Node.js (v16+)
- FFmpeg development libraries
- MCPB CLI tool

### Build Instructions

```bash
# Install dependencies
npm install

# Build Rust binaries
cargo build --release

# Run build script
./build.sh

# Package as MCPB bundle
./package.sh
```

## Usage

### As MCPB Bundle

1. Build the bundle: `./package.sh`
2. Install in Claude Desktop: Open `parallel-transcoder.mcpb`
3. Use via Claude: "Transcode this video to HLS format"

### Standalone

```bash
# Analyze video
./bin/transcoder-coordinator analyze input.mp4

# Transcode with parallel workers
./bin/transcoder-coordinator transcode \
  --input input.mp4 \
  --output output/ \
  --workers 8 \
  --segment-duration 10 \
  --lookahead 40 \
  --format hls
```

## MCP Tools

The bundle exposes these MCP tools:

- **transcode_video**: Transcode a video file using parallel processing
- **analyze_video**: Analyze video and return metadata
- **get_transcode_status**: Get status of ongoing job
- **cancel_transcode**: Cancel a running transcoding job

## Configuration

User-configurable options:

- **output_directory**: Where transcoded videos are saved
- **max_workers**: Number of parallel encoding workers
- **segment_duration**: Target duration for each segment (2-60 seconds)
- **lookahead_frames**: Frames to analyze ahead (0-250)

## Implementation Status

- [x] Project setup and planning
- [ ] Rust video analysis module
- [ ] Rust coordinator binary
- [ ] Rust worker binary
- [ ] Node.js MCP server
- [ ] Build system and packaging
- [ ] Testing and benchmarks

## License

MIT

## Credits

Built with:
- [FFmpeg](https://ffmpeg.org/) - Video processing
- [ffmpeg-next](https://github.com/zmwangx/rust-ffmpeg) - Rust FFmpeg bindings
- [MCP SDK](https://github.com/anthropics/mcp) - Model Context Protocol
- [MCPB](https://github.com/anthropics/mcpb) - MCP Bundle tooling
