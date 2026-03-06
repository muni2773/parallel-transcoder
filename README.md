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
│   │   └── main.rs      # Worker process logic
│   └── Cargo.toml
├── web/                 # Web server + API
│   ├── server.js        # Express + WebSocket API server
│   └── public/
│       └── index.html   # Single-file SPA frontend
├── server/              # Node.js MCP server
│   └── index.js         # MCP protocol implementation
├── docs/                # Documentation
│   ├── PLAN.md          # Detailed implementation plan
│   └── RESEARCH.md      # Research findings
├── API.md               # REST & WebSocket API reference
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

### REST API

The transcoder exposes a full REST API and WebSocket interface for programmatic access:

```bash
# Start the API server
npm run web

# Upload a video
curl -X POST http://localhost:3000/api/upload -F "video=@input.mp4"

# Start transcoding
curl -X POST http://localhost:3000/api/transcode \
  -H "Content-Type: application/json" \
  -d '{"uploadId": "input_1709654400000.mp4", "format": "mp4", "crf": 20}'

# Check job status
curl http://localhost:3000/api/jobs/<jobId>

# Download output
curl -O http://localhost:3000/api/download/<jobId>/output.mp4
```

See **[API.md](API.md)** for the complete API reference with all endpoints, WebSocket protocol, authentication, and code examples in curl, JavaScript, and Python.

### Web UI

A browser-based interface is available at `http://localhost:3000` when the server is running. It provides drag-and-drop upload, real-time progress via WebSocket, and output file downloads.

### As MCPB Bundle

1. Build the bundle: `./package.sh`
2. Install in Claude Desktop: Open `parallel-transcoder.mcpb`
3. Use via Claude: "Transcode this video to HLS format"

### Standalone CLI

```bash
# Transcode with parallel workers
./bin/transcoder-coordinator \
  --input input.mp4 \
  --output output/ \
  --workers 8 \
  --format hls \
  --crf 23 \
  --preset medium \
  --encoder libx264

# Smart mode (skip segments that don't need re-encoding)
./bin/transcoder-coordinator \
  --input input.mp4 \
  --output output/ \
  --smart --smart-tolerance 0.3

# Analyze without encoding
./bin/transcoder-coordinator \
  --input input.mp4 \
  --output output/ \
  --smart-report
```

## API

Full REST API documentation is available in **[API.md](API.md)**.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Server status and job counts |
| `/api/upload` | POST | Upload video file (multipart) |
| `/api/transcode` | POST | Start transcoding job |
| `/api/analyze` | POST | Analyze video complexity |
| `/api/jobs` | GET | List all jobs |
| `/api/jobs/:id` | GET | Job status and progress |
| `/api/jobs/:id/logs` | GET | Paginated job logs |
| `/api/jobs/:id/files` | GET | List output files |
| `/api/download/:jobId/:file` | GET | Download output file |
| `/api/jobs/:id` | DELETE | Cancel and remove job |

**WebSocket:** `ws://localhost:3000/ws` — real-time progress, logs, and completion events.

**Authentication:** Optional API key via `TRANSCODER_API_KEY` env var.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `PORT` | `3000` | Server listen port |
| `TRANSCODER_API_KEY` | *(none)* | API key for authentication (optional) |

### Encoding Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `format` | `hls` | Output format: `hls` or `mp4` |
| `mode` | `normal` | Mode: `normal`, `copy`, `smart`, `smart-auto` |
| `crf` | `23` | Quality (0-51, lower = better) |
| `preset` | `medium` | Speed: `ultrafast` to `veryslow` |
| `encoder` | `libx264` | Encoder: `libx264` or `h264_videotoolbox` |
| `workers` | `0` | Worker count (0 = auto-detect) |

## Implementation Status

- [x] Project setup and planning
- [x] Rust video analysis module
- [x] Rust coordinator binary
- [x] Rust worker binary
- [x] Build system and packaging
- [x] Web UI with real-time progress
- [x] REST API with CORS and optional auth
- [x] WebSocket live updates
- [x] API documentation

## License

MIT

## Credits

Built with:
- [FFmpeg](https://ffmpeg.org/) - Video processing
- [ffmpeg-next](https://github.com/zmwangx/rust-ffmpeg) - Rust FFmpeg bindings
- [MCP SDK](https://github.com/anthropics/mcp) - Model Context Protocol
- [MCPB](https://github.com/anthropics/mcpb) - MCP Bundle tooling
