# Development Guide — Parallel Video Transcoder

Development reference for working on this project.

## Project Overview

Distributed, multi-node video transcoding engine with multi-codec support and hardware acceleration. Rust core with Node.js web interface.

## Architecture

**Four-tier design:**

1. **Cluster Layer** (`cluster/`) — Distributed coordination across machines
   - `protocol.rs` — OBS-websocket-inspired OpCode message format (`{ op, d }`)
   - `transport.rs` — WebSocket transport (tokio-tungstenite), PeerHandle management
   - `srt.rs` — SRT data plane via FFmpeg for segment transfer between nodes
   - `election.rs` — Bully algorithm leader election (priority from UUID)
   - `node.rs` — Node manager with DashMap registry, heartbeat health monitoring
   - `scheduler.rs` — Complexity-aware segment distribution (highest-complexity-first)
   - `main.rs` — `transcoder-node` daemon with tokio::select! event loop

2. **Coordinator** (`coordinator/`) — Video analysis and orchestration
   - `analyzer.rs` — Keyframe detection, scene changes, complexity estimation
   - `segmenter.rs` — Keyframe-aligned segment creation
   - `main.rs` — CLI entry point, local worker spawning, `--cluster` mode

3. **Worker** (`worker/`) — Multi-codec encoding engine
   - Supports H.264, H.265/HEVC, AV1 (CPU and GPU variants)
   - Hardware: VideoToolbox (macOS), NVENC (NVIDIA), VAAPI (Intel/AMD)
   - 10-bit pipeline (YUV420P10LE) for HEVC and AV1
   - SVT-AV1 preset mapping, libaom cpu-used/tiles config

4. **Web + API** (`web/`) — Express server with WebSocket
   - REST API for upload, transcode, jobs, download
   - Cluster endpoints: `/api/cluster/status`, `/api/cluster/nodes`, `/api/cluster/transcode`
   - Dark-themed SPA with platform-aware encoder selection

## Key Technologies

| Layer | Stack |
|-------|-------|
| Cluster | Rust, tokio, tokio-tungstenite, DashMap, uuid, SRT via FFmpeg |
| Coordinator/Worker | Rust, ffmpeg-next, tokio, clap, anyhow |
| Web | Node.js, Express, ws, multer |

## Project Structure

```
parallel-transcoder/
├── cluster/             # Distributed cluster (Rust crate)
│   └── src/             # protocol, transport, srt, election, node, scheduler, main
├── coordinator/         # Video analysis + orchestration (Rust crate)
│   └── src/             # main, analyzer, segmenter
├── worker/              # Encoding engine (Rust crate)
│   └── src/             # main (multi-codec)
├── web/                 # Web server + UI
│   ├── server.js        # Express + WebSocket + cluster API
│   └── public/index.html
├── docs/                # PLAN.md, RESEARCH.md
├── API.md               # Full API reference
├── build.sh             # Cross-platform build
└── Cargo.toml           # Workspace: coordinator, worker, cluster
```

## Development Commands

```bash
# Build everything
./build.sh

# Run tests
cargo test

# Start web server
npm run web

# Start cluster node
./bin/transcoder-node --listen 0.0.0.0:9000 --name node-1

# Join cluster
./bin/transcoder-node --listen 0.0.0.0:9001 --join host:9000 --name node-2

# Local transcode
./bin/transcoder-coordinator --input video.mp4 --output out/ --workers 8 --encoder libx264

# Cluster transcode
./bin/transcoder-coordinator --input video.mp4 --output out/ --cluster --cluster-master host:9000
```

## Coordinator CLI Flags

`--input`, `--output`, `--workers`, `--segment-duration`, `--format` (hls/mp4),
`--crf`, `--preset`, `--encoder` (`-E`), `--verbose`, `--fast`, `--copy`, `--smart`,
`--smart-tolerance`, `--smart-auto`, `--smart-report`, `--cluster`, `--cluster-master`

## Cluster Node CLI Flags

`--listen`, `--join`, `--name`, `--srt-base-port`, `--worker-binary`, `--lib-dir`, `--verbose`

## Encoder Mapping

| Encoder | Codec | Config Notes |
|---------|-------|-------------|
| libx264 | H.264 | Standard x264 presets and CRF |
| h264_videotoolbox | H.264 | macOS HW, no CRF (uses bitrate) |
| h264_nvenc | H.264 | VBR, p5 preset |
| h264_vaapi | H.264 | VBR, profile-aware bitrate |
| libx265 | H.265 | Standard x265, 10-bit capable |
| hevc_videotoolbox | H.265 | macOS HW |
| hevc_nvenc | H.265 | VBR, p5 preset |
| hevc_vaapi | H.265 | VBR |
| libsvtav1 | AV1 | Mapped presets (0-12), 10-bit |
| libaom-av1 | AV1 | cpu-used mapping, row-mt=1, tiles=2x2 |
| av1_nvenc | AV1 | NVIDIA Lovelace+ |

## Cross-Platform Notes

- **Library path**: `DYLD_LIBRARY_PATH` on macOS, `LD_LIBRARY_PATH` on Linux
- **GPU detection**:
  - macOS: VideoToolbox (always available)
  - Linux NVIDIA: check `/dev/nvidia0`
  - Linux VAAPI: check `/dev/dri/renderD128`
- **Build**: `build.sh` uses pkg-config first, then platform-specific fallback paths
- **FFmpeg install**:
  - macOS: `brew install ffmpeg`
  - RHEL/Rocky/Fedora: `dnf install ffmpeg-devel` (requires RPM Fusion)

## Rust Guidelines

- `anyhow::Result` for error handling
- `#[tokio::main]` for async entry points
- `tracing` for structured logging
- Unit tests in each module (58 tests in cluster crate)

## OpCode Protocol (OBS-inspired)

Key OpCodes: Hello(0), Identify(1), Identified(2), ElectionStart(10), ElectionAlive(11), ElectionVictory(12), Heartbeat(20), HeartbeatAck(21), JobSubmit(30), JobAccepted(31), JobProgress(32), JobComplete(33), JobFailed(34), JobCancel(35), SegmentAssign(40), SegmentComplete(42), StatusRequest(50), StatusResponse(51), Event(60), NodeLeave(70), Error(255)

Event subscription bitmask constants in `cluster::protocol::event_subs`.

## Resources

- [API Reference](API.md)
- [Implementation Plan](docs/PLAN.md)
- [Research](docs/RESEARCH.md)
- [ffmpeg-next docs](https://docs.rs/ffmpeg-next/)
