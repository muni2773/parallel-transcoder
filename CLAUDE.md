# CLAUDE.md - Parallel Video Transcoder

This file provides guidance to Claude Code when working on the Parallel Video Transcoder MCPB project.

## Project Overview

A massively parallel video transcoding MCPB (MCP Bundle) with intelligent look-ahead optimization. Built with Rust for performance and Node.js for MCP server integration.

## Architecture

**Three-tier design:**
1. **Node.js MCP Server** (`server/index.js`) - Handles MCP protocol, exposes tools to Claude Desktop
2. **Rust Coordinator** (`coordinator/`) - Analyzes video, creates segments, orchestrates workers
3. **Rust Workers** (`worker/`) - Process individual segments with look-ahead optimization

## Key Technologies

- **Rust**: Core transcoding engine (ffmpeg-next, tokio, clap)
- **FFmpeg**: Video codec library via Rust bindings
- **Node.js**: MCP server (stdio transport, @modelcontextprotocol/sdk)
- **MCPB**: Bundle format for distribution

## Project Structure

```
parallel-transcoder/
├── coordinator/          # Rust coordinator binary
│   ├── src/
│   │   ├── main.rs      # Entry point, CLI, orchestration
│   │   ├── analyzer.rs  # Video analysis (keyframes, scenes, complexity)
│   │   └── segmenter.rs # Segment creation logic
│   └── Cargo.toml
├── worker/              # Rust worker binary
│   ├── src/
│   │   ├── main.rs      # Worker process entry point
│   │   └── lookahead.rs # Look-ahead optimization algorithms
│   └── Cargo.toml
├── server/              # Node.js MCP server
│   └── index.js         # MCP protocol implementation
├── docs/                # Documentation
│   ├── PLAN.md          # Complete implementation plan
│   └── RESEARCH.md      # Video processing research
├── manifest.json        # MCPB bundle manifest
├── package.json         # Node.js dependencies
├── Cargo.toml          # Rust workspace manifest
├── build.sh            # Build script
└── package.sh          # MCPB packaging script
```

## Development Workflow

### Building

```bash
# Build Rust binaries and bundle FFmpeg libraries
./build.sh

# Test coordinator
./bin/transcoder-coordinator --help

# Test worker
./bin/transcoder-worker --help
```

### Testing

```bash
# Run Rust tests
cargo test

# Test MCP server standalone
node server/index.js

# Send test MCP request
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | node server/index.js
```

### Packaging

```bash
# Install Node deps and create MCPB bundle
./package.sh

# Install in Claude Desktop
open parallel-transcoder.mcpb
```

## Implementation Status

### ✅ Completed
- [x] Project structure and scaffolding
- [x] Rust workspace configuration
- [x] Node.js MCP server scaffolding
- [x] MCPB manifest and configuration
- [x] Build system and packaging scripts
- [x] Comprehensive research and planning

### 🚧 In Progress
- [ ] Video analysis module (analyzer.rs)
  - [ ] FFmpeg integration
  - [ ] Keyframe detection
  - [ ] Scene change detection
  - [ ] Complexity estimation

### 📋 TODO
- [ ] Segmentation logic (segmenter.rs)
- [ ] Coordinator orchestration (coordinator/main.rs)
- [ ] Worker transcoding loop (worker/main.rs)
- [ ] Look-ahead algorithms (lookahead.rs)
- [ ] HLS playlist generation
- [ ] MP4 reassembly
- [ ] Progress reporting
- [ ] Error handling and recovery
- [ ] Unit tests
- [ ] Integration tests
- [ ] Performance benchmarks

## Key Implementation Notes

### Video Segmentation Strategy
- **MUST** split at keyframe (I-frame) boundaries
- Use pre-analysis to identify all keyframes
- Target segment duration (default: 10s) aligned to nearest keyframes
- Balance segments by complexity for even workload

### Look-Ahead Optimization
1. **Scene Change Detection**: Compare frame histograms, threshold ~0.3
2. **Complexity Analysis**: Spatial (edges, variance) + Temporal (motion, SAD)
3. **Bitrate Allocation**: More bits for complex frames based on look-ahead
4. **Adaptive Keyframes**: Force keyframes at scene boundaries

### FFmpeg Integration
- Use `ffmpeg-next` crate (Rust bindings to FFmpeg C library)
- Decoder: Read frames, seek to positions
- Encoder: Write frames with custom parameters
- DO NOT attempt to recompile FFmpeg from C to Rust (impractical)

### Output Formats
- **HLS** (Recommended): `.m3u8` playlist + `.ts` segments, simplest reassembly
- **MP4**: Requires complex container reassembly, frame offset rewriting

## Development Guidelines

### Rust Code
- Use `anyhow::Result` for error handling
- Implement `#[tokio::main]` for async operations
- Add tracing/logging with `tracing` crate
- Write unit tests in each module
- Document public APIs with rustdoc comments

### Node.js MCP Server
- Follow MCP SDK patterns (stdio transport)
- Return structured JSON responses
- Handle errors gracefully
- Log to stderr (stdout reserved for MCP protocol)

### Performance Targets
- **4-8x speedup** on 8-core machines vs sequential FFmpeg
- **< 0.5 dB quality loss** compared to reference encoding
- **HLS output** compatible with browsers, VLC, ffplay

## Dependencies

### System Requirements
- FFmpeg (development libraries for building)
- Rust toolchain (stable channel)
- Node.js (v16+)
- MCPB CLI (`npm install -g @anthropic-ai/mcpb`)

### Rust Crates
- `ffmpeg-next` - FFmpeg Rust bindings
- `tokio` - Async runtime
- `serde`/`serde_json` - Serialization
- `clap` - CLI argument parsing
- `anyhow` - Error handling
- `tracing` - Logging

### Node.js Packages
- `@modelcontextprotocol/sdk` - MCP protocol implementation

## Testing Strategy

### Unit Tests
- Keyframe detection accuracy
- Scene change detection threshold tuning
- Complexity calculation verification
- Segment balancing algorithm

### Integration Tests
- End-to-end transcoding pipeline
- HLS playlist generation and playback
- Multi-worker coordination
- Error recovery (invalid input, worker failures)

### Benchmarks
- Compare against `ffmpeg -i input.mp4 output.mp4`
- Measure speedup vs # of workers
- Profile memory usage
- Validate quality (PSNR, SSIM metrics)

## Troubleshooting

### Build Issues
- **FFmpeg not found**: Install FFmpeg dev packages
  - macOS: `brew install ffmpeg`
  - Ubuntu: `apt-get install libavcodec-dev libavformat-dev`
- **Rust version**: Use latest stable: `rustup update`

### Runtime Issues
- **Segmentation artifacts**: Check keyframe alignment
- **Quality loss**: Tune look-ahead parameters, bitrate allocation
- **Performance**: Verify FFmpeg using hardware acceleration if available

## Resources

- [Detailed Implementation Plan](docs/PLAN.md)
- [MCPB Specification](https://github.com/anthropics/mcpb)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [ffmpeg-next Crate](https://docs.rs/ffmpeg-next/)
- [MCP SDK](https://github.com/anthropics/mcp)

## Next Steps

See `docs/PLAN.md` for the complete phase-by-phase implementation roadmap.

**Immediate priorities:**
1. Implement video analysis module with FFmpeg integration
2. Build keyframe and scene change detection
3. Create segment descriptors aligned to keyframes
4. Implement worker transcoding loop with look-ahead buffer
