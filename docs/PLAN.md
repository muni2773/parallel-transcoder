# Massively Parallel Video Transcoding MCPB Bundle - Implementation Plan

## Context

Building an MCPB (MCP Bundle) application for massively parallel video transcoding with intelligent look-ahead optimization. This bundle will enable users to transcode videos significantly faster by:

1. **Parallel Processing**: Splitting video into segments and processing them simultaneously across multiple subagents/workers
2. **Look-Ahead Optimization**: Analyzing future frames to avoid unnecessary compute (scene change detection, complexity analysis, adaptive frame decisions)
3. **Rust + FFmpeg**: Using FFmpeg compiled/integrated with Rust for high-performance transcoding

The goal is to minimize transcoding time while maintaining quality through intelligent work distribution and frame analysis.

---

## Architecture Overview

### Three-Tier Architecture

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

---

## FFmpeg Integration Strategy

### Recommended Approach: Rust FFmpeg Bindings

After research, **using Rust FFmpeg bindings** is the most practical approach:

**Rationale:**
1. **Recompiling FFmpeg C to Rust is impractical**:
   - FFmpeg is ~1.5 million lines of highly optimized C code
   - c2rust produces unsafe Rust that requires extensive manual cleanup
   - Manual rewrite would take years

2. **FFmpeg Rust bindings are production-ready**:
   - `ffmpeg-next` (high-level) and `ffmpeg-sys-next` (low-level) crates
   - Battle-tested in production video applications
   - Provides safe Rust wrappers around FFmpeg's C APIs
   - Native performance (minimal overhead)

3. **Alternative: Bundle FFmpeg CLI**:
   - Simpler but less flexible
   - Higher overhead (process spawning, CLI parsing)
   - Less control over encoding pipeline

**Selected Approach: Use `ffmpeg-next` Rust crate**

---

## Critical Research Findings

### 1. Video Segmentation Strategy

**Must use Keyframe-Aligned Segmentation:**
- Segments MUST split at keyframe (I-frame) boundaries
- Each segment can be independently decoded/encoded
- FFmpeg enforces this automatically in segment muxers
- Pre-analysis phase identifies all keyframe positions

**Process:**
```rust
1. Analyze video → identify all keyframes (I-frames)
2. Calculate segment boundaries (e.g., every 30 seconds, aligned to nearest keyframe)
3. Split metadata (not actual video file yet) into segment descriptors
4. Distribute segment descriptors to workers
5. Workers seek to start position and encode segment
6. Reassemble at keyframe boundaries
```

### 2. Look-Ahead Optimization Techniques

**What "look-ahead" means in video encoding:**

1. **Scene Change Detection**:
   - Analyze upcoming frames for scene changes
   - Force keyframes at scene boundaries
   - Prevents wasted inter-frame predictions across scene cuts
   - Improves visual quality and compression efficiency

2. **Complexity Analysis**:
   - Buffer future frames (lookahead_depth parameter)
   - Measure spatial complexity (texture, detail)
   - Measure temporal complexity (motion)
   - Allocate bits proportionally to complexity

3. **Rate Control Optimization**:
   - Distribute bitrate budget across upcoming frames
   - Give more bits to complex frames, fewer to simple frames
   - Research shows 0.6-1.7 dB quality improvement with look-ahead

4. **Adaptive Frame Type Decisions**:
   - Decide P-frame vs B-frame placement based on future motion
   - Optimize GOP structure dynamically

**Implementation in Workers:**
- Each worker gets assigned segment + look-ahead buffer
- Pre-compute scene changes and complexity estimates in coordinator
- Pass metadata to workers to guide encoding decisions

### 3. Parallel Processing Patterns

**Industry-Proven Approaches:**
- **Chunk-based (2-10 second segments)**: Most successful for streaming
- **GOP-based**: Clean codec boundaries, used in professional VoD
- **Frame-level parallelization**: Maximum throughput for same temporal layer

**Performance Expectations:**
- CTrans (Hadoop-based system): 70% time reduction
- AWS Accelerated Transcoding: Up to 25x faster
- Requires multi-core CPU or distributed workers

### 4. Reassembly Challenges & Solutions

**Key Challenges:**
1. **Keyframe alignment**: Solved by enforcing keyframe boundaries
2. **Rate control consistency**: Coordinator pre-computes bitrate allocation per segment
3. **Metadata coordination**: Workers report stats back to coordinator
4. **Container format complexity**: Use HLS/DASH output (simpler than MP4 reassembly)

**Recommended Output Format: HLS (HTTP Live Streaming)**
- Produces `.m3u8` playlist + `.ts` segment files
- Segments naturally align with our parallel processing model
- No complex reassembly needed (just generate playlist)
- Widely supported for playback

---

## MCPB Bundle Structure

### Directory Layout

```
parallel-transcoder.mcpb (ZIP archive)
├── manifest.json                      # MCPB manifest
├── server/
│   └── index.js                       # Node.js MCP server (stdio transport)
├── bin/
│   ├── transcoder-coordinator         # Rust binary (coordinator)
│   ├── transcoder-coordinator.exe     # Windows version
│   ├── transcoder-worker              # Rust binary (worker)
│   └── transcoder-worker.exe          # Windows version
├── lib/
│   ├── libavcodec.so.60               # FFmpeg shared libraries (Linux)
│   ├── libavformat.so.60
│   ├── libavutil.so.60
│   ├── libswscale.so.60
│   └── ... (macOS .dylib, Windows .dll variants)
├── node_modules/                      # Node.js dependencies
│   └── @modelcontextprotocol/sdk/
├── package.json
├── icon.png
└── README.md
```

### Manifest.json Structure

```json
{
  "manifest_version": "0.2",
  "name": "parallel-video-transcoder",
  "display_name": "Parallel Video Transcoder",
  "version": "1.0.0",
  "description": "Massively parallel video transcoding with look-ahead optimization",
  "author": {
    "name": "Your Name",
    "email": "your.email@example.com"
  },
  "server": {
    "type": "node",
    "entry_point": "server/index.js",
    "mcp_config": {
      "command": "node",
      "args": [
        "${__dirname}/server/index.js"
      ],
      "env": {
        "TRANSCODER_BIN": "${__dirname}/bin",
        "FFMPEG_LIB_PATH": "${__dirname}/lib"
      },
      "platform_overrides": {
        "win32": {
          "env": {
            "PATH": "${__dirname}\\bin;${PATH}"
          }
        },
        "darwin": {
          "env": {
            "DYLD_LIBRARY_PATH": "${__dirname}/lib"
          }
        },
        "linux": {
          "env": {
            "LD_LIBRARY_PATH": "${__dirname}/lib"
          }
        }
      }
    }
  },
  "tools": [
    {
      "name": "transcode_video",
      "description": "Transcode a video file using parallel processing with look-ahead optimization"
    },
    {
      "name": "analyze_video",
      "description": "Analyze video file and return metadata (duration, keyframes, complexity)"
    },
    {
      "name": "get_transcode_status",
      "description": "Get status of an ongoing transcoding job"
    },
    {
      "name": "cancel_transcode",
      "description": "Cancel a running transcoding job"
    }
  ],
  "user_config": {
    "output_directory": {
      "type": "directory",
      "title": "Output Directory",
      "description": "Directory where transcoded videos will be saved",
      "required": true,
      "default": ["${HOME}/Movies/Transcoded"]
    },
    "max_workers": {
      "type": "number",
      "title": "Maximum Worker Processes",
      "description": "Number of parallel encoding workers (default: CPU cores)",
      "default": 0,
      "min": 0,
      "max": 64
    },
    "segment_duration": {
      "type": "number",
      "title": "Segment Duration (seconds)",
      "description": "Target duration for each segment (aligned to keyframes)",
      "default": 10,
      "min": 2,
      "max": 60
    },
    "lookahead_frames": {
      "type": "number",
      "title": "Look-Ahead Frames",
      "description": "Number of frames to analyze ahead for optimization",
      "default": 40,
      "min": 0,
      "max": 250
    }
  },
  "compatibility": {
    "platforms": ["darwin", "win32", "linux"],
    "claude_desktop": ">=0.10.0"
  },
  "keywords": ["video", "transcoding", "ffmpeg", "parallel", "encoding"],
  "license": "MIT"
}
```

---

## Implementation Plan

### Phase 1: Project Setup & Foundation

**1.1 Create Project Structure**
- Create new project directory: `/Volumes/FastDisk3TB/muni/Developer/parallel-transcoder/`
- Initialize Rust workspace with two crates:
  - `transcoder-coordinator`: Main coordination binary
  - `transcoder-worker`: Worker process binary
- Initialize Node.js package for MCP server
- Set up Git repository

**1.2 Add Dependencies**

**Rust Cargo.toml:**
```toml
[workspace]
members = ["coordinator", "worker"]

[workspace.dependencies]
ffmpeg-next = "7.0"
ffmpeg-sys-next = "7.0"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
clap = { version = "4", features = ["derive"] }
anyhow = "1"
tracing = "0.1"
tracing-subscriber = "0.3"
```

**Node.js package.json:**
```json
{
  "name": "parallel-video-transcoder",
  "version": "1.0.0",
  "type": "module",
  "dependencies": {
    "@modelcontextprotocol/sdk": "latest"
  }
}
```

### Phase 2: Rust Video Analysis & Coordinator

**2.1 Video Analysis Module** (`coordinator/src/analyzer.rs`)

**Responsibilities:**
- Open video file with FFmpeg
- Extract metadata (duration, resolution, codec, bitrate)
- Identify all keyframe positions
- Perform scene change detection
- Calculate complexity estimates per frame
- Determine optimal segment boundaries

**Key Functions:**
```rust
pub struct VideoMetadata {
    duration_secs: f64,
    width: u32,
    height: u32,
    fps: f64,
    total_frames: u64,
    keyframe_positions: Vec<u64>,
    scene_changes: Vec<u64>,
    complexity_map: Vec<f32>,
}

pub fn analyze_video(input_path: &Path) -> Result<VideoMetadata>;
pub fn detect_keyframes(decoder: &mut Decoder) -> Result<Vec<u64>>;
pub fn detect_scene_changes(decoder: &mut Decoder, lookahead: usize) -> Result<Vec<u64>>;
pub fn calculate_complexity(frame: &Frame) -> f32;
```

**2.2 Segmentation Module** (`coordinator/src/segmenter.rs`)

**Responsibilities:**
- Split video into segments aligned to keyframes
- Create segment descriptors for workers
- Ensure segments are balanced (similar complexity/duration)

**Key Functions:**
```rust
pub struct Segment {
    id: usize,
    start_frame: u64,
    end_frame: u64,
    start_timestamp: f64,
    end_timestamp: f64,
    complexity_estimate: f32,
    contains_scene_changes: Vec<u64>,
}

pub fn create_segments(
    metadata: &VideoMetadata,
    target_duration_secs: f64
) -> Result<Vec<Segment>>;
```

**2.3 Coordinator Binary** (`coordinator/src/main.rs`)

**Responsibilities:**
- Parse command-line arguments
- Analyze input video
- Create segments
- Spawn worker processes (one per segment)
- Collect results from workers
- Generate HLS playlist or reassemble MP4
- Report progress to MCP server via JSON output

**Key Flow:**
```rust
#[tokio::main]
async fn main() -> Result<()> {
    // 1. Parse arguments
    let args = Args::parse();

    // 2. Analyze video
    let metadata = analyze_video(&args.input)?;

    // 3. Create segments
    let segments = create_segments(&metadata, args.segment_duration)?;

    // 4. Spawn workers (parallel)
    let mut tasks = Vec::new();
    for segment in segments {
        let task = spawn_worker(segment, &args);
        tasks.push(task);
    }

    // 5. Await all workers
    let results = futures::future::join_all(tasks).await;

    // 6. Reassemble or create playlist
    if args.output_format == "hls" {
        create_hls_playlist(&results, &args.output)?;
    } else {
        reassemble_segments(&results, &args.output)?;
    }

    Ok(())
}
```

### Phase 3: Rust Worker Process

**3.1 Worker Binary** (`worker/src/main.rs`)

**Responsibilities:**
- Receive segment descriptor (via command-line args or stdin JSON)
- Seek to start position in input video
- Decode frames for assigned segment
- Apply look-ahead analysis
- Encode frames with optimized parameters
- Write output segment
- Report metadata back to coordinator

**Key Flow:**
```rust
#[tokio::main]
async fn main() -> Result<()> {
    // 1. Parse segment assignment
    let segment: Segment = serde_json::from_stdin()?;

    // 2. Open input video
    let mut decoder = Decoder::new(&segment.input_path)?;
    decoder.seek(segment.start_timestamp)?;

    // 3. Create encoder
    let mut encoder = Encoder::new(
        &segment.output_path,
        &segment.encode_params
    )?;

    // 4. Transcode with look-ahead
    let mut lookahead_buffer = VecDeque::new();

    while let Some(frame) = decoder.next_frame()? {
        if frame.number > segment.end_frame {
            break;
        }

        // Populate look-ahead buffer
        lookahead_buffer.push_back(frame);
        if lookahead_buffer.len() > segment.lookahead_frames {
            let current_frame = lookahead_buffer.pop_front().unwrap();

            // Analyze look-ahead
            let should_force_keyframe = detect_scene_change(&lookahead_buffer)?;
            let frame_complexity = calculate_complexity(&lookahead_buffer);

            // Encode with parameters
            encoder.encode_frame(
                &current_frame,
                should_force_keyframe,
                frame_complexity
            )?;
        }
    }

    // 5. Flush remaining frames
    for frame in lookahead_buffer {
        encoder.encode_frame(&frame, false, 1.0)?;
    }

    // 6. Report metadata
    let metadata = WorkerMetadata {
        segment_id: segment.id,
        frames_encoded: encoder.frames_count(),
        output_size_bytes: encoder.output_size(),
        encoding_time_secs: encoder.elapsed_time(),
    };
    println!("{}", serde_json::to_string(&metadata)?);

    Ok(())
}
```

**3.2 Look-Ahead Implementation** (`worker/src/lookahead.rs`)

**Scene Change Detection:**
```rust
pub fn detect_scene_change(buffer: &VecDeque<Frame>) -> Result<bool> {
    // Compare current frame with next frame
    let current = &buffer[0];
    let next = &buffer[1];

    // Calculate histogram difference
    let diff = histogram_difference(current, next);

    // Threshold for scene change (tunable)
    Ok(diff > 0.3)
}
```

**Complexity Estimation:**
```rust
pub fn calculate_complexity(buffer: &VecDeque<Frame>) -> f32 {
    // Estimate spatial complexity (edge detection, variance)
    let spatial = spatial_complexity(&buffer[0]);

    // Estimate temporal complexity (motion between frames)
    let temporal = temporal_complexity(&buffer[0], &buffer[1]);

    // Combine metrics
    (spatial + temporal) / 2.0
}
```

### Phase 4: Node.js MCP Server

**4.1 MCP Server Implementation** (`server/index.js`)

**Responsibilities:**
- Implement MCP protocol (stdio transport)
- Register tool handlers
- Spawn Rust coordinator process
- Stream progress back to Claude
- Handle errors gracefully

**Core Implementation:**
```javascript
#!/usr/bin/env node
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  ListToolsRequestSchema,
  CallToolRequestSchema
} from "@modelcontextprotocol/sdk/types.js";
import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const COORDINATOR_BIN = path.join(__dirname, "../bin/transcoder-coordinator");

// Create server
const server = new Server(
  {
    name: "parallel-video-transcoder",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Register tool: transcode_video
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  if (name === "transcode_video") {
    return await transcodeVideo(args);
  } else if (name === "analyze_video") {
    return await analyzeVideo(args);
  }

  throw new Error(`Unknown tool: ${name}`);
});

async function transcodeVideo(args) {
  const {
    input_path,
    output_path,
    max_workers = 0,
    segment_duration = 10,
    lookahead_frames = 40,
    output_format = "hls"
  } = args;

  // Spawn coordinator process
  const proc = spawn(COORDINATOR_BIN, [
    "--input", input_path,
    "--output", output_path,
    "--workers", max_workers.toString(),
    "--segment-duration", segment_duration.toString(),
    "--lookahead", lookahead_frames.toString(),
    "--format", output_format
  ]);

  let stdout = "";
  let stderr = "";

  proc.stdout.on("data", (data) => {
    stdout += data.toString();
  });

  proc.stderr.on("data", (data) => {
    stderr += data.toString();
  });

  return new Promise((resolve, reject) => {
    proc.on("close", (code) => {
      if (code === 0) {
        resolve({
          content: [
            {
              type: "text",
              text: `Transcoding completed successfully!\n\nOutput: ${output_path}\n\n${stdout}`
            }
          ]
        });
      } else {
        reject(new Error(`Transcoding failed with code ${code}: ${stderr}`));
      }
    });
  });
}

// Start server
const transport = new StdioServerTransport();
await server.connect(transport);
console.error("Parallel Video Transcoder MCP Server running...");
```

### Phase 5: Build System & Bundling

**5.1 Rust Build Configuration**

Create `build.sh` script:
```bash
#!/bin/bash
set -e

# Build Rust binaries for current platform
cargo build --release

# Copy binaries to bin/
mkdir -p bin/
cp target/release/transcoder-coordinator bin/
cp target/release/transcoder-worker bin/

# Copy FFmpeg shared libraries
# (This requires FFmpeg installed on build system)
if [[ "$OSTYPE" == "darwin"* ]]; then
    mkdir -p lib/
    cp /opt/homebrew/lib/libav*.dylib lib/
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    mkdir -p lib/
    cp /usr/lib/x86_64-linux-gnu/libav*.so* lib/
fi

echo "Build complete!"
```

**5.2 Cross-Compilation for Multiple Platforms**

For production bundles, need to build for:
- macOS (darwin-arm64, darwin-x64)
- Windows (win32-x64)
- Linux (linux-x64)

Use GitHub Actions or cross-compilation tools:
```bash
# Cross-compile for Windows from macOS/Linux
cargo build --release --target x86_64-pc-windows-gnu

# Cross-compile for Linux from macOS
cargo build --release --target x86_64-unknown-linux-gnu
```

**5.3 MCPB Packaging**

```bash
#!/bin/bash
# package.sh

# Install Node.js dependencies
npm install --production

# Build Rust binaries
./build.sh

# Create MCPB bundle
mcpb pack . parallel-transcoder.mcpb

echo "Bundle created: parallel-transcoder.mcpb"
```

### Phase 6: Testing & Validation

**6.1 Unit Tests**

**Rust tests:**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyframe_detection() {
        // Test keyframe detection on sample video
    }

    #[test]
    fn test_scene_change_detection() {
        // Test scene change threshold
    }

    #[test]
    fn test_complexity_calculation() {
        // Test complexity estimation
    }
}
```

**6.2 Integration Tests**

Test scenarios:
1. **Simple transcode**: Single short video (30 seconds)
2. **Long video**: 10+ minute video with multiple scenes
3. **High complexity**: Action video with fast motion
4. **Low complexity**: Static talking head video
5. **Error handling**: Invalid input, permission errors

**6.3 Performance Benchmarks**

Compare against standard FFmpeg:
```bash
# Baseline: Standard ffmpeg
time ffmpeg -i input.mp4 -c:v libx264 output.mp4

# Parallel transcoder
time ./bin/transcoder-coordinator --input input.mp4 --output output.mp4 --workers 8
```

**Expected Results:**
- 4-8x speedup on 8-core machine
- Minimal quality loss (< 0.5 dB PSNR difference)
- HLS output plays correctly in browsers/VLC

---

## Critical Files to Create

### Rust Workspace
- `/Volumes/FastDisk3TB/muni/Developer/parallel-transcoder/Cargo.toml` - Workspace manifest
- `/Volumes/FastDisk3TB/muni/Developer/parallel-transcoder/coordinator/Cargo.toml`
- `/Volumes/FastDisk3TB/muni/Developer/parallel-transcoder/coordinator/src/main.rs`
- `/Volumes/FastDisk3TB/muni/Developer/parallel-transcoder/coordinator/src/analyzer.rs`
- `/Volumes/FastDisk3TB/muni/Developer/parallel-transcoder/coordinator/src/segmenter.rs`
- `/Volumes/FastDisk3TB/muni/Developer/parallel-transcoder/worker/Cargo.toml`
- `/Volumes/FastDisk3TB/muni/Developer/parallel-transcoder/worker/src/main.rs`
- `/Volumes/FastDisk3TB/muni/Developer/parallel-transcoder/worker/src/lookahead.rs`

### Node.js MCP Server
- `/Volumes/FastDisk3TB/muni/Developer/parallel-transcoder/server/index.js`
- `/Volumes/FastDisk3TB/muni/Developer/parallel-transcoder/package.json`

### MCPB Bundle
- `/Volumes/FastDisk3TB/muni/Developer/parallel-transcoder/manifest.json`
- `/Volumes/FastDisk3TB/muni/Developer/parallel-transcoder/README.md`
- `/Volumes/FastDisk3TB/muni/Developer/parallel-transcoder/icon.png`

### Build Scripts
- `/Volumes/FastDisk3TB/muni/Developer/parallel-transcoder/build.sh`
- `/Volumes/FastDisk3TB/muni/Developer/parallel-transcoder/package.sh`

---

## Verification Steps

### End-to-End Test
1. **Build the bundle**: `./package.sh`
2. **Install in Claude Desktop**: Open `parallel-transcoder.mcpb`
3. **Test transcoding**: Ask Claude to transcode a test video
4. **Verify output**:
   - HLS playlist loads correctly
   - Video plays in browser/VLC
   - Quality is acceptable
   - Processing was faster than sequential

### MCP Tool Testing
```bash
# Test MCP server standalone
node server/index.js

# Send test request (in another terminal)
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"analyze_video","arguments":{"input_path":"/path/to/test.mp4"}}}' | node server/index.js
```

---

## Open Questions & Future Enhancements

### Open Questions
1. **GPU Acceleration**: Should we add support for hardware encoders (NVENC, VideoToolbox, QuickSync)?
2. **Distributed Processing**: Should coordinator support distributing work across multiple machines?
3. **Output Formats**: MP4, WebM, or HLS only?
4. **Quality Presets**: Should we expose encoding presets (fast, balanced, high-quality)?

### Future Enhancements
1. **Real-time Progress UI**: WebSocket-based progress reporting
2. **Adaptive Segmentation**: Dynamically adjust segment size based on complexity
3. **Multi-bitrate Output**: Generate multiple quality variants in one pass
4. **Cloud Integration**: S3/GCS input/output support
5. **Resume Capability**: Checkpoint progress and resume interrupted jobs

---

## Summary

This plan outlines a production-ready parallel video transcoding MCPB bundle that:

1. ✅ Uses **Rust + FFmpeg bindings** for high performance
2. ✅ Implements **keyframe-aligned segmentation** for clean parallel processing
3. ✅ Applies **look-ahead optimization** (scene detection, complexity analysis)
4. ✅ Coordinates **multiple worker processes** via a supervisor pattern
5. ✅ Outputs **HLS format** for simple reassembly
6. ✅ Follows **MCPB specification** for bundling and distribution
7. ✅ Provides **MCP tools** for Claude to control transcoding

**Expected Performance**: 4-8x speedup on multi-core machines with minimal quality loss.

**Timeline Estimate**:
- Phase 1-2: 2-3 days (setup, coordinator)
- Phase 3: 2 days (worker implementation)
- Phase 4: 1 day (MCP server)
- Phase 5: 1 day (build system)
- Phase 6: 1-2 days (testing)
- **Total**: ~7-10 days of development time
