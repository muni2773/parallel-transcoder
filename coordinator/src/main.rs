use anyhow::Result;
use clap::Parser;
use tracing::{info, warn};

mod analyzer;
mod segmenter;

#[derive(Parser, Debug)]
#[command(name = "transcoder-coordinator")]
#[command(about = "Coordinator for parallel video transcoding", long_about = None)]
struct Args {
    /// Input video file path
    #[arg(short, long)]
    input: String,

    /// Output path (directory for HLS, file for MP4)
    #[arg(short, long)]
    output: String,

    /// Number of worker processes (0 = auto-detect CPU cores)
    #[arg(short, long, default_value = "0")]
    workers: usize,

    /// Target segment duration in seconds
    #[arg(short, long, default_value = "10.0")]
    segment_duration: f64,

    /// Look-ahead frames for optimization
    #[arg(short, long, default_value = "40")]
    lookahead: usize,

    /// Output format: hls or mp4
    #[arg(short, long, default_value = "hls")]
    format: String,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let log_level = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(log_level)
        .init();

    info!("Parallel Video Transcoder Coordinator v0.1.0");
    info!("Input: {}", args.input);
    info!("Output: {}", args.output);

    let num_workers = if args.workers == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    } else {
        args.workers
    };
    info!("Using {} worker processes", num_workers);

    // Phase 1 — Video Analysis
    info!("Step 1: Analyzing video...");
    let metadata = analyzer::analyze_video(&args.input)?;
    info!(
        "  Video: {}x{} @ {:.2} fps, {:.2}s, {} total frames",
        metadata.width, metadata.height, metadata.fps, metadata.duration_secs, metadata.total_frames
    );
    info!(
        "  Found {} keyframes, {} scene changes",
        metadata.keyframe_positions.len(),
        metadata.scene_changes.len()
    );

    // Phase 2 — Segmentation
    info!("Step 2: Creating segments...");
    let segments = segmenter::create_segments(&metadata, args.segment_duration)?;
    info!("  Created {} segments", segments.len());
    for seg in &segments {
        info!(
            "  Segment {}: frames {}-{} ({:.2}s-{:.2}s) complexity={:.3}",
            seg.id, seg.start_frame, seg.end_frame, seg.start_timestamp, seg.end_timestamp, seg.complexity_estimate
        );
    }

    // Phase 3 — Spawn Workers (TODO)
    info!("Step 3: Spawning {} workers...", num_workers);
    warn!("Worker spawning not yet implemented — segments are ready for distribution");

    // Phase 4 — Reassemble (TODO)
    info!("Step 4: Reassembling output...");
    warn!("Output reassembly not yet implemented");

    info!("Pipeline complete (analysis + segmentation done, transcoding pending)");

    Ok(())
}
