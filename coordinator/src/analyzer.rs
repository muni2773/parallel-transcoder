extern crate ffmpeg_next as ffmpeg;

use anyhow::{Context, Result};
use ffmpeg::format::input;
use ffmpeg::media::Type;
use ffmpeg::software::scaling::{context::Context as ScalerContext, flag::Flags};
use ffmpeg::util::format::pixel::Pixel;
use ffmpeg::util::frame::video::Video;
use serde::{Serialize, Deserialize};
use std::path::Path;
use std::process::Command as StdCommand;
use tracing::{debug, info, warn};

/// Per-GOP statistics computed from packet data (no decoding required).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GopStats {
    pub start_frame: u64,
    pub end_frame: u64,
    pub start_time: f64,
    pub end_time: f64,
    pub size_bytes: u64,
    pub bitrate_bps: f64,
}

/// Audio track information extracted from ffprobe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioTrackInfo {
    pub stream_index: usize,
    pub codec_name: String,
    pub language: Option<String>,
    pub channels: u32,
    pub channel_layout: Option<String>,
    pub sample_rate: u32,
    pub bitrate_bps: Option<u64>,
    pub is_default: bool,
}

/// Subtitle track information extracted from ffprobe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtitleTrackInfo {
    pub stream_index: usize,
    pub codec_name: String,
    pub language: Option<String>,
    pub is_text_based: bool,
    pub is_default: bool,
}

/// Chapter information extracted from ffprobe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterInfo {
    pub id: u64,
    pub start_secs: f64,
    pub end_secs: f64,
    pub title: Option<String>,
}

/// Video metadata extracted from analysis
#[derive(Debug, Clone)]
pub struct VideoMetadata {
    pub duration_secs: f64,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub total_frames: u64,
    pub codec_name: String,
    pub keyframe_positions: Vec<u64>,
    pub keyframe_timestamps: Vec<f64>,
    pub scene_changes: Vec<u64>,
    pub complexity_map: Vec<f32>,
    pub time_base: (i32, i32),
    /// Per-GOP bitrate stats (only populated by fast_analyze_video)
    pub gop_stats: Vec<GopStats>,
    pub profile: Option<String>,
    pub level: Option<i64>,
    pub pix_fmt: Option<String>,
    pub audio_tracks: Vec<AudioTrackInfo>,
    pub subtitle_tracks: Vec<SubtitleTrackInfo>,
    pub chapters: Vec<ChapterInfo>,
}

const SCENE_CHANGE_THRESHOLD: f64 = 0.30;
const HISTOGRAM_BINS: usize = 64;

/// Analyze a video file and extract metadata for segmentation.
///
/// Performs a single pass through the video to collect:
/// - Basic metadata (resolution, fps, duration)
/// - All keyframe positions
/// - Scene change locations (via histogram comparison)
/// - Per-frame complexity estimates (spatial variance)
pub fn analyze_video(input_path: &str) -> Result<VideoMetadata> {
    let path = Path::new(input_path);
    if !path.exists() {
        anyhow::bail!("Input file does not exist: {}", input_path);
    }

    ffmpeg::init().context("Failed to initialize FFmpeg")?;

    let mut ictx = input(&input_path).context("Failed to open input video")?;

    // Find video stream
    let stream = ictx
        .streams()
        .best(Type::Video)
        .ok_or_else(|| anyhow::anyhow!("No video stream found in {}", input_path))?;

    let video_stream_index = stream.index();
    let time_base = stream.time_base();
    let tb_num = time_base.numerator();
    let tb_den = time_base.denominator();
    let avg_frame_rate = stream.avg_frame_rate();

    let fps = if avg_frame_rate.denominator() != 0 {
        avg_frame_rate.numerator() as f64 / avg_frame_rate.denominator() as f64
    } else {
        30.0
    };

    let stream_duration = stream.duration();
    let duration_secs = if stream_duration > 0 {
        stream_duration as f64 * tb_num as f64 / tb_den as f64
    } else {
        // Fall back to format-level duration
        let fmt_dur = ictx.duration();
        if fmt_dur > 0 {
            fmt_dur as f64 / f64::from(ffmpeg::ffi::AV_TIME_BASE)
        } else {
            0.0
        }
    };

    // Create decoder
    let codec_params = stream.parameters();
    let context_decoder =
        ffmpeg::codec::context::Context::from_parameters(codec_params)
            .context("Failed to create codec context")?;
    let mut decoder = context_decoder
        .decoder()
        .video()
        .context("Failed to open video decoder")?;

    let width = decoder.width();
    let height = decoder.height();
    let codec_name = decoder
        .codec()
        .map(|c| c.name().to_string())
        .unwrap_or_else(|| "unknown".into());

    info!(
        "Analyzing: {}x{} @ {:.2} fps, {:.2}s, codec={}",
        width, height, fps, duration_secs, codec_name
    );

    // Scaler for converting frames to grayscale (Y plane) for analysis
    let mut scaler = ScalerContext::get(
        decoder.format(),
        width,
        height,
        Pixel::GRAY8,
        width,
        height,
        Flags::BILINEAR,
    )
    .context("Failed to create pixel format scaler")?;

    let mut keyframe_positions: Vec<u64> = Vec::new();
    let mut keyframe_timestamps: Vec<f64> = Vec::new();
    let mut scene_changes: Vec<u64> = Vec::new();
    let mut complexity_map: Vec<f32> = Vec::new();

    let mut prev_histogram: Option<[f64; HISTOGRAM_BINS]> = None;
    let mut frame_count: u64 = 0;
    let mut decoded_frame = Video::empty();
    let mut gray_frame = Video::empty();

    // Single pass: iterate all packets
    for (stream_ref, packet) in ictx.packets() {
        if stream_ref.index() != video_stream_index {
            continue;
        }

        decoder.send_packet(&packet)?;

        while decoder.receive_frame(&mut decoded_frame).is_ok() {
            // Keyframe detection
            if decoded_frame.is_key() {
                let pts_secs = pts_to_secs(decoded_frame.pts(), tb_num, tb_den);
                keyframe_positions.push(frame_count);
                keyframe_timestamps.push(pts_secs);
                debug!("Keyframe at frame {} ({:.3}s)", frame_count, pts_secs);
            }

            // Convert to grayscale for analysis
            scaler.run(&decoded_frame, &mut gray_frame)?;
            let gray_data = gray_frame.data(0);
            let stride = gray_frame.stride(0) as usize;

            // Spatial complexity (variance of pixel intensities)
            let complexity =
                spatial_complexity(gray_data, width as usize, height as usize, stride);
            complexity_map.push(complexity);

            // Scene change detection via histogram comparison
            let histogram = compute_histogram(gray_data, width as usize, height as usize, stride);
            if let Some(ref prev) = prev_histogram {
                let diff = histogram_chi_square(prev, &histogram);
                if diff > SCENE_CHANGE_THRESHOLD {
                    scene_changes.push(frame_count);
                    debug!(
                        "Scene change at frame {} (diff={:.4})",
                        frame_count, diff
                    );
                }
            }
            prev_histogram = Some(histogram);

            frame_count += 1;

            if frame_count % 1000 == 0 {
                info!("Analyzed {} frames...", frame_count);
            }
        }
    }

    // Flush decoder
    decoder.send_eof()?;
    while decoder.receive_frame(&mut decoded_frame).is_ok() {
        if decoded_frame.is_key() {
            let pts_secs = pts_to_secs(decoded_frame.pts(), tb_num, tb_den);
            keyframe_positions.push(frame_count);
            keyframe_timestamps.push(pts_secs);
        }

        scaler.run(&decoded_frame, &mut gray_frame)?;
        let gray_data = gray_frame.data(0);
        let stride = gray_frame.stride(0) as usize;

        let complexity = spatial_complexity(gray_data, width as usize, height as usize, stride);
        complexity_map.push(complexity);

        let histogram = compute_histogram(gray_data, width as usize, height as usize, stride);
        if let Some(ref prev) = prev_histogram {
            let diff = histogram_chi_square(prev, &histogram);
            if diff > SCENE_CHANGE_THRESHOLD {
                scene_changes.push(frame_count);
            }
        }
        prev_histogram = Some(histogram);

        frame_count += 1;
    }

    info!(
        "Analysis complete: {} frames, {} keyframes, {} scene changes",
        frame_count,
        keyframe_positions.len(),
        scene_changes.len()
    );

    Ok(VideoMetadata {
        duration_secs,
        width,
        height,
        fps,
        total_frames: frame_count,
        codec_name,
        keyframe_positions,
        keyframe_timestamps,
        scene_changes,
        complexity_map,
        time_base: (tb_num, tb_den),
        gop_stats: vec![], // only populated by fast_analyze_video
        profile: None,
        level: None,
        pix_fmt: None,
        audio_tracks: vec![],
        subtitle_tracks: vec![],
        chapters: vec![],
    })
}

/// Fast video analysis using ffprobe subprocesses instead of full decode.
///
/// Runs two ffprobe commands:
/// 1. Stream/format metadata (fps, resolution, duration, codec)
/// 2. Packet-level keyframe positions (flags with 'K')
///
/// Returns VideoMetadata with real keyframes but uniform complexity and no scene changes.
/// Drops analysis time from ~68s to <1s.
pub fn fast_analyze_video(input_path: &str) -> Result<VideoMetadata> {
    let path = Path::new(input_path);
    if !path.exists() {
        anyhow::bail!("Input file does not exist: {}", input_path);
    }

    info!("Fast analysis via ffprobe: {}", input_path);

    // 1. Get stream and format metadata
    let probe_output = StdCommand::new("ffprobe")
        .args([
            "-v", "quiet",
            "-show_streams",
            "-show_format",
            "-show_chapters",
            "-print_format", "json",
            input_path,
        ])
        .output()
        .context("Failed to run ffprobe for metadata")?;

    if !probe_output.status.success() {
        let stderr = String::from_utf8_lossy(&probe_output.stderr);
        anyhow::bail!("ffprobe metadata failed: {}", stderr);
    }

    let probe_json: serde_json::Value =
        serde_json::from_slice(&probe_output.stdout).context("Failed to parse ffprobe JSON")?;

    // Extract video stream info
    let streams = probe_json["streams"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("No streams found in ffprobe output"))?;

    let stream = streams
        .iter()
        .find(|s| s["codec_type"].as_str() == Some("video"))
        .ok_or_else(|| anyhow::anyhow!("No video stream found in ffprobe output"))?;

    let width = stream["width"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("Missing width"))? as u32;
    let height = stream["height"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("Missing height"))? as u32;
    let codec_name = stream["codec_name"]
        .as_str()
        .unwrap_or("unknown")
        .to_string();
    let profile = stream["profile"].as_str().map(|s| s.to_string());
    let level = stream["level"].as_i64();
    let pix_fmt = stream["pix_fmt"].as_str().map(|s| s.to_string());

    // Parse fps from r_frame_rate (e.g. "30000/1001")
    let fps = parse_rational_str(stream["r_frame_rate"].as_str().unwrap_or("30/1"));

    // Parse time_base (e.g. "1/30000")
    let (tb_num, tb_den) = parse_time_base_str(stream["time_base"].as_str().unwrap_or("1/90000"));

    // Duration: try stream duration, then format duration
    let duration_secs = stream["duration"]
        .as_str()
        .and_then(|s| s.parse::<f64>().ok())
        .or_else(|| {
            probe_json["format"]["duration"]
                .as_str()
                .and_then(|s| s.parse::<f64>().ok())
        })
        .unwrap_or(0.0);

    // Total frames: try nb_frames, then estimate from duration * fps
    let total_frames = stream["nb_frames"]
        .as_str()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or_else(|| (duration_secs * fps).round() as u64);

    // Parse audio tracks
    let audio_tracks: Vec<AudioTrackInfo> = streams
        .iter()
        .filter(|s| s["codec_type"].as_str() == Some("audio"))
        .map(|s| {
            let tags = &s["tags"];
            AudioTrackInfo {
                stream_index: s["index"].as_u64().unwrap_or(0) as usize,
                codec_name: s["codec_name"].as_str().unwrap_or("unknown").to_string(),
                language: tags["language"].as_str().map(|l| l.to_string()),
                channels: s["channels"].as_u64().unwrap_or(0) as u32,
                channel_layout: s["channel_layout"].as_str().map(|l| l.to_string()),
                sample_rate: s["sample_rate"]
                    .as_str()
                    .and_then(|sr| sr.parse().ok())
                    .unwrap_or(0),
                bitrate_bps: s["bit_rate"]
                    .as_str()
                    .and_then(|br| br.parse().ok()),
                is_default: s["disposition"]["default"].as_u64() == Some(1),
            }
        })
        .collect();

    // Parse subtitle tracks
    let subtitle_tracks: Vec<SubtitleTrackInfo> = streams
        .iter()
        .filter(|s| s["codec_type"].as_str() == Some("subtitle"))
        .map(|s| {
            let codec = s["codec_name"].as_str().unwrap_or("unknown");
            let tags = &s["tags"];
            SubtitleTrackInfo {
                stream_index: s["index"].as_u64().unwrap_or(0) as usize,
                codec_name: codec.to_string(),
                language: tags["language"].as_str().map(|l| l.to_string()),
                is_text_based: matches!(codec, "srt" | "ass" | "ssa" | "webvtt" | "subrip" | "mov_text" | "text"),
                is_default: s["disposition"]["default"].as_u64() == Some(1),
            }
        })
        .collect();

    // Parse chapters
    let chapters: Vec<ChapterInfo> = probe_json["chapters"]
        .as_array()
        .map(|chs| {
            chs.iter()
                .map(|ch| {
                    let time_base_str = ch["time_base"].as_str().unwrap_or("1/1000000000");
                    let tb = parse_rational_str(time_base_str);
                    ChapterInfo {
                        id: ch["id"].as_u64().unwrap_or(0),
                        start_secs: ch["start"].as_i64().unwrap_or(0) as f64 * tb,
                        end_secs: ch["end"].as_i64().unwrap_or(0) as f64 * tb,
                        title: ch["tags"]["title"].as_str().map(|t| t.to_string()),
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    info!(
        "Fast probe: {}x{} @ {:.2} fps, {:.2}s, {} frames, codec={}",
        width, height, fps, duration_secs, total_frames, codec_name
    );

    // 2. Get keyframe positions and packet sizes via packet probing
    let packets_output = StdCommand::new("ffprobe")
        .args([
            "-v", "quiet",
            "-show_packets",
            "-show_entries", "packet=pts_time,flags,size",
            "-select_streams", "v:0",
            "-print_format", "json",
            input_path,
        ])
        .output()
        .context("Failed to run ffprobe for keyframes")?;

    if !packets_output.status.success() {
        let stderr = String::from_utf8_lossy(&packets_output.stderr);
        anyhow::bail!("ffprobe packets failed: {}", stderr);
    }

    let packets_json: serde_json::Value = serde_json::from_slice(&packets_output.stdout)
        .context("Failed to parse ffprobe packets JSON")?;

    let mut keyframe_positions: Vec<u64> = Vec::new();
    let mut keyframe_timestamps: Vec<f64> = Vec::new();

    // Collect per-packet data for GOP bitrate computation
    struct PacketInfo {
        pts_time: f64,
        size: u64,
        is_key: bool,
    }
    let mut all_packets: Vec<PacketInfo> = Vec::new();

    if let Some(packets) = packets_json["packets"].as_array() {
        let mut frame_index: u64 = 0;
        for pkt in packets {
            let flags = pkt["flags"].as_str().unwrap_or("");
            let pts_time = pkt["pts_time"]
                .as_str()
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0);
            let size = pkt["size"]
                .as_str()
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0);
            let is_key = flags.contains('K');

            if is_key {
                keyframe_positions.push(frame_index);
                keyframe_timestamps.push(pts_time);
            }

            all_packets.push(PacketInfo { pts_time, size, is_key });
            frame_index += 1;
        }
    }

    // Ensure frame 0 is always a keyframe
    if keyframe_positions.is_empty() || keyframe_positions[0] != 0 {
        warn!("No keyframe at frame 0 detected, inserting one");
        keyframe_positions.insert(0, 0);
        keyframe_timestamps.insert(0, 0.0);
    }

    // Compute per-GOP bitrate stats
    let mut gop_stats: Vec<GopStats> = Vec::new();
    let mut gop_start_frame: u64 = 0;
    let mut gop_start_time: f64 = 0.0;
    let mut gop_size: u64 = 0;

    for (i, pkt) in all_packets.iter().enumerate() {
        if pkt.is_key && i > 0 {
            // Close previous GOP
            let gop_end_frame = i as u64 - 1;
            let gop_end_time = all_packets[i - 1].pts_time;
            let gop_duration = gop_end_time - gop_start_time;
            let bitrate = if gop_duration > 0.001 {
                gop_size as f64 * 8.0 / gop_duration
            } else {
                0.0
            };
            gop_stats.push(GopStats {
                start_frame: gop_start_frame,
                end_frame: gop_end_frame,
                start_time: gop_start_time,
                end_time: gop_end_time,
                size_bytes: gop_size,
                bitrate_bps: bitrate,
            });

            // Start new GOP
            gop_start_frame = i as u64;
            gop_start_time = pkt.pts_time;
            gop_size = 0;
        }
        gop_size += pkt.size;
    }

    // Close final GOP
    if !all_packets.is_empty() {
        let last = all_packets.len() - 1;
        let gop_end_time = all_packets[last].pts_time;
        let gop_duration = gop_end_time - gop_start_time;
        let bitrate = if gop_duration > 0.001 {
            gop_size as f64 * 8.0 / gop_duration
        } else {
            0.0
        };
        gop_stats.push(GopStats {
            start_frame: gop_start_frame,
            end_frame: last as u64,
            start_time: gop_start_time,
            end_time: gop_end_time,
            size_bytes: gop_size,
            bitrate_bps: bitrate,
        });
    }

    // Log GOP bitrate stats
    let avg_bitrate: f64 = if !gop_stats.is_empty() {
        gop_stats.iter().map(|g| g.bitrate_bps).sum::<f64>() / gop_stats.len() as f64
    } else {
        0.0
    };
    let min_bitrate = gop_stats.iter().map(|g| g.bitrate_bps).fold(f64::MAX, f64::min);
    let max_bitrate = gop_stats.iter().map(|g| g.bitrate_bps).fold(0.0f64, f64::max);

    info!(
        "Fast analysis complete: {} keyframes, {} GOPs in {:.2}s video",
        keyframe_positions.len(),
        gop_stats.len(),
        duration_secs
    );
    info!(
        "  GOP bitrates: avg={:.1} Mbps, min={:.1} Mbps, max={:.1} Mbps",
        avg_bitrate / 1_000_000.0,
        min_bitrate / 1_000_000.0,
        max_bitrate / 1_000_000.0
    );
    if !audio_tracks.is_empty() {
        info!("  Audio tracks: {}", audio_tracks.len());
        for t in &audio_tracks {
            info!("    #{}: {} {}ch{}", t.stream_index, t.codec_name, t.channels,
                t.language.as_deref().map(|l| format!(" [{}]", l)).unwrap_or_default());
        }
    }
    if !subtitle_tracks.is_empty() {
        info!("  Subtitle tracks: {}", subtitle_tracks.len());
        for t in &subtitle_tracks {
            info!("    #{}: {}{}{}", t.stream_index, t.codec_name,
                if t.is_text_based { " (text)" } else { " (bitmap)" },
                t.language.as_deref().map(|l| format!(" [{}]", l)).unwrap_or_default());
        }
    }
    if !chapters.is_empty() {
        info!("  Chapters: {}", chapters.len());
    }

    // Return metadata with uniform complexity and no scene changes
    let num_frames = total_frames as usize;
    Ok(VideoMetadata {
        duration_secs,
        width,
        height,
        fps,
        total_frames,
        codec_name,
        keyframe_positions,
        keyframe_timestamps,
        scene_changes: vec![],
        complexity_map: vec![0.5; num_frames],
        time_base: (tb_num, tb_den),
        gop_stats,
        profile,
        level,
        pix_fmt,
        audio_tracks,
        subtitle_tracks,
        chapters,
    })
}

/// Parse a rational string like "30000/1001" into a f64.
fn parse_rational_str(s: &str) -> f64 {
    if let Some((num, den)) = s.split_once('/') {
        let n: f64 = num.parse().unwrap_or(30.0);
        let d: f64 = den.parse().unwrap_or(1.0);
        if d != 0.0 { n / d } else { 30.0 }
    } else {
        s.parse().unwrap_or(30.0)
    }
}

/// Parse a time_base string like "1/90000" into (numerator, denominator).
fn parse_time_base_str(s: &str) -> (i32, i32) {
    if let Some((num, den)) = s.split_once('/') {
        let n: i32 = num.parse().unwrap_or(1);
        let d: i32 = den.parse().unwrap_or(90000);
        (n, d)
    } else {
        (1, 90000)
    }
}

/// Convert PTS value to seconds using the stream time base.
fn pts_to_secs(pts: Option<i64>, tb_num: i32, tb_den: i32) -> f64 {
    match pts {
        Some(p) => p as f64 * tb_num as f64 / tb_den as f64,
        None => 0.0,
    }
}

/// Compute a normalized histogram of grayscale pixel intensities.
///
/// Uses `HISTOGRAM_BINS` bins, each covering `256 / HISTOGRAM_BINS` intensity levels.
/// The result is normalized so bins sum to 1.0.
fn compute_histogram(
    data: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> [f64; HISTOGRAM_BINS] {
    let mut bins = [0u64; HISTOGRAM_BINS];
    let bin_width = 256 / HISTOGRAM_BINS;

    for y in 0..height {
        let row = &data[y * stride..y * stride + width];
        for &pixel in row {
            let bin = (pixel as usize) / bin_width;
            let bin = bin.min(HISTOGRAM_BINS - 1);
            bins[bin] += 1;
        }
    }

    let total = (width * height) as f64;
    let mut normalized = [0.0f64; HISTOGRAM_BINS];
    for i in 0..HISTOGRAM_BINS {
        normalized[i] = bins[i] as f64 / total;
    }
    normalized
}

/// Chi-square distance between two histograms.
///
/// Returns a value in [0, 1+] where 0 means identical and higher means more different.
/// Good for scene change detection — values above ~0.3 typically indicate a scene cut.
fn histogram_chi_square(h1: &[f64; HISTOGRAM_BINS], h2: &[f64; HISTOGRAM_BINS]) -> f64 {
    let mut chi2 = 0.0;
    for i in 0..HISTOGRAM_BINS {
        let sum = h1[i] + h2[i];
        if sum > 1e-10 {
            let diff = h1[i] - h2[i];
            chi2 += (diff * diff) / sum;
        }
    }
    chi2 / 2.0 // Normalize to [0, 1] range for identical-length histograms
}

/// Calculate spatial complexity of a grayscale frame.
///
/// Uses the variance of pixel intensities as a proxy — higher variance means more
/// visual detail/texture which requires more bits to encode.
/// Returns a value normalized to approximately [0.0, 1.0].
fn spatial_complexity(data: &[u8], width: usize, height: usize, stride: usize) -> f32 {
    let pixel_count = (width * height) as f64;
    if pixel_count == 0.0 {
        return 0.0;
    }

    // Compute mean
    let mut sum = 0u64;
    for y in 0..height {
        let row = &data[y * stride..y * stride + width];
        for &pixel in row {
            sum += pixel as u64;
        }
    }
    let mean = sum as f64 / pixel_count;

    // Compute variance
    let mut var_sum = 0.0f64;
    for y in 0..height {
        let row = &data[y * stride..y * stride + width];
        for &pixel in row {
            let diff = pixel as f64 - mean;
            var_sum += diff * diff;
        }
    }
    let variance = var_sum / pixel_count;

    // Normalize: max possible variance for 8-bit is ~(127.5)^2 ≈ 16256
    // Typical video frames have variance in 200-4000 range
    let normalized = (variance / 4000.0).min(1.0) as f32;
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_identical() {
        let h = [1.0 / HISTOGRAM_BINS as f64; HISTOGRAM_BINS];
        let diff = histogram_chi_square(&h, &h);
        assert!(diff.abs() < 1e-10, "Identical histograms should have zero distance");
    }

    #[test]
    fn test_histogram_different() {
        let mut h1 = [0.0f64; HISTOGRAM_BINS];
        let mut h2 = [0.0f64; HISTOGRAM_BINS];
        // Concentrate all weight at opposite ends
        h1[0] = 1.0;
        h2[HISTOGRAM_BINS - 1] = 1.0;
        let diff = histogram_chi_square(&h1, &h2);
        assert!(
            diff > SCENE_CHANGE_THRESHOLD,
            "Maximally different histograms should exceed scene change threshold"
        );
    }

    #[test]
    fn test_spatial_complexity_flat() {
        // Uniform gray image — low complexity
        let width = 64;
        let height = 64;
        let data = vec![128u8; width * height];
        let complexity = spatial_complexity(&data, width, height, width);
        assert!(
            complexity < 0.01,
            "Flat image should have near-zero complexity, got {}",
            complexity
        );
    }

    #[test]
    fn test_spatial_complexity_noisy() {
        // Alternating black/white pixels — high complexity
        let width = 64;
        let height = 64;
        let data: Vec<u8> = (0..width * height)
            .map(|i| if i % 2 == 0 { 0 } else { 255 })
            .collect();
        let complexity = spatial_complexity(&data, width, height, width);
        assert!(
            complexity > 0.5,
            "Noisy image should have high complexity, got {}",
            complexity
        );
    }

    #[test]
    fn test_compute_histogram_uniform() {
        // All pixels same value — should concentrate in one bin
        let width = 64;
        let height = 64;
        let data = vec![100u8; width * height];
        let hist = compute_histogram(&data, width, height, width);
        let expected_bin = 100 / (256 / HISTOGRAM_BINS);
        assert!(
            (hist[expected_bin] - 1.0).abs() < 1e-10,
            "All pixels should be in bin {}",
            expected_bin
        );
    }

    #[test]
    fn test_pts_to_secs() {
        assert!((pts_to_secs(Some(90000), 1, 90000) - 1.0).abs() < 1e-10);
        assert!((pts_to_secs(Some(48000), 1, 48000) - 1.0).abs() < 1e-10);
        assert!((pts_to_secs(None, 1, 90000)).abs() < 1e-10);
    }
}
