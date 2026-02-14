extern crate ffmpeg_next as ffmpeg;

use anyhow::{Context, Result};
use clap::Parser;
use ffmpeg::codec;
use ffmpeg::encoder;
use ffmpeg::format::{self, input, Pixel};
use ffmpeg::media::Type;
use ffmpeg::software::scaling::{context::Context as ScalerContext, flag::Flags};
use ffmpeg::util::frame::video::Video;
use ffmpeg::{Dictionary, Rational};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Instant;
use tracing::{debug, info};

mod lookahead;
use lookahead::{analyze_frame, GrayFrame};

#[derive(Parser, Debug)]
#[command(name = "transcoder-worker")]
#[command(about = "Worker process for parallel video transcoding")]
struct Args {
    /// Segment descriptor as JSON string
    #[arg(short, long)]
    segment: Option<String>,

    /// Input video file
    #[arg(short, long)]
    input: String,

    /// Output segment file (.ts for MPEG-TS)
    #[arg(short, long)]
    output: String,

    /// Worker ID
    #[arg(short, long)]
    worker_id: usize,

    /// Look-ahead buffer size (number of frames)
    #[arg(short, long, default_value = "40")]
    lookahead: usize,

    /// Target video bitrate in kbps (0 = CRF mode)
    #[arg(short, long, default_value = "0")]
    bitrate: u32,

    /// CRF value for quality-based encoding (lower = better quality)
    #[arg(short, long, default_value = "23")]
    crf: u32,

    /// Encoding preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
    #[arg(short, long, default_value = "medium")]
    preset: String,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentDescriptor {
    pub id: usize,
    pub start_frame: u64,
    pub end_frame: u64,
    pub start_timestamp: f64,
    pub end_timestamp: f64,
    pub lookahead_frames: Option<usize>,
    pub complexity_estimate: f32,
    pub scene_changes: Vec<u64>,
}

#[derive(Debug, Serialize)]
struct WorkerResult {
    segment_id: usize,
    worker_id: usize,
    frames_encoded: u64,
    output_size_bytes: u64,
    encoding_time_secs: f64,
    average_complexity: f32,
    scene_changes_detected: u64,
    output_path: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let log_level = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(log_level)
        .with_writer(std::io::stderr)
        .init();

    info!("Worker {} starting", args.worker_id);

    let segment: SegmentDescriptor = if let Some(ref json) = args.segment {
        serde_json::from_str(json).context("Failed to parse segment descriptor JSON")?
    } else {
        info!("Reading segment descriptor from stdin...");
        serde_json::from_reader(std::io::stdin()).context("Failed to read segment from stdin")?
    };

    info!(
        "Segment {}: frames {}-{} ({:.2}s-{:.2}s)",
        segment.id, segment.start_frame, segment.end_frame,
        segment.start_timestamp, segment.end_timestamp
    );

    let lookahead_size = segment.lookahead_frames.unwrap_or(args.lookahead);
    let start_time = Instant::now();

    let result = transcode_segment(&args, &segment, lookahead_size)?;

    let encoding_time = start_time.elapsed().as_secs_f64();
    let output_size = std::fs::metadata(&args.output)
        .map(|m| m.len())
        .unwrap_or(0);

    let worker_result = WorkerResult {
        segment_id: segment.id,
        worker_id: args.worker_id,
        frames_encoded: result.frames_encoded,
        output_size_bytes: output_size,
        encoding_time_secs: encoding_time,
        average_complexity: result.avg_complexity,
        scene_changes_detected: result.scene_changes,
        output_path: args.output.clone(),
    };

    // Write result JSON to stdout for coordinator to collect
    println!("{}", serde_json::to_string(&worker_result)?);

    info!(
        "Worker {} complete: {} frames in {:.2}s ({:.1} fps)",
        args.worker_id,
        result.frames_encoded,
        encoding_time,
        result.frames_encoded as f64 / encoding_time
    );

    Ok(())
}

struct TranscodeResult {
    frames_encoded: u64,
    avg_complexity: f32,
    scene_changes: u64,
}

/// Encode a frame and write packets to output.
fn encode_and_write(
    encoder: &mut ffmpeg::encoder::video::Video,
    packet: &mut ffmpeg::Packet,
    frame: &Video,
    encoder_tb: Rational,
    output_tb: Rational,
    octx: &mut format::context::Output,
) -> Result<()> {
    encoder.send_frame(frame)?;
    receive_and_write(encoder, packet, encoder_tb, output_tb, octx)
}

/// Receive all pending packets from encoder and write to output.
fn receive_and_write(
    encoder: &mut ffmpeg::encoder::video::Video,
    packet: &mut ffmpeg::Packet,
    encoder_tb: Rational,
    output_tb: Rational,
    octx: &mut format::context::Output,
) -> Result<()> {
    while encoder.receive_packet(packet).is_ok() {
        packet.rescale_ts(encoder_tb, output_tb);
        packet.set_stream(0);
        packet.write_interleaved(octx)?;
    }
    Ok(())
}

fn transcode_segment(
    args: &Args,
    segment: &SegmentDescriptor,
    lookahead_size: usize,
) -> Result<TranscodeResult> {
    ffmpeg::init().context("Failed to initialize FFmpeg")?;

    // --- Open input ---
    let mut ictx = input(&args.input).context("Failed to open input video")?;

    let input_stream = ictx
        .streams()
        .best(Type::Video)
        .ok_or_else(|| anyhow::anyhow!("No video stream found"))?;

    let video_stream_index = input_stream.index();
    let input_time_base = input_stream.time_base();
    let avg_frame_rate = input_stream.avg_frame_rate();

    let codec_params = input_stream.parameters();
    let decoder_ctx = codec::context::Context::from_parameters(codec_params)
        .context("Failed to create decoder context")?;
    let mut decoder = decoder_ctx
        .decoder()
        .video()
        .context("Failed to open video decoder")?;

    let width = decoder.width();
    let height = decoder.height();
    let src_format = decoder.format();

    info!("Input: {}x{} {:?}", width, height, src_format);

    // --- Scalers ---
    let enc_format = Pixel::YUV420P;

    let mut gray_scaler = ScalerContext::get(
        src_format, width, height,
        Pixel::GRAY8, width, height,
        Flags::BILINEAR,
    )
    .context("Failed to create grayscale scaler")?;

    let mut yuv_scaler = ScalerContext::get(
        src_format, width, height,
        enc_format, width, height,
        Flags::BILINEAR,
    )
    .context("Failed to create YUV420P scaler")?;

    // --- Set up encoder ---
    let h264 = encoder::find(codec::Id::H264)
        .ok_or_else(|| anyhow::anyhow!("H264 encoder not found — is libx264 available?"))?;

    let mut octx = format::output(&args.output).context("Failed to create output file")?;
    let global_header = octx.format().flags().contains(format::Flags::GLOBAL_HEADER);
    let mut ost = octx.add_stream(h264).context("Failed to add output stream")?;

    let enc_time_base = Rational::new(avg_frame_rate.denominator(), avg_frame_rate.numerator());

    let enc_ctx = codec::context::Context::new_with_codec(h264);
    let mut video_enc = enc_ctx.encoder().video()?;
    video_enc.set_width(width);
    video_enc.set_height(height);
    video_enc.set_format(enc_format);
    video_enc.set_time_base(enc_time_base);
    video_enc.set_frame_rate(Some(avg_frame_rate));

    let mut opts = Dictionary::new();
    opts.set("preset", &args.preset);
    if args.bitrate > 0 {
        video_enc.set_bit_rate(args.bitrate as usize * 1000);
    } else {
        opts.set("crf", &args.crf.to_string());
    }

    if global_header {
        video_enc.set_flags(codec::Flags::GLOBAL_HEADER);
    }

    let mut opened_encoder = video_enc
        .open_as_with(h264, opts)
        .context("Failed to open H264 encoder")?;
    ost.set_parameters(&opened_encoder);

    // Get the encoder's actual time_base and the output stream time_base
    let encoder_tb = opened_encoder.time_base();
    let output_tb = ost.time_base();

    octx.write_header().context("Failed to write output header")?;

    // Re-read output_tb after write_header (muxer may change it)
    let output_tb = octx.stream(0).unwrap().time_base();

    info!(
        "Encoder ready: H264 preset={} crf={}, encoder_tb={}/{}, output_tb={}/{}",
        args.preset, args.crf,
        encoder_tb.numerator(), encoder_tb.denominator(),
        output_tb.numerator(), output_tb.denominator()
    );

    // --- Seek to segment start ---
    if segment.start_timestamp > 0.1 {
        let seek_target =
            (segment.start_timestamp * f64::from(ffmpeg::ffi::AV_TIME_BASE)) as i64;
        ictx.seek(seek_target, ..seek_target)
            .context("Failed to seek to segment start")?;
        decoder.flush();
        info!("Seeked to {:.2}s", segment.start_timestamp);
    }

    // --- Decode all frames in segment ---
    // First, decode and buffer all frames, then encode with look-ahead.
    // This avoids complex interleaving and ensures the look-ahead buffer
    // is always full. Memory is bounded by segment size.

    let mut yuv_frames: Vec<Video> = Vec::new();
    let mut gray_frames: Vec<GrayFrame> = Vec::new();
    let mut reached_segment = segment.start_timestamp < 0.1;

    let mut decoded_frame = Video::empty();
    let mut gray_frame = Video::empty();

    for (stream, packet) in ictx.packets() {
        if stream.index() != video_stream_index {
            continue;
        }

        decoder.send_packet(&packet)?;

        while decoder.receive_frame(&mut decoded_frame).is_ok() {
            let pts = decoded_frame.pts().unwrap_or(0);
            let pts_secs = pts as f64 * input_time_base.numerator() as f64
                / input_time_base.denominator() as f64;

            if !reached_segment {
                if pts_secs < segment.start_timestamp - 0.001 {
                    continue;
                }
                reached_segment = true;
                debug!("Reached segment start at PTS {:.3}s", pts_secs);
            }

            if pts_secs > segment.end_timestamp + 0.001 {
                break;
            }

            // Convert to grayscale for analysis
            gray_scaler.run(&decoded_frame, &mut gray_frame)?;
            let gray_data = gray_frame.data(0);
            let stride = gray_frame.stride(0) as usize;
            let mut gray_vec = Vec::with_capacity(width as usize * height as usize);
            for y in 0..height as usize {
                gray_vec.extend_from_slice(&gray_data[y * stride..y * stride + width as usize]);
            }
            gray_frames.push(GrayFrame {
                data: gray_vec,
                width: width as usize,
                height: height as usize,
                frame_number: yuv_frames.len() as u64,
            });

            // Convert to YUV420P for encoding
            let mut yuv_frame = Video::empty();
            yuv_scaler.run(&decoded_frame, &mut yuv_frame)?;
            yuv_frames.push(yuv_frame);
        }
    }

    // Flush decoder
    decoder.send_eof()?;
    while decoder.receive_frame(&mut decoded_frame).is_ok() {
        if !reached_segment {
            continue;
        }
        let pts = decoded_frame.pts().unwrap_or(0);
        let pts_secs = pts as f64 * input_time_base.numerator() as f64
            / input_time_base.denominator() as f64;
        if pts_secs > segment.end_timestamp + 0.001 {
            break;
        }

        gray_scaler.run(&decoded_frame, &mut gray_frame)?;
        let gray_data = gray_frame.data(0);
        let stride = gray_frame.stride(0) as usize;
        let mut gray_vec = Vec::with_capacity(width as usize * height as usize);
        for y in 0..height as usize {
            gray_vec.extend_from_slice(&gray_data[y * stride..y * stride + width as usize]);
        }
        gray_frames.push(GrayFrame {
            data: gray_vec,
            width: width as usize,
            height: height as usize,
            frame_number: yuv_frames.len() as u64,
        });

        let mut yuv_frame = Video::empty();
        yuv_scaler.run(&decoded_frame, &mut yuv_frame)?;
        yuv_frames.push(yuv_frame);
    }

    info!("Decoded {} frames for encoding", yuv_frames.len());

    // --- Encode with look-ahead ---
    let total_frames = yuv_frames.len();
    let mut frames_encoded: u64 = 0;
    let mut complexity_sum: f32 = 0.0;
    let mut scene_change_count: u64 = 0;
    let mut enc_packet = ffmpeg::Packet::empty();

    for i in 0..total_frames {
        // Build look-ahead window for analysis
        let end = (i + lookahead_size + 1).min(gray_frames.len());
        let mut la_buf: VecDeque<GrayFrame> = gray_frames[i..end].iter().cloned().collect();

        let analysis = analyze_frame(&la_buf);

        if analysis.is_scene_change {
            scene_change_count += 1;
            debug!("Scene change at frame {} (complexity={:.3})", i, analysis.complexity);
        }
        complexity_sum += analysis.complexity;

        let frame = &mut yuv_frames[i];
        frame.set_pts(Some(i as i64));

        if analysis.is_scene_change {
            frame.set_kind(ffmpeg::picture::Type::I);
        }

        encode_and_write(
            &mut opened_encoder,
            &mut enc_packet,
            frame,
            encoder_tb,
            output_tb,
            &mut octx,
        )?;

        frames_encoded += 1;

        if frames_encoded % 100 == 0 {
            info!("Encoded {}/{} frames...", frames_encoded, total_frames);
        }
    }

    // Flush encoder
    opened_encoder.send_eof()?;
    receive_and_write(
        &mut opened_encoder,
        &mut enc_packet,
        encoder_tb,
        output_tb,
        &mut octx,
    )?;

    octx.write_trailer()
        .context("Failed to write output trailer")?;

    let avg_complexity = if frames_encoded > 0 {
        complexity_sum / frames_encoded as f32
    } else {
        0.0
    };

    info!(
        "Encoded {} frames, avg complexity={:.3}, scene changes={}",
        frames_encoded, avg_complexity, scene_change_count
    );

    Ok(TranscodeResult {
        frames_encoded,
        avg_complexity,
        scene_changes: scene_change_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_descriptor_deserialize() {
        let json = r#"{
            "id": 0,
            "start_frame": 0,
            "end_frame": 299,
            "start_timestamp": 0.0,
            "end_timestamp": 10.0,
            "complexity_estimate": 0.5,
            "scene_changes": [90, 180]
        }"#;
        let seg: SegmentDescriptor = serde_json::from_str(json).unwrap();
        assert_eq!(seg.id, 0);
        assert_eq!(seg.start_frame, 0);
        assert_eq!(seg.end_frame, 299);
        assert_eq!(seg.scene_changes.len(), 2);
        assert!(seg.lookahead_frames.is_none());
    }

    #[test]
    fn test_segment_descriptor_with_lookahead() {
        let json = r#"{
            "id": 1,
            "start_frame": 300,
            "end_frame": 599,
            "start_timestamp": 10.0,
            "end_timestamp": 20.0,
            "lookahead_frames": 60,
            "complexity_estimate": 0.7,
            "scene_changes": []
        }"#;
        let seg: SegmentDescriptor = serde_json::from_str(json).unwrap();
        assert_eq!(seg.lookahead_frames, Some(60));
    }

    #[test]
    fn test_worker_result_serialize() {
        let result = WorkerResult {
            segment_id: 0,
            worker_id: 1,
            frames_encoded: 300,
            output_size_bytes: 1024000,
            encoding_time_secs: 5.5,
            average_complexity: 0.42,
            scene_changes_detected: 2,
            output_path: "/tmp/seg_0.ts".into(),
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"frames_encoded\":300"));
        assert!(json.contains("\"worker_id\":1"));
    }
}
