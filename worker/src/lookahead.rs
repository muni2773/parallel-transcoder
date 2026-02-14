use std::collections::VecDeque;

const HISTOGRAM_BINS: usize = 64;
const SCENE_CHANGE_THRESHOLD: f64 = 0.30;

/// Grayscale frame data used for look-ahead analysis.
/// Stored separately from the full decoded frame to minimize memory usage.
#[derive(Clone)]
pub struct GrayFrame {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub frame_number: u64,
}

/// Analysis result for a frame based on look-ahead buffer contents.
pub struct FrameAnalysis {
    pub is_scene_change: bool,
    pub complexity: f32,
    pub spatial: f32,
    pub temporal: f32,
}

/// Analyze the current frame using the look-ahead buffer.
///
/// The first element in the buffer is the current frame being encoded.
/// Subsequent elements are future frames used for prediction.
pub fn analyze_frame(buffer: &VecDeque<GrayFrame>) -> FrameAnalysis {
    if buffer.is_empty() {
        return FrameAnalysis {
            is_scene_change: false,
            complexity: 0.5,
            spatial: 0.5,
            temporal: 0.0,
        };
    }

    let current = &buffer[0];
    let spatial = spatial_complexity(&current.data, current.width, current.height);

    let (temporal, is_scene_change) = if buffer.len() >= 2 {
        let next = &buffer[1];
        let temp = temporal_complexity(
            &current.data,
            current.width,
            current.height,
            &next.data,
            next.width,
            next.height,
        );
        let scene = detect_scene_change(
            &current.data,
            current.width,
            current.height,
            &next.data,
        );
        (temp, scene)
    } else {
        (0.0, false)
    };

    // Weighted combination: spatial 0.5, temporal 0.5, scene change bonus
    let mut complexity = spatial * 0.5 + temporal * 0.5;
    if is_scene_change {
        complexity = (complexity + 0.2).min(1.0);
    }

    FrameAnalysis {
        is_scene_change,
        complexity,
        spatial,
        temporal,
    }
}

/// Detect scene change between current and next frame using histogram chi-square distance.
pub fn detect_scene_change(
    current_data: &[u8],
    width: usize,
    height: usize,
    next_data: &[u8],
) -> bool {
    let h1 = compute_histogram(current_data, width, height);
    let h2 = compute_histogram(next_data, width, height);
    histogram_chi_square(&h1, &h2) > SCENE_CHANGE_THRESHOLD
}

/// Calculate spatial complexity of a grayscale frame.
///
/// Uses pixel intensity variance as proxy for visual detail.
/// Returns a value in approximately [0.0, 1.0].
pub fn spatial_complexity(data: &[u8], width: usize, height: usize) -> f32 {
    let pixel_count = width * height;
    if pixel_count == 0 {
        return 0.0;
    }

    let mut sum = 0u64;
    for &p in &data[..pixel_count] {
        sum += p as u64;
    }
    let mean = sum as f64 / pixel_count as f64;

    let mut var_sum = 0.0f64;
    for &p in &data[..pixel_count] {
        let diff = p as f64 - mean;
        var_sum += diff * diff;
    }
    let variance = var_sum / pixel_count as f64;

    (variance / 4000.0).min(1.0) as f32
}

/// Calculate temporal complexity between two consecutive grayscale frames.
///
/// Uses Mean Absolute Difference (MAD) between pixel values as a motion proxy.
/// Returns a value in approximately [0.0, 1.0].
pub fn temporal_complexity(
    current: &[u8],
    cur_w: usize,
    cur_h: usize,
    next: &[u8],
    next_w: usize,
    next_h: usize,
) -> f32 {
    // Frames must be the same size for comparison
    if cur_w != next_w || cur_h != next_h {
        return 1.0; // Assume maximum difference for mismatched frames
    }

    let pixel_count = cur_w * cur_h;
    if pixel_count == 0 {
        return 0.0;
    }

    let mut sad: u64 = 0;
    for i in 0..pixel_count {
        let diff = (current[i] as i32 - next[i] as i32).unsigned_abs();
        sad += diff as u64;
    }

    let mad = sad as f64 / pixel_count as f64;
    // MAD of 0 = no motion, MAD of ~30+ = heavy motion
    (mad / 30.0).min(1.0) as f32
}

/// Compute a normalized histogram of grayscale pixel intensities.
fn compute_histogram(data: &[u8], width: usize, height: usize) -> [f64; HISTOGRAM_BINS] {
    let pixel_count = width * height;
    let bin_width = 256 / HISTOGRAM_BINS;
    let mut bins = [0u64; HISTOGRAM_BINS];

    for &pixel in &data[..pixel_count] {
        let bin = ((pixel as usize) / bin_width).min(HISTOGRAM_BINS - 1);
        bins[bin] += 1;
    }

    let total = pixel_count as f64;
    let mut normalized = [0.0f64; HISTOGRAM_BINS];
    for i in 0..HISTOGRAM_BINS {
        normalized[i] = bins[i] as f64 / total;
    }
    normalized
}

/// Chi-square distance between two normalized histograms.
fn histogram_chi_square(h1: &[f64; HISTOGRAM_BINS], h2: &[f64; HISTOGRAM_BINS]) -> f64 {
    let mut chi2 = 0.0;
    for i in 0..HISTOGRAM_BINS {
        let sum = h1[i] + h2[i];
        if sum > 1e-10 {
            let diff = h1[i] - h2[i];
            chi2 += (diff * diff) / sum;
        }
    }
    chi2 / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_complexity_flat() {
        let data = vec![128u8; 64 * 64];
        let c = spatial_complexity(&data, 64, 64);
        assert!(c < 0.01, "Flat image should be low complexity, got {}", c);
    }

    #[test]
    fn test_spatial_complexity_noisy() {
        let data: Vec<u8> = (0..64 * 64).map(|i| if i % 2 == 0 { 0 } else { 255 }).collect();
        let c = spatial_complexity(&data, 64, 64);
        assert!(c > 0.5, "Noisy image should be high complexity, got {}", c);
    }

    #[test]
    fn test_temporal_complexity_static() {
        let frame = vec![100u8; 64 * 64];
        let t = temporal_complexity(&frame, 64, 64, &frame, 64, 64);
        assert!(t < 0.01, "Identical frames should have zero temporal complexity, got {}", t);
    }

    #[test]
    fn test_temporal_complexity_motion() {
        let frame1 = vec![50u8; 64 * 64];
        let frame2 = vec![200u8; 64 * 64];
        let t = temporal_complexity(&frame1, 64, 64, &frame2, 64, 64);
        assert!(t > 0.9, "Very different frames should have high temporal complexity, got {}", t);
    }

    #[test]
    fn test_scene_change_same() {
        let frame = vec![128u8; 64 * 64];
        assert!(!detect_scene_change(&frame, 64, 64, &frame));
    }

    #[test]
    fn test_scene_change_different() {
        let frame1 = vec![0u8; 64 * 64];
        let frame2 = vec![255u8; 64 * 64];
        assert!(detect_scene_change(&frame1, 64, 64, &frame2));
    }

    #[test]
    fn test_analyze_frame_single() {
        let gf = GrayFrame {
            data: vec![128u8; 64 * 64],
            width: 64,
            height: 64,
            frame_number: 0,
        };
        let mut buf = VecDeque::new();
        buf.push_back(gf);
        let analysis = analyze_frame(&buf);
        assert!(!analysis.is_scene_change);
        assert!(analysis.temporal < 0.01);
    }

    #[test]
    fn test_analyze_frame_with_lookahead() {
        let gf1 = GrayFrame {
            data: vec![50u8; 64 * 64],
            width: 64,
            height: 64,
            frame_number: 0,
        };
        let gf2 = GrayFrame {
            data: vec![50u8; 64 * 64],
            width: 64,
            height: 64,
            frame_number: 1,
        };
        let mut buf = VecDeque::new();
        buf.push_back(gf1);
        buf.push_back(gf2);
        let analysis = analyze_frame(&buf);
        assert!(!analysis.is_scene_change);
        assert!(analysis.temporal < 0.01);
    }
}
