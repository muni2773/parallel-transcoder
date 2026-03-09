//! SRT data plane helper.
//!
//! Uses FFmpeg's native SRT support to serve and receive segment files.
//! No custom streaming code -- FFmpeg handles all SRT protocol details.

use anyhow::{Context, Result};
use std::path::Path;
use tokio::process::Command;
use tracing::{debug, info};

/// SRT data plane manager.
///
/// Uses FFmpeg's native SRT support to serve and receive segment files.
/// No custom streaming code -- FFmpeg handles all SRT protocol details.
pub struct SrtServer {
    /// Base port for SRT listeners (each segment gets its own port).
    base_port: u16,
    next_port: std::sync::atomic::AtomicU16,
}

impl SrtServer {
    pub fn new(base_port: u16) -> Self {
        Self {
            base_port,
            next_port: std::sync::atomic::AtomicU16::new(base_port),
        }
    }

    /// Allocate a port for a new SRT stream.
    pub fn allocate_port(&self) -> u16 {
        self.next_port
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }

    /// Get the base port.
    pub fn base_port(&self) -> u16 {
        self.base_port
    }

    /// Serve a file via SRT using ffmpeg.
    ///
    /// Uses ffmpeg to stream-copy the file to an SRT listener.
    /// The caller should spawn this as a background task.
    /// Returns when the transfer is complete or fails.
    pub async fn serve_file(
        host: &str,
        port: u16,
        file_path: &Path,
        timeout_secs: u32,
    ) -> Result<()> {
        let srt_url = format!(
            "srt://{}:{}?mode=listener&latency=200000&timeout={}000000",
            host, port, timeout_secs
        );

        info!("Serving {} via SRT on port {}", file_path.display(), port);

        let output = Command::new("ffmpeg")
            .args([
                "-y",
                "-i",
                file_path.to_str().unwrap(),
                "-c",
                "copy",
                "-f",
                "mpegts",
                &srt_url,
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .output()
            .await
            .context("Failed to spawn ffmpeg for SRT serving")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!(
                "FFmpeg SRT serve failed: {}",
                stderr.lines().last().unwrap_or("unknown error")
            );
        }

        debug!("SRT serve complete for port {}", port);
        Ok(())
    }

    /// Fetch a file from an SRT source using ffmpeg.
    pub async fn fetch_file(srt_url: &str, output_path: &Path) -> Result<u64> {
        info!("Fetching from {} to {}", srt_url, output_path.display());

        let output = Command::new("ffmpeg")
            .args([
                "-y",
                "-i",
                srt_url,
                "-c",
                "copy",
                "-f",
                "mpegts",
                output_path.to_str().unwrap(),
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .output()
            .await
            .context("Failed to spawn ffmpeg for SRT fetch")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!(
                "FFmpeg SRT fetch failed: {}",
                stderr.lines().last().unwrap_or("unknown error")
            );
        }

        let size = tokio::fs::metadata(output_path)
            .await
            .map(|m| m.len())
            .unwrap_or(0);

        debug!("SRT fetch complete: {} bytes", size);
        Ok(size)
    }

    /// Build an SRT URL for connecting to a remote host.
    pub fn build_url(host: &str, port: u16, mode: SrtMode) -> String {
        let mode_str = match mode {
            SrtMode::Caller => "caller",
            SrtMode::Listener => "listener",
        };
        format!(
            "srt://{}:{}?mode={}&latency=200000",
            host, port, mode_str
        )
    }
}

/// SRT connection mode.
#[derive(Debug, Clone, Copy)]
pub enum SrtMode {
    Caller,
    Listener,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_port_increments() {
        let server = SrtServer::new(10000);
        assert_eq!(server.allocate_port(), 10000);
        assert_eq!(server.allocate_port(), 10001);
        assert_eq!(server.allocate_port(), 10002);
    }

    #[test]
    fn test_build_url_caller() {
        let url = SrtServer::build_url("192.168.1.5", 9100, SrtMode::Caller);
        assert_eq!(url, "srt://192.168.1.5:9100?mode=caller&latency=200000");
    }

    #[test]
    fn test_build_url_listener() {
        let url = SrtServer::build_url("0.0.0.0", 9200, SrtMode::Listener);
        assert_eq!(url, "srt://0.0.0.0:9200?mode=listener&latency=200000");
    }

    #[test]
    fn test_base_port() {
        let server = SrtServer::new(5000);
        assert_eq!(server.base_port(), 5000);
    }
}
