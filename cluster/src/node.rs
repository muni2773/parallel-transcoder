//! Node management and health monitoring.
//!
//! Tracks all nodes in the cluster, their capabilities, and heartbeat status.
//! Detects dead nodes that have missed too many heartbeats.

use std::time::{Duration, Instant};

use dashmap::DashMap;
use tracing::{debug, info, warn};

use crate::protocol::{NodeCapabilities, NodeId, NodeInfo, NodeLoad, NodeStatus};

/// Default heartbeat interval (2 seconds).
const DEFAULT_HEARTBEAT_INTERVAL_SECS: u64 = 2;

/// Default dead timeout (10 seconds = 5 missed heartbeats).
const DEFAULT_DEAD_TIMEOUT_SECS: u64 = 10;

/// Internal tracking data for a node, including the last heartbeat timestamp.
#[derive(Debug, Clone)]
struct TrackedNode {
    info: NodeInfo,
    last_heartbeat: Instant,
}

/// Manages the set of nodes in the cluster.
///
/// Thread-safe via `DashMap` for concurrent access from multiple async tasks.
pub struct NodeManager {
    nodes: DashMap<NodeId, TrackedNode>,
    self_id: NodeId,
    /// How often heartbeats should be sent.
    heartbeat_interval: Duration,
    /// How long before a node is considered dead.
    dead_timeout: Duration,
}

impl std::fmt::Debug for NodeManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeManager")
            .field("self_id", &self.self_id)
            .field("node_count", &self.nodes.len())
            .field("heartbeat_interval", &self.heartbeat_interval)
            .field("dead_timeout", &self.dead_timeout)
            .finish()
    }
}

impl NodeManager {
    /// Create a new node manager for the given local node ID.
    pub fn new(self_id: NodeId) -> Self {
        info!(%self_id, "Node manager initialized");
        Self {
            nodes: DashMap::new(),
            self_id,
            heartbeat_interval: Duration::from_secs(DEFAULT_HEARTBEAT_INTERVAL_SECS),
            dead_timeout: Duration::from_secs(DEFAULT_DEAD_TIMEOUT_SECS),
        }
    }

    /// Builder-style method to set custom timing parameters.
    pub fn with_timeouts(mut self, heartbeat: Duration, dead: Duration) -> Self {
        self.heartbeat_interval = heartbeat;
        self.dead_timeout = dead;
        self
    }

    /// Get this node's ID.
    pub fn self_id(&self) -> NodeId {
        self.self_id
    }

    /// Get the heartbeat interval.
    pub fn heartbeat_interval(&self) -> Duration {
        self.heartbeat_interval
    }

    /// Register a new node in the cluster.
    pub fn register_node(&self, info: NodeInfo) {
        info!(
            node_id = %info.node_id,
            address = %info.address,
            hostname = %info.capabilities.hostname,
            cpu_cores = info.capabilities.cpu_cores,
            "Registering node"
        );
        let id = info.node_id;
        self.nodes.insert(
            id,
            TrackedNode {
                info,
                last_heartbeat: Instant::now(),
            },
        );
    }

    /// Remove a node from the cluster. Returns the node info if it existed.
    pub fn remove_node(&self, id: &NodeId) -> Option<NodeInfo> {
        self.nodes.remove(id).map(|(_, tracked)| {
            info!(%id, "Removed node from cluster");
            tracked.info
        })
    }

    /// Update a node's heartbeat timestamp and load information.
    pub fn update_heartbeat(&self, id: &NodeId, load: NodeLoad) {
        if let Some(mut tracked) = self.nodes.get_mut(id) {
            tracked.last_heartbeat = Instant::now();
            tracked.info.load = load;
            tracked.info.last_heartbeat_ms = 0;

            // Update status based on load
            if tracked.info.load.active_workers >= tracked.info.capabilities.max_concurrent_workers
            {
                tracked.info.status = NodeStatus::Busy;
            } else if tracked.info.status == NodeStatus::Busy
                || tracked.info.status == NodeStatus::Dead
            {
                tracked.info.status = NodeStatus::Active;
            }

            debug!(%id, active_workers = tracked.info.load.active_workers, "Heartbeat updated");
        } else {
            warn!(%id, "Heartbeat from unknown node");
        }
    }

    /// Check for nodes that have missed their heartbeat deadline.
    ///
    /// Returns the IDs of nodes that should be considered dead.
    /// Also updates their status to `Dead`. Skips self.
    pub fn check_dead_nodes(&self) -> Vec<NodeId> {
        let mut dead = Vec::new();
        let now = Instant::now();

        for mut entry in self.nodes.iter_mut() {
            // Skip self — we don't track our own heartbeats
            if entry.key() == &self.self_id {
                continue;
            }

            let elapsed = now.duration_since(entry.last_heartbeat);
            entry.info.last_heartbeat_ms = elapsed.as_millis() as u64;

            if elapsed > self.dead_timeout && entry.info.status != NodeStatus::Dead {
                warn!(
                    node_id = %entry.info.node_id,
                    elapsed_ms = elapsed.as_millis(),
                    "Node declared dead (missed heartbeats)"
                );
                entry.info.status = NodeStatus::Dead;
                dead.push(entry.info.node_id);
            }
        }

        dead
    }

    /// Get information about a specific node.
    pub fn get_node(&self, id: &NodeId) -> Option<NodeInfo> {
        self.nodes.get(id).map(|tracked| {
            let mut info = tracked.info.clone();
            info.last_heartbeat_ms = tracked.last_heartbeat.elapsed().as_millis() as u64;
            info
        })
    }

    /// Get information about all nodes in the cluster.
    pub fn all_nodes(&self) -> Vec<NodeInfo> {
        self.nodes
            .iter()
            .map(|entry| {
                let mut info = entry.info.clone();
                info.last_heartbeat_ms = entry.last_heartbeat.elapsed().as_millis() as u64;
                info
            })
            .collect()
    }

    /// Get only active or busy nodes (excludes Dead, Joining, Draining).
    pub fn active_nodes(&self) -> Vec<NodeInfo> {
        self.nodes
            .iter()
            .filter(|entry| {
                matches!(
                    entry.info.status,
                    NodeStatus::Active | NodeStatus::Busy
                )
            })
            .map(|entry| {
                let mut info = entry.info.clone();
                info.last_heartbeat_ms = entry.last_heartbeat.elapsed().as_millis() as u64;
                info
            })
            .collect()
    }

    /// Get the number of available worker slots across all active nodes.
    pub fn total_available_slots(&self) -> usize {
        self.active_nodes()
            .iter()
            .map(|n| {
                n.capabilities
                    .max_concurrent_workers
                    .saturating_sub(n.load.active_workers)
            })
            .sum()
    }

    /// Get the number of tracked nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Mark a node as draining (finishing work, not accepting new segments).
    pub fn set_draining(&self, id: &NodeId) {
        if let Some(mut entry) = self.nodes.get_mut(id) {
            info!(%id, "Node set to draining");
            entry.info.status = NodeStatus::Draining;
        }
    }

    /// Update the local node's active worker count.
    pub fn update_local_workers(&self, active_workers: usize) {
        if let Some(mut entry) = self.nodes.get_mut(&self.self_id) {
            entry.info.load.active_workers = active_workers;
            entry.info.status =
                if active_workers >= entry.info.capabilities.max_concurrent_workers {
                    NodeStatus::Busy
                } else {
                    NodeStatus::Active
                };
        }
    }

    /// Increment the local node's completed segment count.
    pub fn record_segment_complete(&self) {
        if let Some(mut entry) = self.nodes.get_mut(&self.self_id) {
            entry.info.load.segments_completed += 1;
        }
    }

    /// Increment the local node's failed segment count.
    pub fn record_segment_failed(&self) {
        if let Some(mut entry) = self.nodes.get_mut(&self.self_id) {
            entry.info.load.segments_failed += 1;
        }
    }

    /// Get current load of the local machine.
    pub fn local_load(&self) -> NodeLoad {
        self.nodes
            .get(&self.self_id)
            .map(|entry| entry.info.load.clone())
            .unwrap_or_default()
    }

    /// Detect local machine capabilities.
    ///
    /// Queries the system for CPU count, available memory, and GPU availability.
    pub fn detect_capabilities() -> NodeCapabilities {
        let cpu_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let available_memory_mb = detect_memory_mb();
        let (has_gpu, gpu_encoder) = detect_gpu();
        let hostname = detect_hostname();
        let os = std::env::consts::OS.to_string();

        // Use half of CPU cores as default max workers to leave headroom
        let max_concurrent_workers = std::cmp::max(1, cpu_cores / 2);

        NodeCapabilities {
            cpu_cores,
            available_memory_mb,
            has_gpu,
            gpu_encoder,
            hostname,
            os,
            max_concurrent_workers,
        }
    }
}

/// Detect the system hostname.
fn detect_hostname() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Detect available system memory in megabytes.
fn detect_memory_mb() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output();
        if let Ok(out) = output {
            if let Ok(s) = String::from_utf8(out.stdout) {
                if let Ok(bytes) = s.trim().parse::<u64>() {
                    return bytes / (1024 * 1024);
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
            for line in contents.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if let Some(kb_str) = parts.get(1) {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return kb / 1024;
                        }
                    }
                }
            }
        }
    }

    // Fallback
    8192
}

/// Detect GPU encoder availability.
fn detect_gpu() -> (bool, Option<String>) {
    #[cfg(target_os = "macos")]
    {
        // macOS always has VideoToolbox on Apple Silicon / modern Intel
        return (true, Some("h264_videotoolbox".to_string()));
    }

    #[cfg(target_os = "linux")]
    {
        // Check for NVIDIA GPU via device node
        if std::path::Path::new("/dev/nvidia0").exists() {
            return (true, Some("h264_nvenc".to_string()));
        }
        // Check for VAAPI (Intel/AMD) via render node
        if std::path::Path::new("/dev/dri/renderD128").exists() {
            // Verify vainfo is available to confirm VAAPI support
            if let Ok(output) = std::process::Command::new("vainfo").output() {
                if output.status.success() {
                    return (true, Some("h264_vaapi".to_string()));
                }
            }
            // Even without vainfo, /dev/dri/renderD128 suggests GPU is present
            return (true, Some("h264_vaapi".to_string()));
        }
    }

    #[allow(unreachable_code)]
    {
        (false, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn make_node_info(id: NodeId, cores: usize, max_workers: usize) -> NodeInfo {
        NodeInfo {
            node_id: id,
            address: format!("127.0.0.1:900{}", cores),
            capabilities: NodeCapabilities {
                cpu_cores: cores,
                available_memory_mb: 16384,
                has_gpu: false,
                gpu_encoder: None,
                hostname: format!("test-{}", cores),
                os: "test".into(),
                max_concurrent_workers: max_workers,
            },
            load: NodeLoad::default(),
            is_master: false,
            last_heartbeat_ms: 0,
            status: NodeStatus::Active,
        }
    }

    #[test]
    fn test_register_and_get_node() {
        let self_id = Uuid::new_v4();
        let mgr = NodeManager::new(self_id);
        let node_id = Uuid::new_v4();
        let info = make_node_info(node_id, 8, 4);

        mgr.register_node(info);

        let retrieved = mgr.get_node(&node_id).unwrap();
        assert_eq!(retrieved.node_id, node_id);
        assert_eq!(retrieved.capabilities.cpu_cores, 8);
        assert_eq!(mgr.node_count(), 1);
    }

    #[test]
    fn test_remove_node() {
        let mgr = NodeManager::new(Uuid::new_v4());
        let node_id = Uuid::new_v4();
        mgr.register_node(make_node_info(node_id, 4, 2));
        assert_eq!(mgr.node_count(), 1);

        let removed = mgr.remove_node(&node_id);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().node_id, node_id);
        assert_eq!(mgr.node_count(), 0);
        assert!(mgr.get_node(&node_id).is_none());
    }

    #[test]
    fn test_remove_nonexistent_node() {
        let mgr = NodeManager::new(Uuid::new_v4());
        let result = mgr.remove_node(&Uuid::new_v4());
        assert!(result.is_none());
    }

    #[test]
    fn test_update_heartbeat() {
        let mgr = NodeManager::new(Uuid::new_v4());
        let node_id = Uuid::new_v4();
        mgr.register_node(make_node_info(node_id, 4, 2));

        let load = NodeLoad {
            active_workers: 1,
            cpu_usage_percent: 50.0,
            memory_used_mb: 4096,
            segments_completed: 5,
            segments_failed: 0,
        };
        mgr.update_heartbeat(&node_id, load);

        let info = mgr.get_node(&node_id).unwrap();
        assert_eq!(info.load.active_workers, 1);
        assert_eq!(info.load.segments_completed, 5);
    }

    #[test]
    fn test_busy_status_when_at_capacity() {
        let mgr = NodeManager::new(Uuid::new_v4());
        let node_id = Uuid::new_v4();
        mgr.register_node(make_node_info(node_id, 4, 2));

        let load = NodeLoad {
            active_workers: 2,
            cpu_usage_percent: 90.0,
            memory_used_mb: 8000,
            segments_completed: 0,
            segments_failed: 0,
        };
        mgr.update_heartbeat(&node_id, load);

        let info = mgr.get_node(&node_id).unwrap();
        assert_eq!(info.status, NodeStatus::Busy);
    }

    #[test]
    fn test_check_dead_nodes() {
        let self_id = Uuid::new_v4();
        let mgr = NodeManager::new(self_id).with_timeouts(
            Duration::from_millis(100),
            Duration::from_millis(0), // immediate dead detection
        );
        let node_id = Uuid::new_v4();
        mgr.register_node(make_node_info(node_id, 4, 2));

        // Node should be dead immediately (0ms timeout)
        let dead = mgr.check_dead_nodes();
        assert_eq!(dead.len(), 1);
        assert_eq!(dead[0], node_id);

        // Check status was updated
        let info = mgr.get_node(&node_id).unwrap();
        assert_eq!(info.status, NodeStatus::Dead);
    }

    #[test]
    fn test_check_dead_skips_self() {
        let self_id = Uuid::new_v4();
        let mgr = NodeManager::new(self_id).with_timeouts(
            Duration::from_millis(100),
            Duration::from_millis(0),
        );
        // Register self
        mgr.register_node(make_node_info(self_id, 4, 2));

        // Self should not be marked dead
        let dead = mgr.check_dead_nodes();
        assert!(dead.is_empty());
    }

    #[test]
    fn test_active_nodes_excludes_dead() {
        let self_id = Uuid::new_v4();
        let mgr = NodeManager::new(self_id).with_timeouts(
            Duration::from_millis(100),
            Duration::from_millis(0),
        );
        let alive_id = Uuid::new_v4();
        let dead_id = Uuid::new_v4();

        mgr.register_node(make_node_info(alive_id, 4, 2));
        mgr.register_node(make_node_info(dead_id, 4, 2));

        // Mark as dead
        mgr.check_dead_nodes();

        // Update heartbeat for the alive node to revive it
        mgr.update_heartbeat(&alive_id, NodeLoad::default());
        let info = mgr.get_node(&alive_id).unwrap();
        assert_eq!(info.status, NodeStatus::Active);
    }

    #[test]
    fn test_set_draining() {
        let mgr = NodeManager::new(Uuid::new_v4());
        let node_id = Uuid::new_v4();
        mgr.register_node(make_node_info(node_id, 4, 2));

        mgr.set_draining(&node_id);
        let info = mgr.get_node(&node_id).unwrap();
        assert_eq!(info.status, NodeStatus::Draining);
    }

    #[test]
    fn test_all_nodes() {
        let mgr = NodeManager::new(Uuid::new_v4());
        mgr.register_node(make_node_info(Uuid::new_v4(), 4, 2));
        mgr.register_node(make_node_info(Uuid::new_v4(), 8, 4));

        let all = mgr.all_nodes();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_total_available_slots() {
        let mgr = NodeManager::new(Uuid::new_v4());
        let n1 = Uuid::new_v4();
        let n2 = Uuid::new_v4();

        mgr.register_node(make_node_info(n1, 8, 4)); // 4 slots, 0 active
        mgr.register_node(make_node_info(n2, 4, 2)); // 2 slots, 0 active

        assert_eq!(mgr.total_available_slots(), 6);

        // Fill some slots
        mgr.update_heartbeat(&n1, NodeLoad {
            active_workers: 3,
            ..Default::default()
        });
        assert_eq!(mgr.total_available_slots(), 3); // 1 + 2
    }

    #[test]
    fn test_update_local_workers() {
        let self_id = Uuid::new_v4();
        let mgr = NodeManager::new(self_id);
        mgr.register_node(make_node_info(self_id, 4, 2));

        mgr.update_local_workers(1);
        let info = mgr.get_node(&self_id).unwrap();
        assert_eq!(info.load.active_workers, 1);
        assert_eq!(info.status, NodeStatus::Active);

        mgr.update_local_workers(2);
        let info = mgr.get_node(&self_id).unwrap();
        assert_eq!(info.load.active_workers, 2);
        assert_eq!(info.status, NodeStatus::Busy);
    }

    #[test]
    fn test_record_segment_complete_and_failed() {
        let self_id = Uuid::new_v4();
        let mgr = NodeManager::new(self_id);
        mgr.register_node(make_node_info(self_id, 4, 2));

        mgr.record_segment_complete();
        mgr.record_segment_complete();
        mgr.record_segment_failed();

        let load = mgr.local_load();
        assert_eq!(load.segments_completed, 2);
        assert_eq!(load.segments_failed, 1);
    }

    #[test]
    fn test_local_load_default_when_not_registered() {
        let mgr = NodeManager::new(Uuid::new_v4());
        let load = mgr.local_load();
        assert_eq!(load.active_workers, 0);
        assert_eq!(load.segments_completed, 0);
    }

    #[test]
    fn test_detect_capabilities() {
        let caps = NodeManager::detect_capabilities();
        assert!(caps.cpu_cores >= 1);
        assert!(caps.available_memory_mb > 0);
        assert!(!caps.hostname.is_empty());
        assert!(!caps.os.is_empty());
        assert!(caps.max_concurrent_workers >= 1);
    }
}
