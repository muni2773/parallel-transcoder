//! `transcoder-node` — Distributed cluster daemon for parallel video transcoding.
//!
//! Each machine in the cluster runs this binary. It participates in leader election
//! (Bully algorithm), and either acts as master (accepting jobs, distributing segments)
//! or worker (transcoding assigned segments locally).
//!
//! # Usage
//!
//! ```bash
//! # Bootstrap a new cluster (first node becomes master):
//! transcoder-node --listen 0.0.0.0:9900
//!
//! # Join an existing cluster as a worker:
//! transcoder-node --listen 0.0.0.0:9901 --join 192.168.1.10:9900
//! ```

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use tokio::signal;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};
use uuid::Uuid;

// Import from the crate library.
// NOTE: lib.rs must export these modules. If election/node/scheduler are still
// commented out in lib.rs, uncomment them before building.
use transcoder_cluster::election::ElectionManager;
use transcoder_cluster::node::NodeManager;
use transcoder_cluster::protocol::*;
use transcoder_cluster::scheduler::Scheduler;
use transcoder_cluster::srt::{SrtMode, SrtServer};
use transcoder_cluster::transport::{PeerMessage, Transport};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// Distributed video transcoding cluster node.
///
/// Runs on each machine in the cluster. Participates in leader election,
/// accepts jobs (as master), or transcodes segments (as worker).
#[derive(Parser, Debug)]
#[command(name = "transcoder-node", version, about)]
struct Cli {
    /// Listen address for cluster communication.
    #[arg(short, long, default_value = "0.0.0.0:9900")]
    listen: String,

    /// Join an existing cluster at this address.
    #[arg(short, long)]
    join: Option<String>,

    /// Human-readable node name (defaults to hostname).
    #[arg(short, long)]
    name: Option<String>,

    /// Base port for SRT data streams.
    #[arg(long, default_value_t = 9910)]
    srt_base_port: u16,

    /// Path to the transcoder-worker binary.
    #[arg(long, default_value = "./bin/transcoder-worker")]
    worker_binary: PathBuf,

    /// Path to FFmpeg shared libraries.
    #[arg(long, default_value = "./lib/")]
    lib_dir: PathBuf,

    /// Enable verbose (debug-level) logging.
    #[arg(short, long)]
    verbose: bool,
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CLUSTER_NAME: &str = "transcoder-cluster";
const PROTOCOL_VERSION: u32 = 1;

const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);
const ELECTION_CHECK_INTERVAL: Duration = Duration::from_millis(500);
const HEALTH_CHECK_INTERVAL: Duration = Duration::from_secs(5);
const SCHEDULE_INTERVAL: Duration = Duration::from_secs(1);

const ELECTION_TIMEOUT: Duration = Duration::from_secs(3);
const VICTORY_TIMEOUT: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

/// Per-job metadata tracked by the master.
#[allow(dead_code)]
struct JobContext {
    config: EncodingConfig,
    input_filename: String,
    segments: Vec<SegmentDescriptor>,
}

/// Top-level application state shared across the event loop.
struct App {
    node_id: NodeId,
    listen_addr: SocketAddr,
    capabilities: NodeCapabilities,
    transport: Transport,
    election: ElectionManager,
    nodes: NodeManager,
    scheduler: Scheduler,
    srt: SrtServer,
    /// Segment descriptors kept by the master for reassignment.
    job_contexts: HashMap<JobId, JobContext>,
    /// Active local worker child processes (segment_id → cancel sender).
    local_workers: HashMap<usize, mpsc::Sender<()>>,
    /// Pending worker results waiting to be relayed to the master.
    pending_results: HashMap<usize, (mpsc::UnboundedReceiver<ResultMessage>, SocketAddr)>,
    /// CLI configuration.
    worker_binary: PathBuf,
    lib_dir: PathBuf,
}

impl App {
    fn new(
        node_id: NodeId,
        listen_addr: SocketAddr,
        capabilities: NodeCapabilities,
        transport: Transport,
        worker_binary: PathBuf,
        lib_dir: PathBuf,
        srt_base_port: u16,
    ) -> Self {
        Self {
            node_id,
            listen_addr,
            capabilities,
            transport,
            election: ElectionManager::new(node_id, ELECTION_TIMEOUT, VICTORY_TIMEOUT),
            nodes: NodeManager::new(node_id),
            scheduler: Scheduler::new(),
            srt: SrtServer::new(srt_base_port),
            job_contexts: HashMap::new(),
            local_workers: HashMap::new(),
            pending_results: HashMap::new(),
            worker_binary,
            lib_dir,
        }
    }

    /// Returns true if this node is the cluster master.
    fn is_master(&self) -> bool {
        self.election.is_leader()
    }

    // --------------------------------------------------------------------
    // Self-registration: register ourselves in the node table.
    // --------------------------------------------------------------------

    fn register_self(&self) {
        let info = NodeInfo {
            node_id: self.node_id,
            address: self.listen_addr.to_string(),
            capabilities: self.capabilities.clone(),
            load: NodeLoad::default(),
            is_master: self.is_master(),
            last_heartbeat_ms: 0,
            status: NodeStatus::Active,
        };
        self.nodes.register_node(info);
    }

    // ====================================================================
    // Message dispatch
    // ====================================================================

    async fn handle_message(&mut self, peer: PeerMessage) {
        let PeerMessage {
            peer_addr,
            message,
            ..
        } = peer;

        match message.op {
            // -- Handshake --
            OpCode::Hello => self.handle_hello(peer_addr),
            OpCode::Identify => self.handle_identify(peer_addr, &message),
            OpCode::Identified => self.handle_identified(&message),

            // -- Election --
            OpCode::ElectionStart => self.handle_election_start(peer_addr, &message),
            OpCode::ElectionAlive => self.handle_election_alive(&message),
            OpCode::ElectionVictory => self.handle_election_victory(&message),

            // -- Heartbeat --
            OpCode::Heartbeat => self.handle_heartbeat(&message),
            OpCode::HeartbeatAck => { /* acknowledged */ }

            // -- Jobs --
            OpCode::JobSubmit => self.handle_job_submit(peer_addr, &message).await,

            // -- Segments --
            OpCode::SegmentAssign => self.handle_segment_assign(peer_addr, &message).await,
            OpCode::SegmentComplete => self.handle_segment_complete(&message),
            OpCode::SegmentFailed => self.handle_segment_failed(&message),

            // -- Status --
            OpCode::StatusRequest => self.handle_status_request(peer_addr),

            // -- Lifecycle --
            OpCode::NodeLeave => self.handle_node_leave(&message),

            _ => {
                debug!(op = ?message.op, "Unhandled opcode");
            }
        }
    }

    // ====================================================================
    // Handshake
    // ====================================================================

    /// Respond to a Hello from a connecting peer by sending our Identify.
    fn handle_hello(&self, peer_addr: SocketAddr) {
        info!(%peer_addr, "Received Hello, sending Identify");
        let identify = IdentifyData {
            node_id: self.node_id,
            listen_addr: self.listen_addr.to_string(),
            capabilities: self.capabilities.clone(),
            event_subscriptions: event_subs::ALL,
        };
        if let Ok(msg) = Message::new(OpCode::Identify, &identify) {
            let _ = self.transport.send_to(&peer_addr, msg);
        }
    }

    /// Process a peer's Identify: register them and send Identified back.
    fn handle_identify(&mut self, peer_addr: SocketAddr, msg: &Message) {
        let Ok(data) = msg.parse_data::<IdentifyData>() else {
            warn!("Failed to parse Identify payload");
            return;
        };
        info!(
            node_id = %data.node_id,
            addr = %data.listen_addr,
            hostname = %data.capabilities.hostname,
            cores = data.capabilities.cpu_cores,
            "Peer identified"
        );

        // Associate the node_id with this transport address.
        self.transport.set_peer_node_id(&peer_addr, data.node_id);

        // Register the peer in the node table.
        let info = NodeInfo {
            node_id: data.node_id,
            address: data.listen_addr,
            capabilities: data.capabilities,
            load: NodeLoad::default(),
            is_master: false,
            last_heartbeat_ms: 0,
            status: NodeStatus::Active,
        };
        self.nodes.register_node(info);

        // Send our cluster state back.
        let identified = IdentifiedData {
            node_id: self.node_id,
            master_id: self.election.current_leader(),
            cluster_nodes: self.nodes.all_nodes(),
        };
        if let Ok(resp) = Message::new(OpCode::Identified, &identified) {
            let _ = self.transport.send_to(&peer_addr, resp);
        }
    }

    /// Process an Identified response: learn about the cluster.
    fn handle_identified(&mut self, msg: &Message) {
        let Ok(data) = msg.parse_data::<IdentifiedData>() else {
            warn!("Failed to parse Identified payload");
            return;
        };
        info!(
            peer = %data.node_id,
            master = ?data.master_id,
            cluster_size = data.cluster_nodes.len(),
            "Handshake complete"
        );

        // Register all advertised cluster nodes.
        for node in data.cluster_nodes {
            if node.node_id != self.node_id {
                self.nodes.register_node(node);
            }
        }

        // Accept their master if we don't have one yet.
        if let Some(master_id) = data.master_id {
            if self.election.current_leader().is_none() {
                let victory = ElectionVictoryData { master_id };
                self.election.handle_victory(&victory);
                info!(%master_id, "Accepted existing cluster master");
            }
        }
    }

    // ====================================================================
    // Election
    // ====================================================================

    fn handle_election_start(&mut self, peer_addr: SocketAddr, msg: &Message) {
        let Ok(data) = msg.parse_data::<ElectionStartData>() else {
            return;
        };

        let was_leader = self.election.is_leader();

        if let Some(alive_msg) = self.election.handle_election_start(&data) {
            // Send ElectionAlive back to the lower-priority candidate.
            let _ = self.transport.send_to(&peer_addr, alive_msg);

            // If we were already leader, just re-announce victory instead of
            // restarting the full election cycle.
            if was_leader {
                let victory = ElectionVictoryData {
                    master_id: self.node_id,
                };
                if let Ok(msg) = Message::new(OpCode::ElectionVictory, &victory) {
                    self.transport.broadcast(&msg, None);
                }
            } else {
                // We're now a Candidate (set by handle_election_start).
                // Broadcast our own ElectionStart so other nodes know.
                let start_msg = self.election.start_election();
                self.transport.broadcast(&start_msg, Some(&peer_addr));
            }
        }
    }

    fn handle_election_alive(&mut self, msg: &Message) {
        let Ok(data) = msg.parse_data::<ElectionAliveData>() else {
            return;
        };
        self.election.handle_alive(&data);
    }

    fn handle_election_victory(&mut self, msg: &Message) {
        let Ok(data) = msg.parse_data::<ElectionVictoryData>() else {
            return;
        };
        info!(master_id = %data.master_id, "New master elected");
        self.election.handle_victory(&data);
    }

    // ====================================================================
    // Heartbeat
    // ====================================================================

    fn handle_heartbeat(&self, msg: &Message) {
        let Ok(data) = msg.parse_data::<HeartbeatData>() else {
            return;
        };
        self.nodes.update_heartbeat(&data.node_id, data.load);
    }

    fn send_heartbeat(&self) {
        let hb = HeartbeatData {
            node_id: self.node_id,
            load: NodeLoad {
                active_workers: self.local_workers.len(),
                cpu_usage_percent: 0.0,
                memory_used_mb: 0,
                segments_completed: 0,
                segments_failed: 0,
            },
        };
        if let Ok(msg) = Message::new(OpCode::Heartbeat, &hb) {
            self.transport.broadcast(&msg, None);
        }
    }

    // ====================================================================
    // Job management (master only)
    // ====================================================================

    async fn handle_job_submit(&mut self, peer_addr: SocketAddr, msg: &Message) {
        if !self.is_master() {
            warn!("Received JobSubmit but not master — rejecting");
            let err = ErrorData {
                code: 403,
                message: "Not the master node".into(),
            };
            if let Ok(resp) = Message::new(OpCode::Error, &err) {
                let _ = self.transport.send_to(&peer_addr, resp);
            }
            return;
        }

        let Ok(data) = msg.parse_data::<JobSubmitData>() else {
            warn!("Failed to parse JobSubmit payload");
            return;
        };

        info!(
            job_id = %data.job_id,
            input = %data.input_filename,
            size_mb = data.input_size_bytes / (1024 * 1024),
            "Job submitted"
        );

        // Analyze the video to produce segment descriptors.
        let segments = analyze_video(&data);
        let total_segments = segments.len();

        // Store job context for later segment distribution.
        self.job_contexts.insert(
            data.job_id,
            JobContext {
                config: data.config,
                input_filename: data.input_filename,
                segments: segments.clone(),
            },
        );

        // Enqueue segments for scheduling.
        self.scheduler.add_job(data.job_id, segments);

        // Acknowledge to the submitter.
        let accepted = JobAcceptedData {
            job_id: data.job_id,
            total_segments,
        };
        if let Ok(resp) = Message::new(OpCode::JobAccepted, &accepted) {
            let _ = self.transport.send_to(&peer_addr, resp);
        }

        info!(
            job_id = %data.job_id,
            total_segments,
            "Job accepted, segments queued"
        );
    }

    // ====================================================================
    // Segment scheduling (master → workers)
    // ====================================================================

    fn run_scheduler(&mut self) {
        if !self.is_master() {
            return;
        }

        let active = self.nodes.active_nodes();
        if active.is_empty() {
            return;
        }

        let assignments = self.scheduler.schedule(&active);
        for assignment in assignments {
            let node_id = assignment.node_id;
            let job_id = assignment.job_id;
            let segment = assignment.segment;
            let ctx = match self.job_contexts.get(&job_id) {
                Some(c) => c,
                None => continue,
            };

            let srt_port = self.srt.allocate_port();
            let srt_url = SrtServer::build_url(
                &self.listen_addr.ip().to_string(),
                srt_port,
                SrtMode::Caller,
            );

            let assign_data = SegmentAssignData {
                job_id,
                segment: segment.clone(),
                srt_url,
                encoding_config: ctx.config.clone(),
            };

            if let Ok(msg) = Message::new(OpCode::SegmentAssign, &assign_data) {
                if let Some(addr) = self.transport.find_peer_addr(&node_id) {
                    if let Err(e) = self.transport.send_to(&addr, msg) {
                        warn!(
                            %node_id, segment_id = segment.id,
                            "Failed to send SegmentAssign: {}", e
                        );
                    } else {
                        debug!(
                            %job_id, segment_id = segment.id, %node_id,
                            "Segment assigned"
                        );
                    }
                } else {
                    warn!(%node_id, "No peer address, cannot assign segment");
                }
            }
        }
    }

    // ====================================================================
    // Segment processing (worker side)
    // ====================================================================

    async fn handle_segment_assign(&mut self, peer_addr: SocketAddr, msg: &Message) {
        let Ok(data) = msg.parse_data::<SegmentAssignData>() else {
            warn!("Failed to parse SegmentAssign payload");
            return;
        };

        info!(
            job_id = %data.job_id,
            segment_id = data.segment.id,
            time_range = format!("{:.1}s–{:.1}s", data.segment.start_timestamp, data.segment.end_timestamp),
            "Received segment assignment"
        );

        // Acknowledge the assignment.
        let ack = SegmentAssignAckData {
            job_id: data.job_id,
            segment_id: data.segment.id,
            node_id: self.node_id,
        };
        if let Ok(ack_msg) = Message::new(OpCode::SegmentAssignAck, &ack) {
            let _ = self.transport.send_to(&peer_addr, ack_msg);
        }

        // Track the local worker.
        let (cancel_tx, _cancel_rx) = mpsc::channel::<()>(1);
        self.local_workers.insert(data.segment.id, cancel_tx);

        // Spawn the worker process in a background task.
        let node_id = self.node_id;
        let worker_binary = self.worker_binary.clone();
        let lib_dir = self.lib_dir.clone();
        let listen_addr = self.listen_addr;
        let srt_port = self.srt.allocate_port();

        // Channel for the spawned task to send results back to the event loop.
        // We don't have access to the transport from a spawned task, so we use
        // a channel and process the result on the next iteration.
        let (result_tx, result_rx) = mpsc::unbounded_channel::<ResultMessage>();

        let job_id = data.job_id;
        let segment_id = data.segment.id;

        tokio::spawn(async move {
            let outcome = spawn_worker(
                &worker_binary,
                &lib_dir,
                &data,
                node_id,
                listen_addr,
                srt_port,
            )
            .await;

            let _ = result_tx.send(ResultMessage {
                job_id,
                segment_id,
                node_id,
                srt_port,
                outcome,
            });
        });

        // Store the result receiver so we can poll it from the event loop.
        // We can't use the transport from a spawned task, so results are
        // relayed via poll_worker_results() on each schedule tick.
        self.pending_results
            .insert(segment_id, (result_rx, peer_addr));
    }

    /// Poll pending worker results and relay them to the master.
    fn poll_worker_results(&mut self) {
        let mut completed_keys = Vec::new();

        for (&key, (rx, master_addr)) in &mut self.pending_results {
            match rx.try_recv() {
                Ok(result_msg) => {
                    self.local_workers.remove(&result_msg.segment_id);

                    match result_msg.outcome {
                        Ok(seg_result) => {
                            let srt_output_url = SrtServer::build_url(
                                &self.listen_addr.ip().to_string(),
                                result_msg.srt_port,
                                SrtMode::Listener,
                            );
                            let complete = SegmentCompleteData {
                                job_id: result_msg.job_id,
                                result: seg_result,
                                srt_output_url,
                            };
                            if let Ok(msg) = Message::new(OpCode::SegmentComplete, &complete) {
                                let _ = self.transport.send_to(master_addr, msg);
                            }
                        }
                        Err(e) => {
                            let failed = SegmentFailedData {
                                job_id: result_msg.job_id,
                                segment_id: result_msg.segment_id,
                                node_id: result_msg.node_id,
                                error: format!("{:#}", e),
                            };
                            if let Ok(msg) = Message::new(OpCode::SegmentFailed, &failed) {
                                let _ = self.transport.send_to(master_addr, msg);
                            }
                        }
                    }

                    completed_keys.push(key);
                }
                Err(mpsc::error::TryRecvError::Empty) => {}
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    completed_keys.push(key);
                }
            }
        }

        for key in completed_keys {
            self.pending_results.remove(&key);
        }
    }

    // ====================================================================
    // Segment results (master side)
    // ====================================================================

    fn handle_segment_complete(&mut self, msg: &Message) {
        if !self.is_master() {
            return;
        }

        let Ok(data) = msg.parse_data::<SegmentCompleteData>() else {
            warn!("Failed to parse SegmentComplete payload");
            return;
        };

        info!(
            job_id = %data.job_id,
            segment_id = data.result.segment_id,
            node = %data.result.node_id,
            time = format!("{:.1}s", data.result.encoding_time_secs),
            size_kb = data.result.output_size_bytes / 1024,
            "Segment completed"
        );

        let job_id = data.job_id;
        self.scheduler.mark_complete(&job_id, data.result);
        self.check_job_completion(job_id);
    }

    fn handle_segment_failed(&mut self, msg: &Message) {
        if !self.is_master() {
            return;
        }

        let Ok(data) = msg.parse_data::<SegmentFailedData>() else {
            warn!("Failed to parse SegmentFailed payload");
            return;
        };

        warn!(
            job_id = %data.job_id,
            segment_id = data.segment_id,
            node = %data.node_id,
            error = %data.error,
            "Segment failed"
        );

        let job_id = data.job_id;
        self.scheduler.mark_failed(&job_id, data.segment_id, data.error);
        self.check_job_completion(job_id);
    }

    /// Check if a job has finished and broadcast the final status.
    fn check_job_completion(&mut self, job_id: JobId) {
        let Some((completed, total, failed)) = self.scheduler.job_progress(&job_id) else {
            return;
        };

        if !self.scheduler.is_job_complete(&job_id) {
            // Broadcast progress.
            let progress = JobProgressData {
                job_id,
                completed_segments: completed,
                total_segments: total,
                failed_segments: failed,
                phase: "processing".into(),
            };
            if let Ok(msg) = Message::new(OpCode::JobProgress, &progress) {
                self.transport.broadcast(&msg, None);
            }
            return;
        }

        // Job is done.
        if self.scheduler.has_failures(&job_id) {
            warn!(%job_id, completed, total, failed, "Job finished with failures");
            let fail_data = serde_json::json!({
                "job_id": job_id,
                "error": format!("{} of {} segments failed", failed, total),
            });
            if let Ok(msg) = Message::new(OpCode::JobFailed, &fail_data) {
                self.transport.broadcast(&msg, None);
            }
        } else {
            info!(%job_id, completed, total, "Job completed successfully");
            let complete_data = serde_json::json!({
                "job_id": job_id,
                "total_segments": total,
            });
            if let Ok(msg) = Message::new(OpCode::JobComplete, &complete_data) {
                self.transport.broadcast(&msg, None);
            }
        }

        self.job_contexts.remove(&job_id);
    }

    // ====================================================================
    // Status
    // ====================================================================

    fn handle_status_request(&self, peer_addr: SocketAddr) {
        let mut job_infos = Vec::new();
        for (&job_id, _ctx) in &self.job_contexts {
            let (completed, total, failed) = self.scheduler.job_progress(&job_id).unwrap_or((0, 0, 0));
            let state = if self.scheduler.is_job_complete(&job_id) {
                if self.scheduler.has_failures(&job_id) {
                    JobState::Failed
                } else {
                    JobState::Complete
                }
            } else {
                JobState::Processing
            };
            job_infos.push(JobStatusInfo {
                job_id,
                state,
                total_segments: total,
                completed_segments: completed,
                failed_segments: failed,
                assigned_nodes: self.scheduler.job_assigned_nodes(&job_id),
            });
        }

        let status = StatusResponseData {
            master_id: self.election.current_leader(),
            nodes: self.nodes.all_nodes(),
            active_jobs: job_infos,
        };
        if let Ok(msg) = Message::new(OpCode::StatusResponse, &status) {
            let _ = self.transport.send_to(&peer_addr, msg);
        }
    }

    // ====================================================================
    // Node leave / health checks / election timeout
    // ====================================================================

    fn handle_node_leave(&mut self, msg: &Message) {
        #[derive(serde::Deserialize)]
        struct NodeLeaveData {
            node_id: NodeId,
        }

        let Ok(data) = msg.parse_data::<NodeLeaveData>() else {
            return;
        };
        info!(node_id = %data.node_id, "Node leaving cluster");
        self.nodes.remove_node(&data.node_id);

        if self.is_master() {
            let reassigned = self.scheduler.reassign_from_node(&data.node_id);
            if reassigned > 0 {
                info!(count = reassigned, "Reassigned segments from departing node");
            }
        }

        // If the departing node was master, start an election.
        if self.election.current_leader() == Some(data.node_id) {
            warn!("Master left the cluster, starting election");
            self.election.reset();
            let start_msg = self.election.start_election();
            self.transport.broadcast(&start_msg, None);
        }
    }

    fn check_health(&mut self) {
        let dead = self.nodes.check_dead_nodes();
        for dead_id in &dead {
            warn!(node_id = %dead_id, "Detected dead node");

            if self.is_master() {
                let reassigned = self.scheduler.reassign_from_node(dead_id);
                if reassigned > 0 {
                    info!(
                        node_id = %dead_id,
                        count = reassigned,
                        "Reassigned segments from dead node"
                    );
                }
            }

            if self.election.current_leader() == Some(*dead_id) {
                warn!("Dead node was master, starting election");
                self.election.reset();
                let start_msg = self.election.start_election();
                self.transport.broadcast(&start_msg, None);
            }
        }
    }

    fn check_election(&mut self) {
        if let Some(msg) = self.election.check_timeout() {
            self.transport.broadcast(&msg, None);
        }
    }

    fn send_leave(&self) {
        let leave = serde_json::json!({ "node_id": self.node_id });
        if let Ok(msg) = Message::new(OpCode::NodeLeave, &leave) {
            self.transport.broadcast(&msg, None);
        }
    }
}

// ---------------------------------------------------------------------------
// Worker result relay
// ---------------------------------------------------------------------------

/// Message sent from the spawned worker task back to the event loop.
struct ResultMessage {
    job_id: JobId,
    segment_id: usize,
    node_id: NodeId,
    srt_port: u16,
    outcome: Result<SegmentResult>,
}

// ---------------------------------------------------------------------------
// Video analysis (simplified)
// ---------------------------------------------------------------------------

/// Analyze input video and produce segment descriptors.
///
/// In a full implementation this invokes the coordinator's analyzer module.
/// For the initial version we produce a simple time-based segmentation.
fn analyze_video(data: &JobSubmitData) -> Vec<SegmentDescriptor> {
    // Estimate duration assuming ~5 MB/s average bitrate.
    let estimated_duration_secs = (data.input_size_bytes as f64) / (5.0 * 1024.0 * 1024.0);
    let segment_duration = 10.0_f64;
    let num_segments = ((estimated_duration_secs / segment_duration).ceil() as usize).max(1);

    let fps_estimate = 30.0_f64;
    let frames_per_segment = (segment_duration * fps_estimate) as u64;

    (0..num_segments)
        .map(|i| SegmentDescriptor {
            id: i,
            start_frame: i as u64 * frames_per_segment,
            end_frame: (i as u64 + 1) * frames_per_segment,
            start_timestamp: i as f64 * segment_duration,
            end_timestamp: ((i + 1) as f64 * segment_duration).min(estimated_duration_secs),
            lookahead_frames: Some(30),
            complexity_estimate: 0.5,
            scene_changes: vec![],
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Worker process spawning
// ---------------------------------------------------------------------------

/// Spawn the transcoder-worker binary for a single segment.
///
/// Returns a `SegmentResult` on success.
async fn spawn_worker(
    worker_binary: &PathBuf,
    lib_dir: &PathBuf,
    data: &SegmentAssignData,
    node_id: NodeId,
    _listen_addr: SocketAddr,
    _srt_port: u16,
) -> Result<SegmentResult> {
    use tokio::process::Command;

    let start_time = std::time::Instant::now();
    let tmp_dir = std::env::temp_dir().join(format!("transcoder-{}", data.job_id));
    tokio::fs::create_dir_all(&tmp_dir).await.ok();

    let input_path = tmp_dir.join(format!("segment_{}.ts", data.segment.id));
    let output_path = tmp_dir.join(format!("segment_{}_out.ts", data.segment.id));

    // Step 1: Fetch the segment data from the master via SRT.
    if !data.srt_url.is_empty() {
        info!(
            segment_id = data.segment.id,
            srt_url = %data.srt_url,
            "Fetching segment data via SRT"
        );
        SrtServer::fetch_file(&data.srt_url, &input_path)
            .await
            .context("Failed to fetch segment via SRT")?;
    }

    // Step 2: Spawn the worker binary.
    let lib_path_key = if cfg!(target_os = "macos") {
        "DYLD_LIBRARY_PATH"
    } else {
        "LD_LIBRARY_PATH"
    };

    let mut cmd = Command::new(worker_binary);
    cmd.env(lib_path_key, lib_dir)
        .arg("--input")
        .arg(&input_path)
        .arg("--output")
        .arg(&output_path)
        .arg("--segment-id")
        .arg(data.segment.id.to_string())
        .arg("--start-frame")
        .arg(data.segment.start_frame.to_string())
        .arg("--end-frame")
        .arg(data.segment.end_frame.to_string())
        .arg("--crf")
        .arg(data.encoding_config.crf.to_string())
        .arg("--preset")
        .arg(&data.encoding_config.preset)
        .arg("--encoder")
        .arg(&data.encoding_config.encoder)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    info!(
        segment_id = data.segment.id,
        binary = %worker_binary.display(),
        "Spawning worker process"
    );

    let output = cmd
        .output()
        .await
        .context("Failed to spawn worker process")?;

    let elapsed = start_time.elapsed();

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "Worker exited with {}: {}",
            output.status,
            stderr.trim()
        );
    }

    // Step 3: Parse the result. Try structured JSON from stdout first.
    let stdout = String::from_utf8_lossy(&output.stdout);
    if let Ok(result) = serde_json::from_str::<SegmentResult>(stdout.trim()) {
        return Ok(result);
    }

    // Fallback: construct result from available metadata.
    let output_size = tokio::fs::metadata(&output_path)
        .await
        .map(|m| m.len())
        .unwrap_or(0);

    Ok(SegmentResult {
        segment_id: data.segment.id,
        worker_id: 0,
        node_id,
        frames_encoded: data.segment.end_frame.saturating_sub(data.segment.start_frame),
        output_size_bytes: output_size,
        encoding_time_secs: elapsed.as_secs_f64(),
        average_complexity: data.segment.complexity_estimate,
        scene_changes_detected: data.segment.scene_changes.len() as u64,
    })
}

// ---------------------------------------------------------------------------
// Banner
// ---------------------------------------------------------------------------

fn print_banner(node_id: NodeId, listen_addr: &SocketAddr, role: &str, caps: &NodeCapabilities) {
    eprintln!();
    eprintln!("  +================================================+");
    eprintln!("  |       TRANSCODER NODE  v{}               |", env!("CARGO_PKG_VERSION"));
    eprintln!("  +================================================+");
    eprintln!();
    eprintln!("  Node ID   : {}", node_id);
    eprintln!("  Listen    : {}", listen_addr);
    eprintln!("  Role      : {}", role);
    eprintln!("  Hostname  : {}", caps.hostname);
    eprintln!("  CPU cores : {}", caps.cpu_cores);
    eprintln!("  Memory    : {} MB", caps.available_memory_mb);
    eprintln!(
        "  GPU       : {}",
        caps.gpu_encoder.as_deref().unwrap_or("none")
    );
    eprintln!("  Workers   : {} max", caps.max_concurrent_workers);
    eprintln!();
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize structured logging.
    let filter = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(filter)),
        )
        .with_target(false)
        .init();

    // Generate a unique node identity.
    let node_id = Uuid::new_v4();
    let listen_addr: SocketAddr = cli
        .listen
        .parse()
        .context("Invalid --listen address")?;

    // Detect local machine capabilities.
    let mut capabilities = NodeManager::detect_capabilities();
    if let Some(ref name) = cli.name {
        capabilities.hostname = name.clone();
    }

    // Determine initial role for the banner.
    let initial_role = if cli.join.is_some() {
        "worker (joining cluster)"
    } else {
        "master (bootstrap)"
    };
    print_banner(node_id, &listen_addr, initial_role, &capabilities);

    // Create the WebSocket transport layer.
    let mut transport = Transport::new(listen_addr);
    let mut incoming_rx = transport
        .take_incoming()
        .expect("incoming receiver already taken");

    // Start accepting WebSocket connections.
    transport.listen().await?;
    info!("WebSocket transport listening on {}", listen_addr);

    // Build application state.
    let mut app = App::new(
        node_id,
        listen_addr,
        capabilities,
        transport,
        cli.worker_binary,
        cli.lib_dir,
        cli.srt_base_port,
    );

    // Register ourselves in the node table.
    app.register_self();

    // --- Cluster bootstrap or join ---

    if let Some(ref join_addr) = cli.join {
        info!(addr = %join_addr, "Joining existing cluster");
        app.transport
            .connect(join_addr)
            .await
            .with_context(|| format!("Failed to connect to {}", join_addr))?;

        // Send Hello to initiate the handshake.
        let join_sock: SocketAddr = join_addr
            .parse()
            .context("Invalid --join address")?;
        let hello = HelloData {
            cluster_name: CLUSTER_NAME.into(),
            protocol_version: PROTOCOL_VERSION,
            master_id: None,
        };
        if let Ok(msg) = Message::new(OpCode::Hello, &hello) {
            let _ = app.transport.send_to(&join_sock, msg);
        }

        // Start an election to discover or contest the current master.
        let start_msg = app.election.start_election();
        app.transport.broadcast(&start_msg, None);
    } else {
        // Single-node bootstrap — we are the master.
        info!("Bootstrapping as single-node cluster master");
        app.election.force_leader();
        info!("Master ready, waiting for workers to join");
    }

    // ------------------------------------------------------------------
    // Main event loop
    // ------------------------------------------------------------------

    let mut heartbeat_tick = tokio::time::interval(HEARTBEAT_INTERVAL);
    let mut election_tick = tokio::time::interval(ELECTION_CHECK_INTERVAL);
    let mut health_tick = tokio::time::interval(HEALTH_CHECK_INTERVAL);
    let mut schedule_tick = tokio::time::interval(SCHEDULE_INTERVAL);

    // Consume the first immediate tick.
    heartbeat_tick.tick().await;
    election_tick.tick().await;
    health_tick.tick().await;
    schedule_tick.tick().await;

    info!("Entering main event loop");

    loop {
        tokio::select! {
            // Incoming WebSocket messages from peers.
            Some(peer_msg) = incoming_rx.recv() => {
                app.handle_message(peer_msg).await;
            }

            // Periodic heartbeat broadcast.
            _ = heartbeat_tick.tick() => {
                app.send_heartbeat();
            }

            // Election timeout check.
            _ = election_tick.tick() => {
                app.check_election();
            }

            // Health check: detect dead nodes.
            _ = health_tick.tick() => {
                app.check_health();
            }

            // Schedule pending segments to available workers (master only).
            _ = schedule_tick.tick() => {
                app.run_scheduler();
                app.poll_worker_results();
            }

            // Graceful shutdown on Ctrl+C.
            _ = signal::ctrl_c() => {
                info!("Received shutdown signal");
                eprintln!("\n  Shutting down gracefully...");
                app.send_leave();
                tokio::time::sleep(Duration::from_millis(250)).await;
                info!("Goodbye.");
                break;
            }
        }
    }

    Ok(())
}
