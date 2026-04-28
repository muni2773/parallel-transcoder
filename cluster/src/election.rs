//! Leader election using the Bully Algorithm.
//!
//! Each node has a priority derived from its UUID. When the master is detected
//! as down, a node starts an election by sending `ElectionStart` to all nodes
//! with higher priority. If no higher-priority node responds within the timeout,
//! the candidate declares itself master with `ElectionVictory`.

use std::time::{Duration, Instant};

use tracing::{debug, info, warn};

use crate::protocol::{
    ElectionAliveData, ElectionStartData, ElectionVictoryData, Message, NodeId, OpCode,
};

/// State of this node in the election process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ElectionState {
    /// Not participating in an election; following the current master.
    Follower,
    /// Actively running for master.
    Candidate,
    /// Won the election and is the master.
    Leader,
}

/// Manages leader election using the Bully Algorithm.
///
/// The algorithm guarantees that the node with the highest priority among
/// active nodes will become master. Priority is derived deterministically
/// from the node UUID so that all nodes agree on ordering.
#[derive(Debug)]
pub struct ElectionManager {
    node_id: NodeId,
    /// Deterministic priority derived from node_id (higher = wins election).
    priority: u64,
    state: ElectionState,
    /// Known master node ID, if any.
    current_leader: Option<NodeId>,
    /// When the current election started (if in Candidate state).
    election_started_at: Option<Instant>,
    /// Whether we received an ElectionAlive from a higher-priority node.
    received_alive: bool,
    /// When we received the last ElectionAlive (to wait for their victory).
    alive_received_at: Option<Instant>,
    /// How long to wait for ElectionAlive responses before declaring victory.
    election_timeout: Duration,
    /// How long to wait for a victory announcement after receiving ElectionAlive.
    victory_timeout: Duration,
}

impl ElectionManager {
    /// Create a new election manager for the given node.
    ///
    /// # Arguments
    /// * `node_id` - This node's unique identifier
    /// * `election_timeout` - Time to wait for higher-priority nodes to respond
    /// * `victory_timeout` - Time to wait for victory announcement after hearing from a higher node
    pub fn new(node_id: NodeId, election_timeout: Duration, victory_timeout: Duration) -> Self {
        let priority = Self::priority_from_id(&node_id);
        info!(
            %node_id,
            priority,
            "Election manager initialized"
        );
        Self {
            node_id,
            priority,
            state: ElectionState::Follower,
            current_leader: None,
            election_started_at: None,
            received_alive: false,
            alive_received_at: None,
            election_timeout,
            victory_timeout,
        }
    }

    /// Derive a deterministic priority from a UUID.
    ///
    /// Uses the upper 64 bits of the UUID as the priority value.
    /// Higher values win elections.
    fn priority_from_id(id: &NodeId) -> u64 {
        let bytes = id.as_bytes();
        u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    }

    /// Returns this node's ID.
    pub fn node_id(&self) -> NodeId {
        self.node_id
    }

    /// Returns this node's priority value.
    pub fn priority(&self) -> u64 {
        self.priority
    }

    /// Returns `true` if this node is currently the leader.
    pub fn is_leader(&self) -> bool {
        self.state == ElectionState::Leader
    }

    /// Returns the current leader's node ID, if known.
    pub fn current_leader(&self) -> Option<NodeId> {
        self.current_leader
    }

    /// Start a new election. Returns the ElectionStart message to broadcast.
    pub fn start_election(&mut self) -> Message {
        info!(%self.node_id, self.priority, "Starting election");
        self.state = ElectionState::Candidate;
        self.election_started_at = Some(Instant::now());
        self.received_alive = false;
        self.alive_received_at = None;
        self.current_leader = None;

        Message::new(
            OpCode::ElectionStart,
            &ElectionStartData {
                candidate_id: self.node_id,
                priority: self.priority,
            },
        )
        .unwrap()
    }

    /// Handle an incoming `ElectionStart` message from another candidate.
    ///
    /// If this node has higher priority, it responds with `ElectionAlive`
    /// and starts its own election. Returns Some(ElectionAlive) if we have
    /// higher priority, None otherwise.
    pub fn handle_election_start(&mut self, data: &ElectionStartData) -> Option<Message> {
        debug!(
            candidate_id = %data.candidate_id,
            candidate_priority = data.priority,
            self_priority = self.priority,
            "Received ElectionStart"
        );

        if self.priority > data.priority {
            // We have higher priority: send ElectionAlive and start our own election
            info!(
                %self.node_id,
                "Higher priority than candidate {}, responding with ElectionAlive",
                data.candidate_id
            );

            // Start our own election if not already leader or candidate
            if self.state != ElectionState::Leader && self.state != ElectionState::Candidate {
                self.state = ElectionState::Candidate;
                self.election_started_at = Some(Instant::now());
                self.received_alive = false;
                self.alive_received_at = None;
            }

            Some(
                Message::new(
                    OpCode::ElectionAlive,
                    &ElectionAliveData {
                        node_id: self.node_id,
                    },
                )
                .unwrap(),
            )
        } else {
            // Lower or equal priority: back off
            debug!("Lower priority than candidate, backing off");
            self.state = ElectionState::Follower;
            self.election_started_at = None;
            None
        }
    }

    /// Handle an incoming `ElectionAlive` response from a higher-priority node.
    ///
    /// This means we should back off and wait for their victory announcement.
    pub fn handle_alive(&mut self, data: &ElectionAliveData) {
        info!(
            node_id = %data.node_id,
            "Received ElectionAlive from higher-priority node, backing off"
        );
        self.received_alive = true;
        self.alive_received_at = Some(Instant::now());
        // Stay in Candidate state but don't declare victory — wait for their victory
    }

    /// Handle a victory announcement from another node.
    pub fn handle_victory(&mut self, data: &ElectionVictoryData) {
        info!(master_id = %data.master_id, "Received ElectionVictory, accepting new master");
        self.current_leader = Some(data.master_id);
        self.state = if data.master_id == self.node_id {
            ElectionState::Leader
        } else {
            ElectionState::Follower
        };
        self.election_started_at = None;
        self.received_alive = false;
        self.alive_received_at = None;
    }

    /// Check whether the election timeout has expired.
    ///
    /// Call this periodically. Returns `Some(ElectionVictory)` if this node
    /// should declare itself master, or `Some(ElectionStart)` if we need to
    /// restart the election (waited too long for a victory after receiving
    /// ElectionAlive).
    pub fn check_timeout(&mut self) -> Option<Message> {
        if self.state != ElectionState::Candidate {
            return None;
        }

        if self.received_alive {
            // We heard from a higher-priority node. Wait for their victory.
            if let Some(alive_at) = self.alive_received_at {
                if alive_at.elapsed() > self.victory_timeout {
                    // The higher-priority node didn't declare victory in time.
                    // They may have died. Restart election.
                    warn!("Victory timeout expired after ElectionAlive, restarting election");
                    return Some(self.start_election());
                }
            }
            None
        } else if let Some(started_at) = self.election_started_at {
            if started_at.elapsed() > self.election_timeout {
                // No higher-priority node responded. We win!
                info!(
                    %self.node_id,
                    "Election timeout expired with no higher-priority response, declaring victory"
                );
                self.state = ElectionState::Leader;
                self.current_leader = Some(self.node_id);
                self.election_started_at = None;
                Some(
                    Message::new(
                        OpCode::ElectionVictory,
                        &ElectionVictoryData {
                            master_id: self.node_id,
                        },
                    )
                    .unwrap(),
                )
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Force this node to become leader (for single-node clusters or initial bootstrap).
    pub fn force_leader(&mut self) {
        self.state = ElectionState::Leader;
        self.current_leader = Some(self.node_id);
        self.election_started_at = None;
        self.received_alive = false;
        self.alive_received_at = None;
        info!(%self.node_id, "Forced leader");
    }

    /// Reset election state (e.g., when detecting master is down).
    pub fn reset(&mut self) {
        debug!(%self.node_id, "Resetting election state");
        self.state = ElectionState::Follower;
        self.current_leader = None;
        self.election_started_at = None;
        self.received_alive = false;
        self.alive_received_at = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn make_manager(id: NodeId) -> ElectionManager {
        ElectionManager::new(id, Duration::from_secs(3), Duration::from_secs(5))
    }

    #[test]
    fn test_priority_deterministic() {
        let id = Uuid::new_v4();
        let p1 = ElectionManager::priority_from_id(&id);
        let p2 = ElectionManager::priority_from_id(&id);
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_start_election_becomes_candidate() {
        let id = Uuid::new_v4();
        let mut mgr = make_manager(id);
        let msg = mgr.start_election();
        assert_eq!(msg.op, OpCode::ElectionStart);

        let data: ElectionStartData = msg.parse_data().unwrap();
        assert_eq!(data.candidate_id, id);
        assert_eq!(data.priority, mgr.priority());
    }

    #[test]
    fn test_higher_priority_responds_with_alive() {
        let low_bytes = [0u8; 16];
        let high_bytes = [0xFF; 16];
        let node_low = Uuid::from_bytes(low_bytes);
        let node_high = Uuid::from_bytes(high_bytes);

        let mut mgr_high = make_manager(node_high);
        let low_priority = ElectionManager::priority_from_id(&node_low);

        let start_data = ElectionStartData {
            candidate_id: node_low,
            priority: low_priority,
        };

        let response = mgr_high.handle_election_start(&start_data);
        assert!(response.is_some());
        let msg = response.unwrap();
        assert_eq!(msg.op, OpCode::ElectionAlive);

        let alive_data: ElectionAliveData = msg.parse_data().unwrap();
        assert_eq!(alive_data.node_id, node_high);
    }

    #[test]
    fn test_lower_priority_ignores_higher_candidate() {
        let low_bytes = [0u8; 16];
        let high_bytes = [0xFF; 16];
        let node_low = Uuid::from_bytes(low_bytes);
        let node_high = Uuid::from_bytes(high_bytes);

        let mut mgr_low = make_manager(node_low);
        let high_priority = ElectionManager::priority_from_id(&node_high);

        let start_data = ElectionStartData {
            candidate_id: node_high,
            priority: high_priority,
        };

        let response = mgr_low.handle_election_start(&start_data);
        assert!(response.is_none());
    }

    #[test]
    fn test_election_timeout_declares_victory() {
        let id = Uuid::new_v4();
        let mut mgr = ElectionManager::new(
            id,
            Duration::from_millis(1),
            Duration::from_secs(5),
        );
        mgr.start_election();

        // Sleep past election_timeout deterministically — `> Duration::ZERO`
        // can race the monotonic clock when two Instant reads tie.
        std::thread::sleep(Duration::from_millis(5));
        let result = mgr.check_timeout();
        assert!(result.is_some());
        let msg = result.unwrap();
        assert_eq!(msg.op, OpCode::ElectionVictory);

        let data: ElectionVictoryData = msg.parse_data().unwrap();
        assert_eq!(data.master_id, id);
        assert!(mgr.is_leader());
        assert_eq!(mgr.current_leader(), Some(id));
    }

    #[test]
    fn test_handle_alive_backs_off() {
        let id = Uuid::new_v4();
        let other = Uuid::new_v4();
        let mut mgr = make_manager(id);
        mgr.start_election();

        let alive_data = ElectionAliveData { node_id: other };
        mgr.handle_alive(&alive_data);
        assert!(!mgr.is_leader());
    }

    #[test]
    fn test_handle_victory_from_other() {
        let id = Uuid::new_v4();
        let master = Uuid::new_v4();
        let mut mgr = make_manager(id);

        let victory_data = ElectionVictoryData { master_id: master };
        mgr.handle_victory(&victory_data);

        assert!(!mgr.is_leader());
        assert_eq!(mgr.current_leader(), Some(master));
    }

    #[test]
    fn test_handle_victory_self() {
        let id = Uuid::new_v4();
        let mut mgr = make_manager(id);

        let victory_data = ElectionVictoryData { master_id: id };
        mgr.handle_victory(&victory_data);

        assert!(mgr.is_leader());
        assert_eq!(mgr.current_leader(), Some(id));
    }

    #[test]
    fn test_not_leader_initially() {
        let mgr = make_manager(Uuid::new_v4());
        assert!(!mgr.is_leader());
        assert_eq!(mgr.current_leader(), None);
    }

    #[test]
    fn test_force_leader() {
        let id = Uuid::new_v4();
        let mut mgr = make_manager(id);
        assert!(!mgr.is_leader());

        mgr.force_leader();
        assert!(mgr.is_leader());
        assert_eq!(mgr.current_leader(), Some(id));
    }

    #[test]
    fn test_reset() {
        let id = Uuid::new_v4();
        let mut mgr = ElectionManager::new(id, Duration::from_millis(1), Duration::from_secs(5));
        mgr.start_election();
        std::thread::sleep(Duration::from_millis(5));
        let _ = mgr.check_timeout(); // becomes leader
        assert!(mgr.is_leader());

        mgr.reset();
        assert!(!mgr.is_leader());
        assert_eq!(mgr.current_leader(), None);
    }

    #[test]
    fn test_victory_timeout_restarts_election() {
        let id = Uuid::new_v4();
        let other = Uuid::new_v4();
        let mut mgr = ElectionManager::new(
            id,
            Duration::from_secs(5),
            Duration::from_millis(1),
        );
        mgr.start_election();

        // Receive alive from a higher-priority node
        let alive_data = ElectionAliveData { node_id: other };
        mgr.handle_alive(&alive_data);

        // Sleep past victory_timeout deterministically — `> Duration::ZERO`
        // can race the monotonic clock when two Instant reads tie.
        std::thread::sleep(Duration::from_millis(5));
        let result = mgr.check_timeout();
        assert!(result.is_some());
        let msg = result.unwrap();
        assert_eq!(msg.op, OpCode::ElectionStart);
    }

    #[test]
    fn test_check_timeout_as_follower_returns_none() {
        let id = Uuid::new_v4();
        let mut mgr = make_manager(id);
        // Follower state, no election in progress
        let result = mgr.check_timeout();
        assert!(result.is_none());
    }

    #[test]
    fn test_full_election_scenario() {
        // Simulate a 3-node cluster election
        let low_bytes = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0, 0, 0, 0, 0, 0, 0, 0];
        let mid_bytes = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0, 0, 0, 0, 0, 0, 0, 0];
        let high_bytes = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0, 0, 0, 0, 0, 0, 0, 0];
        let node_low = Uuid::from_bytes(low_bytes);
        let node_mid = Uuid::from_bytes(mid_bytes);
        let node_high = Uuid::from_bytes(high_bytes);

        let mut mgr_low = ElectionManager::new(node_low, Duration::from_millis(1), Duration::from_secs(5));
        let mut mgr_mid = ElectionManager::new(node_mid, Duration::from_millis(1), Duration::from_secs(5));
        let mut mgr_high = ElectionManager::new(node_high, Duration::from_millis(1), Duration::from_secs(5));

        // Low-priority node starts election
        let start_msg = mgr_low.start_election();
        let start_data: ElectionStartData = start_msg.parse_data().unwrap();

        // Mid-priority node responds with Alive (higher than low)
        let mid_response = mgr_mid.handle_election_start(&start_data);
        assert!(mid_response.is_some());

        // High-priority node also responds with Alive
        let high_response = mgr_high.handle_election_start(&start_data);
        assert!(high_response.is_some());

        // Low node receives Alive, backs off
        let alive_data = ElectionAliveData { node_id: node_high };
        mgr_low.handle_alive(&alive_data);
        assert!(!mgr_low.is_leader());

        // High-priority node times out (no one higher), declares victory
        std::thread::sleep(Duration::from_millis(5));
        let victory = mgr_high.check_timeout();
        assert!(victory.is_some());
        let victory_msg = victory.unwrap();
        assert_eq!(victory_msg.op, OpCode::ElectionVictory);

        let victory_data: ElectionVictoryData = victory_msg.parse_data().unwrap();
        assert_eq!(victory_data.master_id, node_high);

        // All nodes accept the victory
        mgr_low.handle_victory(&victory_data);
        mgr_mid.handle_victory(&victory_data);
        mgr_high.handle_victory(&victory_data);

        assert!(mgr_high.is_leader());
        assert!(!mgr_mid.is_leader());
        assert!(!mgr_low.is_leader());
        assert_eq!(mgr_low.current_leader(), Some(node_high));
        assert_eq!(mgr_mid.current_leader(), Some(node_high));
    }
}
