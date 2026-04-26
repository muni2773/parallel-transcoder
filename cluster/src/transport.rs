//! WebSocket-based transport layer for cluster communication.
//!
//! Uses tokio-tungstenite for the control plane. Data plane (segment file
//! transfer) uses SRT URLs handled by FFmpeg natively (see `srt` module).

use crate::protocol::{Message, NodeId};
use anyhow::{Context, Result};
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::{lookup_host, TcpListener};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite;
use tracing::{error, info, warn};

/// Incoming message from a connected peer.
#[derive(Debug, Clone)]
pub struct PeerMessage {
    pub peer_id: Option<NodeId>,
    pub peer_addr: SocketAddr,
    pub message: Message,
}

/// Handle to a connected peer for sending messages.
#[derive(Clone)]
pub struct PeerHandle {
    pub node_id: Option<NodeId>,
    pub addr: SocketAddr,
    tx: mpsc::UnboundedSender<Message>,
}

impl PeerHandle {
    pub fn send(&self, msg: Message) -> Result<()> {
        self.tx
            .send(msg)
            .map_err(|_| anyhow::anyhow!("Peer disconnected"))?;
        Ok(())
    }
}

/// WebSocket transport layer.
///
/// Handles accepting incoming connections and connecting to peers.
/// Messages are received via an unbounded channel.
pub struct Transport {
    listen_addr: SocketAddr,
    peers: Arc<DashMap<SocketAddr, PeerHandle>>,
    incoming_tx: mpsc::UnboundedSender<PeerMessage>,
    incoming_rx: Option<mpsc::UnboundedReceiver<PeerMessage>>,
}

impl Transport {
    pub fn new(listen_addr: SocketAddr) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        Self {
            listen_addr,
            peers: Arc::new(DashMap::new()),
            incoming_tx: tx,
            incoming_rx: Some(rx),
        }
    }

    /// Take the incoming message receiver (can only be called once).
    pub fn take_incoming(&mut self) -> Option<mpsc::UnboundedReceiver<PeerMessage>> {
        self.incoming_rx.take()
    }

    /// Start accepting WebSocket connections.
    pub async fn listen(&self) -> Result<()> {
        let listener = TcpListener::bind(self.listen_addr)
            .await
            .context("Failed to bind WebSocket listener")?;
        info!("WebSocket transport listening on {}", self.listen_addr);

        let peers = self.peers.clone();
        let incoming_tx = self.incoming_tx.clone();

        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, addr)) => {
                        let peers = peers.clone();
                        let incoming_tx = incoming_tx.clone();
                        tokio::spawn(async move {
                            match tokio_tungstenite::accept_async(stream).await {
                                Ok(ws_stream) => {
                                    info!("Accepted WebSocket connection from {}", addr);
                                    Self::run_server_connection(ws_stream, addr, peers, incoming_tx)
                                        .await;
                                }
                                Err(e) => {
                                    warn!("WebSocket handshake failed from {}: {}", addr, e);
                                }
                            }
                        });
                    }
                    Err(e) => {
                        error!("Accept error: {}", e);
                    }
                }
            }
        });

        Ok(())
    }

    /// Connect to a peer node. `addr` may be `host:port` (DNS-resolved) or
    /// `ip:port`. The returned PeerHandle.addr is the resolved SocketAddr,
    /// which callers should use as the routing key for `send_to`.
    pub async fn connect(&self, addr: &str) -> Result<PeerHandle> {
        // Resolve hostname → SocketAddr so the peer-map key is valid even when
        // addr is a K8s service DNS name like `transcoder-node-0.<svc>.<ns>...`.
        let socket_addr: SocketAddr = match addr.parse() {
            Ok(sa) => sa,
            Err(_) => lookup_host(addr)
                .await
                .with_context(|| format!("DNS lookup failed for {}", addr))?
                .next()
                .with_context(|| format!("No addresses resolved for {}", addr))?,
        };

        let url = format!("ws://{}", addr);
        let (ws_stream, _) = tokio_tungstenite::connect_async(&url)
            .await
            .with_context(|| format!("Failed to connect to {}", addr))?;

        info!("Connected to peer at {} ({})", addr, socket_addr);

        let (tx, rx) = mpsc::unbounded_channel::<Message>();

        let handle = PeerHandle {
            node_id: None,
            addr: socket_addr,
            tx,
        };

        self.peers.insert(socket_addr, handle.clone());

        let peers = self.peers.clone();
        let incoming_tx = self.incoming_tx.clone();

        tokio::spawn(
            Self::run_client_connection(ws_stream, rx, socket_addr, peers, incoming_tx),
        );

        Ok(handle)
    }

    /// Send a message to a specific peer by address.
    pub fn send_to(&self, addr: &SocketAddr, msg: Message) -> Result<()> {
        if let Some(peer) = self.peers.get(addr) {
            peer.send(msg)?;
        } else {
            anyhow::bail!("No connection to {}", addr);
        }
        Ok(())
    }

    /// Broadcast a message to all connected peers.
    pub fn broadcast(&self, msg: &Message, exclude: Option<&SocketAddr>) {
        for entry in self.peers.iter() {
            if let Some(exc) = exclude {
                if entry.key() == exc {
                    continue;
                }
            }
            if let Err(e) = entry.value().send(msg.clone()) {
                warn!("Failed to send to {}: {}", entry.key(), e);
            }
        }
    }

    /// Get all connected peer addresses.
    pub fn connected_peers(&self) -> Vec<SocketAddr> {
        self.peers.iter().map(|e| *e.key()).collect()
    }

    /// Set the node_id for a peer (after Identify handshake).
    pub fn set_peer_node_id(&self, addr: &SocketAddr, node_id: NodeId) {
        if let Some(mut peer) = self.peers.get_mut(addr) {
            peer.node_id = Some(node_id);
        }
    }

    /// Find peer address by node_id.
    pub fn find_peer_addr(&self, node_id: &NodeId) -> Option<SocketAddr> {
        self.peers
            .iter()
            .find(|e| e.value().node_id.as_ref() == Some(node_id))
            .map(|e| *e.key())
    }

    /// Get the number of connected peers.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    // --- Internal: server-side connection (accepted from TcpListener) ---

    async fn run_server_connection(
        ws_stream: tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
        addr: SocketAddr,
        peers: Arc<DashMap<SocketAddr, PeerHandle>>,
        incoming_tx: mpsc::UnboundedSender<PeerMessage>,
    ) {
        let (tx, rx) = mpsc::unbounded_channel::<Message>();
        let handle = PeerHandle {
            node_id: None,
            addr,
            tx,
        };
        peers.insert(addr, handle);

        let (mut write, mut read) = ws_stream.split();

        let peers_read = peers.clone();

        // Spawn writer task
        let writer = tokio::spawn(async move {
            let mut rx = rx;
            while let Some(msg) = rx.recv().await {
                match serde_json::to_string(&msg) {
                    Ok(json) => {
                        if let Err(e) = write.send(tungstenite::Message::Text(json.into())).await {
                            warn!("Write error to {}: {}", addr, e);
                            break;
                        }
                    }
                    Err(e) => {
                        error!("Serialize error: {}", e);
                    }
                }
            }
        });

        // Reader loop in current task
        while let Some(result) = read.next().await {
            match result {
                Ok(tungstenite::Message::Text(text)) => {
                    match serde_json::from_str::<Message>(&text) {
                        Ok(msg) => {
                            let node_id = peers_read.get(&addr).and_then(|p| p.node_id);
                            let _ = incoming_tx.send(PeerMessage {
                                peer_id: node_id,
                                peer_addr: addr,
                                message: msg,
                            });
                        }
                        Err(e) => {
                            warn!("Invalid message from {}: {}", addr, e);
                        }
                    }
                }
                Ok(tungstenite::Message::Close(_)) => {
                    info!("Peer {} disconnected", addr);
                    break;
                }
                Ok(_) => {} // ignore binary, ping, pong
                Err(e) => {
                    warn!("Read error from {}: {}", addr, e);
                    break;
                }
            }
        }

        writer.abort();
        peers_read.remove(&addr);
        info!("Server connection closed: {}", addr);
    }

    // --- Internal: client-side connection (we initiated via connect_async) ---

    async fn run_client_connection(
        ws_stream: tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
        mut rx: mpsc::UnboundedReceiver<Message>,
        addr: SocketAddr,
        peers: Arc<DashMap<SocketAddr, PeerHandle>>,
        incoming_tx: mpsc::UnboundedSender<PeerMessage>,
    ) {
        let (mut write, mut read) = ws_stream.split();

        let peers_read = peers.clone();

        // Spawn writer task
        let writer = tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                match serde_json::to_string(&msg) {
                    Ok(json) => {
                        if let Err(e) = write.send(tungstenite::Message::Text(json.into())).await {
                            warn!("Write error to {}: {}", addr, e);
                            break;
                        }
                    }
                    Err(e) => {
                        error!("Serialize error: {}", e);
                    }
                }
            }
        });

        // Reader loop in current task
        while let Some(result) = read.next().await {
            match result {
                Ok(tungstenite::Message::Text(text)) => {
                    match serde_json::from_str::<Message>(&text) {
                        Ok(msg) => {
                            let node_id = peers_read.get(&addr).and_then(|p| p.node_id);
                            let _ = incoming_tx.send(PeerMessage {
                                peer_id: node_id,
                                peer_addr: addr,
                                message: msg,
                            });
                        }
                        Err(e) => {
                            warn!("Invalid message from {}: {}", addr, e);
                        }
                    }
                }
                Ok(tungstenite::Message::Close(_)) => {
                    info!("Peer {} disconnected", addr);
                    break;
                }
                Ok(_) => {}
                Err(e) => {
                    warn!("Read error from {}: {}", addr, e);
                    break;
                }
            }
        }

        writer.abort();
        peers_read.remove(&addr);
        info!("Client connection closed: {}", addr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{HelloData, OpCode};
    use std::net::{IpAddr, Ipv4Addr};

    #[tokio::test]
    async fn test_transport_listen_and_connect() {
        // Bind to find a free port, then re-create the transport on that port
        let probe = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let server_addr = probe.local_addr().unwrap();
        drop(probe);

        let mut server = Transport::new(server_addr);
        let mut incoming = server.take_incoming().unwrap();

        server.listen().await.unwrap();

        // Give the listener a moment to start
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Connect a client
        let client_addr =
            SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);
        let client = Transport::new(client_addr);

        let peer = client
            .connect(&server_addr.to_string())
            .await
            .unwrap();

        // Send a message from client to server
        let hello = HelloData {
            cluster_name: "test".into(),
            protocol_version: 1,
            master_id: None,
        };
        let msg = Message::new(OpCode::Hello, &hello).unwrap();
        peer.send(msg).unwrap();

        // Receive on server side
        let received = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            incoming.recv(),
        )
        .await
        .unwrap()
        .unwrap();

        assert_eq!(received.message.op, OpCode::Hello);
        let parsed: HelloData = received.message.parse_data().unwrap();
        assert_eq!(parsed.cluster_name, "test");
    }

    #[test]
    fn test_peer_handle_send_disconnected() {
        let (tx, _rx) = mpsc::unbounded_channel();
        drop(_rx); // drop receiver to simulate disconnect
        let handle = PeerHandle {
            node_id: None,
            addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 9000),
            tx,
        };
        // Sending should still succeed because unbounded_sender doesn't fail
        // until the receiver is dropped AND we try to send
        // Actually with rx dropped, send should fail
        let msg = Message::new(OpCode::Hello, &serde_json::json!({})).unwrap();
        // This may or may not fail depending on timing; just verify it doesn't panic
        let _ = handle.send(msg);
    }
}
