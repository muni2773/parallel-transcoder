# Parallel Video Transcoder — API Reference

## Overview

REST API, WebSocket interface, and cluster endpoints for parallel video transcoding. The server accepts video uploads, runs transcoding jobs using a Rust coordinator with multiple workers (local or distributed across a cluster), and streams real-time progress via WebSocket.

**Base URL:** `http://localhost:3000`

## Authentication

Optional. Set `TRANSCODER_API_KEY` environment variable to require an API key.

- Header: `X-API-Key: your-key-here`
- Query parameter: `?api_key=your-key-here`

---

## Transcoding Endpoints

### Health Check
`GET /api/health`

Returns server status, job counts, and platform info.

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "uptime": 3600,
  "platform": "darwin",
  "arch": "arm64",
  "jobs": { "total": 5, "running": 1 }
}
```

### Upload Video
`POST /api/upload`

Upload a video file (multipart/form-data, field name `video`). Max size: 10 GB.

**Response:**
```json
{
  "uploadId": "my_video_1709654400000.mp4",
  "originalName": "my_video.mp4",
  "size": 104857600
}
```

**Example:**
```bash
curl -X POST http://localhost:3000/api/upload -F "video=@video.mp4"
```

### Start Transcode Job
`POST /api/transcode`

**Request body (JSON):**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `uploadId` | string | *required* | ID from upload |
| `format` | string | `"hls"` | `"hls"` or `"mp4"` |
| `mode` | string | `"normal"` | `"normal"`, `"copy"`, `"smart"`, `"smart-auto"` |
| `crf` | number | `23` | Quality (0-51 for H.264/H.265, 0-63 for AV1) |
| `preset` | string | `"medium"` | `"ultrafast"` to `"veryslow"` |
| `encoder` | string | `"libx264"` | See supported encoders below |
| `workers` | number | `0` | Worker count (0 = auto-detect) |
| `smartTolerance` | number | `0.3` | Smart mode tolerance (0.05-0.50) |
| `verbose` | boolean | `false` | Verbose logging |

**Supported encoders:**

| Codec | CPU | macOS GPU | Linux NVIDIA | Linux Intel/AMD |
|-------|-----|-----------|-------------|-----------------|
| H.264 | `libx264` | `h264_videotoolbox` | `h264_nvenc` | `h264_vaapi` |
| H.265 | `libx265` | `hevc_videotoolbox` | `hevc_nvenc` | `hevc_vaapi` |
| AV1   | `libsvtav1`, `libaom-av1` | — | `av1_nvenc` | — |

**Response:**
```json
{
  "jobId": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running"
}
```

**Example:**
```bash
curl -X POST http://localhost:3000/api/transcode \
  -H "Content-Type: application/json" \
  -d '{"uploadId": "my_video_1709654400000.mp4", "format": "mp4", "encoder": "libx265", "crf": 28}'
```

### Analyze Video
`POST /api/analyze`

Analyze complexity without encoding.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `uploadId` | string | *required* | ID from upload |
| `crf` | number | `23` | CRF for analysis |
| `encoder` | string | `"libx264"` | Encoder for analysis |
| `smartTolerance` | number | `0.3` | Complexity threshold |

### List Jobs
`GET /api/jobs`

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "running",
    "phase": "encoding",
    "percent": 65,
    "config": { "format": "mp4", "crf": 28, "preset": "fast", "encoder": "libx265" },
    "createdAt": "2024-03-05T10:30:00.000Z",
    "logCount": 42
  }
]
```

### Get Job Status
`GET /api/jobs/:id`

**Phases:** `starting` → `analyzing` → `splitting` → `encoding` → `finalizing` → `complete`

**Statuses:** `queued`, `running`, `complete`, `error`, `cancelled`

### Get Job Logs
`GET /api/jobs/:id/logs?offset=0&limit=100`

```json
{
  "total": 142,
  "offset": 0,
  "lines": ["2024-03-05T10:30:00 INFO coordinator: Analyzing video..."]
}
```

### List Output Files
`GET /api/jobs/:id/files`

```json
[
  { "name": "output.mp4", "size": 52428800 }
]
```

### Download Output File
`GET /api/download/:jobId/:filename`

### Cancel Job
`DELETE /api/jobs/:id`

```json
{ "deleted": true }
```

---

## Cluster Endpoints

These endpoints interact with a running cluster of `transcoder-node` daemons.

### Cluster Status
`GET /api/cluster/status`

Returns the current cluster state from the master node.

**Response:**
```json
{
  "master": "node-1",
  "nodes": 3,
  "activeJobs": 1,
  "totalCapacity": { "cores": 32, "memoryGb": 96 }
}
```

### List Cluster Nodes
`GET /api/cluster/nodes`

Returns information about all nodes in the cluster.

**Response:**
```json
[
  {
    "id": "550e8400-...",
    "name": "node-1",
    "address": "192.168.1.10:9000",
    "role": "master",
    "capabilities": {
      "cpuCores": 8,
      "memoryMb": 32768,
      "gpus": ["videotoolbox"]
    },
    "load": { "activeTasks": 2, "cpuPercent": 45.2 },
    "status": "alive"
  }
]
```

### Submit Cluster Job
`POST /api/cluster/transcode`

Submit a transcoding job to the cluster. The master node distributes segments across all available nodes.

**Request body (JSON):**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `uploadId` | string | *required* | ID from upload |
| `format` | string | `"hls"` | Output format |
| `encoder` | string | `"libx264"` | Video encoder |
| `crf` | number | `23` | Quality level |
| `preset` | string | `"medium"` | Speed preset |

**Response:**
```json
{
  "jobId": "cluster-job-550e8400",
  "status": "submitted",
  "assignedNodes": ["node-1", "node-2", "node-3"]
}
```

---

## WebSocket API

Connect to `ws://localhost:3000/ws` for real-time updates.

### Subscribe
```json
{ "type": "subscribe", "jobId": "JOB_ID" }
```

### Server Events

**progress:**
```json
{ "type": "progress", "jobId": "...", "phase": "encoding", "percent": 65, "message": "Encoding segment 6/10" }
```

**log:**
```json
{ "type": "log", "jobId": "...", "line": "INFO worker: Segment 6 complete" }
```

**complete:**
```json
{ "type": "complete", "jobId": "...", "outputFiles": [{ "name": "output.mp4", "size": 52428800 }] }
```

**error:**
```json
{ "type": "error", "jobId": "...", "message": "Coordinator exited with code 1" }
```

### JavaScript Example
```javascript
const ws = new WebSocket("ws://localhost:3000/ws");
ws.onopen = () => ws.send(JSON.stringify({ type: "subscribe", jobId: "JOB_ID" }));
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === "progress") console.log(`${msg.phase}: ${msg.percent}%`);
  if (msg.type === "complete") console.log("Done!", msg.outputFiles);
  if (msg.type === "error") console.error(msg.message);
};
```

### Python Example
```python
import asyncio, json, websockets

async def monitor(job_id):
    async with websockets.connect("ws://localhost:3000/ws") as ws:
        await ws.send(json.dumps({"type": "subscribe", "jobId": job_id}))
        async for message in ws:
            msg = json.loads(message)
            if msg["type"] == "complete":
                break
            print(f"{msg.get('phase', '')}: {msg.get('percent', '')}%")

asyncio.run(monitor("JOB_ID"))
```

---

## Cluster OpCode Protocol

The cluster control plane uses an OBS-websocket-inspired binary protocol over WebSocket. Each message has the shape `{ "op": <number>, "d": <payload> }`.

| OpCode | Name | Direction | Description |
|--------|------|-----------|-------------|
| 0 | Hello | Server→Client | Initial handshake with node capabilities |
| 1 | Identify | Client→Server | Node identification and auth |
| 2 | Identified | Server→Client | Successful identification |
| 10 | ElectionStart | Any→All | Bully election initiation |
| 11 | ElectionAlive | Any→Candidate | "I'm alive" response to election |
| 12 | ElectionVictory | Leader→All | New leader announcement |
| 20 | Heartbeat | Any→Peer | Periodic heartbeat |
| 21 | HeartbeatAck | Peer→Any | Heartbeat acknowledgment |
| 30 | JobSubmit | Client→Master | Submit new transcoding job |
| 31 | JobAccepted | Master→Client | Job accepted confirmation |
| 32 | JobProgress | Worker→Master | Segment progress update |
| 33 | JobComplete | Master→Client | Job finished successfully |
| 34 | JobFailed | Master→Client | Job failed |
| 35 | JobCancel | Client→Master | Cancel a running job |
| 40 | SegmentAssign | Master→Worker | Assign segment to worker |
| 41 | SegmentAssignAck | Worker→Master | Segment assignment acknowledged |
| 42 | SegmentComplete | Worker→Master | Segment encoding finished |
| 43 | SegmentFailed | Worker→Master | Segment encoding failed |
| 50 | StatusRequest | Any→Master | Request cluster status |
| 51 | StatusResponse | Master→Any | Cluster status response |
| 60 | Event | Master→Subscribers | Event broadcast |
| 70 | NodeLeave | Any→All | Graceful node departure |
| 255 | Error | Any→Any | Error notification |

---

## Error Responses

All errors return JSON:
```json
{ "error": "Description of what went wrong" }
```

| Code | Meaning |
|------|---------|
| 400 | Bad request (missing/invalid parameters) |
| 401 | Unauthorized (invalid API key) |
| 404 | Not found (job or file) |
| 500 | Internal server error |

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `PORT` | `3000` | Server listen port |
| `TRANSCODER_API_KEY` | *(none)* | API key for authentication |
| `CLUSTER_MASTER` | *(none)* | Cluster master address for cluster endpoints |
