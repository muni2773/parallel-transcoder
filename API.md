# Parallel Video Transcoder — API Reference

## Overview
REST API and WebSocket interface for parallel video transcoding. The server accepts video uploads, runs transcoding jobs using a Rust-based coordinator with multiple workers, and streams real-time progress via WebSocket.

**Base URL:** `http://localhost:3000`

## Authentication
Authentication is optional. Set the `TRANSCODER_API_KEY` environment variable to require an API key. When enabled, include the key in requests via:
- Header: `X-API-Key: your-key-here`
- Query parameter: `?api_key=your-key-here`

## Endpoints

### Health Check
`GET /api/health`

Returns server status and job counts.

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "uptime": 3600,
  "jobs": { "total": 5, "running": 1 }
}
```

**Example:**
```bash
curl http://localhost:3000/api/health
```

### Upload Video
`POST /api/upload`

Upload a video file for transcoding. Max size: 10 GB.

**Request:** multipart/form-data with field name `video`

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
curl -X POST http://localhost:3000/api/upload \
  -F "video=@/path/to/video.mp4"
```

### Start Transcode Job
`POST /api/transcode`

Start a transcoding job with the uploaded video.

**Request body (JSON):**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `uploadId` | string | *required* | ID returned from upload |
| `format` | string | `"hls"` | Output format: `"hls"` or `"mp4"` |
| `mode` | string | `"normal"` | Encoding mode: `"normal"`, `"copy"`, `"smart"`, `"smart-auto"` |
| `crf` | number | `23` | Quality (0-51, lower = better) |
| `preset` | string | `"medium"` | Speed preset: `"ultrafast"` to `"veryslow"` |
| `encoder` | string | `"libx264"` | Video encoder: `"libx264"` or `"h264_videotoolbox"` |
| `workers` | number | `0` | Worker count (0 = auto-detect CPU cores) |
| `smartTolerance` | number | `0.3` | Smart mode tolerance (0.05-0.50) |
| `verbose` | boolean | `false` | Enable verbose logging |

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
  -d '{
    "uploadId": "my_video_1709654400000.mp4",
    "format": "mp4",
    "crf": 20,
    "preset": "fast",
    "encoder": "libx264"
  }'
```

### Analyze Video
`POST /api/analyze`

Analyze a video without full transcoding. Returns a segment-by-segment complexity report useful for tuning encoding parameters.

**Request body (JSON):**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `uploadId` | string | *required* | ID returned from upload |
| `crf` | number | `23` | CRF value for analysis |
| `encoder` | string | `"libx264"` | Encoder to analyze with |
| `smartTolerance` | number | `0.3` | Complexity tolerance threshold |

**Response:** JSON analysis report with segments table, complexity metrics, and encoding decisions.

**Example:**
```bash
curl -X POST http://localhost:3000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"uploadId": "my_video_1709654400000.mp4"}'
```

### List Jobs
`GET /api/jobs`

List all transcoding jobs.

**Response:**
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "running",
    "phase": "encoding",
    "percent": 65,
    "config": { "format": "mp4", "crf": 20, "preset": "fast" },
    "errorMessage": null,
    "createdAt": "2024-03-05T10:30:00.000Z",
    "logCount": 42
  }
]
```

**Example:**
```bash
curl http://localhost:3000/api/jobs
```

### Get Job Status
`GET /api/jobs/:id`

Get status of a specific job.

**Phases:** `starting` -> `analyzing` -> `splitting` -> `encoding` -> `finalizing` -> `complete`
**Statuses:** `queued`, `running`, `complete`, `error`, `cancelled`

**Example:**
```bash
curl http://localhost:3000/api/jobs/550e8400-e29b-41d4-a716-446655440000
```

### Get Job Logs
`GET /api/jobs/:id/logs`

Retrieve log lines for a job.

**Query parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `offset` | number | `0` | Starting line index |
| `limit` | number | `100` | Max lines to return |

**Response:**
```json
{
  "total": 142,
  "offset": 0,
  "lines": ["2024-03-05T10:30:00 INFO coordinator: Analyzing video...", "..."]
}
```

**Example:**
```bash
curl "http://localhost:3000/api/jobs/550e8400/logs?offset=0&limit=50"
```

### List Output Files
`GET /api/jobs/:id/files`

List output files for a completed job.

**Response:**
```json
[
  { "name": "output.mp4", "size": 52428800 },
  { "name": "output.m3u8", "size": 1024 }
]
```

**Example:**
```bash
curl http://localhost:3000/api/jobs/550e8400-e29b-41d4-a716-446655440000/files
```

### Download Output File
`GET /api/download/:jobId/:filename`

Download a specific output file.

**Example:**
```bash
curl -O http://localhost:3000/api/download/550e8400-e29b-41d4-a716-446655440000/output.mp4
```

### Cancel Job
`DELETE /api/jobs/:id`

Cancel a running job and remove its output files.

**Response:**
```json
{ "deleted": true }
```

**Example:**
```bash
curl -X DELETE http://localhost:3000/api/jobs/550e8400-e29b-41d4-a716-446655440000
```

## WebSocket API

Connect to `ws://localhost:3000/ws` for real-time job updates.

### Subscribe to Job
Send after connecting to receive updates for a specific job:
```json
{ "type": "subscribe", "jobId": "550e8400-e29b-41d4-a716-446655440000" }
```

### Server Events

**progress** -- Phase and percent updates:
```json
{
  "type": "progress",
  "jobId": "550e8400",
  "phase": "encoding",
  "percent": 65,
  "message": "Encoding segment 6/10"
}
```

**log** -- Individual log lines:
```json
{
  "type": "log",
  "jobId": "550e8400",
  "line": "2024-03-05T10:30:00 INFO worker: Segment 6 complete"
}
```

**complete** -- Job finished successfully:
```json
{
  "type": "complete",
  "jobId": "550e8400",
  "outputFiles": [{ "name": "output.mp4", "size": 52428800 }]
}
```

**error** -- Job failed or was cancelled:
```json
{
  "type": "error",
  "jobId": "550e8400",
  "message": "Coordinator exited with code 1"
}
```

### WebSocket Example (JavaScript)
```javascript
const ws = new WebSocket("ws://localhost:3000/ws");

ws.onopen = () => {
  ws.send(JSON.stringify({ type: "subscribe", jobId: "YOUR_JOB_ID" }));
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  switch (msg.type) {
    case "progress":
      console.log(`Phase: ${msg.phase} — ${msg.percent}%`);
      break;
    case "complete":
      console.log("Done!", msg.outputFiles);
      break;
    case "error":
      console.error("Error:", msg.message);
      break;
  }
};
```

### WebSocket Example (Python)
```python
import asyncio, json, websockets

async def monitor(job_id):
    async with websockets.connect("ws://localhost:3000/ws") as ws:
        await ws.send(json.dumps({"type": "subscribe", "jobId": job_id}))
        async for message in ws:
            msg = json.loads(message)
            if msg["type"] == "progress":
                print(f"{msg['phase']}: {msg['percent']}%")
            elif msg["type"] == "complete":
                print("Done!", msg["outputFiles"])
                break
            elif msg["type"] == "error":
                print("Error:", msg["message"])
                break

asyncio.run(monitor("YOUR_JOB_ID"))
```

## Full Workflow Example

Complete end-to-end transcoding workflow using curl and wscat:

```bash
# 1. Check server health
curl http://localhost:3000/api/health

# 2. Upload a video
UPLOAD=$(curl -s -X POST http://localhost:3000/api/upload \
  -F "video=@input.mp4")
UPLOAD_ID=$(echo $UPLOAD | python3 -c "import sys,json; print(json.load(sys.stdin)['uploadId'])")
echo "Upload ID: $UPLOAD_ID"

# 3. (Optional) Analyze the video first
curl -s -X POST http://localhost:3000/api/analyze \
  -H "Content-Type: application/json" \
  -d "{\"uploadId\": \"$UPLOAD_ID\"}" | python3 -m json.tool

# 4. Start transcoding
JOB=$(curl -s -X POST http://localhost:3000/api/transcode \
  -H "Content-Type: application/json" \
  -d "{\"uploadId\": \"$UPLOAD_ID\", \"format\": \"mp4\", \"crf\": 20}")
JOB_ID=$(echo $JOB | python3 -c "import sys,json; print(json.load(sys.stdin)['jobId'])")
echo "Job ID: $JOB_ID"

# 5. Monitor progress (WebSocket via wscat)
# Install: npm install -g wscat
# wscat -c ws://localhost:3000/ws -x "{\"type\":\"subscribe\",\"jobId\":\"$JOB_ID\"}"

# 5b. Or poll the REST API
while true; do
  STATUS=$(curl -s http://localhost:3000/api/jobs/$JOB_ID)
  PHASE=$(echo $STATUS | python3 -c "import sys,json; print(json.load(sys.stdin)['phase'])")
  PCT=$(echo $STATUS | python3 -c "import sys,json; print(json.load(sys.stdin)['percent'])")
  echo "$PHASE: $PCT%"
  [ "$PHASE" = "complete" ] && break
  [ "$PHASE" = "error" ] && break
  sleep 2
done

# 6. List output files
curl -s http://localhost:3000/api/jobs/$JOB_ID/files | python3 -m json.tool

# 7. Download the output
curl -O http://localhost:3000/api/download/$JOB_ID/output.mp4
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `PORT` | `3000` | Server listen port |
| `TRANSCODER_API_KEY` | *(none)* | API key for authentication (optional) |

## Error Responses

All errors return JSON with an `error` field:
```json
{ "error": "Description of what went wrong" }
```

Common HTTP status codes:
| Code | Meaning |
|------|---------|
| 400 | Bad request (missing parameters, invalid input) |
| 401 | Unauthorized (invalid or missing API key) |
| 404 | Not found (job or file doesn't exist) |
| 500 | Internal server error |

## Rate Limits

No rate limiting is applied by default. For production deployments, consider placing the API behind a reverse proxy (nginx, Caddy) with rate limiting configured.
