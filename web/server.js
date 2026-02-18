import express from "express";
import multer from "multer";
import { WebSocketServer } from "ws";
import { spawn } from "child_process";
import crypto from "crypto";
import path from "path";
import fs from "fs";
import fsp from "fs/promises";
import { fileURLToPath } from "url";
import http from "http";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const COORDINATOR_BIN = path.join(__dirname, "../bin/transcoder-coordinator");
const LIB_DIR = path.join(__dirname, "../lib/");
const UPLOAD_DIR = path.join(__dirname, "uploads");
const OUTPUT_DIR = path.join(__dirname, "outputs");

const PORT = parseInt(process.env.PORT || "3000", 10);
const MAX_LOG_LINES = 200;
const UPLOAD_LIMIT = 10 * 1024 * 1024 * 1024; // 10 GB
const PID_FILE = path.join(__dirname, ".web.pid");

// ---------------------------------------------------------------------------
// Ensure directories
// ---------------------------------------------------------------------------
fs.mkdirSync(UPLOAD_DIR, { recursive: true });
fs.mkdirSync(OUTPUT_DIR, { recursive: true });

// ---------------------------------------------------------------------------
// Job Manager
// ---------------------------------------------------------------------------

/**
 * @typedef {Object} JobState
 * @property {string}              id
 * @property {"queued"|"running"|"complete"|"error"|"cancelled"} status
 * @property {import("child_process").ChildProcess|null} process
 * @property {object}              config
 * @property {string}              outputDir
 * @property {string[]}            logs
 * @property {string}              phase
 * @property {number}              percent
 * @property {string|null}         errorMessage
 * @property {Date}                createdAt
 */

/** @type {Map<string, JobState>} */
const jobs = new Map();

/** @type {Map<string, Set<import("ws").WebSocket>>} */
const subscribers = new Map();

// ---------------------------------------------------------------------------
// Express app
// ---------------------------------------------------------------------------
const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// Multer for uploads
const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, UPLOAD_DIR),
  filename: (_req, file, cb) => {
    const ext = path.extname(file.originalname);
    const base = path.basename(file.originalname, ext).replace(/[^a-zA-Z0-9_-]/g, "_");
    cb(null, `${base}_${Date.now()}${ext}`);
  },
});
const upload = multer({ storage, limits: { fileSize: UPLOAD_LIMIT } });

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

// Upload video
app.post("/api/upload", upload.single("video"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No file uploaded" });
  }
  res.json({
    uploadId: req.file.filename,
    originalName: req.file.originalname,
    size: req.file.size,
  });
});

// Start transcode job
app.post("/api/transcode", async (req, res) => {
  const {
    uploadId,
    format = "hls",
    mode = "normal",
    smartTolerance = 0.3,
    crf = 23,
    preset = "medium",
    encoder = "libx264",
    workers = 0,
    verbose = false,
  } = req.body;

  if (!uploadId) {
    return res.status(400).json({ error: "uploadId is required" });
  }

  const inputPath = path.join(UPLOAD_DIR, path.basename(uploadId));
  if (!fs.existsSync(inputPath)) {
    return res.status(404).json({ error: "Uploaded file not found" });
  }

  const jobId = crypto.randomUUID();
  const jobOutputDir = path.join(OUTPUT_DIR, jobId);
  fs.mkdirSync(jobOutputDir, { recursive: true });

  // Build coordinator args
  const args = [
    "--input", inputPath,
    "--output", jobOutputDir,
    "--format", format,
    "--workers", String(workers),
  ];

  switch (mode) {
    case "copy":
      args.push("--copy");
      break;
    case "smart":
      args.push("--smart", "--smart-tolerance", String(smartTolerance));
      break;
    case "smart-auto":
      args.push("--smart-auto", "--smart-tolerance", String(smartTolerance));
      break;
    case "normal":
    default:
      args.push("--fast");
      break;
  }

  if (mode !== "copy") {
    args.push("--crf", String(crf), "--preset", preset, "--encoder", encoder);
  }

  if (verbose) {
    args.push("--verbose");
  }

  /** @type {JobState} */
  const job = {
    id: jobId,
    status: "running",
    process: null,
    config: { uploadId, format, mode, crf, preset, encoder, workers, verbose, smartTolerance },
    outputDir: jobOutputDir,
    logs: [],
    phase: "starting",
    percent: 0,
    errorMessage: null,
    createdAt: new Date(),
  };
  jobs.set(jobId, job);

  const child = spawn(COORDINATOR_BIN, args, {
    env: { ...process.env, DYLD_LIBRARY_PATH: LIB_DIR },
    stdio: ["ignore", "pipe", "pipe"],
  });
  job.process = child;

  child.stderr.on("data", (data) => {
    const text = data.toString();
    for (const line of text.split("\n")) {
      if (!line.trim()) continue;
      pushLog(jobId, line);
      parsePhase(jobId, line);
    }
  });

  child.stdout.on("data", (data) => {
    // stdout is generally not used for progress, but capture it
    const text = data.toString();
    for (const line of text.split("\n")) {
      if (line.trim()) pushLog(jobId, line);
    }
  });

  child.on("close", (code) => {
    job.process = null;
    if (job.status === "cancelled") return; // already handled

    if (code === 0) {
      job.status = "complete";
      job.phase = "complete";
      job.percent = 100;
      const outputFiles = listOutputFiles(jobOutputDir);
      broadcast(jobId, { type: "complete", jobId, outputFiles });
    } else {
      job.status = "error";
      job.phase = "error";
      job.errorMessage = `Coordinator exited with code ${code}`;
      broadcast(jobId, { type: "error", jobId, message: job.errorMessage });
    }
  });

  child.on("error", (err) => {
    job.process = null;
    job.status = "error";
    job.phase = "error";
    job.errorMessage = `Failed to start coordinator: ${err.message}`;
    broadcast(jobId, { type: "error", jobId, message: job.errorMessage });
  });

  res.json({ jobId, status: "running" });
});

// Analyze (smart report)
app.post("/api/analyze", async (req, res) => {
  const {
    uploadId,
    crf = 23,
    encoder = "libx264",
    smartTolerance = 0.3,
  } = req.body;

  if (!uploadId) {
    return res.status(400).json({ error: "uploadId is required" });
  }

  const inputPath = path.join(UPLOAD_DIR, path.basename(uploadId));
  if (!fs.existsSync(inputPath)) {
    return res.status(404).json({ error: "Uploaded file not found" });
  }

  const tmpOutputDir = path.join(OUTPUT_DIR, `analyze_${Date.now()}`);
  fs.mkdirSync(tmpOutputDir, { recursive: true });

  const args = [
    "--input", inputPath,
    "--output", tmpOutputDir,
    "--smart-report",
    "--smart-tolerance", String(smartTolerance),
    "--crf", String(crf),
    "--encoder", encoder,
  ];

  try {
    const result = await runCoordinatorSync(args);
    // Clean up temp directory
    fsp.rm(tmpOutputDir, { recursive: true, force: true }).catch(() => {});
    // stdout should contain the JSON report
    try {
      const report = JSON.parse(result.stdout);
      res.json(report);
    } catch {
      res.json({ raw: result.stdout, stderr: result.stderr });
    }
  } catch (err) {
    fsp.rm(tmpOutputDir, { recursive: true, force: true }).catch(() => {});
    res.status(500).json({ error: err.message });
  }
});

// List all jobs
app.get("/api/jobs", (_req, res) => {
  const list = [];
  for (const job of jobs.values()) {
    list.push(jobSummary(job));
  }
  res.json(list);
});

// Single job
app.get("/api/jobs/:id", (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) return res.status(404).json({ error: "Job not found" });
  res.json(jobSummary(job));
});

// Job output files
app.get("/api/jobs/:id/files", (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) return res.status(404).json({ error: "Job not found" });
  const files = listOutputFiles(job.outputDir);
  res.json(files);
});

// Cancel / remove job
app.delete("/api/jobs/:id", async (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) return res.status(404).json({ error: "Job not found" });

  // Kill child process if still running
  if (job.process) {
    job.status = "cancelled";
    job.phase = "cancelled";
    job.process.kill("SIGTERM");
    job.process = null;
    broadcast(job.id, { type: "error", jobId: job.id, message: "Job cancelled" });
  }

  // Clean up output files
  fsp.rm(job.outputDir, { recursive: true, force: true }).catch(() => {});

  jobs.delete(req.params.id);
  subscribers.delete(req.params.id);
  res.json({ deleted: true });
});

// Download output file
app.get("/api/download/:jobId/:filename", (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) return res.status(404).json({ error: "Job not found" });

  // Path join safety: ensure filename doesn't escape the output dir
  const filename = path.basename(req.params.filename);
  const filePath = path.join(job.outputDir, filename);

  if (!filePath.startsWith(job.outputDir)) {
    return res.status(403).json({ error: "Invalid filename" });
  }

  if (!fs.existsSync(filePath)) {
    return res.status(404).json({ error: "File not found" });
  }

  res.download(filePath, filename);
});

// ---------------------------------------------------------------------------
// HTTP + WebSocket server
// ---------------------------------------------------------------------------
const server = http.createServer(app);
const wss = new WebSocketServer({ noServer: true });

server.on("upgrade", (request, socket, head) => {
  const { pathname } = new URL(request.url, `http://${request.headers.host}`);
  if (pathname === "/ws") {
    wss.handleUpgrade(request, socket, head, (ws) => {
      wss.emit("connection", ws, request);
    });
  } else {
    socket.destroy();
  }
});

wss.on("connection", (ws) => {
  ws.on("message", (data) => {
    try {
      const msg = JSON.parse(data.toString());
      if (msg.type === "subscribe" && msg.jobId) {
        let subs = subscribers.get(msg.jobId);
        if (!subs) {
          subs = new Set();
          subscribers.set(msg.jobId, subs);
        }
        subs.add(ws);

        // Send current state to the newly subscribed client
        const job = jobs.get(msg.jobId);
        if (job) {
          safeSend(ws, {
            type: "progress",
            jobId: msg.jobId,
            phase: job.phase,
            percent: job.percent,
            message: `Subscribed — current phase: ${job.phase}`,
          });
          // Send recent logs
          for (const line of job.logs) {
            safeSend(ws, { type: "log", jobId: msg.jobId, line });
          }
        }
      }
    } catch {
      // ignore malformed messages
    }
  });

  ws.on("close", () => {
    // Remove from all subscriber sets
    for (const subs of subscribers.values()) {
      subs.delete(ws);
    }
  });
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Push a log line to job history and broadcast it. */
function pushLog(jobId, line) {
  const job = jobs.get(jobId);
  if (!job) return;
  job.logs.push(line);
  if (job.logs.length > MAX_LOG_LINES) {
    job.logs.shift();
  }
  broadcast(jobId, { type: "log", jobId, line });
}

/**
 * Parse coordinator stderr lines to extract phase information.
 * The coordinator uses the `tracing` crate, so lines look like:
 *   2024-01-15T10:30:00.000Z  INFO coordinator: Analyzing video ...
 */
function parsePhase(jobId, line) {
  const job = jobs.get(jobId);
  if (!job) return;

  const lower = line.toLowerCase();

  if (lower.includes("analyzing video") || lower.includes("analyzing")) {
    job.phase = "analyzing";
    job.percent = 10;
  } else if (lower.includes("creating segments") || lower.includes("segments")) {
    // Check for segment count patterns like "Created N segments"
    const match = line.match(/(\d+)\s+segments/i);
    if (match) {
      job.phase = "splitting";
      job.percent = 25;
    }
  } else if (lower.includes("pre-split") || lower.includes("presplit_") || lower.includes("splitting")) {
    job.phase = "splitting";
    job.percent = 30;
    // Try to extract progress from bar patterns like "3/10 segments"
    const match = line.match(/(\d+)\/(\d+)\s+segments/);
    if (match) {
      const done = parseInt(match[1], 10);
      const total = parseInt(match[2], 10);
      if (total > 0) {
        job.percent = 25 + Math.round((done / total) * 15); // 25-40%
      }
    }
  } else if (lower.includes("segment_") || lower.includes("transcoding") || lower.includes("encoding")) {
    job.phase = "encoding";
    // Try to extract progress
    const match = line.match(/(\d+)\/(\d+)\s+segments/);
    if (match) {
      const done = parseInt(match[1], 10);
      const total = parseInt(match[2], 10);
      if (total > 0) {
        job.percent = 40 + Math.round((done / total) * 45); // 40-85%
      }
    } else if (job.percent < 40) {
      job.percent = 40;
    }
  } else if (lower.includes("playlist") || lower.includes("concatenat") || lower.includes("assembling")) {
    job.phase = "finalizing";
    job.percent = 90;
  } else if (lower.includes("pipeline complete")) {
    job.phase = "complete";
    job.percent = 100;
  }

  broadcast(jobId, {
    type: "progress",
    jobId,
    phase: job.phase,
    percent: job.percent,
    message: line.trim(),
  });
}

/** Broadcast a message to all subscribers of a job. */
function broadcast(jobId, message) {
  const subs = subscribers.get(jobId);
  if (!subs) return;
  const payload = JSON.stringify(message);
  for (const ws of subs) {
    if (ws.readyState === ws.OPEN) {
      ws.send(payload);
    }
  }
}

/** Send a message to a single WebSocket, swallowing errors. */
function safeSend(ws, message) {
  if (ws.readyState === ws.OPEN) {
    ws.send(JSON.stringify(message));
  }
}

/** Build a job summary (no process handle or internal state). */
function jobSummary(job) {
  return {
    id: job.id,
    status: job.status,
    phase: job.phase,
    percent: job.percent,
    config: job.config,
    errorMessage: job.errorMessage,
    createdAt: job.createdAt,
    logCount: job.logs.length,
  };
}

/** List files in a job's output directory. */
function listOutputFiles(dir) {
  try {
    return fs.readdirSync(dir).map((name) => {
      const stat = fs.statSync(path.join(dir, name));
      return { name, size: stat.size };
    });
  } catch {
    return [];
  }
}

/** Run the coordinator synchronously and return stdout/stderr. */
function runCoordinatorSync(args) {
  return new Promise((resolve, reject) => {
    const child = spawn(COORDINATOR_BIN, args, {
      env: { ...process.env, DYLD_LIBRARY_PATH: LIB_DIR },
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (d) => { stdout += d.toString(); });
    child.stderr.on("data", (d) => { stderr += d.toString(); });

    child.on("close", (code) => {
      if (code === 0) {
        resolve({ stdout, stderr });
      } else {
        const err = new Error(`Coordinator exited with code ${code}`);
        err.stderr = stderr;
        err.stdout = stdout;
        reject(err);
      }
    });

    child.on("error", (err) => {
      reject(new Error(`Failed to start coordinator: ${err.message}`));
    });
  });
}

// ---------------------------------------------------------------------------
// Graceful shutdown
// ---------------------------------------------------------------------------
function shutdown(signal) {
  console.log(`\n${signal} received — shutting down...`);

  // Kill all running coordinator processes
  for (const job of jobs.values()) {
    if (job.process) {
      job.status = "cancelled";
      job.phase = "cancelled";
      job.process.kill("SIGTERM");
      job.process = null;
    }
  }

  // Close all WebSocket connections
  for (const client of wss.clients) {
    client.close(1001, "Server shutting down");
  }

  // Stop accepting new connections, then exit
  server.close(() => {
    // Remove PID file
    try { fs.unlinkSync(PID_FILE); } catch {}
    console.log("Server stopped.");
    process.exit(0);
  });

  // Force exit after 5 seconds if graceful shutdown stalls
  setTimeout(() => {
    try { fs.unlinkSync(PID_FILE); } catch {}
    console.error("Forced exit after timeout.");
    process.exit(1);
  }, 5000);
}

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
server.listen(PORT, () => {
  // Write PID file so `npm run web:stop` can find us
  fs.writeFileSync(PID_FILE, String(process.pid));
  console.log(`Parallel Transcoder web server listening on http://localhost:${PORT} (pid ${process.pid})`);
});
