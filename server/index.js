#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  ListToolsRequestSchema,
  CallToolRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs/promises";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const COORDINATOR_BIN = path.join(
  __dirname,
  "../bin/transcoder-coordinator"
);

// Active transcoding jobs
const activeJobs = new Map();

// Create MCP server
const server = new Server(
  {
    name: "parallel-video-transcoder",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Register tools list handler
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "transcode_video",
        description:
          "Transcode a video file using parallel processing with look-ahead optimization. Supports HLS and MP4 output formats.",
        inputSchema: {
          type: "object",
          properties: {
            input_path: {
              type: "string",
              description: "Path to the input video file",
            },
            output_path: {
              type: "string",
              description:
                "Path for output (directory for HLS, file for MP4)",
            },
            max_workers: {
              type: "number",
              description:
                "Number of parallel workers (0 = auto-detect CPU cores)",
              default: 0,
            },
            segment_duration: {
              type: "number",
              description: "Target segment duration in seconds",
              default: 10,
            },
            lookahead_frames: {
              type: "number",
              description: "Number of frames to analyze ahead",
              default: 40,
            },
            output_format: {
              type: "string",
              enum: ["hls", "mp4"],
              description: "Output format",
              default: "hls",
            },
          },
          required: ["input_path", "output_path"],
        },
      },
      {
        name: "analyze_video",
        description:
          "Analyze a video file and return detailed metadata including duration, resolution, keyframes, scene changes, and complexity.",
        inputSchema: {
          type: "object",
          properties: {
            input_path: {
              type: "string",
              description: "Path to the video file to analyze",
            },
          },
          required: ["input_path"],
        },
      },
      {
        name: "get_transcode_status",
        description:
          "Get the current status of an ongoing transcoding job.",
        inputSchema: {
          type: "object",
          properties: {
            job_id: {
              type: "string",
              description: "ID of the transcoding job",
            },
          },
          required: ["job_id"],
        },
      },
      {
        name: "cancel_transcode",
        description:
          "Cancel a running transcoding job and clean up temporary files.",
        inputSchema: {
          type: "object",
          properties: {
            job_id: {
              type: "string",
              description: "ID of the transcoding job to cancel",
            },
          },
          required: ["job_id"],
        },
      },
    ],
  };
});

// Register tool call handler
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case "transcode_video":
        return await transcodeVideo(args);
      case "analyze_video":
        return await analyzeVideo(args);
      case "get_transcode_status":
        return await getTranscodeStatus(args);
      case "cancel_transcode":
        return await cancelTranscode(args);
      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    return {
      content: [
        {
          type: "text",
          text: `Error: ${error.message}`,
        },
      ],
      isError: true,
    };
  }
});

async function transcodeVideo(args) {
  const {
    input_path,
    output_path,
    max_workers = 0,
    segment_duration = 10,
    lookahead_frames = 40,
    output_format = "hls",
  } = args;

  // Validate input file exists
  try {
    await fs.access(input_path);
  } catch {
    throw new Error(`Input file not found: ${input_path}`);
  }

  const jobId = `job_${Date.now()}`;

  // Spawn coordinator process
  const proc = spawn(COORDINATOR_BIN, [
    "--input",
    input_path,
    "--output",
    output_path,
    "--workers",
    max_workers.toString(),
    "--segment-duration",
    segment_duration.toString(),
    "--lookahead",
    lookahead_frames.toString(),
    "--format",
    output_format,
  ]);

  let stdout = "";
  let stderr = "";

  proc.stdout.on("data", (data) => {
    stdout += data.toString();
  });

  proc.stderr.on("data", (data) => {
    stderr += data.toString();
  });

  // Store job info
  activeJobs.set(jobId, {
    process: proc,
    input_path,
    output_path,
    status: "running",
    startTime: Date.now(),
  });

  return new Promise((resolve, reject) => {
    proc.on("close", (code) => {
      activeJobs.delete(jobId);

      if (code === 0) {
        resolve({
          content: [
            {
              type: "text",
              text: `✅ Transcoding completed successfully!\n\n**Output:** ${output_path}\n\n**Details:**\n${stdout}`,
            },
          ],
        });
      } else {
        reject(
          new Error(`Transcoding failed with exit code ${code}:\n${stderr}`)
        );
      }
    });

    proc.on("error", (error) => {
      activeJobs.delete(jobId);
      reject(new Error(`Failed to start coordinator: ${error.message}`));
    });
  });
}

async function analyzeVideo(args) {
  const { input_path } = args;

  // Validate input file exists
  try {
    await fs.access(input_path);
  } catch {
    throw new Error(`Input file not found: ${input_path}`);
  }

  // TODO: Run coordinator in analyze-only mode
  // For now, return placeholder
  return {
    content: [
      {
        type: "text",
        text: `Video analysis for: ${input_path}\n\n⚠️  Analysis feature not yet implemented.\n\nThis will return:\n- Duration, resolution, FPS\n- Keyframe positions\n- Scene change locations\n- Complexity estimates`,
      },
    ],
  };
}

async function getTranscodeStatus(args) {
  const { job_id } = args;

  const job = activeJobs.get(job_id);
  if (!job) {
    throw new Error(`Job not found: ${job_id}`);
  }

  const elapsedMs = Date.now() - job.startTime;
  const elapsedSecs = (elapsedMs / 1000).toFixed(1);

  return {
    content: [
      {
        type: "text",
        text: `**Job Status:** ${job.status}\n**Elapsed Time:** ${elapsedSecs}s\n**Input:** ${job.input_path}\n**Output:** ${job.output_path}`,
      },
    ],
  };
}

async function cancelTranscode(args) {
  const { job_id } = args;

  const job = activeJobs.get(job_id);
  if (!job) {
    throw new Error(`Job not found: ${job_id}`);
  }

  job.process.kill("SIGTERM");
  activeJobs.delete(job_id);

  return {
    content: [
      {
        type: "text",
        text: `✅ Transcoding job ${job_id} has been cancelled.`,
      },
    ],
  };
}

// Start server
const transport = new StdioServerTransport();
await server.connect(transport);
console.error("Parallel Video Transcoder MCP Server running...");
