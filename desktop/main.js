const { app, BrowserWindow, shell, dialog, ipcMain } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const os = require('os');

const READY_TIMEOUT_MS = 15000;
const SHUTDOWN_GRACE_MS = 3000;
const NODE_LOG_BUFFER = 400;

let serverProcess = null;
let mainWindow = null;
let allowedOrigin = null;
let isQuitting = false;

// Cluster node state
let nodeProcess = null;
let nodeState = {
  running: false,
  pid: null,
  listen: null,
  joinedMaster: null,
  name: null,
  startedAt: null,
};
const nodeLogs = [];

function log(...args) {
  console.log('[desktop]', ...args);
}

function isSafeExternal(url) {
  try {
    const u = new URL(url);
    return u.protocol === 'http:' || u.protocol === 'https:';
  } catch {
    return false;
  }
}

function resolvePaths() {
  if (app.isPackaged) {
    return {
      serverPath: path.join(process.resourcesPath, 'app', 'web', 'server.js'),
      resourcesDir: process.resourcesPath,
    };
  }
  return {
    serverPath: path.join(__dirname, '..', 'web', 'server.js'),
    resourcesDir: path.join(__dirname, '..'),
  };
}

function resolveNodeBinary() {
  const { resourcesDir } = resolvePaths();
  return {
    nodeBin: path.join(resourcesDir, 'bin', 'transcoder-node'),
    workerBin: path.join(resourcesDir, 'bin', 'transcoder-worker'),
    libDir: path.join(resourcesDir, 'lib') + path.sep,
  };
}

function pushNodeLog(stream, line) {
  const entry = { t: Date.now(), stream, line };
  nodeLogs.push(entry);
  if (nodeLogs.length > NODE_LOG_BUFFER) nodeLogs.shift();
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('cluster:log', entry);
  }
}

function emitNodeState() {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('cluster:state', nodeState);
  }
}

function startNode(opts = {}) {
  if (nodeProcess && nodeProcess.exitCode === null) {
    throw new Error('Cluster node already running');
  }
  const { nodeBin, workerBin, libDir } = resolveNodeBinary();

  const listen = String(opts.listen || '0.0.0.0:9900').trim();
  const name = (opts.name || os.hostname() || 'node').trim();
  const join = opts.join ? String(opts.join).trim() : null;
  const srtBasePort = Number.isFinite(opts.srtBasePort) ? opts.srtBasePort : 9910;
  const verbose = !!opts.verbose;

  const args = [
    '--listen', listen,
    '--name', name,
    '--worker-binary', workerBin,
    '--lib-dir', libDir,
    '--srt-base-port', String(srtBasePort),
  ];
  if (join) args.push('--join', join);
  if (verbose) args.push('--verbose');

  log('spawning cluster node', nodeBin, args.join(' '));

  const libPathKey = process.platform === 'darwin' ? 'DYLD_LIBRARY_PATH' : 'LD_LIBRARY_PATH';
  const child = spawn(nodeBin, args, {
    env: { ...process.env, [libPathKey]: libDir },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  let bufOut = '';
  let bufErr = '';
  const splitLines = (chunk, bufRef, stream) => {
    let buf = bufRef.value + chunk.toString('utf8');
    let idx;
    while ((idx = buf.indexOf('\n')) !== -1) {
      const line = buf.slice(0, idx).replace(/\r$/, '');
      buf = buf.slice(idx + 1);
      if (line) pushNodeLog(stream, line);
    }
    bufRef.value = buf;
  };
  const outRef = { value: bufOut };
  const errRef = { value: bufErr };

  child.stdout.on('data', (chunk) => splitLines(chunk, outRef, 'stdout'));
  child.stderr.on('data', (chunk) => splitLines(chunk, errRef, 'stderr'));

  child.on('exit', (code, signal) => {
    log('cluster node exited', { code, signal });
    pushNodeLog('event', `exited code=${code} signal=${signal}`);
    nodeProcess = null;
    nodeState = {
      running: false,
      pid: null,
      listen: null,
      joinedMaster: null,
      name: null,
      startedAt: null,
    };
    emitNodeState();
  });

  child.on('error', (err) => {
    log('cluster node spawn error', err.message);
    pushNodeLog('event', `spawn error: ${err.message}`);
  });

  nodeProcess = child;
  nodeState = {
    running: true,
    pid: child.pid,
    listen,
    joinedMaster: join,
    name,
    startedAt: Date.now(),
  };
  emitNodeState();
  return { ...nodeState };
}

async function stopNode() {
  if (!nodeProcess || nodeProcess.exitCode !== null) return { stopped: false };
  log('stopping cluster node');
  const proc = nodeProcess;
  proc.kill('SIGTERM');
  await new Promise((resolve) => {
    const t = setTimeout(() => {
      if (proc.exitCode === null) {
        try { proc.kill('SIGKILL'); } catch {}
      }
      resolve();
    }, SHUTDOWN_GRACE_MS);
    proc.once('exit', () => { clearTimeout(t); resolve(); });
  });
  return { stopped: true };
}

function startServer() {
  const { serverPath, resourcesDir } = resolvePaths();
  log('spawning server', serverPath);

  const child = spawn(process.execPath, [serverPath], {
    env: {
      ...process.env,
      DESKTOP_MODE: '1',
      TRANSCODER_RESOURCES_DIR: resourcesDir,
      ELECTRON_RUN_AS_NODE: '1',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  return new Promise((resolve, reject) => {
    let settled = false;
    let stdoutBuf = '';

    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      reject(new Error('Server did not become ready within 15s'));
    }, READY_TIMEOUT_MS);

    child.stdout.on('data', (chunk) => {
      stdoutBuf += chunk.toString('utf8');
      let idx;
      while ((idx = stdoutBuf.indexOf('\n')) !== -1) {
        const line = stdoutBuf.slice(0, idx).replace(/\r$/, '');
        stdoutBuf = stdoutBuf.slice(idx + 1);
        log('[server:out]', line);
        const m = line.match(/^__DESKTOP_READY__PORT=(\d+)$/);
        if (m && !settled) {
          settled = true;
          clearTimeout(timer);
          resolve({ child, port: Number(m[1]) });
        }
      }
    });

    child.stderr.on('data', (chunk) => {
      log('[server:err]', chunk.toString('utf8').trimEnd());
    });

    child.on('exit', (code, signal) => {
      log('server exited', { code, signal });
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        reject(new Error(`Server exited before ready (code=${code}, signal=${signal})`));
      } else if (!isQuitting) {
        dialog.showErrorBox('Transcoder server stopped', `The server exited unexpectedly (code=${code}, signal=${signal}).`);
        app.quit();
      }
    });
  });
}

function createWindow(port) {
  allowedOrigin = `http://127.0.0.1:${port}`;
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 1024,
    minHeight: 700,
    backgroundColor: '#0E0E12',
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (isSafeExternal(url)) shell.openExternal(url);
    return { action: 'deny' };
  });

  mainWindow.webContents.on('will-navigate', (e, url) => {
    try {
      if (new URL(url).origin !== allowedOrigin) e.preventDefault();
    } catch {
      e.preventDefault();
    }
  });

  mainWindow.on('closed', () => { mainWindow = null; });
  mainWindow.loadURL(allowedOrigin);
}

async function shutdownServer() {
  if (!serverProcess || serverProcess.exitCode !== null) return;
  log('shutting down server');
  serverProcess.kill('SIGTERM');
  await new Promise((resolve) => {
    const t = setTimeout(() => {
      if (serverProcess && serverProcess.exitCode === null) {
        log('SIGKILL fallback');
        serverProcess.kill('SIGKILL');
      }
      resolve();
    }, SHUTDOWN_GRACE_MS);
    serverProcess.once('exit', () => { clearTimeout(t); resolve(); });
  });
}

// IPC handlers for cluster node management
ipcMain.handle('cluster:start-node', async (_event, opts) => {
  try {
    return { ok: true, state: startNode(opts || {}) };
  } catch (err) {
    return { ok: false, error: err.message };
  }
});

ipcMain.handle('cluster:stop-node', async () => {
  try {
    const res = await stopNode();
    return { ok: true, ...res };
  } catch (err) {
    return { ok: false, error: err.message };
  }
});

ipcMain.handle('cluster:node-state', async () => ({ ok: true, state: nodeState }));

ipcMain.handle('cluster:node-logs', async (_event, limit) => {
  const n = Math.max(1, Math.min(Number(limit) || 100, NODE_LOG_BUFFER));
  return { ok: true, logs: nodeLogs.slice(-n) };
});

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });

  app.whenReady().then(async () => {
    try {
      const { child, port } = await startServer();
      serverProcess = child;
      log('server ready on port', port);
      createWindow(port);
    } catch (err) {
      log('startup failed:', err.message);
      dialog.showErrorBox('Failed to start Parallel Transcoder', err.message);
      app.quit();
    }
  });

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0 && allowedOrigin) {
      const port = Number(new URL(allowedOrigin).port);
      createWindow(port);
    }
  });

  app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit();
  });

  app.on('before-quit', async (e) => {
    if (isQuitting) return;
    isQuitting = true;
    e.preventDefault();
    await stopNode().catch(() => {});
    await shutdownServer();
    app.exit(0);
  });
}
