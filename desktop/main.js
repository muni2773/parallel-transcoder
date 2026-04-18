const { app, BrowserWindow, shell, dialog } = require('electron');
const { spawn } = require('child_process');
const path = require('path');

const READY_TIMEOUT_MS = 15000;
const SHUTDOWN_GRACE_MS = 3000;

let serverProcess = null;
let mainWindow = null;
let allowedOrigin = null;
let isQuitting = false;

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
    await shutdownServer();
    app.exit(0);
  });
}
