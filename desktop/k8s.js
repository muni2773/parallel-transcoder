'use strict';

const path = require('path');
const fs = require('fs');

const K8S_LOG_BUFFER = 600;
const K8S_CLUSTER_NAME = 'transcoder';
const K8S_NAMESPACE = 'transcoder';
const K8S_STATEFULSET = 'transcoder-node';
const K8S_IMAGE = 'transcoder-node:dev';
const K8S_TOOL_TIMEOUT_MS = 10 * 60 * 1000;
const K8S_QUICK_TIMEOUT_MS = 30 * 1000;
const SHUTDOWN_GRACE_MS = 3000;
const POD_NAME_RE = /^[a-z0-9.\-]+$/i;

/**
 * Parse `kubectl get pods -o json` output into a flat list of pod summaries.
 * Exported separately from the manager so it's trivially testable.
 */
function parsePodList(json) {
  const items = Array.isArray(json && json.items) ? json.items : [];
  return items.map((p) => {
    const meta = p.metadata || {};
    const status = p.status || {};
    const spec = p.spec || {};
    const containerStatuses = Array.isArray(status.containerStatuses) ? status.containerStatuses : [];
    const ready = containerStatuses.length > 0 && containerStatuses.every((c) => c.ready);
    const restarts = containerStatuses.reduce((a, c) => a + (Number(c.restartCount) || 0), 0);
    const ordinal = (meta.labels || {})['apps.kubernetes.io/pod-index'] || null;
    return {
      name: meta.name || null,
      phase: status.phase || null,
      ready,
      restarts,
      startedAt: status.startTime || null,
      role: ordinal === '0' ? 'master' : 'worker',
      ordinal,
      node: spec.nodeName || null,
    };
  });
}

/** Validate a replica count for `kubectl scale`. Throws on invalid input. */
function validateReplicas(replicas) {
  const n = Number(replicas);
  if (!Number.isFinite(n) || n < 1 || n > 50 || Math.floor(n) !== n) {
    throw new Error('replicas must be an integer between 1 and 50');
  }
  return n;
}

function validatePodName(name) {
  if (typeof name !== 'string' || !POD_NAME_RE.test(name)) {
    throw new Error('Invalid pod name');
  }
  return name;
}

function validatePort(port) {
  const n = Number(port);
  if (!Number.isFinite(n) || n < 1 || n > 65535 || Math.floor(n) !== n) {
    throw new Error('Invalid local port');
  }
  return n;
}

function kubectlContextArg() {
  return `kind-${K8S_CLUSTER_NAME}`;
}

function kubectlBaseArgs() {
  return ['--context', kubectlContextArg(), '-n', K8S_NAMESPACE];
}

/**
 * Build a manager around an injected child_process surface.
 *
 * deps:
 *   spawn:     same shape as child_process.spawn
 *   execFile:  same shape as child_process.execFile
 *   repoRoot:  absolute path to the parallel-transcoder repo (for k8s/ + Dockerfile)
 *   platform:  process.platform (used for `which` / `where`)
 *   onLog:     optional (entry) => void — called for each log line we emit
 *   onBusy:    optional (label|null) => void — called when the busy state changes
 *   fsExists:  optional (path) => boolean — defaults to fs.existsSync
 */
function createK8sManager(deps) {
  if (!deps || typeof deps.spawn !== 'function' || typeof deps.execFile !== 'function') {
    throw new Error('createK8sManager requires { spawn, execFile }');
  }
  const spawn = deps.spawn;
  const execFile = deps.execFile;
  const repoRoot = deps.repoRoot || process.cwd();
  const platform = deps.platform || process.platform;
  const fsExists = deps.fsExists || fs.existsSync;
  const onLog = typeof deps.onLog === 'function' ? deps.onLog : () => {};
  const onBusy = typeof deps.onBusy === 'function' ? deps.onBusy : () => {};

  const logs = [];
  let busy = null;
  let portForwardProc = null;

  function pushLog(stream, line) {
    const entry = { t: Date.now(), stream, line };
    logs.push(entry);
    if (logs.length > K8S_LOG_BUFFER) logs.shift();
    onLog(entry);
  }

  function setBusy(label) {
    busy = label;
    onBusy(label);
  }

  function hasBin(bin) {
    return new Promise((resolve) => {
      const which = platform === 'win32' ? 'where' : 'which';
      execFile(which, [bin], { timeout: 5000 }, (err, stdout) => {
        if (err) return resolve(null);
        const p = String(stdout || '').split(/\r?\n/)[0].trim();
        resolve(p || null);
      });
    });
  }

  function runStreamed(cmd, args, opts = {}) {
    return new Promise((resolve, reject) => {
      pushLog('event', `$ ${cmd} ${args.join(' ')}`);
      const child = spawn(cmd, args, {
        cwd: opts.cwd,
        env: opts.env || process.env,
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      let stdout = '';
      let stderr = '';
      let timer = null;
      if (opts.timeout) {
        timer = setTimeout(() => {
          try { child.kill('SIGTERM'); } catch { /* ignore */ }
          pushLog('event', `command timed out after ${opts.timeout}ms`);
        }, opts.timeout);
      }
      const splitter = (stream) => {
        let buf = '';
        return (chunk) => {
          const text = chunk.toString('utf8');
          if (stream === 'stdout') stdout += text; else stderr += text;
          buf += text;
          let idx;
          while ((idx = buf.indexOf('\n')) !== -1) {
            const line = buf.slice(0, idx).replace(/\r$/, '');
            buf = buf.slice(idx + 1);
            if (line) pushLog(stream, line);
          }
        };
      };
      if (child.stdout) child.stdout.on('data', splitter('stdout'));
      if (child.stderr) child.stderr.on('data', splitter('stderr'));
      child.on('error', (err) => {
        if (timer) clearTimeout(timer);
        pushLog('event', `spawn error: ${err.message}`);
        reject(err);
      });
      child.on('exit', (code, signal) => {
        if (timer) clearTimeout(timer);
        pushLog('event', `exit code=${code} signal=${signal}`);
        resolve({ code: code == null ? -1 : code, signal, stdout, stderr });
      });
    });
  }

  async function checkTools() {
    const [kubectl, kind, docker] = await Promise.all([hasBin('kubectl'), hasBin('kind'), hasBin('docker')]);
    const dockerfile = path.join(repoRoot, 'Dockerfile');
    const kindOverlay = path.join(repoRoot, 'k8s', 'overlays', 'kind');
    return {
      kubectl,
      kind,
      docker,
      repoRoot,
      dockerfileExists: fsExists(dockerfile),
      kindOverlayExists: fsExists(path.join(kindOverlay, 'kustomization.yaml')),
      cluster: K8S_CLUSTER_NAME,
      namespace: K8S_NAMESPACE,
      statefulset: K8S_STATEFULSET,
      image: K8S_IMAGE,
    };
  }

  async function clusterExists() {
    try {
      const r = await runStreamed('kind', ['get', 'clusters'], { timeout: K8S_QUICK_TIMEOUT_MS });
      if (r.code !== 0) return false;
      return r.stdout.split(/\r?\n/).map((s) => s.trim()).includes(K8S_CLUSTER_NAME);
    } catch {
      return false;
    }
  }

  async function getPods() {
    const r = await runStreamed('kubectl', [...kubectlBaseArgs(), 'get', 'pods', '-o', 'json'], { timeout: K8S_QUICK_TIMEOUT_MS });
    if (r.code !== 0) throw new Error(`kubectl get pods failed (${r.code}): ${r.stderr.trim() || 'unknown'}`);
    let parsed;
    try { parsed = JSON.parse(r.stdout); }
    catch (e) { throw new Error(`Failed to parse kubectl output: ${e.message}`); }
    return parsePodList(parsed);
  }

  async function getReplicas() {
    const r = await runStreamed('kubectl', [
      ...kubectlBaseArgs(),
      'get', 'statefulset', K8S_STATEFULSET,
      '-o', 'jsonpath={.spec.replicas}',
    ], { timeout: K8S_QUICK_TIMEOUT_MS });
    if (r.code !== 0) return null;
    const n = parseInt(String(r.stdout).trim(), 10);
    return Number.isFinite(n) ? n : null;
  }

  async function getStatus() {
    const tools = await checkTools();
    const exists = tools.kind ? await clusterExists() : false;
    let pods = [];
    let replicas = null;
    let podsError = null;
    if (exists && tools.kubectl) {
      try { pods = await getPods(); }
      catch (e) { podsError = e.message; }
      try { replicas = await getReplicas(); }
      catch { /* ignore */ }
    }
    return { tools, clusterExists: exists, pods, replicas, podsError, busy };
  }

  async function createCluster(opts = {}) {
    if (busy) throw new Error(`Already running: ${busy}`);
    const tools = await checkTools();
    const missing = [];
    if (!tools.kind) missing.push('kind');
    if (!tools.kubectl) missing.push('kubectl');
    if (!tools.docker) missing.push('docker');
    if (missing.length) throw new Error(`Missing required tools: ${missing.join(', ')}`);
    if (!tools.dockerfileExists) throw new Error(`Dockerfile not found at ${repoRoot}/Dockerfile`);
    if (!tools.kindOverlayExists) throw new Error('k8s/overlays/kind/kustomization.yaml not found');

    const kindConfig = path.join(repoRoot, 'k8s', 'overlays', 'kind', 'kind-cluster.yaml');
    const overlayDir = path.join(repoRoot, 'k8s', 'overlays', 'kind');
    const skipBuild = !!opts.skipBuild;

    setBusy('Creating cluster');
    try {
      if (await clusterExists()) {
        pushLog('event', `kind cluster "${K8S_CLUSTER_NAME}" already exists — skipping create`);
      } else {
        setBusy('Creating kind cluster');
        const r = await runStreamed('kind', ['create', 'cluster', '--name', K8S_CLUSTER_NAME, '--config', kindConfig], { timeout: K8S_TOOL_TIMEOUT_MS });
        if (r.code !== 0) throw new Error(`kind create cluster failed with code ${r.code}`);
      }

      if (!skipBuild) {
        setBusy('Building image');
        const r2 = await runStreamed('docker', ['build', '-t', K8S_IMAGE, '.'], { cwd: repoRoot, timeout: K8S_TOOL_TIMEOUT_MS });
        if (r2.code !== 0) throw new Error(`docker build failed with code ${r2.code}`);

        setBusy('Loading image into kind');
        const r3 = await runStreamed('kind', ['load', 'docker-image', K8S_IMAGE, '--name', K8S_CLUSTER_NAME], { timeout: K8S_TOOL_TIMEOUT_MS });
        if (r3.code !== 0) throw new Error(`kind load docker-image failed with code ${r3.code}`);
      }

      setBusy('Applying overlay');
      const r4 = await runStreamed('kubectl', ['--context', kubectlContextArg(), 'apply', '-k', overlayDir], { timeout: K8S_TOOL_TIMEOUT_MS });
      if (r4.code !== 0) throw new Error(`kubectl apply -k failed with code ${r4.code}`);

      pushLog('event', 'cluster ready');
      return { ok: true };
    } finally {
      setBusy(null);
    }
  }

  async function deleteCluster() {
    if (busy) throw new Error(`Already running: ${busy}`);
    setBusy('Deleting cluster');
    try {
      await stopPortForward();
      const r = await runStreamed('kind', ['delete', 'cluster', '--name', K8S_CLUSTER_NAME], { timeout: K8S_TOOL_TIMEOUT_MS });
      if (r.code !== 0) throw new Error(`kind delete cluster failed with code ${r.code}`);
      return { ok: true };
    } finally {
      setBusy(null);
    }
  }

  async function scale(replicas) {
    const n = validateReplicas(replicas);
    setBusy(`Scaling to ${n} replicas`);
    try {
      const r = await runStreamed('kubectl', [
        ...kubectlBaseArgs(),
        'scale', `statefulset/${K8S_STATEFULSET}`,
        `--replicas=${n}`,
      ], { timeout: K8S_QUICK_TIMEOUT_MS });
      if (r.code !== 0) throw new Error(`kubectl scale failed with code ${r.code}`);
      return { ok: true, replicas: n };
    } finally {
      setBusy(null);
    }
  }

  async function deletePod(name) {
    const validated = validatePodName(name);
    setBusy(`Deleting pod ${validated}`);
    try {
      const r = await runStreamed('kubectl', [
        ...kubectlBaseArgs(),
        'delete', 'pod', validated,
      ], { timeout: K8S_QUICK_TIMEOUT_MS });
      if (r.code !== 0) throw new Error(`kubectl delete pod failed with code ${r.code}`);
      return { ok: true };
    } finally {
      setBusy(null);
    }
  }

  async function stopPortForward() {
    if (!portForwardProc || portForwardProc.exitCode !== null) {
      portForwardProc = null;
      return;
    }
    const proc = portForwardProc;
    proc.kill('SIGTERM');
    await new Promise((resolve) => {
      const t = setTimeout(() => {
        try { if (proc.exitCode === null) proc.kill('SIGKILL'); } catch { /* ignore */ }
        resolve();
      }, SHUTDOWN_GRACE_MS);
      proc.once('exit', () => { clearTimeout(t); resolve(); });
    });
    portForwardProc = null;
  }

  async function startPortForward(localPort) {
    await stopPortForward();
    const port = validatePort(localPort);
    pushLog('event', `$ kubectl --context ${kubectlContextArg()} -n ${K8S_NAMESPACE} port-forward svc/transcoder-master ${port}:9900`);
    const child = spawn('kubectl', [
      ...kubectlBaseArgs(),
      'port-forward', 'svc/transcoder-master',
      `${port}:9900`,
    ], { stdio: ['ignore', 'pipe', 'pipe'] });
    portForwardProc = child;
    const splitter = (stream) => {
      let buf = '';
      return (chunk) => {
        buf += chunk.toString('utf8');
        let idx;
        while ((idx = buf.indexOf('\n')) !== -1) {
          const line = buf.slice(0, idx).replace(/\r$/, '');
          buf = buf.slice(idx + 1);
          if (line) pushLog(stream, `[port-forward] ${line}`);
        }
      };
    };
    if (child.stdout) child.stdout.on('data', splitter('stdout'));
    if (child.stderr) child.stderr.on('data', splitter('stderr'));
    child.on('exit', (code, signal) => {
      pushLog('event', `port-forward exited code=${code} signal=${signal}`);
      if (portForwardProc === child) portForwardProc = null;
    });
    return { ok: true, pid: child.pid, localPort: port };
  }

  function getLogs(limit) {
    const n = Math.max(1, Math.min(Number(limit) || 200, K8S_LOG_BUFFER));
    return { logs: logs.slice(-n), busy };
  }

  function getBusy() { return busy; }

  return {
    checkTools,
    clusterExists,
    getPods,
    getReplicas,
    getStatus,
    createCluster,
    deleteCluster,
    scale,
    deletePod,
    startPortForward,
    stopPortForward,
    getLogs,
    getBusy,
  };
}

module.exports = {
  createK8sManager,
  parsePodList,
  validateReplicas,
  validatePodName,
  validatePort,
  K8S_LOG_BUFFER,
  K8S_CLUSTER_NAME,
  K8S_NAMESPACE,
  K8S_STATEFULSET,
  K8S_IMAGE,
};
