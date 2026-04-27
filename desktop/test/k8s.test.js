'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');
const path = require('node:path');

const {
  createK8sManager,
  parsePodList,
  validateReplicas,
  validatePodName,
  validatePort,
  K8S_CLUSTER_NAME,
  K8S_NAMESPACE,
  K8S_STATEFULSET,
} = require('../k8s');

// ---------------------------------------------------------------------------
// Fake child_process — records calls and lets each test scenario script the
// streams + exit code that runStreamed should observe.
// ---------------------------------------------------------------------------

function makeFakeChild({ stdout = '', stderr = '', exitCode = 0, emitError = null } = {}) {
  const child = new EventEmitter();
  child.stdout = new EventEmitter();
  child.stderr = new EventEmitter();
  child.exitCode = null;
  child.kill = () => true;
  // Schedule stream + exit on the next microtask so the manager has a chance
  // to attach 'data' / 'exit' listeners before they fire.
  queueMicrotask(() => {
    if (emitError) {
      child.emit('error', emitError);
      return;
    }
    if (stdout) child.stdout.emit('data', Buffer.from(stdout));
    if (stderr) child.stderr.emit('data', Buffer.from(stderr));
    child.exitCode = exitCode;
    child.emit('exit', exitCode, null);
  });
  return child;
}

/**
 * Build a fake spawn that pulls scripted responses off a queue keyed by
 * a substring match against `${cmd} ${args.join(' ')}`.
 *
 * scenarios is an array of { match, response } where match is a string or
 * regex and response is the makeFakeChild option bag (or a function that
 * receives [cmd, args] and returns one).
 */
function makeFakeSpawn(scenarios) {
  const calls = [];
  const remaining = scenarios.slice();
  function spawn(cmd, args, opts) {
    calls.push({ cmd, args, opts });
    const line = `${cmd} ${(args || []).join(' ')}`;
    const idx = remaining.findIndex((s) => {
      if (s.match instanceof RegExp) return s.match.test(line);
      return line.includes(s.match);
    });
    if (idx === -1) {
      throw new Error(`Unscripted spawn: ${line}`);
    }
    const s = remaining.splice(idx, 1)[0];
    const response = typeof s.response === 'function' ? s.response(cmd, args) : s.response;
    return makeFakeChild(response);
  }
  spawn.calls = calls;
  spawn.remaining = remaining;
  return spawn;
}

function makeFakeExecFile(map) {
  // map: { binName: pathOrNullForMissing }
  return function execFile(which, args, _opts, cb) {
    const target = args[0];
    const found = Object.prototype.hasOwnProperty.call(map, target) ? map[target] : null;
    if (found) cb(null, found + '\n', '');
    else cb(new Error(`not found: ${target}`), '', '');
  };
}

function buildManager(overrides = {}) {
  return createK8sManager({
    spawn: overrides.spawn || (() => { throw new Error('spawn not stubbed'); }),
    execFile: overrides.execFile || makeFakeExecFile({
      kubectl: '/usr/local/bin/kubectl',
      kind: '/usr/local/bin/kind',
      docker: '/usr/local/bin/docker',
    }),
    repoRoot: overrides.repoRoot || '/tmp/repo',
    platform: overrides.platform || 'linux',
    fsExists: overrides.fsExists || (() => true),
    onLog: overrides.onLog,
    onBusy: overrides.onBusy,
  });
}

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

test('validateReplicas accepts integers in [1,50]', () => {
  assert.equal(validateReplicas(1), 1);
  assert.equal(validateReplicas(3), 3);
  assert.equal(validateReplicas(50), 50);
  assert.equal(validateReplicas('7'), 7); // numeric strings are coerced
});

test('validateReplicas rejects out-of-range and non-integers', () => {
  assert.throws(() => validateReplicas(0));
  assert.throws(() => validateReplicas(-1));
  assert.throws(() => validateReplicas(51));
  assert.throws(() => validateReplicas(2.5));
  assert.throws(() => validateReplicas('abc'));
  assert.throws(() => validateReplicas(NaN));
});

test('validatePodName accepts DNS-1123-ish names and rejects shell metachars', () => {
  assert.equal(validatePodName('transcoder-node-0'), 'transcoder-node-0');
  assert.equal(validatePodName('abc.def-1'), 'abc.def-1');
  assert.throws(() => validatePodName(''));
  assert.throws(() => validatePodName('foo;rm -rf /'));
  assert.throws(() => validatePodName('foo bar'));
  assert.throws(() => validatePodName('foo$BAR'));
  assert.throws(() => validatePodName(null));
});

test('validatePort accepts 1..65535 ints', () => {
  assert.equal(validatePort(9900), 9900);
  assert.equal(validatePort(1), 1);
  assert.equal(validatePort(65535), 65535);
  assert.throws(() => validatePort(0));
  assert.throws(() => validatePort(65536));
  assert.throws(() => validatePort(9900.5));
  assert.throws(() => validatePort('not-a-port'));
});

test('parsePodList shapes kubectl JSON into flat summaries', () => {
  const json = {
    items: [
      {
        metadata: {
          name: 'transcoder-node-0',
          labels: { 'apps.kubernetes.io/pod-index': '0' },
        },
        status: {
          phase: 'Running',
          startTime: '2025-01-01T00:00:00Z',
          containerStatuses: [{ ready: true, restartCount: 1 }],
        },
        spec: { nodeName: 'transcoder-worker' },
      },
      {
        metadata: {
          name: 'transcoder-node-1',
          labels: { 'apps.kubernetes.io/pod-index': '1' },
        },
        status: {
          phase: 'Pending',
          containerStatuses: [
            { ready: false, restartCount: 0 },
            { ready: true, restartCount: 2 },
          ],
        },
        spec: {},
      },
      {
        metadata: { name: 'orphan' },
        status: {},
        spec: {},
      },
    ],
  };
  const out = parsePodList(json);
  assert.equal(out.length, 3);
  assert.deepEqual(out[0], {
    name: 'transcoder-node-0',
    phase: 'Running',
    ready: true,
    restarts: 1,
    startedAt: '2025-01-01T00:00:00Z',
    role: 'master',
    ordinal: '0',
    node: 'transcoder-worker',
  });
  assert.equal(out[1].role, 'worker');
  assert.equal(out[1].ready, false);    // not all containers ready
  assert.equal(out[1].restarts, 2);     // restarts summed
  assert.equal(out[2].role, 'worker');  // missing label defaults to worker
  assert.equal(out[2].ready, false);    // no statuses → not ready
});

test('parsePodList tolerates empty/null input', () => {
  assert.deepEqual(parsePodList(null), []);
  assert.deepEqual(parsePodList({}), []);
  assert.deepEqual(parsePodList({ items: null }), []);
});

// ---------------------------------------------------------------------------
// Manager-level behaviour
// ---------------------------------------------------------------------------

test('checkTools reports presence and manifest layout', async () => {
  const mgr = buildManager({
    execFile: makeFakeExecFile({ kubectl: '/k', kind: null, docker: '/d' }),
    fsExists: (p) => p.endsWith('Dockerfile'),
  });
  const tools = await mgr.checkTools();
  assert.equal(tools.kubectl, '/k');
  assert.equal(tools.kind, null);
  assert.equal(tools.docker, '/d');
  assert.equal(tools.dockerfileExists, true);
  assert.equal(tools.kindOverlayExists, false);
  assert.equal(tools.cluster, K8S_CLUSTER_NAME);
  assert.equal(tools.namespace, K8S_NAMESPACE);
  assert.equal(tools.statefulset, K8S_STATEFULSET);
});

test('createCluster surfaces missing-tools error', async () => {
  const mgr = buildManager({
    execFile: makeFakeExecFile({ kubectl: null, kind: null, docker: null }),
  });
  await assert.rejects(() => mgr.createCluster(), /Missing required tools/);
});

test('createCluster validates Dockerfile/overlay presence', async () => {
  const mgr = buildManager({
    fsExists: () => false,
  });
  await assert.rejects(() => mgr.createCluster(), /Dockerfile not found/);
});

test('createCluster runs kind→docker→kind→kubectl in order with correct args', async () => {
  // Pre-script: clusterExists() is called first (returns no clusters), then create, build, load, apply.
  const spawn = makeFakeSpawn([
    { match: 'kind get clusters',           response: { stdout: '' } },
    { match: 'kind create cluster --name',   response: { stdout: '' } },
    { match: 'docker build -t transcoder-node:dev .', response: { stdout: '' } },
    { match: 'kind load docker-image',       response: { stdout: '' } },
    { match: 'kubectl --context kind-transcoder apply -k', response: { stdout: '' } },
  ]);

  const busyEvents = [];
  const mgr = buildManager({
    spawn,
    onBusy: (label) => busyEvents.push(label),
  });

  const r = await mgr.createCluster({ skipBuild: false });
  assert.deepEqual(r, { ok: true });

  // All scripted responses consumed.
  assert.equal(spawn.remaining.length, 0);

  // Inspect the kind create invocation specifically.
  const create = spawn.calls.find((c) => c.cmd === 'kind' && c.args.includes('create'));
  assert.ok(create, 'expected kind create call');
  assert.deepEqual(create.args.slice(0, 4), ['create', 'cluster', '--name', K8S_CLUSTER_NAME]);
  assert.ok(create.args.includes('--config'));
  const cfgIdx = create.args.indexOf('--config');
  assert.equal(create.args[cfgIdx + 1], path.join('/tmp/repo', 'k8s', 'overlays', 'kind', 'kind-cluster.yaml'));

  // Docker build runs in repo root.
  const build = spawn.calls.find((c) => c.cmd === 'docker');
  assert.equal(build.opts.cwd, '/tmp/repo');

  // Apply targets the overlay dir.
  const apply = spawn.calls.find((c) => c.cmd === 'kubectl' && c.args.includes('apply'));
  assert.ok(apply.args.includes('-k'));
  assert.equal(apply.args[apply.args.indexOf('-k') + 1], path.join('/tmp/repo', 'k8s', 'overlays', 'kind'));

  // Busy state was set then cleared.
  assert.ok(busyEvents.length >= 2);
  assert.equal(busyEvents[busyEvents.length - 1], null);
  assert.equal(mgr.getBusy(), null);
});

test('createCluster with skipBuild omits docker build + kind load', async () => {
  const spawn = makeFakeSpawn([
    { match: 'kind get clusters',           response: { stdout: '' } },
    { match: 'kind create cluster --name',   response: { stdout: '' } },
    { match: 'kubectl --context kind-transcoder apply -k', response: { stdout: '' } },
  ]);
  const mgr = buildManager({ spawn });
  await mgr.createCluster({ skipBuild: true });
  assert.equal(spawn.remaining.length, 0);
  assert.equal(spawn.calls.filter((c) => c.cmd === 'docker').length, 0);
  assert.equal(spawn.calls.filter((c) => c.cmd === 'kind' && c.args.includes('load')).length, 0);
});

test('createCluster skips create when cluster already exists', async () => {
  const spawn = makeFakeSpawn([
    { match: 'kind get clusters',           response: { stdout: 'transcoder\n' } },
    { match: 'docker build -t',             response: { stdout: '' } },
    { match: 'kind load docker-image',      response: { stdout: '' } },
    { match: 'kubectl --context kind-transcoder apply -k', response: { stdout: '' } },
  ]);
  const mgr = buildManager({ spawn });
  await mgr.createCluster();
  assert.equal(spawn.remaining.length, 0);
  // No kind create call should have happened.
  assert.equal(spawn.calls.filter((c) => c.cmd === 'kind' && c.args.includes('create')).length, 0);
});

test('createCluster surfaces non-zero exit codes', async () => {
  const spawn = makeFakeSpawn([
    { match: 'kind get clusters',           response: { stdout: '' } },
    { match: 'kind create cluster',         response: { stdout: '', stderr: 'boom', exitCode: 2 } },
  ]);
  const mgr = buildManager({ spawn });
  await assert.rejects(() => mgr.createCluster(), /kind create cluster failed with code 2/);
});

test('scale rejects bad input before spawning anything', async () => {
  const spawn = makeFakeSpawn([]);
  const mgr = buildManager({ spawn });
  await assert.rejects(() => mgr.scale(0), /replicas must be an integer/);
  await assert.rejects(() => mgr.scale(100), /replicas must be an integer/);
  await assert.rejects(() => mgr.scale('foo'), /replicas must be an integer/);
  assert.equal(spawn.calls.length, 0);
});

test('scale issues kubectl scale with the right arguments', async () => {
  const spawn = makeFakeSpawn([
    { match: 'kubectl --context kind-transcoder -n transcoder scale', response: { stdout: '' } },
  ]);
  const mgr = buildManager({ spawn });
  const r = await mgr.scale(5);
  assert.deepEqual(r, { ok: true, replicas: 5 });
  const call = spawn.calls[0];
  assert.equal(call.cmd, 'kubectl');
  assert.deepEqual(call.args, [
    '--context', 'kind-transcoder',
    '-n', 'transcoder',
    'scale', `statefulset/${K8S_STATEFULSET}`,
    '--replicas=5',
  ]);
});

test('deletePod rejects shell-meta names without spawning', async () => {
  const spawn = makeFakeSpawn([]);
  const mgr = buildManager({ spawn });
  await assert.rejects(() => mgr.deletePod('foo;ls'), /Invalid pod name/);
  assert.equal(spawn.calls.length, 0);
});

test('deletePod issues kubectl delete with namespace + context', async () => {
  const spawn = makeFakeSpawn([
    { match: 'kubectl --context kind-transcoder -n transcoder delete pod transcoder-node-2', response: { stdout: '' } },
  ]);
  const mgr = buildManager({ spawn });
  const r = await mgr.deletePod('transcoder-node-2');
  assert.deepEqual(r, { ok: true });
});

test('getStatus combines tool detection, cluster presence, pods, and replicas', async () => {
  const podsJson = JSON.stringify({
    items: [
      {
        metadata: { name: 'transcoder-node-0', labels: { 'apps.kubernetes.io/pod-index': '0' } },
        status: { phase: 'Running', containerStatuses: [{ ready: true, restartCount: 0 }] },
        spec: { nodeName: 'kind-control-plane' },
      },
    ],
  });
  const spawn = makeFakeSpawn([
    { match: 'kind get clusters',                                  response: { stdout: 'transcoder\n' } },
    { match: 'kubectl --context kind-transcoder -n transcoder get pods -o json', response: { stdout: podsJson } },
    { match: 'kubectl --context kind-transcoder -n transcoder get statefulset transcoder-node', response: { stdout: '3' } },
  ]);
  const mgr = buildManager({ spawn });
  const status = await mgr.getStatus();
  assert.equal(status.clusterExists, true);
  assert.equal(status.replicas, 3);
  assert.equal(status.pods.length, 1);
  assert.equal(status.pods[0].role, 'master');
  assert.equal(status.podsError, null);
});

test('getStatus reports clusterExists=false when kind missing', async () => {
  const spawn = makeFakeSpawn([]); // no spawn calls at all
  const mgr = buildManager({
    spawn,
    execFile: makeFakeExecFile({ kubectl: '/k', kind: null, docker: '/d' }),
  });
  const status = await mgr.getStatus();
  assert.equal(status.clusterExists, false);
  assert.deepEqual(status.pods, []);
  assert.equal(spawn.calls.length, 0);
});

test('busy guard prevents concurrent destructive ops', async () => {
  // First create runs; while it's mid-flight we attempt a delete.
  let releaseCreate;
  const slowChild = new EventEmitter();
  slowChild.stdout = new EventEmitter();
  slowChild.stderr = new EventEmitter();
  slowChild.exitCode = null;
  slowChild.kill = () => true;

  let call = 0;
  const spawn = (_cmd, _args) => {
    call += 1;
    if (call === 1) {
      // kind get clusters — resolve immediately so createCluster proceeds to busy state.
      return makeFakeChild({ stdout: '' });
    }
    if (call === 2) {
      // kind create cluster — slow, so createCluster is mid-flight.
      releaseCreate = () => {
        slowChild.exitCode = 0;
        slowChild.emit('exit', 0, null);
      };
      return slowChild;
    }
    // Any subsequent call (e.g. kubectl apply) — fresh fast child each time.
    return makeFakeChild({ stdout: '' });
  };

  const mgr = buildManager({
    spawn,
    fsExists: () => true,
  });

  const createPromise = mgr.createCluster({ skipBuild: true });
  // Yield until busy state flips on (manager has to walk through clusterExists()).
  for (let i = 0; i < 10 && mgr.getBusy() == null; i++) {
    await new Promise((r) => setImmediate(r));
  }
  assert.ok(mgr.getBusy(), 'expected busy to be set while createCluster is in flight');

  await assert.rejects(() => mgr.deleteCluster(), /Already running/);

  releaseCreate();
  await createPromise.catch(() => { /* don't care if apply succeeds, just that guard fired */ });
});

test('logs are buffered and getLogs honors limit', async () => {
  const spawn = makeFakeSpawn([
    { match: 'kind get clusters', response: { stdout: 'transcoder\nother\n' } },
  ]);
  const mgr = buildManager({ spawn });
  await mgr.clusterExists();
  const { logs } = mgr.getLogs(50);
  // We should see at least the command-echo event and the exit event.
  assert.ok(logs.some((l) => l.line.startsWith('$ kind get clusters')));
  assert.ok(logs.some((l) => l.line.startsWith('exit code=')));
});
