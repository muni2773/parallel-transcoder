const { contextBridge, ipcRenderer } = require('electron');

const stateListeners = new Set();
const logListeners = new Set();
const k8sLogListeners = new Set();
const k8sBusyListeners = new Set();

ipcRenderer.on('cluster:state', (_e, state) => {
  stateListeners.forEach((fn) => { try { fn(state); } catch {} });
});

ipcRenderer.on('cluster:log', (_e, entry) => {
  logListeners.forEach((fn) => { try { fn(entry); } catch {} });
});

ipcRenderer.on('k8s:log', (_e, entry) => {
  k8sLogListeners.forEach((fn) => { try { fn(entry); } catch {} });
});

ipcRenderer.on('k8s:busy', (_e, payload) => {
  k8sBusyListeners.forEach((fn) => { try { fn(payload); } catch {} });
});

contextBridge.exposeInMainWorld('desktop', {
  isDesktop: true,
  platform: process.platform,
  versions: {
    electron: process.versions.electron,
    node: process.versions.node,
    chrome: process.versions.chrome,
  },
  cluster: {
    startNode: (opts) => ipcRenderer.invoke('cluster:start-node', opts),
    stopNode: () => ipcRenderer.invoke('cluster:stop-node'),
    getNodeState: () => ipcRenderer.invoke('cluster:node-state'),
    getLogs: (limit) => ipcRenderer.invoke('cluster:node-logs', limit),
    onState: (cb) => {
      stateListeners.add(cb);
      return () => stateListeners.delete(cb);
    },
    onLog: (cb) => {
      logListeners.add(cb);
      return () => logListeners.delete(cb);
    },
  },
  k8s: {
    checkTools: () => ipcRenderer.invoke('k8s:check-tools'),
    getStatus: () => ipcRenderer.invoke('k8s:get-status'),
    createCluster: (opts) => ipcRenderer.invoke('k8s:create-cluster', opts),
    deleteCluster: () => ipcRenderer.invoke('k8s:delete-cluster'),
    scale: (replicas) => ipcRenderer.invoke('k8s:scale', replicas),
    deletePod: (name) => ipcRenderer.invoke('k8s:delete-pod', name),
    portForwardStart: (localPort) => ipcRenderer.invoke('k8s:port-forward-start', localPort),
    portForwardStop: () => ipcRenderer.invoke('k8s:port-forward-stop'),
    getLogs: (limit) => ipcRenderer.invoke('k8s:get-logs', limit),
    onLog: (cb) => {
      k8sLogListeners.add(cb);
      return () => k8sLogListeners.delete(cb);
    },
    onBusy: (cb) => {
      k8sBusyListeners.add(cb);
      return () => k8sBusyListeners.delete(cb);
    },
  },
});
