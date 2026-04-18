const { contextBridge, ipcRenderer } = require('electron');

const stateListeners = new Set();
const logListeners = new Set();

ipcRenderer.on('cluster:state', (_e, state) => {
  stateListeners.forEach((fn) => { try { fn(state); } catch {} });
});

ipcRenderer.on('cluster:log', (_e, entry) => {
  logListeners.forEach((fn) => { try { fn(entry); } catch {} });
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
});
