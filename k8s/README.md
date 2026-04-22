# Kubernetes deployment

StatefulSet-based deployment of the `transcoder-node` cluster daemon.

## Layout

```
k8s/
├── base/              # StatefulSet + headless Service + master Service +
│                      # ConfigMap + ServiceAccount — no assumptions about
│                      # node pools, GPU, or object storage.
└── overlays/
    ├── kind/          # Local dev on kind: in-cluster MinIO, CPU-only
    │                  # encoder, tight resource limits.
    └── cloud/         # EKS/GKE: NVIDIA + VAAPI + AVX-512 node pools
                       # selected via NFD labels, S3 via IRSA / Workload Identity.
```

## Kind workflow

```bash
# 1. Create a multi-node kind cluster.
kind create cluster --name transcoder --config k8s/overlays/kind/kind-cluster.yaml

# 2. Build the container image for the right architecture.
docker build -t transcoder-node:dev .

# 3. Load it into every kind node (kind uses its own container runtime).
kind load docker-image transcoder-node:dev --name transcoder

# 4. Apply the overlay.
kubectl apply -k k8s/overlays/kind

# 5. Watch the cluster come up.
kubectl -n transcoder get pods -w
```

Pod-0 bootstraps as deterministic master (`--k8s-mode` short-circuits the
bully election); pods 1..N join `transcoder-node-0.transcoder`. SIGTERM
triggers a clean `NodeLeave` broadcast before K8s kills the pod.

## Submitting a job

The coordinator submits jobs through the `transcoder-master` Service, which
endpoints to whichever pod has `statefulset.kubernetes.io/pod-name=transcoder-node-0`:

```bash
kubectl -n transcoder port-forward svc/transcoder-master 9900:9900
./bin/transcoder-coordinator --input video.mp4 --cluster --cluster-master 127.0.0.1:9900
```

## Encoder selection

`transcoder-node` picks its default encoder from env vars (see
`cluster/src/node.rs::detect_encoder_from_env`):

| `ENCODER_GPU` | `ENCODER_CPU_FEATURE` | Encoder chosen |
|---------------|-----------------------|----------------|
| `nvidia`      | —                     | `hevc_nvenc`   |
| `vaapi`       | —                     | `hevc_vaapi`   |
| `none`        | `avx512`              | `libsvtav1`    |
| `none`        | `baseline`            | `libx264`      |

The cloud overlay sets these via nodeAffinity onto labeled node pools; kind
hardcodes `none` / `baseline`.
