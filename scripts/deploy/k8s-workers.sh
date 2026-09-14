#!/usr/bin/env bash
# Deploy the workers to a Kubernetes cluster (k3s, or any other): one worker per
# node, the GPU or the CPU image picked from each node's own labels.
#
#   scripts/deploy/k8s-workers.sh render           write deploy/k8s/local/ from .env and the cluster
#   scripts/deploy/k8s-workers.sh apply            render, then apply it
#   scripts/deploy/k8s-workers.sh status           every node, its kind, and its worker
#   scripts/deploy/k8s-workers.sh targets          rewrite the Prometheus targets from the pods
#   scripts/deploy/k8s-workers.sh restart <node>   restart one node's worker
#   scripts/deploy/k8s-workers.sh delete           remove the workers, config and secrets
#
# GPU nodes are the ones with an nvidia.com/gpu.count label, which NVIDIA's GPU
# feature discovery sets (standalone with the device plugin, or in the GPU Operator).
# Kubernetes gives a pod a fixed number of GPUs, so `render` writes one GPU
# DaemonSet per count it finds; run `apply` again after adding a node with a count
# the cluster did not have. Every other node gets the CPU worker. Label a node
# atelier/worker=false to keep workers off it.
#
# Settings, as environment variables:
#   CONTROL_HOST   the address the nodes reach the control machine at (required, or
#                  ATELIER_CONTROL_HOST in ENV_FILE)
#   ENV_FILE       where the secrets come from                default .env
#   MATERIALS_NFS  server:/export/path (or ATELIER_MATERIALS_NFS in ENV_FILE): an
#                  init container copies the materials from it into the node's folder
#                  before each worker starts — needs nfs-common on every node
#   PULL_SECRET    name of a docker-registry secret in the atelier namespace, for a
#                  registry that needs a login
#   RUNTIME_CLASS  the GPU pods' runtime class; default nvidia when the cluster has
#                  it, none otherwise
#   GPU_COUNTS     e.g. "8 4": render without asking the cluster
#   KUBECTL        default kubectl
#
# experiments/ and materials are the nodes' own /srv/atelier folders, as with the
# SSH deploy: keep experiments/ identical with scripts/deploy/deploy-workers.sh sync,
# and set ATELIER_MATERIALS_NFS to have the materials copied in.
set -euo pipefail
cd "$(dirname "$0")/../.."

ENV_FILE="${ENV_FILE:-.env}"
KUBECTL="${KUBECTL:-kubectl}"
NS=atelier
OUT=deploy/k8s/local
LABEL=app.kubernetes.io/name=atelier-worker

say()  { printf '\033[36m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32m%s\033[0m\n' "$*"; }
warn() { printf '    \033[33m%s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }
k()    { $KUBECTL "$@"; }

envget() {  # a value from ENV_FILE, without sourcing it
  local v=""
  [[ -f "$ENV_FILE" ]] && v=$(grep -E "^[[:space:]]*$1=" "$ENV_FILE" | tail -1 | cut -d= -f2-) || true
  v="${v%$'\r'}"
  if [[ "$v" == \"*\" || "$v" == \'*\' ]]; then v="${v:1:${#v}-2}"; fi
  printf '%s' "${v:-${2:-}}"
}

# An init container that copies the materials from NFS into the node's folder before
# the worker starts: all of them on a new node, what changed on the share otherwise,
# never deleting. Same script as deploy-workers.sh; backticks rather than $(…), which
# the kubelet would read as a variable reference. Needs the NFS client (nfs-common)
# on every node, since the kubelet mounts the share.
materials_patch() {  # materials_patch <image> <server> <path>
  cat <<EOF
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: patched-by-target
spec:
  template:
    spec:
      initContainers:
        - name: materials
          image: $1
          imagePullPolicy: IfNotPresent
          command: [sh, -c]
          args:
            - |
              set -e
              need=\`du -sb /nfs | cut -f1\`
              used=\`du -sb /dst | cut -f1\`
              free=\`df -B1 --output=avail /dst | tail -1\`
              if [ "\$need" -gt \`expr \$used + \$free\` ]; then
                echo "the materials are \`expr \$need / 1048576\` MB and this node has room for \`expr \\( \$used + \$free \\) / 1048576\` MB" >&2
                exit 3
              fi
              cp -au /nfs/. /dst/
              echo "materials on this node: \`du -sh /dst | cut -f1\`"
          volumeMounts:
            - {name: materials-nfs, mountPath: /nfs, readOnly: true}
            - {name: materials, mountPath: /dst}
      volumes:
        - name: materials-nfs
          nfs: {server: "$2", path: "$3", readOnly: true}
EOF
}

# kubectl is supported within one minor version of the API server
skew_check() {
  local client server
  client=$(k version -o json 2>/dev/null | grep -A3 '"clientVersion"' | sed -n 's/.*"minor": *"\([0-9]*\).*/\1/p' | head -1)
  server=$(k version -o json 2>/dev/null | grep -A3 '"serverVersion"' | sed -n 's/.*"minor": *"\([0-9]*\).*/\1/p' | head -1)
  if [[ -n "$client" && -n "$server" ]] && (( client - server > 1 || server - client > 1 )); then
    warn "kubectl 1.$client against a 1.$server cluster is outside the supported skew of one minor version — use kubectl 1.$((server - 1))–1.$((server + 1))"
  fi
}

render() {
  local counts registry tag n missing runtime
  CONTROL_HOST="${CONTROL_HOST:-$(envget ATELIER_CONTROL_HOST)}"
  [[ -n "$CONTROL_HOST" ]] || die "set CONTROL_HOST to the address the nodes reach the control machine at, e.g. CONTROL_HOST=192.168.1.20 $0 render"
  [[ -n "$(envget ATELIER_WORKER_TOKEN)" ]] || die "$ENV_FILE has no ATELIER_WORKER_TOKEN"

  say "GPU nodes"
  if [[ -n "${GPU_COUNTS:-}" ]]; then
    counts=$(tr ' ' '\n' <<< "$GPU_COUNTS" | grep -E '^[0-9]+$' | sort -un || true)
    runtime="${RUNTIME_CLASS-nvidia}"
  else
    k get nodes >/dev/null || die "$KUBECTL cannot reach the cluster"
    skew_check
    counts=$(k get nodes -l nvidia.com/gpu.count -o jsonpath='{range .items[*]}{.metadata.labels.nvidia\.com/gpu\.count}{"\n"}{end}' | grep -E '^[0-9]+$' | sort -un || true)
    # the GPU Operator's own label, on a node GFD has not labelled yet
    missing=$(k get nodes -l 'nvidia.com/gpu.present=true,!nvidia.com/gpu.count' -o name)
    [[ -z "$missing" ]] || warn "these have GPUs but no nvidia.com/gpu.count yet, so they get the CPU worker until GPU feature discovery labels them: $missing"
    if [[ -z "$counts" ]]; then
      warn "no node has an nvidia.com/gpu.count label — every node gets the CPU worker. Deploy NVIDIA's device plugin with GPU feature discovery (or the GPU Operator) for the GPUs."
    fi
    if [[ -n "${RUNTIME_CLASS+set}" ]]; then runtime="$RUNTIME_CLASS"
    elif k get runtimeclass nvidia >/dev/null 2>&1; then runtime=nvidia
    else
      runtime=""
      [[ -z "$counts" ]] || warn "no RuntimeClass 'nvidia': the GPU pods use the nodes' default runtime, which must then be the nvidia one (nvidia-ctk runtime configure --runtime=containerd --set-as-default)"
    fi
  fi
  for n in $counts; do ok "nodes with $n GPU(s)"; done
  [[ -z "$counts" ]] || ok "runtime class: $([[ -n "$runtime" ]] && echo "$runtime" || echo "none, the nodes' default runtime")"

  say "writing $OUT/"
  mkdir -p "$OUT"
  rm -f "$OUT"/worker-gpu-*.yaml
  registry="$(envget ATELIER_REGISTRY)"; tag="$(envget ATELIER_TAG latest)"

  cat > "$OUT/config.env" <<EOF
ATELIER_ROLE=worker
ATELIER_CONTROL_HOST=$CONTROL_HOST
ATELIER_HTTP_PORT=$(envget ATELIER_HTTP_PORT 80)
ATELIER_DB_PORT=$(envget ATELIER_DB_PORT 5432)
ATELIER_REDIS_PORT=$(envget ATELIER_REDIS_PORT 6379)
ATELIER_STORAGE_MODE=$(envget ATELIER_STORAGE_MODE upload)
ATELIER_UPLOAD_MAX_MB=$(envget ATELIER_UPLOAD_MAX_MB 64)
ATELIER_ROOT=/srv/atelier
ATELIER_EXPERIMENTS_DIR=/srv/atelier/experiments
ATELIER_MATERIALS_DIR=/srv/atelier/materials
ATELIER_DATA_DIR=/srv/atelier/data
ATELIER_RUNNER_BACKEND=subprocess
ATELIER_WORKER_METRICS_PORT=9101
ATELIER_CPU_SLOTS=$(envget ATELIER_CPU_SLOTS 1)
EOF
  (
    umask 077
    cat > "$OUT/secrets.env" <<EOF
ATELIER_WORKER_TOKEN=$(envget ATELIER_WORKER_TOKEN)
ATELIER_DB_PASSWORD=$(envget ATELIER_DB_PASSWORD atelier)
ATELIER_REDIS_PASSWORD=$(envget ATELIER_REDIS_PASSWORD atelier)
EOF
  )
  for n in $counts; do
    if [[ -n "$runtime" ]]; then
      sed -e "s/__GPUS__/$n/g" -e "s/__RUNTIME_CLASS__/$runtime/" deploy/k8s/worker-gpu.template.yaml
    else
      sed -e "s/__GPUS__/$n/g" -e '/runtimeClassName: __RUNTIME_CLASS__/d' deploy/k8s/worker-gpu.template.yaml
    fi > "$OUT/worker-gpu-$n.yaml"
  done

  rm -f "$OUT"/materials-*.patch.yaml
  local nfs server path
  nfs="${MATERIALS_NFS:-$(envget ATELIER_MATERIALS_NFS)}"
  if [[ -n "$nfs" ]]; then
    server="${nfs%%:*}"; path="${nfs#*:}"
    [[ "$server" != "$nfs" && "$path" == /* ]] || die "ATELIER_MATERIALS_NFS must look like server:/export/path, not '$nfs'"
    materials_patch atelier-worker-cpu "$server" "$path" > "$OUT/materials-cpu.patch.yaml"
    materials_patch atelier-worker "$server" "$path" > "$OUT/materials-gpu.patch.yaml"
    ok "materials copied from $nfs by an init container before each worker starts"
  fi

  {
    echo "# Written by scripts/deploy/k8s-workers.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ). Re-render rather than edit."
    echo "apiVersion: kustomize.config.k8s.io/v1beta1"
    echo "kind: Kustomization"
    echo "namespace: $NS"
    echo "labels:"
    echo "  - pairs: {app.kubernetes.io/part-of: atelier}"
    echo "resources:"
    echo "  - ../base"
    for n in $counts; do echo "  - worker-gpu-$n.yaml"; done
    echo "configMapGenerator:"
    echo "  - {name: atelier-worker-config, envs: [config.env]}"
    echo "secretGenerator:"
    echo "  - {name: atelier-worker-secrets, envs: [secrets.env]}"
    echo "images:"
    # without a registry the images must already be on each node:
    #   gunzip -c atelier-worker-cu128.tar.gz | sudo k3s ctr images import -
    echo "  - {name: atelier-worker, newName: ${registry}atelier-worker, newTag: \"$tag\"}"
    echo "  - {name: atelier-worker-cpu, newName: ${registry}atelier-worker-cpu, newTag: \"$tag\"}"
    if [[ -n "${PULL_SECRET:-}" || -n "$nfs" ]]; then echo "patches:"; fi
    if [[ -n "${PULL_SECRET:-}" ]]; then
      echo "  - target: {kind: DaemonSet}"
      echo "    patch: |-"
      echo "      - {op: add, path: /spec/template/spec/imagePullSecrets, value: [{name: $PULL_SECRET}]}"
    fi
    if [[ -n "$nfs" ]]; then
      echo "  - {path: materials-cpu.patch.yaml, target: {kind: DaemonSet, name: atelier-worker-cpu}}"
      echo "  - {path: materials-gpu.patch.yaml, target: {kind: DaemonSet, labelSelector: atelier/kind=gpu}}"
    fi
  } > "$OUT/kustomization.yaml"
  ok "kustomization.yaml, config.env, secrets.env$(for n in $counts; do printf ', worker-gpu-%s.yaml' "$n"; done)"
  RENDERED_COUNTS="$counts"
}

apply() {
  local ds n keep
  render
  say "applying"
  k apply -k "$OUT" | sed 's/^/    /'
  # GPU DaemonSets for counts the cluster no longer has
  for ds in $(k -n "$NS" get ds -l "$LABEL,atelier/kind=gpu" -o name); do
    keep=""
    for n in $RENDERED_COUNTS; do [[ "$ds" == "daemonset.apps/atelier-worker-gpu-$n" ]] && keep=1; done
    [[ -n "$keep" ]] || { k -n "$NS" delete "$ds" | sed 's/^/    /'; }
  done
  say "rolling out"
  for ds in $(k -n "$NS" get ds -l "$LABEL" -o name); do
    k -n "$NS" rollout status "$ds" --timeout=180s | sed 's/^/    /' || warn "$ds not ready yet — see: $0 status"
  done
  targets
}

status() {
  local pod node
  say "nodes"
  k get nodes -L nvidia.com/gpu.count,nvidia.com/gpu.product,atelier/worker | sed 's/^/    /'
  say "workers"
  k -n "$NS" get pods -l "$LABEL" -o wide | sed 's/^/    /'
  while read -r pod node; do
    [[ -n "$pod" ]] || continue
    printf '    %s: %s\n' "$node" "$(k -n "$NS" logs "$pod" 2>/dev/null | grep -E 'worker up on|no GPUs visible|refusing to start' | tail -1 || true)"
  done < <(k -n "$NS" get pods -l "$LABEL" -o jsonpath='{range .items[*]}{.metadata.name} {.spec.nodeName}{"\n"}{end}')
}

targets() {
  local out=deploy/prometheus/targets/gpu-nodes.yml node ip
  say "Prometheus targets"
  {
    echo "# Written by scripts/deploy/k8s-workers.sh from the worker pods on $(date -u +%Y-%m-%dT%H:%M:%SZ)."
    echo "# Port 9101 is each worker's metrics listener (a hostPort); Prometheus re-reads this within 30 s."
    while read -r node ip; do
      [[ -n "$ip" ]] || continue
      printf -- '- targets: ["%s:9101"]\n  labels:\n    node: %s\n    role: worker\n' "$ip" "$node"
    done < <(k -n "$NS" get pods -l "$LABEL" -o jsonpath='{range .items[*]}{.spec.nodeName} {.status.hostIP}{"\n"}{end}' | sort)
  } > "$out"
  ok "wrote $out"
}

cmd="${1:-}"; shift || true
case "$cmd" in
  render)  render ;;
  apply)   apply ;;
  status)  status ;;
  targets) targets ;;
  restart)
    [[ $# -eq 1 ]] || die "usage: $0 restart <node>"
    k -n "$NS" delete pod -l "$LABEL" --field-selector "spec.nodeName=$1" ;;
  delete)
    k -n "$NS" delete ds,configmap,secret -l app.kubernetes.io/part-of=atelier ;;
  *) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 2 ;;
esac
