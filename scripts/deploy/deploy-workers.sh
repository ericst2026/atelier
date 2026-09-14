#!/usr/bin/env bash
# Deploy, update and inspect the worker on every node in deploy/workers, from the
# control machine, over SSH. A node needs nothing but SSH and Docker.
#
#   scripts/deploy/deploy-workers.sh check              can every node be deployed to?
#   scripts/deploy/deploy-workers.sh up                 deploy or update every node
#   scripts/deploy/deploy-workers.sh up gpu-a cpu-a     only these (name or ssh host)
#   scripts/deploy/deploy-workers.sh sync               copy experiments/ only, no restart
#   scripts/deploy/deploy-workers.sh materials [node…]  copy materials from NFS only
#   scripts/deploy/deploy-workers.sh status | down [node…] | logs <node> | targets
#
# `up` does, for each node: find its GPUs and pick the GPU or the CPU image, copy
# the compose files and experiments/, write a worker-only .env from this machine's
# .env, pull or load the image, copy the materials from NFS (when
# ATELIER_MATERIALS_NFS is set), start the worker, and wait for it to report in.
# Then it rewrites deploy/prometheus/targets/gpu-nodes.yml.
#
# The materials copy is incremental: a new node gets all of them before its worker
# first starts, an existing one only what was added or changed on the share. Nothing
# already on the node is deleted. A copy that fails leaves the worker unstarted.
#
# A worker is recreated only when its image or settings changed, and a recreated
# worker fails the runs it had in progress. experiments/ is updated in place, which
# a running worker picks up within a minute without a restart.
#
# Settings, as environment variables:
#   CONTROL_HOST        the address the nodes reach this machine at (required, or
#                       ATELIER_CONTROL_HOST in ENV_FILE)
#   WORKERS             the node list                    default deploy/workers
#   ENV_FILE            where the secrets come from      default .env
#   REMOTE_ROOT         the Atelier folder on each node  default /srv/atelier
#   IMAGES_DIR          export-images.sh tarballs, for nodes without a registry
#   WORKER_VARIANT      which GPU tarball to load        default cu128
#   MATERIALS_NFS       server:/export/path of the materials (or ATELIER_MATERIALS_NFS
#                       in ENV_FILE); unset means materials are left alone
#   MATERIALS_NFS_OPTS  NFS mount options                 default nfsvers=4.1
#   MATERIALS_SKIP=1    do not copy materials on this `up`
#   DASHBOARDS=1        also run node-exporter and write node-exporters.yml
#   SSH_OPTS            extra ssh options, e.g. "-p 2222 -i ~/.ssh/atelier"
#
# On Windows use deploy-workers.cmd, which runs this in Git Bash: PowerShell 5.1
# corrupts the binary streams piped over ssh here.
set -euo pipefail
cd "$(dirname "$0")/../.."

WORKERS="${WORKERS:-deploy/workers}"
ENV_FILE="${ENV_FILE:-.env}"
REMOTE_ROOT="${REMOTE_ROOT:-/srv/atelier}"
APP="$REMOTE_ROOT/app"
WORKER_VARIANT="${WORKER_VARIANT:-cu128}"
SSH_OPTS="${SSH_OPTS:-}"

say()  { printf '\033[36m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32m%s\033[0m\n' "$*"; }
warn() { printf '    \033[33m%s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }
indent() { sed 's/^/    /'; }

# a value from ENV_FILE, without sourcing it (passwords may hold shell characters)
envget() {
  local v=""
  [[ -f "$ENV_FILE" ]] && v=$(grep -E "^[[:space:]]*$1=" "$ENV_FILE" | tail -1 | cut -d= -f2-) || true
  v="${v%$'\r'}"
  if [[ "$v" == \"*\" || "$v" == \'*\' ]]; then v="${v:1:${#v}-2}"; fi
  printf '%s' "${v:-${2:-}}"
}

# a value for the node's .env: single quotes, or compose would expand a $ in a password
q() {
  local v="$1"
  [[ "$v" == *"'"* ]] && die "a value in $ENV_FILE contains a single quote, which compose's .env cannot hold alongside a \$: ${v:0:3}…"
  printf "'%s'" "$v"
}

rsh() {  # rsh <host> <command>
  local host="$1"; shift
  # shellcheck disable=SC2086
  ssh -o BatchMode=yes -o ConnectTimeout=10 $SSH_OPTS "$host" "$@"
}

case "${1:-}" in
  up|check|sync|materials|status|down|logs|targets) ;;
  *) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 2 ;;
esac

# --- the node list -----------------------------------------------------------
HOSTS=(); NAMES=(); KINDS=(); SLOTS=(); GPUS=()
[[ -f "$WORKERS" ]] || die "no $WORKERS — copy deploy/workers.example to it and list your nodes"
while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line%%#*}"; line="${line%$'\r'}"
  read -r -a f <<< "$line" || true
  [[ ${#f[@]} -eq 0 ]] && continue
  name="" kind=auto slots="" gpus=""
  for kv in "${f[@]:1}"; do
    case "$kv" in
      name=*) name="${kv#name=}" ;; kind=*) kind="${kv#kind=}" ;;
      slots=*) slots="${kv#slots=}" ;; gpus=*) gpus="${kv#gpus=}" ;;
      *) die "$WORKERS: unknown setting '$kv' for ${f[0]}" ;;
    esac
  done
  [[ "$kind" =~ ^(auto|gpu|cpu)$ ]] || die "$WORKERS: kind must be auto, gpu or cpu for ${f[0]}"
  HOSTS+=("${f[0]}"); NAMES+=("$name"); KINDS+=("$kind"); SLOTS+=("$slots"); GPUS+=("$gpus")
done < "$WORKERS"
[[ ${#HOSTS[@]} -gt 0 ]] || die "$WORKERS lists no nodes"

select_nodes() {  # sets SEL to the indices named on the command line, or all
  SEL=()
  if [[ $# -eq 0 ]]; then SEL=("${!HOSTS[@]}"); return 0; fi
  local want i found
  for want in "$@"; do
    found=""
    for i in "${!HOSTS[@]}"; do
      if [[ "$want" == "${NAMES[$i]}" || "$want" == "${HOSTS[$i]}" || "$want" == "${HOSTS[$i]#*@}" ]]; then SEL+=("$i"); found=1; fi
    done
    [[ -n "$found" ]] || die "no node '$want' in $WORKERS"
  done
}

need_control_host() {
  CONTROL_HOST="${CONTROL_HOST:-$(envget ATELIER_CONTROL_HOST)}"
  [[ -n "$CONTROL_HOST" ]] || die "set CONTROL_HOST to the address the nodes reach this machine at, e.g. CONTROL_HOST=192.168.1.20 $0 ${1:-up}"
}

# resolve <index>: asks the node, then sets NODE KIND NGPU ADDR. Returns 1 when the
# node cannot be used, without exiting, so a caller can go on to the next one.
resolve() {
  local i="$1" facts n rt hn cv
  facts=$(rsh "${HOSTS[$i]}" "n=0; command -v nvidia-smi >/dev/null 2>&1 && n=\$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
    rt=no; docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia && rt=yes
    ip=\$(ip -o route get '$CONTROL_HOST' 2>/dev/null | sed -n 's/.* src \([0-9a-f.:]*\).*/\1/p')
    cv=\$(docker compose version --short 2>/dev/null || echo none)
    echo \"\${n:-0} \$rt \$(hostname -s) \${ip:-unknown} \$cv\"") || { warn "ssh to ${HOSTS[$i]} failed"; return 1; }
  read -r n rt hn ADDR cv <<< "$facts"
  NODE="${NAMES[$i]:-$hn}"; NGPU="$n"; KIND="${KINDS[$i]}"
  if [[ "$KIND" == auto ]]; then
    if [[ "$n" -gt 0 && "$rt" == yes ]]; then KIND=gpu
    else
      KIND=cpu
      if [[ "$n" -gt 0 ]]; then warn "$NODE has $n GPU(s) but docker has no nvidia runtime — deploying it as a CPU node. Install nvidia-container-toolkit to use the GPUs."; fi
    fi
  fi
  if [[ "$KIND" == gpu && "$rt" != yes ]]; then warn "$NODE is kind=gpu but docker has no nvidia runtime"; return 1; fi
  if [[ "$cv" == none ]]; then warn "$NODE has no docker compose v2"; return 1; fi
}

compose_files() {  # compose_files <kind>
  if [[ "$1" == cpu ]]; then echo "-f docker-compose.worker.yml -f docker-compose.worker-cpu.yml"
  else echo "-f docker-compose.worker.yml"; fi
}

worker_env() {  # worker_env <index>
  local i="$1"
  cat <<EOF
# Written by scripts/deploy/deploy-workers.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ) for $NODE ($KIND).
# Edits here are overwritten on the next deploy: change the control machine's .env
# or deploy/workers instead.
ATELIER_CONTROL_HOST=$(q "$CONTROL_HOST")
ATELIER_NODE_NAME=$(q "$NODE")
ATELIER_WORKER_TOKEN=$(q "$(envget ATELIER_WORKER_TOKEN)")
ATELIER_DB_PASSWORD=$(q "$(envget ATELIER_DB_PASSWORD atelier)")
ATELIER_REDIS_PASSWORD=$(q "$(envget ATELIER_REDIS_PASSWORD atelier)")
ATELIER_HTTP_PORT=$(q "$(envget ATELIER_HTTP_PORT 80)")
ATELIER_DB_PORT=$(q "$(envget ATELIER_DB_PORT 5432)")
ATELIER_REDIS_PORT=$(q "$(envget ATELIER_REDIS_PORT 6379)")
ATELIER_REGISTRY=$(q "$(envget ATELIER_REGISTRY)")
ATELIER_TAG=$(q "$(envget ATELIER_TAG latest)")
ATELIER_STORAGE_MODE=$(q "$(envget ATELIER_STORAGE_MODE upload)")
ATELIER_UPLOAD_MAX_MB=$(q "$(envget ATELIER_UPLOAD_MAX_MB 64)")
ATELIER_RUNNER_BACKEND=$(q "$(envget ATELIER_RUNNER_BACKEND subprocess)")
ATELIER_WORKER_METRICS_PORT=$(q "$(envget ATELIER_WORKER_METRICS_PORT 9101)")
ATELIER_EXPERIMENTS_DIR=$(q "$REMOTE_ROOT/experiments")
ATELIER_MATERIALS_DIR=$(q "$REMOTE_ROOT/materials")
ATELIER_DATA_DIR=$(q "$REMOTE_ROOT/data")
ATELIER_CPU_SLOTS=$(q "${SLOTS[$i]:-1}")
ATELIER_GPU_COUNT=$(q "${GPUS[$i]}")
EOF
}

# experiments/ is updated in place: a running worker has the folder bind-mounted, and
# a folder swapped for a new one would stay invisible to it until a restart
sync_experiments() {  # sync_experiments <host>
  tar -cf - --exclude=__pycache__ --exclude='*.pyc' experiments docker-compose.worker.yml docker-compose.worker-cpu.yml |
    rsh "$1" "set -e
      mkdir -p '$APP' '$REMOTE_ROOT/experiments' '$REMOTE_ROOT/materials' '$REMOTE_ROOT/data'
      rm -rf '$APP/.incoming' && mkdir '$APP/.incoming' && tar -xf - -C '$APP/.incoming'
      cd '$APP/.incoming/experiments' && find . -type f | sort > ../new.list
      cd '$REMOTE_ROOT/experiments' && find . -type f | sort > '$APP/.incoming/old.list'
      comm -23 '$APP/.incoming/old.list' '$APP/.incoming/new.list' | xargs -r -d '\n' rm -f --
      cp -a '$APP/.incoming/experiments/.' '$REMOTE_ROOT/experiments/'
      find '$REMOTE_ROOT/experiments' -mindepth 1 -type d -empty -delete
      mv -f '$APP/.incoming/'docker-compose.worker*.yml '$APP/'
      rm -rf '$APP/.incoming'"
}

ensure_image() {  # ensure_image <host>   (uses KIND)
  local host="$1" registry tag image file sum
  registry="$(envget ATELIER_REGISTRY)"; tag="$(envget ATELIER_TAG latest)"
  if [[ "$KIND" == cpu ]]; then image="${registry}atelier-worker-cpu:$tag"; file="atelier-worker-cpu.tar.gz"
  else image="${registry}atelier-worker:$tag"; file="atelier-worker-$WORKER_VARIANT.tar.gz"; fi
  if [[ -n "$registry" ]]; then
    rsh "$host" "cd '$APP' && docker compose $(compose_files "$KIND") pull --quiet worker" 2>&1 | indent
    ok "pulled $image"
  elif [[ -n "${IMAGES_DIR:-}" ]]; then
    [[ -f "$IMAGES_DIR/$file" ]] || { warn "no $IMAGES_DIR/$file — make it with scripts/offline/export-images.sh"; return 1; }
    sum=$(sha256sum "$IMAGES_DIR/$file" | cut -d' ' -f1)
    if [[ "$(rsh "$host" "cat '$APP/.loaded-$file' 2>/dev/null || true")" == "$sum" ]]; then
      ok "$file already loaded"
    else
      echo "    loading $file (a GPU image is several GB; this takes a while)"
      gunzip -c "$IMAGES_DIR/$file" | rsh "$host" "docker load >/dev/null"
      if [[ "$KIND" == gpu ]]; then  # only the cu128 tarball carries atelier-worker:latest
        rsh "$host" "docker image inspect atelier-worker:$tag >/dev/null 2>&1 || docker tag atelier-worker:$WORKER_VARIANT atelier-worker:$tag"
      fi
      rsh "$host" "echo $sum > '$APP/.loaded-$file'"
      ok "loaded $file"
    fi
  else
    if ! rsh "$host" "docker image inspect '$image' >/dev/null 2>&1"; then
      warn "$image is not on $host — set ATELIER_REGISTRY in $ENV_FILE, or IMAGES_DIR to a folder of export-images.sh tarballs"
      return 1
    fi
    ok "$image present"
  fi
}

MATERIALS_NFS="${MATERIALS_NFS:-$(envget ATELIER_MATERIALS_NFS)}"
MATERIALS_NFS_OPTS="${MATERIALS_NFS_OPTS:-nfsvers=4.1}"

# Runs inside a container with the share at /nfs and the node's materials at /dst.
# Kept in sync with the init container k8s-workers.sh writes.
COPY_MATERIALS='set -e
need=`du -sb /nfs | cut -f1`
used=`du -sb /dst | cut -f1`
free=`df -B1 --output=avail /dst | tail -1`
if [ "$need" -gt `expr $used + $free` ]; then
  echo "the materials are `expr $need / 1048576` MB and this node has room for `expr \( $used + $free \) / 1048576` MB" >&2
  exit 3
fi
cp -au /nfs/. /dst/
echo "materials on this node: `du -sh /dst | cut -f1`"'

copy_materials() {  # copy_materials <host>   (uses KIND for the image)
  local host="$1" server path image
  server="${MATERIALS_NFS%%:*}"; path="${MATERIALS_NFS#*:}"
  [[ "$server" != "$MATERIALS_NFS" && "$path" == /* ]] || { warn "ATELIER_MATERIALS_NFS must look like server:/export/path, not '$MATERIALS_NFS'"; return 1; }
  image="$(envget ATELIER_REGISTRY)atelier-worker$([[ $KIND == cpu ]] && echo -cpu || true):$(envget ATELIER_TAG latest)"
  echo "    copying materials from $MATERIALS_NFS (only what the node does not have yet)"
  # docker mounts the share itself, so the ssh user needs no sudo; the files are
  # written as that user, who owns $REMOTE_ROOT
  rsh "$host" "set -e
    addr=\$(getent ahostsv4 '$server' | awk 'NR == 1 { print \$1 }'); addr=\${addr:-$server}
    docker volume rm -f atelier-materials-nfs >/dev/null 2>&1 || true
    docker volume create --driver local --opt type=nfs --opt o=addr=\$addr,ro,$MATERIALS_NFS_OPTS --opt 'device=:$path' atelier-materials-nfs >/dev/null
    mkdir -p '$REMOTE_ROOT/materials'
    docker run --rm -i --user \$(id -u):\$(id -g) -v atelier-materials-nfs:/nfs:ro -v '$REMOTE_ROOT/materials':/dst '$image' sh -s
    docker volume rm atelier-materials-nfs >/dev/null" <<< "$COPY_MATERIALS" 2>&1 | indent
}

wait_for_worker() {  # wait_for_worker <host>
  local out
  for _ in $(seq 1 45); do
    # the whole log of the current container: an unchanged worker reported in long ago
    out=$(rsh "$1" "cd '$APP' && docker compose logs --no-log-prefix worker 2>&1 | grep -E 'worker up on|refusing to start|Traceback' | tail -1" || true)
    case "$out" in
      *"worker up on"*) ok "${out#*atelier.worker INFO }"; return 0 ;;
      *refusing*|*Traceback*) warn "$out — see: $0 logs $NODE"; return 1 ;;
    esac
    sleep 2
  done
  warn "the worker did not report in within 90 s — see: $0 logs $NODE"
  return 1
}

deploy_node() {  # deploy_node <index>
  local i="$1" host="${HOSTS[$1]}" profile="" services="worker"
  resolve "$i"
  say "$NODE ($host): $KIND node$([[ $KIND == gpu ]] && echo ", $NGPU GPU(s)" || true)"
  sync_experiments "$host"; ok "compose files and experiments/ copied"
  worker_env "$i" | rsh "$host" "umask 077 && cat > '$APP/.env'"; ok ".env written"
  ensure_image "$host"
  if [[ -n "$MATERIALS_NFS" && -z "${MATERIALS_SKIP:-}" ]]; then
    copy_materials "$host"
  fi
  if [[ -n "${DASHBOARDS:-}" ]]; then profile="--profile dashboards"; services="worker node-exporter"; fi
  # --remove-orphans: a node that changed kind keeps nothing of the other definition
  rsh "$host" "cd '$APP' && docker compose $(compose_files "$KIND") $profile up -d --no-build --remove-orphans $services" 2>&1 | indent
  wait_for_worker "$host"
}

write_targets() {  # from NODE_NAMES / NODE_ADDRS, filled by resolve_all
  local port i out
  port="$(envget ATELIER_WORKER_METRICS_PORT 9101)"
  [[ "$port" == 0 ]] && return 0
  out=deploy/prometheus/targets/gpu-nodes.yml
  {
    echo "# Written by scripts/deploy/deploy-workers.sh from $WORKERS on $(date -u +%Y-%m-%dT%H:%M:%SZ)."
    echo "# Port $port is each worker's metrics listener; Prometheus re-reads this within 30 s."
    for i in "${!NODE_NAMES[@]}"; do
      printf -- '- targets: ["%s:%s"]\n  labels:\n    node: %s\n    role: worker\n' "${NODE_ADDRS[$i]}" "$port" "${NODE_NAMES[$i]}"
    done
  } > "$out"
  ok "wrote $out"
  [[ -n "${DASHBOARDS:-}" ]] || return 0
  out=deploy/prometheus/targets/node-exporters.yml
  {
    echo "# Written by scripts/deploy/deploy-workers.sh from $WORKERS on $(date -u +%Y-%m-%dT%H:%M:%SZ)."
    for i in "${!NODE_NAMES[@]}"; do
      printf -- '- targets: ["%s:9100"]\n  labels:\n    node: %s\n' "${NODE_ADDRS[$i]}" "${NODE_NAMES[$i]}"
    done
  } > "$out"
  ok "wrote $out"
}

resolve_all() {  # every node in the list, for the Prometheus targets
  NODE_NAMES=(); NODE_ADDRS=()
  local i
  for i in "${!HOSTS[@]}"; do
    if resolve "$i" 2>/dev/null && [[ "$ADDR" != unknown ]]; then
      NODE_NAMES[$i]="$NODE"; NODE_ADDRS[$i]="$ADDR"
    else
      warn "left ${NAMES[$i]:-${HOSTS[$i]}} out of the Prometheus targets: unreachable, or no route to $CONTROL_HOST"
    fi
  done
}

# Runs fn for each selected node in a subshell with errexit on, carrying on past a
# failed node. Must be called as a plain command — inside `if`, `||` or `&&` bash
# ignores errexit in everything underneath, and a failing step would not stop.
FAILED=()
for_nodes() {  # for_nodes <fn> [node…]
  local fn="$1" i rc
  shift; select_nodes "$@"
  for i in "${SEL[@]}"; do
    set +e; ( set -e; "$fn" "$i" ); rc=$?; set -e
    [[ $rc -eq 0 ]] || FAILED+=("${NAMES[$i]:-${HOSTS[$i]}}")
  done
}
finish() {
  if [[ ${#FAILED[@]} -gt 0 ]]; then printf '\033[31mfailed: %s\033[0m\n' "${FAILED[*]}" >&2; exit 1; fi
}

check_node() {
  local i="$1" host="${HOSTS[$1]}" port
  say "${NAMES[$i]:-$host} ($host)"
  rsh "$host" true || { warn "ssh failed — the key must be accepted without a prompt"; return 1; }
  ok "ssh"
  resolve "$i"
  ok "hostname $NODE, address towards $CONTROL_HOST: $ADDR, $([[ $KIND == gpu ]] && echo "GPU node with $NGPU GPU(s)" || echo "CPU node")"
  rsh "$host" "docker info >/dev/null 2>&1" || { warn "docker needs sudo here: add the user to the docker group"; return 1; }
  ok "docker usable without sudo"
  if rsh "$host" "v=\$(docker compose version --short); [ \"\$(printf '%s\n' 2.24.0 \"\${v#v}\" | sort -V | head -1)\" = 2.24.0 ]"; then
    ok "compose 2.24 or newer"
  elif [[ "$KIND" == cpu ]]; then
    warn "compose is older than 2.24 and cannot read the CPU file's !reset"; return 1
  else
    warn "compose is older than 2.24 (fine for a GPU node, not for a CPU one)"
  fi
  rsh "$host" "mkdir -p '$APP' 2>/dev/null && test -w '$APP'" || { warn "cannot write $APP: sudo mkdir -p $REMOTE_ROOT && sudo chown \$USER $REMOTE_ROOT"; return 1; }
  ok "$APP writable"
  for port in "$(envget ATELIER_HTTP_PORT 80)" "$(envget ATELIER_DB_PORT 5432)" "$(envget ATELIER_REDIS_PORT 6379)"; do
    rsh "$host" "timeout 3 bash -c '</dev/tcp/$CONTROL_HOST/$port'" 2>/dev/null || { warn "cannot reach $CONTROL_HOST:$port — a firewall, or the control plane is not up"; return 1; }
    ok "reaches $CONTROL_HOST:$port"
  done
  if [[ -n "$MATERIALS_NFS" ]]; then
    if rsh "$host" "timeout 3 bash -c '</dev/tcp/${MATERIALS_NFS%%:*}/2049'" 2>/dev/null; then ok "reaches the materials share on ${MATERIALS_NFS%%:*}:2049"
    else warn "cannot reach ${MATERIALS_NFS%%:*}:2049 — the materials copy before the worker starts will fail"; return 1; fi
  elif [[ -n "$(rsh "$host" "ls -A '$REMOTE_ROOT/materials' 2>/dev/null | head -1")" ]]; then ok "materials present"
  else warn "$REMOTE_ROOT/materials is empty and ATELIER_MATERIALS_NFS is not set: only the generated-data experiments will run here"; fi
}

materials_node() {
  local i="$1"
  [[ -n "$MATERIALS_NFS" ]] || { warn "set ATELIER_MATERIALS_NFS in $ENV_FILE, or MATERIALS_NFS"; return 1; }
  resolve "$i"
  say "$NODE (${HOSTS[$i]}): materials"
  ensure_image "${HOSTS[$i]}"
  copy_materials "${HOSTS[$i]}"
}

status_node() {
  local host="${HOSTS[$1]}"
  say "${NAMES[$1]:-$host} ($host)"
  rsh "$host" "cd '$APP' 2>/dev/null || { echo 'not deployed'; exit 0; }
    docker compose ps --all --format '{{.Service}}: {{.State}} ({{.Status}}) {{.Image}}'
    docker compose logs --no-log-prefix worker 2>/dev/null | grep -E 'worker up on|no GPUs visible' | tail -1" | indent
}

down_node() {
  local host="${HOSTS[$1]}"
  say "stopping ${NAMES[$1]:-$host} ($host)"
  rsh "$host" "cd '$APP' && docker compose -f docker-compose.worker.yml --profile dashboards down" 2>&1 | indent
}

sync_node() {
  say "${NAMES[$1]:-${HOSTS[$1]}}: experiments/"
  sync_experiments "${HOSTS[$1]}"
  ok "copied"
}

cmd="${1:-}"; shift || true
case "$cmd" in
  up)
    need_control_host up
    [[ -n "$(envget ATELIER_WORKER_TOKEN)" ]] || die "$ENV_FILE has no ATELIER_WORKER_TOKEN"
    for key in ATELIER_WORKER_TOKEN ATELIER_DB_PASSWORD ATELIER_REDIS_PASSWORD ATELIER_REGISTRY; do
      [[ "$(envget $key)" != *"'"* ]] || die "$key in $ENV_FILE contains a single quote, which a compose .env cannot hold safely — pick another"
    done
    for_nodes deploy_node "$@"
    say "Prometheus targets"
    resolve_all
    write_targets
    finish ;;
  check)   need_control_host check; for_nodes check_node "$@"; finish ;;
  sync)    for_nodes sync_node "$@"; finish ;;
  materials) need_control_host materials; for_nodes materials_node "$@"; finish ;;
  status)  for_nodes status_node "$@"; finish ;;
  down)    for_nodes down_node "$@"; finish ;;
  logs)
    [[ $# -eq 1 ]] || die "usage: $0 logs <node>"
    select_nodes "$1"
    # shellcheck disable=SC2086
    exec ssh -t $SSH_OPTS "${HOSTS[${SEL[0]}]}" "cd '$APP' && docker compose logs --tail 200 -f worker" ;;
  targets) need_control_host targets; resolve_all; write_targets ;;
  *) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 2 ;;
esac
