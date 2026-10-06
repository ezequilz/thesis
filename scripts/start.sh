#!/usr/bin/env bash
# One-command start/restart of the full dev stack. Safe to re-run at any time.
#
#   0. Docker daemon       started automatically if not running (macOS)
#   1. CliRelay proxy      http://localhost:8317  (panel at /manage;
#                          cloned to $CLIRELAY_DIR on first run)
#   2. Host libraries      reinstall only when a required version changed;
#                          downloads stay in the pip wheel cache
#   3. splat-explorer      image rebuild only when the Dockerfile changed;
#                          viser viewer :8080, dashboard :8090, and the
#                          scene-run manager reload when src/ or configs
#                          changed, or when they are not already healthy
#
# Usage:
#   scripts/start.sh                 (re)start CliRelay + viewer + dashboard
#   scripts/start.sh --render-test   ... and render sanity views first
#   scripts/start.sh --episode       ... and run an agent episode at the end
#   scripts/start.sh --force         reinstall, rebuild, and restart everything
#   scripts/start.sh --stop          stop everything (project + CliRelay)
set -euo pipefail
cd "$(dirname "$0")/.."

CLIRELAY_DIR="${CLIRELAY_DIR:-$HOME/CliRelay}"
CLIRELAY_REPO="https://github.com/kittors/CliRelay.git"
CLIRELAY_URL="http://localhost:8317"

stop_host_dashboard() {
  if [ -f outputs/dashboard.pid ]; then
    pid=$(cat outputs/dashboard.pid 2>/dev/null || true)
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      echo "    Stopping host dashboard (pid $pid)"
      kill "$pid" || true
    fi
    rm -f outputs/dashboard.pid
  fi
  local leftover pid cmd
  leftover=$(pgrep -f "[s]plat-explorer dashboard" || true)
  for pid in $leftover; do
    echo "    Stopping leftover dashboard (pid $pid)"
    kill "$pid" || true
  done
  # Don't let a lingering :8090 listener make the restart skip the dashboard.
  for _ in $(seq 1 20); do
    leftover=$(lsof -ti "tcp:8090" -sTCP:LISTEN || true)
    [ -z "$leftover" ] && break
    for pid in $leftover; do
      cmd=$(ps -p "$pid" -o command= || true)
      if [[ "$cmd" == *"splat-explorer dashboard"* ]]; then
        echo "    Waiting for dashboard pid $pid to release :8090"
        kill "$pid" 2>/dev/null || true
      fi
    done
    sleep 0.25
  done
}

stop_scene_run_manager() {
  if [ -f outputs/scene-run-manager.pid ]; then
    local pid
    pid=$(cat outputs/scene-run-manager.pid 2>/dev/null || true)
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      echo "    Stopping scene-run manager (pid $pid)"
      kill "$pid" || true
    fi
    rm -f outputs/scene-run-manager.pid
  fi
  local leftover pid
  leftover=$(pgrep -f "[s]plat-explorer scene-run-manager" || true)
  for pid in $leftover; do
    echo "    Stopping leftover scene-run manager (pid $pid)"
    kill "$pid" || true
  done
  for _ in $(seq 1 20); do
    leftover=$(pgrep -f "[s]plat-explorer scene-run-manager" || true)
    [ -z "$leftover" ] && break
    sleep 0.25
  done
  leftover=$(pgrep -f "[s]plat-explorer scene-run-manager" || true)
  for pid in $leftover; do
    echo "    Force-stopping scene-run manager (pid $pid)"
    kill -9 "$pid" 2>/dev/null || true
  done
}

start_scene_run_manager() {
  stop_scene_run_manager
  mkdir -p outputs/scene-runs
  echo "    Starting persistent scene-run manager"
  export CLIRELAY_BASE_URL="${CLIRELAY_BASE_URL:-http://localhost:8317/v1}"
  export VISER_RENDER_URL="${VISER_RENDER_URL:-http://localhost:8081}"
  export VISER_VIEWER_URL="${VISER_VIEWER_URL:-http://localhost:8080}"
  nohup .venv/bin/splat-explorer scene-run-manager >> outputs/scene-run-manager.log 2>&1 </dev/null &
  echo $! > outputs/scene-run-manager.pid
  disown $! 2>/dev/null || true
  local pid
  pid=$(cat outputs/scene-run-manager.pid)
  echo "    Scene-run manager pid $pid  (logs: outputs/scene-run-manager.log)"
  printf "    Waiting for scene-run manager"
  local up=0
  for _ in $(seq 1 40); do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null \
      && [ -f outputs/scene-runs/manager.json ] \
      && .venv/bin/python -c "import json,sys,time; from pathlib import Path; want=int(sys.argv[1]); body=json.loads(Path('outputs/scene-runs/manager.json').read_text()); sys.exit(0 if int(body.get('pid') or 0)==want and body.get('status')=='running' and time.time()-float(body.get('updated_at') or 0)<15 else 1)" "$pid"
    then
      echo "  up"
      up=1
      break
    fi
    printf "."
    sleep 0.25
  done
  if [ "$up" != 1 ]; then
    echo "  ERROR"
    echo "    Scene-run manager did not heartbeat. Last log lines:"
    tail -n 40 outputs/scene-run-manager.log || true
    exit 1
  fi
}

# True when $1 exists and its contents are exactly $2. Stamps are one line.
stamp_matches() {
  [ -f "$1" ] && [ "$(<"$1")" = "$2" ]
}

# free_port <port> <cmd substring> ; returns 1 if a foreign process holds it.
free_port() {
  local pids pid cmd blocked=0
  pids=$(lsof -ti "tcp:$1" -sTCP:LISTEN || true)
  for pid in $pids; do
    cmd=$(ps -p "$pid" -o command= || true)
    if [[ "$cmd" == *"$2"* ]]; then
      echo "    Killing stale local process on :$1 (pid $pid)"
      kill "$pid" || true
    else
      echo "    WARNING: port $1 is in use by another process:"
      echo "      $pid  $cmd"
      echo "    Free it with: kill $pid"
      blocked=1
    fi
  done
  [ -n "$pids" ] && sleep 1
  return $blocked
}

dashboard_up() {
  local up=0
  curl -sf --connect-timeout 1 --max-time 5 -o /dev/null "http://127.0.0.1:8090" \
    && curl -sf --connect-timeout 1 --max-time 5 -o /dev/null "http://127.0.0.1:8090/repair" \
    && curl -sf --connect-timeout 1 --max-time 5 -o /dev/null "http://127.0.0.1:8090/repair/gpu" \
    && curl -sf --connect-timeout 1 --max-time 5 -o /dev/null "http://127.0.0.1:8090/scene-runs" \
    && curl -sf --connect-timeout 1 --max-time 5 -o /dev/null "http://127.0.0.1:8090/scene-runs-ext" \
    && curl -sf --connect-timeout 1 --max-time 5 -o /dev/null "http://127.0.0.1:8090/api/scene-runs-ext/state" \
    && up=1 || true
  [ "$up" = 1 ] || return 1
}

wait_for_dashboard() {
  local where="$1"
  printf "    Waiting for dashboard pages"
  local up=0
  for _ in $(seq 1 40); do
    if curl -sf -o /dev/null "http://127.0.0.1:8090/scene-runs" \
      && curl -sf -o /dev/null "http://127.0.0.1:8090/scene-runs-ext" \
      && curl -sf -o /dev/null "http://127.0.0.1:8090/api/scene-runs-ext/state"; then
      echo "  up"
      up=1
      break
    fi
    printf "."
    sleep 0.5
  done
  if [ "$up" != 1 ]; then
    echo "  ERROR"
    echo "    Dashboard did not become ready. Last log lines:"
    if [ "$where" = host ]; then
      tail -n 40 outputs/dashboard.log || true
    else
      docker compose logs --tail 40 dashboard || true
    fi
    exit 1
  fi
}

launch_host_dashboard() {
  mkdir -p outputs
  echo "    Starting host dashboard on :8090 (Metal / gsplat-mlx; not Docker)"
  export CLIRELAY_BASE_URL="${CLIRELAY_BASE_URL:-http://localhost:8317/v1}"
  export VISER_RENDER_URL="${VISER_RENDER_URL:-http://localhost:8081}"
  export VISER_VIEWER_URL="${VISER_VIEWER_URL:-http://localhost:8080}"
  nohup .venv/bin/splat-explorer dashboard >> outputs/dashboard.log 2>&1 </dev/null &
  echo $! > outputs/dashboard.pid
  disown $! 2>/dev/null || true
  echo "    Host dashboard pid $(cat outputs/dashboard.pid)  (logs: outputs/dashboard.log)"
  printf "    Waiting for host dashboard"
  local up=0 listen_pid
  for _ in $(seq 1 40); do
    if curl -sf -o /dev/null "http://127.0.0.1:8090" \
      && curl -sf -o /dev/null "http://127.0.0.1:8090/repair" \
      && curl -sf -o /dev/null "http://127.0.0.1:8090/repair/gpu" \
      && curl -sf -o /dev/null "http://127.0.0.1:8090/scene-runs" \
      && curl -sf -o /dev/null "http://127.0.0.1:8090/scene-runs-ext" \
      && curl -sf -o /dev/null "http://127.0.0.1:8090/api/scene-runs-ext/state"; then
      echo "  up"
      up=1
      listen_pid=$(lsof -ti "tcp:8090" -sTCP:LISTEN | head -1 || true)
      if [ -n "${listen_pid:-}" ]; then
        echo "$listen_pid" > outputs/dashboard.pid
        echo "    Listening pid $listen_pid"
      fi
      break
    fi
    printf "."
    sleep 0.5
  done
  if [ "$up" != 1 ]; then
    echo "  ERROR"
    echo "    Host dashboard did not bind :8090. Last log lines:"
    tail -n 40 outputs/dashboard.log || true
    exit 1
  fi
  printf "    Waiting for catalog scene"
  local status
  for _ in $(seq 1 60); do
    status=$(curl -s "http://127.0.0.1:8090/api/state" | python3 -c "import json,sys; print((json.load(sys.stdin).get('scene') or {}).get('status') or '')" 2>/dev/null || true)
    if [ "$status" = "ready" ]; then echo "  ready"; break; fi
    if [ "$status" = "error" ]; then echo "  ERROR"; break; fi
    printf "."
    sleep 0.5
  done
  echo ""
}

# Returns 1 when a foreign process holds the viewer ports.
reload_viewer() {
  local running
  running=$(docker compose ps --status running -q viewer 2>/dev/null || true)
  if [ -z "$running" ]; then
    # 86: a foreign process holds the port. Other statuses are docker failures.
    free_port 8080 "splat-explorer viewer" || return 86
    free_port 8081 "splat-explorer viewer" || true
  fi
  if [ "$VIEWER_RECREATE" = 1 ]; then
    echo "    Recreating viewer (image or compose file changed)"
    docker compose up -d --force-recreate --remove-orphans viewer || return $?
  elif [ -n "$running" ] && [ "$CODE_CHANGED" = 1 ]; then
    echo "    Reloading viewer (src or configs changed)"
    docker compose restart viewer || return $?
  elif [ -z "$running" ]; then
    echo "    Starting viewer"
    docker compose up -d --remove-orphans viewer || return $?
  else
    echo "    Viewer already running; libraries unchanged"
  fi
}

# Returns 1 when a foreign process holds :8090.
reload_docker_dashboard() {
  local running
  running=$(docker compose ps --status running -q dashboard 2>/dev/null || true)
  if [ -z "$running" ]; then
    free_port 8090 "splat-explorer dashboard" || return 86
  fi
  if [ "$VIEWER_RECREATE" = 1 ]; then
    echo "    Recreating dashboard (image or compose file changed)"
    rm -f outputs/live/scene.json
    docker compose up -d --force-recreate --remove-orphans dashboard || return $?
  elif [ -n "$running" ] && [ "$CODE_CHANGED" = 1 ]; then
    echo "    Reloading dashboard (src or configs changed)"
    rm -f outputs/live/scene.json
    docker compose restart dashboard || return $?
  elif [ -z "$running" ]; then
    echo "    Starting dashboard"
    rm -f outputs/live/scene.json
    docker compose up -d --remove-orphans dashboard || return $?
  else
    echo "    Dashboard already running; libraries unchanged"
    return 0
  fi
  wait_for_dashboard docker
}

RENDER_TEST=0 EPISODE=0 FORCE=0
for arg in "$@"; do
  case "$arg" in
    --render-test) RENDER_TEST=1 ;;
    --episode)     EPISODE=1 ;;
    --force)       FORCE=1 ;;
    --stop)
      echo "==> Stopping splat-explorer stack"
      stop_scene_run_manager
      stop_host_dashboard
      docker compose down --remove-orphans
      if [ -d "$CLIRELAY_DIR" ]; then
        echo "==> Stopping CliRelay"
        (cd "$CLIRELAY_DIR" && docker compose down)
      fi
      exit 0 ;;
    *) echo "Unknown option: $arg (see header of this script)"; exit 2 ;;
  esac
done

echo "==> [0/4] Checking the Docker daemon"
if ! docker info >/dev/null 2>&1; then
  if [ -d "/Applications/OrbStack.app" ]; then
    echo "    Docker daemon not running — starting OrbStack"
    open -a OrbStack
  elif [ -d "/Applications/Docker.app" ]; then
    echo "    Docker daemon not running — starting Docker Desktop"
    open -a Docker
  else
    echo "    ERROR: Docker daemon not running and no Docker Desktop/OrbStack found."
    exit 1
  fi
  printf "    Waiting for the daemon"
  for _ in $(seq 1 60); do
    if docker info >/dev/null 2>&1; then DOCKER_READY=1; break; fi
    printf "."
    sleep 2
  done
  echo ""
  if [ "${DOCKER_READY:-0}" != 1 ]; then
    echo "    ERROR: Docker daemon did not come up within 2 minutes."
    exit 1
  fi
fi

echo "==> [1/4] Starting CliRelay at $CLIRELAY_URL"
if [ ! -d "$CLIRELAY_DIR" ]; then
  echo "    First run: cloning CliRelay to $CLIRELAY_DIR"
  git clone --depth 1 "$CLIRELAY_REPO" "$CLIRELAY_DIR"
fi
(cd "$CLIRELAY_DIR" && docker compose up -d)

printf "    Waiting for CliRelay to become ready"
for _ in $(seq 1 90); do
  if curl -s -o /dev/null "$CLIRELAY_URL"; then READY=1; break; fi
  printf "."
  sleep 2
done
echo ""
if [ "${READY:-0}" != 1 ]; then
  echo "    ERROR: CliRelay did not respond within 3 minutes."
  echo "    Check: (cd $CLIRELAY_DIR && docker compose logs -f cli-proxy-api)"
  exit 1
fi
echo "    Panel: $CLIRELAY_URL/manage"
if [ -f "$CLIRELAY_DIR/.env" ]; then
  ADMIN_PW=$(grep -E '^CLIRELAY_ADMIN_PASSWORD=' "$CLIRELAY_DIR/.env" | cut -d= -f2- || true)
  [ -n "${ADMIN_PW:-}" ] && echo "    Admin password (from $CLIRELAY_DIR/.env): $ADMIN_PW"
fi

echo "==> [2/4] Host libraries and splat-explorer image"
# A parent environment can set this and force pip to re-download every wheel.
unset PIP_NO_CACHE_DIR || true
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
mkdir -p .cache/start outputs

HOST_DASHBOARD=0
HOST_EXTRAS="viewer,vlm"
if [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ]; then
  HOST_DASHBOARD=1
  HOST_EXTRAS="viewer,vlm,apple"
  echo "    Apple Silicon: host tools include [apple] (gsplat-mlx / MLX)"
fi
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi

DEPS_CHANGED=0
install_args=(install --extras "$HOST_EXTRAS")
if [ "$FORCE" = 1 ]; then
  install_args+=(--force)
fi
if .venv/bin/python scripts/start_deps.py "${install_args[@]}"; then
  :
else
  deps_status=$?
  if [ "$deps_status" -eq 10 ]; then
    DEPS_CHANGED=1
  else
    exit "$deps_status"
  fi
fi

image_fp=$(.venv/bin/python scripts/start_deps.py fingerprint Dockerfile)
compose_fp=$(.venv/bin/python scripts/start_deps.py fingerprint docker-compose.yml)
code_fp=$(.venv/bin/python scripts/start_deps.py fingerprint src configs --exclude lrz.local.yaml)

VIEWER_RECREATE=0
CODE_CHANGED=0
if [ "$FORCE" = 1 ] \
  || ! docker image inspect splat-explorer:latest >/dev/null 2>&1 \
  || ! stamp_matches .cache/start/image.sha "$image_fp"; then
  echo "    Building splat-explorer image (unchanged wheels stay in the BuildKit pip cache)"
  docker compose build
  printf '%s\n' "$image_fp" > .cache/start/image.sha
  VIEWER_RECREATE=1
else
  echo "    Image matches the Dockerfile; skipping build"
fi
if [ "$FORCE" = 1 ] || ! stamp_matches .cache/start/compose.sha "$compose_fp"; then
  VIEWER_RECREATE=1
fi
if [ "$FORCE" = 1 ] || ! stamp_matches .cache/start/code.sha "$code_fp"; then
  CODE_CHANGED=1
fi

echo "==> [3/4] Viewer (:8080), dashboard (:8090), scene-run manager"
# Code and host-library changes have to bounce the manager so it imports the
# new modules. Leave a healthy manager alone when nothing it loads changed,
# so a repeat start does not drop a queued scene run.
manager_restart=0
manager_pid=""
if [ -f outputs/scene-run-manager.pid ]; then
  manager_pid=$(cat outputs/scene-run-manager.pid 2>/dev/null || true)
fi
manager_up=0
if [ "$CODE_CHANGED" != 1 ] && [ "$DEPS_CHANGED" != 1 ] \
  && [ -n "$manager_pid" ] && kill -0 "$manager_pid" 2>/dev/null; then
  # An active run can sit inside one job for a long time without rewriting
  # manager.json, so a live pid with status=running is enough. Freshness is
  # only required when this script just started the process.
  if .venv/bin/python -c "import json,sys; from pathlib import Path; want=int(sys.argv[1]); body=json.loads(Path('outputs/scene-runs/manager.json').read_text()); sys.exit(0 if int(body.get('pid') or 0)==want and body.get('status')=='running' else 1)" "$manager_pid"; then
    manager_up=1
  fi
fi
if [ "$manager_up" != 1 ]; then
  manager_restart=1
fi

STAMPS_OK=1
viewer_status=0
reload_viewer || viewer_status=$?
if [ "$viewer_status" -eq 86 ]; then
  echo "    Skipping viewer (port busy)"
  STAMPS_OK=0
elif [ "$viewer_status" -ne 0 ]; then
  exit "$viewer_status"
fi

if [ "$HOST_DASHBOARD" = 1 ]; then
  host_restart=0
  if [ "$CODE_CHANGED" = 1 ] || [ "$DEPS_CHANGED" = 1 ]; then
    host_restart=1
  else
    dash_up=0
    dashboard_up && dash_up=1 || true
    [ "$dash_up" = 1 ] || host_restart=1
  fi
  if [ "$host_restart" = 1 ]; then
    stop_host_dashboard
    if free_port 8090 "splat-explorer dashboard"; then
      # Drop leftover episode-repair PLY pointers. Dashboard boot then prefers a
      # queued/active scene-run room over the YAML Starter Scene default.
      rm -f outputs/live/scene.json
      launch_host_dashboard
    else
      echo "    Skipping dashboard (port busy)"
      STAMPS_OK=0
    fi
  else
    echo "    Host dashboard already running; libraries unchanged"
  fi
else
  dash_status=0
  reload_docker_dashboard || dash_status=$?
  if [ "$dash_status" -eq 86 ]; then
    echo "    Skipping dashboard (port busy)"
    STAMPS_OK=0
  elif [ "$dash_status" -ne 0 ]; then
    exit "$dash_status"
  fi
fi

if [ "$manager_restart" = 1 ]; then
  start_scene_run_manager
else
  echo "    Scene-run manager already running; libraries unchanged"
fi

if [ "$STAMPS_OK" = 1 ]; then
  printf '%s\n' "$code_fp" > .cache/start/code.sha
  printf '%s\n' "$compose_fp" > .cache/start/compose.sha
fi

echo "==> [4/4] Optional one-off jobs"
if [ "$RENDER_TEST" = 1 ]; then
  echo "    Rendering test views -> outputs/test_views/"
  docker compose run --rm render-test
fi
if [ "$EPISODE" = 1 ]; then
  echo "    Running an agent episode -> outputs/episodes/"
  docker compose run --rm harness
fi
[ "$RENDER_TEST$EPISODE" = "00" ] && echo "    (none requested; --render-test / --episode)"

echo ""
echo "Done."
echo "  CliRelay panel : $CLIRELAY_URL/manage   (create an API key here)"
echo "  Debug viewer   : http://localhost:8080"
if [ "${HOST_DASHBOARD:-0}" = 1 ]; then
  echo "  Dashboard      : http://localhost:8090  (host process — gsplat-mlx / Metal)"
else
  echo "  Dashboard      : http://localhost:8090  (start/watch episodes, VLM debug)"
fi
echo "  Repair review  : http://localhost:8090/repair"
echo "  GPU / LRZ      : http://localhost:8090/repair/gpu  (reserve 8h/24h, probe, jobs)"
echo "  Scene runs     : http://localhost:8090/scene-runs  (automated VLM → Qwen → GSFix3D)"
echo "  Extended runs  : http://localhost:8090/scene-runs-ext  (ArtiFixer multiview)"
echo "  Spectator (HD) : http://localhost:8090/spectator  (viewing only)"
echo "  CLI episode    : export CLIRELAY_API_KEY=sk-... && \\"
echo "                   splat-explorer --config configs/cli_relay.yaml explore"
echo "  Force restart  : scripts/start.sh --force"
echo "  Stop all       : scripts/start.sh --stop"
