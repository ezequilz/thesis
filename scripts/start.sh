#!/usr/bin/env bash
# One-command start/restart of the full dev stack. Safe to re-run at any time.
#
#   0. Docker daemon       started automatically if not running (macOS)
#   1. CliRelay proxy      http://localhost:8317  (panel at /manage;
#                          cloned to $CLIRELAY_DIR on first run)
#   2. splat-explorer      image build + viser debug viewer at :8080
#                          + episode dashboard at :8090
#
# Usage:
#   scripts/start.sh                 (re)start CliRelay + viewer + dashboard
#   scripts/start.sh --render-test   ... and render sanity views first
#   scripts/start.sh --episode       ... and run an agent episode at the end
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
  local leftover
  leftover=$(pgrep -f "[.]venv/bin/splat-explorer dashboard" || true)
  for pid in $leftover; do
    echo "    Stopping leftover dashboard (pid $pid)"
    kill "$pid" || true
  done
  sleep 0.4
}

RENDER_TEST=0 EPISODE=0
for arg in "$@"; do
  case "$arg" in
    --render-test) RENDER_TEST=1 ;;
    --episode)     EPISODE=1 ;;
    --stop)
      echo "==> Stopping splat-explorer stack"
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

echo "==> [2/4] Building splat-explorer image + host extras"
docker compose build

HOST_DASHBOARD=0
if [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ]; then
  HOST_DASHBOARD=1
  echo "    Apple Silicon: installing [apple] extra (gsplat-mlx / MLX) into .venv"
  if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
  fi
  .venv/bin/pip install -e ".[viewer,vlm,apple]"
fi

echo "==> [3/4] (Re)starting viser viewer (:8080) + episode dashboard (:8090)"
stop_host_dashboard
# Drop the last visor pointer so a restart loads the catalog room, not a leftover repair PLY.
rm -f outputs/live/scene.json
docker compose down --remove-orphans
# Ports may be held by stale locally-run instances ("splat-explorer viewer" /
# "splat-explorer dashboard" outside Docker) — kill our own, warn about others.
SERVICES=""
free_port() {  # free_port <port> <cmd substring> ; returns 1 if a foreign process holds it
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
free_port 8080 "splat-explorer viewer"    && SERVICES="$SERVICES viewer" \
  || echo "    Skipping viewer (port busy)"
free_port 8081 "splat-explorer viewer"    || true
free_port 8090 "splat-explorer dashboard" && DASH_FREE=1 \
  || { echo "    Skipping dashboard (port busy)"; DASH_FREE=0; }
if [ -n "$SERVICES" ]; then
  # shellcheck disable=SC2086
  docker compose up -d $SERVICES
fi
if [ "$HOST_DASHBOARD" = 1 ] && [ "$DASH_FREE" = 1 ]; then
  mkdir -p outputs
  echo "    Starting host dashboard on :8090 (Metal / gsplat-mlx; not Docker)"
  export CLIRELAY_BASE_URL="${CLIRELAY_BASE_URL:-http://localhost:8317/v1}"
  export VISER_RENDER_URL="${VISER_RENDER_URL:-http://localhost:8081}"
  export VISER_VIEWER_URL="${VISER_VIEWER_URL:-http://localhost:8080}"
  nohup .venv/bin/splat-explorer dashboard >> outputs/dashboard.log 2>&1 &
  echo $! > outputs/dashboard.pid
  echo "    Host dashboard pid $(cat outputs/dashboard.pid)  (logs: outputs/dashboard.log)"
  printf "    Waiting for host dashboard"
  for _ in $(seq 1 40); do
    if curl -s -o /dev/null "http://127.0.0.1:8090"; then echo "  up"; break; fi
    printf "."
    sleep 0.5
  done
  printf "    Waiting for catalog scene"
  for _ in $(seq 1 60); do
    status=$(curl -s "http://127.0.0.1:8090/api/state" | python3 -c "import json,sys; print((json.load(sys.stdin).get('scene') or {}).get('status') or '')" 2>/dev/null || true)
    if [ "$status" = "ready" ]; then echo "  ready"; break; fi
    if [ "$status" = "error" ]; then echo "  ERROR"; break; fi
    printf "."
    sleep 0.5
  done
  echo ""
elif [ "$HOST_DASHBOARD" != 1 ] && [ "$DASH_FREE" = 1 ]; then
  docker compose up -d dashboard
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
echo "  Spectator (HD) : http://localhost:8090/spectator  (viewing only)"
echo "  CLI episode    : export CLIRELAY_API_KEY=sk-... && \\"
echo "                   splat-explorer --config configs/cli_relay.yaml explore"
echo "  Stop all       : scripts/start.sh --stop"
