#!/usr/bin/env bash
# One-command start/restart of the full dev stack. Safe to re-run at any time.
#
#   1. CliRelay proxy      http://localhost:8317  (panel at /manage;
#                          cloned to $CLIRELAY_DIR on first run)
#   2. splat-explorer      image build + viser debug viewer at :8080
#
# Usage:
#   scripts/start.sh                 (re)start CliRelay + viewer
#   scripts/start.sh --render-test   ... and render sanity views first
#   scripts/start.sh --episode       ... and run an agent episode at the end
#   scripts/start.sh --stop          stop everything (project + CliRelay)
set -euo pipefail
cd "$(dirname "$0")/.."

CLIRELAY_DIR="${CLIRELAY_DIR:-$HOME/CliRelay}"
CLIRELAY_REPO="https://github.com/kittors/CliRelay.git"
CLIRELAY_URL="http://localhost:8317"

RENDER_TEST=0 EPISODE=0
for arg in "$@"; do
  case "$arg" in
    --render-test) RENDER_TEST=1 ;;
    --episode)     EPISODE=1 ;;
    --stop)
      echo "==> Stopping splat-explorer stack"
      docker compose down --remove-orphans
      if [ -d "$CLIRELAY_DIR" ]; then
        echo "==> Stopping CliRelay"
        (cd "$CLIRELAY_DIR" && docker compose down)
      fi
      exit 0 ;;
    *) echo "Unknown option: $arg (see header of this script)"; exit 2 ;;
  esac
done

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

echo "==> [2/4] Building splat-explorer image"
docker compose build

echo "==> [3/4] (Re)starting viser debug viewer at http://localhost:8080"
docker compose down --remove-orphans
# Port 8080 may be held by a stale locally-run viewer ("splat-explorer viewer"
# outside Docker) — kill our own, but only warn about anything else.
PORT_PID=$(lsof -ti tcp:8080 -sTCP:LISTEN || true)
if [ -n "$PORT_PID" ]; then
  PORT_CMD=$(ps -p "$PORT_PID" -o command= || true)
  if [[ "$PORT_CMD" == *"splat-explorer viewer"* ]]; then
    echo "    Killing stale local viewer (pid $PORT_PID)"
    kill "$PORT_PID" && sleep 1
  else
    echo "    WARNING: port 8080 is in use by another process, skipping viewer:"
    echo "      $PORT_PID  $PORT_CMD"
    echo "    Free it with: kill $PORT_PID"
    SKIP_VIEWER=1
  fi
fi
if [ "${SKIP_VIEWER:-0}" != 1 ]; then
  docker compose up -d viewer
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
echo "  Live episode   : export CLIRELAY_API_KEY=sk-... && \\"
echo "                   splat-explorer --config configs/cli_relay.yaml explore"
echo "  Stop all       : scripts/start.sh --stop"
