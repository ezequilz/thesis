#!/usr/bin/env bash
# Startup sequence: build the image, sanity-render test views, start the debug
# viewer, and run an agent episode (scripted policy unless configured otherwise).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> [1/4] Building image"
docker compose build

echo "==> [2/4] Rendering initial test views -> outputs/test_views/"
docker compose run --rm render-test

echo "==> [3/4] Starting viser debug viewer at http://localhost:8080"
docker compose up -d viewer

echo "==> [4/4] Running an agent episode -> outputs/episodes/"
docker compose run --rm harness

echo "Done. Viewer is still running (docker compose down to stop it)."
