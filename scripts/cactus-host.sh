#!/bin/sh
set -eu

CHAT_MODEL="${CACTUS_CHAT_MODEL:-google/gemma-4-E2B-it}"

if ! command -v cactus >/dev/null 2>&1; then
  echo "Install Cactus first: brew install cactus-compute/cactus/cactus"
  exit 1
fi

cactus download "$CHAT_MODEL"

cactus serve "$CHAT_MODEL" --host 0.0.0.0 --port 8080 --backend metal --no-cloud-handoff &
chat_pid=$!

trap 'kill "$chat_pid" 2>/dev/null || true' INT TERM EXIT
wait