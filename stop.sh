#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PID_FILE="${PID_FILE:-$ROOT_DIR/.rag_chatbot.pid}"

usage() {
  cat <<'EOF'
Usage:
  ./stop.sh            Stop the running RAG chatbot (uses PID file)
  ./stop.sh --status   Show running status
  ./stop.sh --force    SIGKILL if it doesn't stop quickly

Environment variables:
  PID_FILE=.rag_chatbot.pid
EOF
}

is_running() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "$pid" 2>/dev/null
}

status() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if is_running "$pid"; then
      echo "RAG Chatbot is running (pid=$pid)"
      return 0
    fi
    echo "Stale PID file found (pid=${pid}); not running."
    return 1
  fi
  echo "RAG Chatbot is not running."
  return 1
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--status" ]]; then
  status
  exit $?
fi

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file found at $PID_FILE. Nothing to stop."
  exit 0
fi

PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -z "$PID" ]]; then
  echo "PID file is empty; removing it."
  rm -f "$PID_FILE"
  exit 0
fi

if ! is_running "$PID"; then
  echo "Process not running (pid=$PID); removing PID file."
  rm -f "$PID_FILE"
  exit 0
fi

echo "Stopping RAG Chatbot (pid=$PID)"

# Try to stop children first (Streamlit can spawn subprocesses)
if command -v pkill >/dev/null 2>&1; then
  pkill -TERM -P "$PID" 2>/dev/null || true
fi

kill -TERM "$PID" 2>/dev/null || true

# Wait up to ~10s
for _ in {1..20}; do
  if ! is_running "$PID"; then
    rm -f "$PID_FILE"
    echo "Stopped."
    exit 0
  fi
  sleep 0.5
  if command -v pkill >/dev/null 2>&1; then
    pkill -TERM -P "$PID" 2>/dev/null || true
  fi
  kill -TERM "$PID" 2>/dev/null || true
done

if [[ "${1:-}" == "--force" ]]; then
  echo "Force killing (pid=$PID)"
  if command -v pkill >/dev/null 2>&1; then
    pkill -KILL -P "$PID" 2>/dev/null || true
  fi
  kill -KILL "$PID" 2>/dev/null || true
  rm -f "$PID_FILE"
  echo "Killed."
  exit 0
fi

echo "Still running. Re-run with --force if needed."
exit 1
