#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

APP_FILE="${APP_FILE:-app.py}"
# Default to all interfaces so the app is reachable remotely.
# If you only want localhost access, run with: HOST=127.0.0.1 ./start.sh
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8501}"
VENV_DIR="${VENV_DIR:-myenv}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
PID_FILE="${PID_FILE:-$ROOT_DIR/.rag_chatbot.pid}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/streamlit.log}"

# Performance / stability defaults for embedding models (CPU-only).
# Override if you know CUDA is correctly configured.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Security-related toggles (keep enabled by default).
# Set to 1 to disable if you're behind a trusted reverse proxy / internal network.
DISABLE_CORS="${DISABLE_CORS:-0}"
DISABLE_XSRF="${DISABLE_XSRF:-0}"

usage() {
  cat <<'EOF'
Usage:
  ./start.sh              Start the Streamlit RAG chatbot in background
  ./start.sh --foreground Run in foreground (no PID file)
  ./start.sh --status     Show running status

Environment variables:
  APP_FILE=app.py         Streamlit entry file
  HOST=0.0.0.0            Bind address (use 127.0.0.1 for localhost-only)
  PORT=8501               Port
  VENV_DIR=myenv          Virtualenv directory (expects bin/activate)
  PID_FILE=.rag_chatbot.pid
  LOG_DIR=logs
  LOG_FILE=logs/streamlit.log
  DISABLE_CORS=0          Set to 1 to pass --server.enableCORS false
  DISABLE_XSRF=0          Set to 1 to pass --server.enableXsrfProtection false
  CUDA_VISIBLE_DEVICES=   Defaults to empty (CPU-only). Set to e.g. 0 to enable GPU.
  TOKENIZERS_PARALLELISM=false  Disable tokenizer parallelism warnings

Examples:
  HOST=0.0.0.0 PORT=8501 ./start.sh
  DISABLE_CORS=1 DISABLE_XSRF=1 ./start.sh
  ./start.sh --foreground
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
      echo "URL: http://${HOST}:${PORT}"
      echo "Log: ${LOG_FILE}"
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

mkdir -p "$LOG_DIR"

if [[ "${1:-}" != "--foreground" ]]; then
  if [[ -f "$PID_FILE" ]]; then
    existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if is_running "$existing_pid"; then
      echo "Already running (pid=$existing_pid)."
      exit 0
    fi
    echo "Removing stale PID file."
    rm -f "$PID_FILE"
  fi
fi

# Prefer using the repo venv if present.
PYTHON_BIN="python3"
if [[ -x "$ROOT_DIR/$VENV_DIR/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/$VENV_DIR/bin/python"
fi

# Ensure streamlit can be invoked.
STREAMLIT_CMD=()
if command -v streamlit >/dev/null 2>&1; then
  STREAMLIT_CMD=(streamlit)
else
  STREAMLIT_CMD=("$PYTHON_BIN" -m streamlit)
fi

if [[ ! -f "$APP_FILE" ]]; then
  echo "ERROR: APP_FILE not found: $APP_FILE"
  exit 1
fi

EXTRA_ARGS=(--server.headless true)

if [[ "$DISABLE_CORS" == "1" ]]; then
  EXTRA_ARGS+=(--server.enableCORS false)
fi

if [[ "$DISABLE_XSRF" == "1" ]]; then
  EXTRA_ARGS+=(--server.enableXsrfProtection false)
fi

COMMON_ARGS=(run "$APP_FILE" --server.address "$HOST" --server.port "$PORT" "${EXTRA_ARGS[@]}")

if [[ "${1:-}" == "--foreground" ]]; then
  echo "Starting in foreground: ${STREAMLIT_CMD[*]} ${COMMON_ARGS[*]}"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM" \
    exec "${STREAMLIT_CMD[@]}" "${COMMON_ARGS[@]}"
fi

# Run in background with log capture.
nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM" \
  "${STREAMLIT_CMD[@]}" "${COMMON_ARGS[@]}" >>"$LOG_FILE" 2>&1 &
PID=$!

echo "$PID" >"$PID_FILE"

echo "Started RAG Chatbot (pid=$PID)"
echo "URL: http://${HOST}:${PORT}"
echo "Log: ${LOG_FILE}"
