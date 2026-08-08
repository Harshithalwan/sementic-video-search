#!/usr/bin/env bash
#
# run_mac_latency_log.sh
# ---------------------
# One-shot runner for the semantic-video-search latency logging experiment.
# Designed for macOS.
#
#   - installs git/python via Homebrew if missing
#   - clones the repo if not already inside one
#   - creates a Python venv and installs requirements.txt
#   - runs the streaming latency-logging CLI against a video in testData/
#   - waits for a new latency_logs/*.jsonl, lets it accumulate events for ~60s
#   - commits the new log with a Mac identity and pushes to the remote
#
# Usage:
#   bash run_mac_latency_log.sh

set -euo pipefail

REPO_URL="https://github.com/Harshithalwan/sementic-video-search.git"
GIT_NAME="Harshit Halwan (mac Air m5)"
GIT_EMAIL="harshithalwan@gmail.com"
WAIT_FOR_EVENTS_SECS=60
LOG_FILE_TIMEOUT_SECS=1200   # 20 min; first run may download the HF model
VIDEO_WAIT_SECS=300          # 5 min for the user to drop a video into testData/

log()  { printf '\n[run] %s\n' "$*"; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0. Move to the directory that contains this script
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
log "Working directory: $PWD"

# ---------------------------------------------------------------------------
# 1. git (Homebrew only)
# ---------------------------------------------------------------------------
if ! command -v git >/dev/null 2>&1; then
  log "git not found."
  command -v brew >/dev/null 2>&1 || die "Homebrew is required. Install it first:
  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"
then re-run this script."
  log "Installing git via Homebrew..."
  brew install git
fi
log "git: $(git --version)"

# ---------------------------------------------------------------------------
# 2. Clone the repo if needed, else ensure origin points at the HTTPS URL
# ---------------------------------------------------------------------------
if git rev-parse --git-dir >/dev/null 2>&1; then
  log "Already inside a git repo: $PWD"
  if git remote get-url origin >/dev/null 2>&1; then
    CURRENT="$(git remote get-url origin)"
    if [ "$CURRENT" != "$REPO_URL" ]; then
      git remote set-url origin "$REPO_URL"
      log "origin updated to $REPO_URL"
    fi
  else
    git remote add origin "$REPO_URL"
    log "origin added as $REPO_URL"
  fi
else
  log "Not a git repo. Cloning $REPO_URL into a temporary folder..."
  CLONE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/semantic-video-search.XXXXXX")"
  git clone "$REPO_URL" "$CLONE_DIR"
  cd "$CLONE_DIR"
  log "Cloned into $CLONE_DIR - all subsequent work happens there."
fi

# ---------------------------------------------------------------------------
# 3. Python (Homebrew only, needs >= 3.10)
# ---------------------------------------------------------------------------
PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
  MAJ="$(python3 -c 'import sys; print(sys.version_info.major)' 2>/dev/null || echo 0)"
  MIN="$(python3 -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo 0)"
  if [ "$MAJ" -gt 3 ] || { [ "$MAJ" -eq 3 ] && [ "$MIN" -ge 10 ]; }; then
    PYTHON_BIN="$(command -v python3)"
  else
    log "python3 is too old ($MAJ.$MIN); installing a newer one via Homebrew."
  fi
fi
if [ -z "$PYTHON_BIN" ]; then
  command -v brew >/dev/null 2>&1 || die "Homebrew is required to install Python. Install it first:
  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"
then re-run this script."
  log "Installing Python via Homebrew..."
  brew install python
  PYTHON_BIN="$(command -v python3)"
  [ -n "$PYTHON_BIN" ] || die "python3 still not found after 'brew install python'."
fi
log "python: $("$PYTHON_BIN" -c 'import platform; print(platform.python_version())')"

# ---------------------------------------------------------------------------
# 4. Virtual environment + project dependencies
# ---------------------------------------------------------------------------
if [ ! -d .venv ]; then
  log "Creating virtual environment (.venv)..."
  "$PYTHON_BIN" -m venv .venv
fi
log "Upgrading pip..."
.venv/bin/python -m pip install --upgrade pip
log "Installing project dependencies (torch/transformers may take a while)..."
.venv/bin/pip install -r requirements.txt
log "Dependencies installed."

# ---------------------------------------------------------------------------
# 5. MPS/Apple-Silicon runtime environment
# ---------------------------------------------------------------------------
# Let ops without an MPS kernel fall back to CPU instead of crashing, and
# keep memory un-fragmented for the larger models.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# ---------------------------------------------------------------------------
# 6. Pick a video from testData
# ---------------------------------------------------------------------------
mkdir -p testData
VIDEO="$(find testData -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | head -n1 || true)"
if [ -z "$VIDEO" ]; then
  log "No .mp4 in testData/. Drop a video file in:
    $PWD/testData/
  It will be picked up automatically."
  DEADLINE=$(( $(date +%s) + VIDEO_WAIT_SECS ))
  while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    sleep 10
    VIDEO="$(find testData -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | head -n1 || true)"
    [ -n "$VIDEO" ] && break
  done
fi
[ -n "$VIDEO" ] || die "No video found in testData/ within $((VIDEO_WAIT_SECS/60)) min. Re-run once a video is present."
log "Using video: $VIDEO"

# ---------------------------------------------------------------------------
# 7. Run the latency logging command in the background
# ---------------------------------------------------------------------------
PRE_EXISTING="$(find latency_logs -maxdepth 1 -type f -name '*.jsonl' 2>/dev/null | sort || true)"
RUN_LOG="$(mktemp "${TMPDIR:-/tmp}/latency_run.XXXXXX")"
log "Starting stream with latency logging (output: $RUN_LOG)"
.venv/bin/python main.py \
  --mode stream \
  --source "$VIDEO" \
  --enable-yolo \
  --enable-activity-detection \
  --enable-latency-logging \
  >"$RUN_LOG" 2>&1 &
PID=$!

# ---------------------------------------------------------------------------
# 8. Wait for a new latency log file, then give it ~60s of events
# ---------------------------------------------------------------------------
NEW_FILE=""
DEADLINE=$(( $(date +%s) + LOG_FILE_TIMEOUT_SECS ))
log "Waiting for a new latency_logs/*.jsonl (first run may download the model)..."
while [ -z "$NEW_FILE" ] && [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if ! kill -0 "$PID" 2>/dev/null; then
    log "Process exited before a log file appeared. Last lines of run output:"
    tail -n 30 "$RUN_LOG" || true
    exit 1
  fi
  for f in latency_logs/*.jsonl; do
    [ -e "$f" ] || continue
    if ! grep -Fqx "$f" <<<"$PRE_EXISTING"; then
      NEW_FILE="$f"
      break
    fi
  done
  [ -z "$NEW_FILE" ] && sleep 5
done

if [ -z "$NEW_FILE" ]; then
  log "Timed out after $((LOG_FILE_TIMEOUT_SECS/60)) min waiting for a log file. Stopping run."
  kill -INT "$PID" 2>/dev/null || true
  exit 1
fi
log "New latency log detected: $NEW_FILE"

log "Giving the run $WAIT_FOR_EVENTS_SECS seconds to log events..."
WAIT_UNTIL=$(( $(date +%s) + WAIT_FOR_EVENTS_SECS ))
while [ "$(date +%s)" -lt "$WAIT_UNTIL" ]; do
  if ! kill -0 "$PID" 2>/dev/null; then
    log "Run finished on its own."
    break
  fi
  sleep 2
done

if kill -0 "$PID" 2>/dev/null; then
  log "Stopping the run (SIGINT, then SIGTERM if needed)..."
  kill -INT "$PID" 2>/dev/null || true
  for ((i=0; i<10; i++)); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 2
  done
  if kill -0 "$PID" 2>/dev/null; then
    kill -TERM "$PID" 2>/dev/null || true
    sleep 3
  fi
  if kill -0 "$PID" 2>/dev/null; then
    kill -9 "$PID" 2>/dev/null || true
  fi
  wait "$PID" 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 9. Git identity (repo-local), commit, push
# ---------------------------------------------------------------------------
log "Configuring repo-local git identity..."
git config --local user.name "$GIT_NAME"
git config --local user.email "$GIT_EMAIL"
log "Identity: $GIT_NAME <$GIT_EMAIL>"

log "Committing $NEW_FILE"
git add -- "$NEW_FILE"
git commit -m "Add latency logs for $(hostname)"

BRANCH="$(git branch --show-current)"
log "Pushing branch '$BRANCH' to origin..."
if git push origin "$BRANCH"; then
  log "Done. Latency log pushed successfully."
else
  printf 'Push failed. For HTTPS pushes GitHub needs authentication, e.g.:
  gh auth login
or configure a personal access token / credential helper, then re-run:
  git push origin %s\n' "$BRANCH" >&2
  exit 1
fi
