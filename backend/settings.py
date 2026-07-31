"""Runtime settings for the FastAPI backend."""

# Set from the CLI (python -m backend.main --enable-latency-logging).
# Read by the processing routers so per-component latency logging can be
# enabled for sessions started through the web UI.
LATENCY_LOGGING_ENABLED = False
