"""
Gunicorn configuration file for production deployment
Optimized for 1K concurrent users
"""

import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
# Formula: (2 × CPU cores) + 1
# For 8 cores: (2 × 8) + 1 = 17, but use 8-12 for stability
workers = int(os.getenv("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000

# Timeout settings
timeout = 300  # 5 minutes (for AI processing)
keepalive = 5
graceful_timeout = 30

# Worker lifecycle
max_requests = 1000  # Restart worker after N requests (prevent memory leaks)
max_requests_jitter = 100  # Add randomness to prevent all workers restarting together

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"  # Log to stderr
loglevel = os.getenv("LOG_LEVEL", "info")
access_log_format = (
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" '
    "%(D)s"  # Request duration in microseconds
)

# Process naming
proc_name = "studyguru-pro"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (if using HTTPS directly)
# keyfile = None
# certfile = None


def when_ready(server):
    """Called just after the server is started."""
    server.log.info("StudyGuru Pro API server is ready. Spawning workers")


def worker_int(worker):
    """Called when a worker receives INT or QUIT signal."""
    worker.log.info("Worker received INT or QUIT signal")


def pre_fork(server, worker):
    """Called just before a worker is forked."""
    pass


def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info("Worker spawned (pid: %s)", worker.pid)


def post_worker_init(worker):
    """Called just after a worker has initialized the application."""
    worker.log.info("Worker initialized (pid: %s)", worker.pid)


def worker_abort(worker):
    """Called when a worker times out."""
    worker.log.warning("Worker timeout (pid: %s)", worker.pid)
