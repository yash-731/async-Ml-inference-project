#!/usr/bin/env sh
set -eu
exec celery -A app.worker.celery_app:celery_app worker \
    --loglevel="${LOG_LEVEL:-INFO}" \
    --concurrency="${CELERY_CONCURRENCY:-2}" \
    --queues="${TASK_QUEUE:-forecasts}"
