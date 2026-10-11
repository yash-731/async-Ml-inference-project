FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies in their own layer: an app code change should not
# invalidate the (slow) dependency install. The stub package satisfies the
# editable install before the real source is copied in.
COPY pyproject.toml README.md ./
RUN mkdir -p app && touch app/__init__.py && pip install -e .

COPY alembic.ini ./
COPY alembic ./alembic
COPY scripts ./scripts
COPY app ./app

RUN chmod +x scripts/*.sh \
    && useradd --create-home --uid 1000 appuser \
    && mkdir -p /data/uploads \
    && chown -R appuser:appuser /app /data

USER appuser
EXPOSE 8000

CMD ["/app/scripts/start-api.sh"]
