# =============================================================================
# TraderBot Dockerfile
# Multi-stage build for Python 3.10 application
# =============================================================================

FROM python:3.10-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1000 traderbot \
    && useradd --uid 1000 --gid traderbot --shell /bin/bash --create-home traderbot

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Runtime Stage
# =============================================================================
FROM python:3.10-slim AS runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 traderbot \
    && useradd --uid 1000 --gid traderbot --shell /bin/bash --create-home traderbot

# Set working directory
WORKDIR /app

# Copy installed packages from base stage
COPY --from=base /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=traderbot:traderbot . .

# Make entrypoint executable
RUN chmod +x docker-entrypoint.sh

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data \
    && chown -R traderbot:traderbot /app

# Switch to non-root user
USER traderbot

# Expose ports
# 8001 - FastAPI Backend
# 8501 - Streamlit Frontend
EXPOSE 8001 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Set entrypoint
ENTRYPOINT ["./docker-entrypoint.sh"]
