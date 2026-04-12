# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# --- Build Stage ---
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# 1. Install system dependencies (git for VCS, curl for uv)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# 2. Ensure uv is available for high-speed dependency management
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# 3. Copy only dependency files first to leverage Docker layer caching
COPY pyproject.toml uv.lock* ./

# 4. ⚡️ SPEED UP: Force lightweight CPU-only PyTorch to avoid 4GB+ timeouts
ENV UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"

# 5. Install dependencies using uv cache mounts
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-editable

# 6. Copy the entire project (since Dockerfile is now at root)
COPY . .

# 7. Finalize the environment installation
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-editable

# --- Final Runtime Stage ---
FROM ${BASE_IMAGE}
WORKDIR /app

# 8. Copy the virtual environment and project from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

# 9. Set PATH and PYTHONPATH for consistent imports
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# 10. Enable the OpenEnv interactive Web UI
ENV ENABLE_WEB_INTERFACE=true

# 11. Health check to ensure the server is responding
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 12. Expose the standard hackathon port
EXPOSE 8000

# 13. Start the FastAPI server
CMD ["uvicorn", "bug_triage_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]