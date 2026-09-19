# Two stages: resolve dependencies once, then build a runtime image that also
# carries Chromium.
#
# Chromium is the reason this image is large and it is not optional.
# `crawl/fetcher.py` starts a real headless browser, and that is the only path
# here that renders JavaScript - vendor contact details frequently sit in a nav
# or footer that does not exist until the page has run. SearchRoute's extraction
# providers cannot see those. If the browser fails to start the agent still
# works (the crawler disables itself for the process and paid extraction takes
# over), but it loses the tier that makes contact-page following free.

# --------------------------------------------------------------------------- #
# builder
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# The routers are copied whole, not just their manifests. SearchRoute takes its
# version dynamically from `src/searchroute/__init__.py`, so a manifest-only
# copy cannot even be resolved - uv fails reading the version. They are still
# copied before the agent's own `src/`, because they change far less often, so
# the expensive dependency layer stays cached across ordinary edits.
COPY pyproject.toml uv.lock README.md ./
COPY LLMRoute/ LLMRoute/
COPY SearchRoute/ SearchRoute/

# `--no-install-project` builds the dependency tree without this project, so
# editing `src/` below does not invalidate it.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-editable

COPY src/ src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable

# --------------------------------------------------------------------------- #
# runtime
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    # Playwright's default is under $HOME, which differs between the root build
    # step and the non-root runtime user. Pinning it means the browser installed
    # below is the one crawl4ai finds.
    PLAYWRIGHT_BROWSERS_PATH=/opt/playwright

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

# Chromium plus its system libraries. `--with-deps` is what pulls the shared
# objects a headless browser needs on a slim image; installing the browser
# without them fails at launch rather than at build.
RUN playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*

COPY LLMRoute/ LLMRoute/
COPY SearchRoute/ SearchRoute/
COPY src/ src/

# Runs as a non-root user. The browser directory has to be readable by it, and
# /app/runs is the mount point for the archive volume.
RUN useradd --create-home --uid 10001 agent \
    && mkdir -p /app/runs /home/agent/.cache/procurement_agent \
    && chown -R agent:agent /app /home/agent /opt/playwright
USER agent

EXPOSE 8000

# `/health` needs the API key, so this checks the unauthenticated OpenAPI route
# instead - enough to prove the process is serving, which is what a healthcheck
# is for. `depends_on: service_healthy` in compose relies on this.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:8000/openapi.json', timeout=4).status_code==200 else 1)"

CMD ["uvicorn", "procurement_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
