"""Application entry point."""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack, asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from procurement_agent.api.routes import router
from procurement_agent.config import get_settings

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Hold the checkpointer's connection pool open for the app's lifetime."""
    load_dotenv()
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(levelname)-7s %(name)-45s %(message)s",
    )

    # Imported here so module import does not require a database.
    from procurement_agent.graph.build import open_graph
    from procurement_agent.llm.models import validate_models

    async with AsyncExitStack() as stack:
        app.state.graph = await stack.enter_async_context(open_graph(settings))

        # Providers deprecate models without notice; surface it at boot rather
        # than as a 404 deep inside a research run. Never fatal.
        for problem in await validate_models():
            log.warning("model config: %s", problem)

        yield

    app.state.graph = None


app = FastAPI(
    title="Procurement Research Agent",
    version="0.1.0",
    description=(
        "Clarifies a raw-material specification, researches the material, and "
        "returns verified vendor leads with contact details traceable to their "
        "source pages."
    ),
    lifespan=lifespan,
)

# Without this, a browser frontend cannot call this API at all - every
# cross-origin request fails before it reaches a route.
#
# The origin list is explicit and never `*`. These endpoints take an
# `X-API-Key` header and `GET /runs` returns real companies' contact details,
# which is exactly the combination a wildcard origin would expose to any page
# the user happens to visit.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    # `X-API-Key` has to be listed or the preflight rejects it. `Accept` and
    # `Cache-Control` are what a streaming client sends for text/event-stream.
    allow_headers=["X-API-Key", "Content-Type", "Accept", "Cache-Control"],
)
app.include_router(router)
