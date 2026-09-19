"""Settings for this application.

Only what this app decides lives here. Provider topology - model IDs, rate
limits, fallback order, API key names - belongs to LLMRoute and SearchRoute and
is configured in their own files; duplicating any of it here would mean two
places to update and one of them silently wrong.

Secrets are read from the environment by those libraries directly, which is why
no API key appears below.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TaskType = Literal["extraction", "clarification", "reasoning", "bulk", "drafting"]


class Settings(BaseSettings):
    """Secrets and deployment values, read from the environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str = "postgresql://procurement:procurement@localhost:5433/procurement_agent"

    #: Where LLMRoute persists its quota ledger. Outside the project so several
    #: checkouts share one count of what the free tiers have actually spent -
    #: the quotas are per account, not per working copy.
    llm_ledger_path: Path = Path.home() / ".cache" / "procurement_agent" / "llm_ledger.json"

    api_key: str = "dev-local-key"

    #: Browser origins allowed to call this API, comma-separated.
    #:
    #: An explicit list, never `*`. The requests carry an `X-API-Key` header,
    #: and a wildcard origin on a credentialed API is precisely the case the
    #: same-origin policy exists to prevent - any page the user visits could
    #: then drive this agent and read its archived contact data.
    #:
    #: Defaults cover the ports the common frontend dev servers use, so a local
    #: UI works without configuration.
    cors_origins: str = (
        "http://localhost:3000,http://localhost:5173,http://localhost:8080,"
        "http://127.0.0.1:3000,http://127.0.0.1:5173,http://127.0.0.1:8080"
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    #: Ceiling on what ONE run may spend, across every node. Distinct from
    #: SearchRoute's own ledger, which is a monthly per-provider balance shared
    #: machine-wide: this stops a single wide fan-out consuming the month in one
    #: request.
    #:
    #: 25 rather than the previous 12 because 12 was never enough - a measured
    #: run spent it on the vendor search alone and still truncated. It is
    #: affordable now that keyless providers are no longer charged: the academic
    #: pass and the reference lookups cost nothing, so this is spent almost
    #: entirely on finding vendors.
    search_credits_per_session: int = 25
    #: Hits requested per query.
    results_per_query: int = 5
    max_fanout: int = 3
    #: Extra credits for the academic-literature pass, which runs only when the
    #: open-web search found few real vendors.
    research_search_credits: int = 4
    #: Turn the academic pass off entirely without touching code.
    enable_research_search: bool = True

    # --- Local crawling -----------------------------------------------------
    #: crawl4ai runs ahead of the paid extract tier. Every page it fetches is a
    #: search credit not spent, which is what makes contact-page following
    #: affordable. Disable to fall back to paid extraction only.
    enable_crawl4ai: bool = True
    #: Contact pages fetched per vendor, on top of the product page.
    crawl_max_contact_pages: int = 3
    #: These are other companies' websites. Keep concurrency low and identify
    #: ourselves honestly.
    crawl_concurrency: int = 2
    crawl_timeout: int = 25
    crawl_respect_robots: bool = True
    crawl_user_agent: str = (
        "ProcurementResearchBot/0.1 (industrial sourcing research; "
        "contact via the operator of this deployment)"
    )

    # --- Run archive --------------------------------------------------------
    #: Every completed run is written here as one JSON file. This is a record,
    #: not a cache: the graph never reads it back, so a stored run can never
    #: silently stand in for a fresh search. Retrieval is an explicit CLI or API
    #: call.
    runs_dir: Path = PROJECT_ROOT / "runs"
    enable_run_archive: bool = True

    # --- Observability ------------------------------------------------------
    #: Langfuse is optional and off unless both keys are set. The local trace in
    #: every archived run does not depend on it.
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    #: Cloud, or a self-hosted instance. Read by the Langfuse SDK itself; named
    #: here so it is documented rather than invisible.
    langfuse_host: str = "https://cloud.langfuse.com"

    #: Pin a run to whichever provider served its first call. Off by default:
    #: on free tiers, spreading across providers as quota allows beats
    #: concentrating load on one provider's limit, and this graph's tasks
    #: deliberately span tiers so a single pin does not apply cleanly anyway.
    #: Turn on when consistent model behaviour within one run matters more.
    llm_sticky_session: bool = False

    # --- Search-grounded research -------------------------------------------
    #: Look the material up before asking clarifying questions, so the questions
    #: are not built on a misreading of a trade name. Off-switch provided
    #: because this is the one node the buyer waits on.
    enable_spec_lookup: bool = True

    log_level: str = "INFO"


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
