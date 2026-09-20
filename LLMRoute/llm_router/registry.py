"""Loads models.yaml + limits.yaml into Endpoint objects.

This is the only module that knows the YAML file format. Everything downstream
works with `Endpoint`, which already carries its tier, its quota scope and its
resolved limits, so no other module needs to look anything up again.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.parse import urlsplit

import yaml

logger = logging.getLogger("llm_router.registry")

CONFIG_DIR = Path(__file__).parent / "config"

TIERS: tuple[str, ...] = ("S", "A", "B")
_TIER_RANK = {t: i for i, t in enumerate(TIERS)}

# Window suffix -> seconds. `mo` is a 30-day rolling window, not a calendar month.
_WINDOWS: dict[str, int] = {"m": 60, "h": 3600, "d": 86400, "w": 604800, "mo": 2592000}
_WINDOW_NAMES: dict[int, str] = {60: "minute", 3600: "hour", 86400: "day",
                                 604800: "week", 2592000: "month"}

# rpm / tpd / tpm_uncached / rpmo ...
_LIMIT_RE = re.compile(r"^(?P<kind>[rt])p(?P<window>mo|[mhdw])(?:_(?:un)?cached)?$")

REQUESTS = "requests"
TOKENS = "tokens"

VALID_SCOPES = ("per_model", "per_account", "per_project", "per_session")

#: How a provider is asked to produce schema-conforming output.
#:
#: This cannot be one global setting, because the right answer is a property of
#: the model, not of the request. Two measured cases:
#:
#: - Groq's gpt-oss models, under `function_calling`, intermittently answer in
#:   prose instead of calling the tool, and Groq rejects that with a 400.
#:   `json_schema` constrains decoding directly and does not have the failure.
#: - Gemma on Ollama has weak-to-absent tool calling, so `function_calling` does
#:   not reliably produce a call at all.
#:
#: LangChain implements `json_schema` and `json_mode` on the concrete chat model
#: classes, not on BaseChatModel, so a router that binds one method for a whole
#: ladder gets it wrong for at least one provider in that ladder.
VALID_STRUCTURED_OUTPUT_METHODS = ("function_calling", "json_schema", "json_mode")


class ConfigError(ValueError):
    """models.yaml or limits.yaml is malformed or internally inconsistent."""


def tier_rank(tier: str) -> int:
    """S=0, A=1, B=2. Lower is better."""
    try:
        return _TIER_RANK[tier]
    except KeyError:
        raise ConfigError(f"unknown tier {tier!r}, expected one of {TIERS}") from None


def window_name(seconds: int) -> str:
    return _WINDOW_NAMES.get(seconds, f"{seconds}s")


# --------------------------------------------------------------------------- #
# Limits
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Limit:
    """One quota counter: `value` units of `kind` per `window` seconds."""

    kind: str          # REQUESTS | TOKENS
    window: int        # seconds
    value: int

    @property
    def label(self) -> str:
        return f"{self.value} {self.kind}/{window_name(self.window)}"


def parse_limits(raw: Mapping[str, Any] | None, *, where: str) -> tuple[Limit, ...]:
    """Turn {'rpm': 30, 'tpd_uncached': 1000} into Limit objects.

    Unknown keys are an error rather than a silent no-op: a typo'd limit key
    would otherwise read as "unlimited", which is exactly the wrong default for
    a free tier.
    """
    if not raw:
        return ()
    out: list[Limit] = []
    for key, value in raw.items():
        match = _LIMIT_RE.match(str(key))
        if not match:
            raise ConfigError(f"{where}: unrecognised limit key {key!r}")
        if value is None:
            continue
        if not isinstance(value, int) or value <= 0:
            raise ConfigError(f"{where}: limit {key!r} must be a positive int, got {value!r}")
        kind = REQUESTS if match["kind"] == "r" else TOKENS
        out.append(Limit(kind=kind, window=_WINDOWS[match["window"]], value=int(value)))
    # tightest window first so can_call() reports the most immediate blocker
    return tuple(sorted(out, key=lambda l: (l.window, l.kind)))


# --------------------------------------------------------------------------- #
# Endpoint
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Endpoint:
    """One concrete, callable (provider, model_id) pair.

    Carries everything the ledger, the ladder and the provider wrapper need, so
    it can be passed around on its own.
    """

    provider: str
    model_id: str
    logical_model: str
    tier: str
    quota_scope: str = "per_model"
    limits: tuple[Limit, ...] = ()
    supports_streaming: bool = True
    priority: int = 100
    min_interval: float = 0.0
    cooldown_seconds: float = 60.0
    api_key_env: tuple[str, ...] = ()
    token_accounting: str = "total"
    #: How `with_structured_output()` should constrain this endpoint. Carried on
    #: the endpoint because the right answer differs by provider and the ladder
    #: only learns which provider it is using at call time. See
    #: VALID_STRUCTURED_OUTPUT_METHODS for why one global method is not enough.
    structured_output_method: str = "function_calling"
    #: Whether this endpoint can produce schema-conforming output at all. Some
    #: models support neither tool calling nor constrained decoding - Groq's
    #: allam-2-7b rejects both - and for a caller whose every request is
    #: structured, such an endpoint is not a fallback but a guaranteed failure.
    #: Declared rather than discovered, because discovering it costs a 400.
    supports_structured_output: bool = True

    @property
    def key(self) -> str:
        """Stable identity of this endpoint. Two logical models never share one."""
        return f"{self.provider}:{self.model_id}"

    @property
    def ledger_key(self) -> str:
        """The bucket this endpoint's usage counts against.

        per_model providers get one bucket per model. per_account, per_project
        and per_session providers pool every model into a single bucket, because
        that is how the provider actually meters them.
        """
        if self.quota_scope == "per_model":
            return f"{self.provider}:{self.model_id}"
        return f"{self.provider}:*"

    @property
    def api_key(self) -> str | None:
        """First non-empty value among this provider's candidate env vars."""
        for name in self.api_key_env:
            value = os.environ.get(name)
            if value:
                return value
        return None

    @property
    def has_credentials(self) -> bool:
        return not self.api_key_env or self.api_key is not None

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.provider}/{self.model_id} ({self.logical_model}, tier {self.tier})"


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    quota_scope: str
    enabled: bool
    api_key_env: tuple[str, ...]
    supports_streaming: bool
    cooldown_seconds: float
    token_accounting: str
    default_limits: tuple[Limit, ...]
    model_limits: Mapping[str, tuple[Limit, ...]]
    min_interval_default: float
    min_interval_by_model: Mapping[str, float]
    structured_output_method: str = "function_calling"
    supports_structured_output: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict)

    def limits_for(self, model_id: str) -> tuple[Limit, ...]:
        return self.model_limits.get(model_id, self.default_limits)

    def min_interval_for(self, model_id: str) -> float:
        return self.min_interval_by_model.get(model_id, self.min_interval_default)


def _as_env_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value)


def _parse_provider(name: str, raw: Mapping[str, Any]) -> ProviderConfig:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"provider {name!r}: expected a mapping")
    scope = raw.get("quota_scope", "per_model")
    if scope not in VALID_SCOPES:
        raise ConfigError(f"provider {name!r}: quota_scope {scope!r} not in {VALID_SCOPES}")

    raw_interval = raw.get("min_interval_seconds", 0.0)
    if isinstance(raw_interval, Mapping):
        interval_default = 0.0
        interval_by_model = {str(k): float(v) for k, v in raw_interval.items()}
    else:
        interval_default = float(raw_interval or 0.0)
        interval_by_model = {}

    model_limits = {
        str(model_id): parse_limits(limits, where=f"{name}.models.{model_id}")
        for model_id, limits in (raw.get("models") or {}).items()
    }

    structured = str(raw.get("structured_output_method", "function_calling"))
    if structured not in VALID_STRUCTURED_OUTPUT_METHODS:
        raise ConfigError(
            f"provider {name!r}: structured_output_method {structured!r} not in "
            f"{VALID_STRUCTURED_OUTPUT_METHODS}"
        )

    known = {
        "quota_scope", "api_key_env", "enabled", "supports_streaming",
        "cooldown_seconds", "token_accounting", "default_limits", "models",
        "min_interval_seconds", "structured_output_method",
        "supports_structured_output",
    }
    return ProviderConfig(
        name=name,
        quota_scope=scope,
        enabled=bool(raw.get("enabled", True)),
        api_key_env=_as_env_tuple(raw.get("api_key_env")),
        supports_streaming=bool(raw.get("supports_streaming", True)),
        cooldown_seconds=float(raw.get("cooldown_seconds", 60.0)),
        token_accounting=str(raw.get("token_accounting", "total")),
        default_limits=parse_limits(raw.get("default_limits"), where=f"{name}.default_limits"),
        model_limits=model_limits,
        min_interval_default=interval_default,
        min_interval_by_model=interval_by_model,
        structured_output_method=structured,
        supports_structured_output=bool(raw.get("supports_structured_output", True)),
        extra={k: v for k, v in raw.items() if k not in known},
    )


# --------------------------------------------------------------------------- #
# Ollama reachability
# --------------------------------------------------------------------------- #

#: Ollama is the only keyless endpoint: with no `api_key_env` to check,
#: nothing else stops it from being offered on a machine that has no daemon at
#: all - exactly the case on a hosted deploy (e.g. Streamlit Community Cloud)
#: with no local Ollama process. A cheap reachability probe against its
#: configured address settles that before the endpoint is ever handed to the
#: ladder, rather than the ladder discovering it only after a real call burns
#: its connection timeout against a closed port.
OLLAMA_BASE_URL_ENV = "OLLAMA_BASE_URL"
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"
OLLAMA_PROBE_TIMEOUT = 0.3
#: How long a probe result is trusted before checking again. Long enough that
#: a hot path of many calls in one turn does not re-probe for each of them,
#: short enough that starting the daemon mid-session is noticed quickly.
OLLAMA_PROBE_CACHE_SECONDS = 15.0

_ollama_probe_lock = threading.Lock()
_ollama_probe_cache: tuple[float, bool] | None = None


def ollama_base_url() -> str:
    """The Ollama daemon's address: `OLLAMA_BASE_URL`, or its localhost default."""
    return os.environ.get(OLLAMA_BASE_URL_ENV) or OLLAMA_DEFAULT_BASE_URL


def _ollama_daemon_reachable() -> bool:
    """Whether the configured Ollama daemon actually answers right now.

    A plain TCP connect, not a full HTTP round trip: fast enough to run on
    every call that touches the ladder, and a closed or refused port is
    exactly what "no daemon here" looks like. Cached briefly so it costs one
    connect per window rather than one per call.
    """
    global _ollama_probe_cache
    now = time.monotonic()
    with _ollama_probe_lock:
        if _ollama_probe_cache is not None:
            checked_at, reachable = _ollama_probe_cache
            if now - checked_at < OLLAMA_PROBE_CACHE_SECONDS:
                return reachable

    url = ollama_base_url()
    parsed = urlsplit(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=OLLAMA_PROBE_TIMEOUT):
            reachable = True
    except OSError:
        reachable = False

    with _ollama_probe_lock:
        previous = _ollama_probe_cache
        _ollama_probe_cache = (now, reachable)
    if not reachable and (previous is None or previous[1] is not False):
        logger.info(
            "ollama daemon not reachable at %s; excluding it from the ladder "
            "until it is (re-checked every %.0fs)",
            url, OLLAMA_PROBE_CACHE_SECONDS,
        )
    return reachable


def reset_ollama_probe_cache() -> None:
    """Drop the cached reachability result. Tests, or after starting Ollama."""
    global _ollama_probe_cache
    with _ollama_probe_lock:
        _ollama_probe_cache = None


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

class Registry:
    """The loaded model + limit configuration.

    Endpoints are built once at load time. `enabled_endpoints()` additionally
    drops providers that are switched off or have no API key in the environment,
    which is what the ladder should actually consider.
    """

    def __init__(
        self,
        models_config: Mapping[str, Any],
        limits_config: Mapping[str, Any],
    ) -> None:
        self._providers: dict[str, ProviderConfig] = {
            str(name): _parse_provider(str(name), raw or {})
            for name, raw in (limits_config.get("providers") or {}).items()
        }
        if not self._providers:
            raise ConfigError("limits.yaml defines no providers")

        self._non_chat_patterns: tuple[str, ...] = tuple(
            models_config.get("non_chat_patterns") or ()
        )
        self._tiers: dict[str, str] = {}
        self._by_model: dict[str, tuple[Endpoint, ...]] = {}
        self._rejected: list[str] = []
        self._load_models(models_config)

    # -- loading ----------------------------------------------------------- #

    def _is_non_chat(self, model_id: str) -> bool:
        low = model_id.lower()
        return any(fnmatch.fnmatch(low, p.lower()) for p in self._non_chat_patterns)

    def _load_models(self, models_config: Mapping[str, Any]) -> None:
        raw_models = models_config.get("models") or {}
        if not raw_models:
            raise ConfigError("models.yaml defines no models")

        for logical_model, spec in raw_models.items():
            logical_model = str(logical_model)
            if not isinstance(spec, Mapping):
                raise ConfigError(f"model {logical_model!r}: expected a mapping")
            tier = str(spec.get("tier", "")).upper()
            tier_rank(tier)  # validates
            self._tiers[logical_model] = tier

            endpoints: list[Endpoint] = []
            for raw_ep in spec.get("endpoints") or ():
                endpoint = self._build_endpoint(logical_model, tier, raw_ep)
                if endpoint is not None:
                    endpoints.append(endpoint)
            if not endpoints:
                logger.warning("model %r has no usable endpoints", logical_model)
            endpoints.sort(key=lambda e: (e.priority, e.provider))
            self._by_model[logical_model] = tuple(endpoints)

    def _build_endpoint(
        self, logical_model: str, tier: str, raw: Mapping[str, Any]
    ) -> Endpoint | None:
        where = f"model {logical_model!r}"
        if not isinstance(raw, Mapping):
            raise ConfigError(f"{where}: endpoint entries must be mappings")
        provider_name = str(raw.get("provider") or "")
        model_id = str(raw.get("model_id") or "")
        if not provider_name or not model_id:
            raise ConfigError(f"{where}: endpoint needs both `provider` and `model_id`")

        provider = self._providers.get(provider_name)
        if provider is None:
            raise ConfigError(
                f"{where}: provider {provider_name!r} has no entry in limits.yaml"
            )

        # Classifiers, guard models, audio and embedding models are never
        # selectable as chat endpoints, so they are refused at load time rather
        # than filtered later where a caller could bypass the filter.
        if self._is_non_chat(model_id):
            self._rejected.append(f"{provider_name}/{model_id}")
            logger.warning(
                "refusing non-chat model %s/%s declared under %r",
                provider_name, model_id, logical_model,
            )
            return None

        limits = provider.limits_for(model_id)
        if not limits:
            logger.warning(
                "no limits configured for %s/%s; it will be treated as unmetered",
                provider_name, model_id,
            )
        return Endpoint(
            provider=provider_name,
            model_id=model_id,
            logical_model=logical_model,
            tier=tier,
            quota_scope=provider.quota_scope,
            limits=limits,
            supports_streaming=bool(
                raw.get("supports_streaming", provider.supports_streaming)
            ),
            priority=int(raw.get("priority", 100)),
            min_interval=provider.min_interval_for(model_id),
            cooldown_seconds=provider.cooldown_seconds,
            api_key_env=provider.api_key_env,
            token_accounting=provider.token_accounting,
            # Per-endpoint override, so a single model that behaves differently
            # to the rest of its provider can be corrected without splitting the
            # provider in two.
            structured_output_method=str(
                raw.get("structured_output_method", provider.structured_output_method)
            ),
            supports_structured_output=bool(
                raw.get(
                    "supports_structured_output",
                    provider.supports_structured_output,
                )
            ),
        )

    # -- queries ------------------------------------------------------------ #

    @property
    def rejected_endpoints(self) -> tuple[str, ...]:
        """Endpoints dropped at load time for being non-chat models."""
        return tuple(self._rejected)

    def provider(self, name: str) -> ProviderConfig:
        try:
            return self._providers[name]
        except KeyError:
            raise KeyError(f"unknown provider {name!r}") from None

    def providers(self) -> tuple[str, ...]:
        return tuple(self._providers)

    def has_model(self, logical_model: str) -> bool:
        return logical_model in self._by_model

    def models(self) -> tuple[str, ...]:
        return tuple(self._by_model)

    def tier_of(self, logical_model: str) -> str:
        try:
            return self._tiers[logical_model]
        except KeyError:
            raise KeyError(f"unknown model {logical_model!r}") from None

    def get_endpoints(self, logical_model: str) -> tuple[Endpoint, ...]:
        """Every configured endpoint for a logical model, best priority first."""
        try:
            return self._by_model[logical_model]
        except KeyError:
            raise KeyError(
                f"unknown model {logical_model!r}; known models: {', '.join(self.models())}"
            ) from None

    def all_endpoints(self) -> Iterator[Endpoint]:
        for endpoints in self._by_model.values():
            yield from endpoints

    def is_usable(self, provider_name: str) -> bool:
        """Provider is switched on, its API key is present, and it answers.

        Ollama is exempt from the key check but not from this one: it is only
        "usable" when its daemon is actually reachable, so a deploy with no
        local Ollama at all does not have it offered anyway.
        """
        provider = self._providers.get(provider_name)
        if provider is None or not provider.enabled:
            return False
        if provider_name == "ollama" and not _ollama_daemon_reachable():
            return False
        if not provider.api_key_env:
            return True
        return any(os.environ.get(name) for name in provider.api_key_env)

    def enabled_endpoints(
        self,
        *,
        logical_model: str | None = None,
        tier: str | None = None,
        providers: Sequence[str] | None = None,
        streaming: bool = False,
        structured: bool = False,
        require_credentials: bool = True,
    ) -> tuple[Endpoint, ...]:
        """Endpoints the router may actually consider right now."""
        if logical_model is not None:
            pool: Iterable[Endpoint] = self.get_endpoints(logical_model)
        else:
            pool = self.all_endpoints()
        allowed = set(providers) if providers else None

        out = []
        for endpoint in pool:
            if allowed is not None and endpoint.provider not in allowed:
                continue
            if tier is not None and endpoint.tier != tier:
                continue
            if streaming and not endpoint.supports_streaming:
                continue
            if structured and not endpoint.supports_structured_output:
                continue
            provider = self._providers[endpoint.provider]
            if not provider.enabled:
                continue
            if endpoint.provider == "ollama" and not _ollama_daemon_reachable():
                continue
            if require_credentials and not endpoint.has_credentials:
                continue
            out.append(endpoint)
        return tuple(out)

    def models_in_tier(self, tier: str) -> tuple[str, ...]:
        return tuple(m for m, t in self._tiers.items() if t == tier)

    def _unusable_reason(self, provider_name: str) -> str:
        if self.is_usable(provider_name):
            return ""
        provider = self._providers.get(provider_name)
        if provider is None or not provider.enabled:
            return " (disabled)"
        if provider_name == "ollama":
            return f" (daemon unreachable at {ollama_base_url()})"
        return " (no key)"

    def describe(self) -> str:  # pragma: no cover - diagnostics
        lines = []
        for model in self.models():
            eps = self.get_endpoints(model)
            live = [e for e in eps if self.is_usable(e.provider)]
            lines.append(
                f"{model} [tier {self.tier_of(model)}] "
                f"{len(live)}/{len(eps)} endpoints usable: "
                + ", ".join(
                    f"{e.provider}/{e.model_id}{self._unusable_reason(e.provider)}"
                    for e in eps
                )
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def load_registry(
    models_path: str | os.PathLike[str] | None = None,
    limits_path: str | os.PathLike[str] | None = None,
) -> Registry:
    """Build a Registry from the two YAML files (defaults to the bundled ones)."""
    models_file = Path(models_path) if models_path else CONFIG_DIR / "models.yaml"
    limits_file = Path(limits_path) if limits_path else CONFIG_DIR / "limits.yaml"
    return Registry(_read_yaml(models_file), _read_yaml(limits_file))


_default_registry: Registry | None = None
_registry_lock = threading.Lock()


def default_registry() -> Registry:
    """Process-wide registry loaded from the bundled config, built once."""
    global _default_registry
    if _default_registry is None:
        with _registry_lock:
            if _default_registry is None:
                _default_registry = load_registry()
    return _default_registry


def reset_default_registry() -> None:
    """Drop the cached registry (tests, or after editing the YAML at runtime)."""
    global _default_registry
    with _registry_lock:
        _default_registry = None
