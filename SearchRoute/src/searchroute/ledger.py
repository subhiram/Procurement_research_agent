"""Where quota counters are persisted.

Default is a JSON file under the user's state dir. It uses advisory file locking
and re-reads inside the lock on every write, so several worker processes sharing
one machine converge on a single count instead of clobbering each other — the
common case for an agent app running under gunicorn or a process pool.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

from .quota import QuotaState

try:  # POSIX only; Windows falls back to the in-process lock alone.
    import fcntl
except ImportError:  # pragma: no cover - platform dependent
    fcntl = None  # type: ignore[assignment]


class MemoryStore:
    """Non-persistent. Fine for tests and short-lived scripts; counters reset
    with the process, so a long-running app should not use this."""

    def __init__(self) -> None:
        self._data: dict[str, QuotaState] = {}
        self._lock = threading.Lock()

    def get(self, provider: str) -> QuotaState:
        with self._lock:
            state = self._data.get(provider)
            return QuotaState(**state.to_dict()) if state else QuotaState()

    def set(self, provider: str, state: QuotaState) -> None:
        with self._lock:
            self._data[provider] = state

    def all(self) -> dict[str, QuotaState]:
        with self._lock:
            return dict(self._data)

    def update(self, provider: str, mutate: Callable[[QuotaState], QuotaState]) -> QuotaState:
        with self._lock:
            current = self._data.get(provider) or QuotaState()
            updated = mutate(QuotaState(**current.to_dict()))
            self._data[provider] = updated
            return updated


def default_state_path() -> Path:
    """``$XDG_STATE_HOME/searchroute/quota.json``, or the platform equivalent."""
    base = os.environ.get("XDG_STATE_HOME")
    if base:
        return Path(base) / "searchroute" / "quota.json"
    return Path.home() / ".local" / "state" / "searchroute" / "quota.json"


class JSONFileStore:
    """The default store. Cross-process safe via an advisory lock."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path).expanduser() if path else default_state_path()
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict[str, QuotaState]:
        try:
            raw = json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            # A corrupt ledger must never take the app down — quota tracking is
            # advisory, so we start over rather than raise.
            return {}
        providers = raw.get("providers", {}) if isinstance(raw, dict) else {}
        return {k: QuotaState.from_dict(v) for k, v in providers.items()}

    def _write(self, data: dict[str, QuotaState]) -> None:
        payload = {
            "version": 1,
            "providers": {k: v.to_dict() for k, v in data.items()},
        }
        # Atomic replace so a crash mid-write can't leave a truncated ledger.
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def get(self, provider: str) -> QuotaState:
        with self._lock:
            return self._read().get(provider, QuotaState())

    def set(self, provider: str, state: QuotaState) -> None:
        with self._lock, self._file_lock():
            data = self._read()  # re-read inside the lock: another process may have written
            data[provider] = state
            self._write(data)

    def update(self, provider: str, mutate: Callable[[QuotaState], QuotaState]) -> QuotaState:
        """Read, mutate and write while holding the lock.

        This is the only safe way to change a counter. ``get`` then ``set``
        reads outside the lock, so concurrent workers all start from the same
        stale value and overwrite each other — most of the spend vanishes.
        """
        with self._lock, self._file_lock():
            data = self._read()
            updated = mutate(data.get(provider) or QuotaState())
            data[provider] = updated
            self._write(data)
            return updated

    def all(self) -> dict[str, QuotaState]:
        with self._lock:
            return self._read()

    def _file_lock(self):
        return _FileLock(self.path.with_suffix(".lock")) if fcntl else _NullLock()


class _FileLock:
    def __init__(self, path: Path):
        self.path = path
        self._fh = None

    def __enter__(self):
        self._fh = open(self.path, "w")
        fcntl.flock(self._fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if self._fh is not None:
            fcntl.flock(self._fh, fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None
        return False


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def make_store(spec: str | Path | object | None) -> object:
    """Accept a path, ``"memory"``, or an already-built store."""
    if spec is None:
        return JSONFileStore()
    if isinstance(spec, (str, Path)):
        if str(spec) == "memory":
            return MemoryStore()
        return JSONFileStore(spec)
    return spec
