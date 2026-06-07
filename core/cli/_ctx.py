"""CliContext — shared context object passed through the Click call stack.

Imports from core.server.config, core.server.registry, and
core.storage.sqlite are performed lazily inside methods to avoid
triggering the heavy core/__init__.py import chain at module load
time.  This keeps ``--version`` and ``--help`` fast (< 200 ms).
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional


def _get_defaults() -> dict:
    """Lazy import of DEFAULTS to avoid triggering core/__init__.py."""
    from core.server.config import DEFAULTS
    return DEFAULTS


def _load_config(config_path: str) -> dict:
    """Lazy import of load_config."""
    from core.server.config import load_config
    return load_config(config_path)


def _get_library_id() -> str:
    """Lazy import of LIBRARY_ID."""
    from core.server.registry import LIBRARY_ID
    return LIBRARY_ID


class CliContext:
    """Wraps config loading, storage access, and registry.

    Instantiated once by the root Click group and attached to
    ``click.Context.obj`` so every subcommand can access it.

    All heavy imports are deferred until first use so that ``--version``
    and ``--help`` remain fast.
    """

    def __init__(self) -> None:
        self._config: Optional[Dict[str, Any]] = None
        self._config_path: Optional[str] = None
        self._click_params: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and cache the service config, falling back to DEFAULTS."""
        self._config_path = config_path
        if self._config is not None:
            return self._config
        try:
            self._config = _load_config(config_path)
        except Exception:
            self._config = copy.deepcopy(_get_defaults())
        if not self._config.get("storage_path"):
            self._config["storage_path"] = "./library"
        return self._config

    @property
    def config(self) -> Dict[str, Any]:
        if self._config is None:
            default_path = "service_config.json"
            if self._click_params:
                default_path = self._click_params.get("config", default_path)
            return self.load_config(default_path)
        return self._config

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    def get_registry(self):
        """Return a fresh GraphRegistry for the current config."""
        from core.server.registry import GraphRegistry
        return GraphRegistry(
            self.config.get("storage_path", "./library"),
            self.config,
        )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    @contextmanager
    def get_storage(self, graph_id: str, *, ensure: bool = False):
        """Context manager yielding a :class:`SQLiteGraphStorageManager`.

        Parameters
        ----------
        graph_id:
            Graph identifier (normalised to LIBRARY_ID in single-library mode).
        ensure:
            If *True*, create the graph directory and metadata when missing.
        """
        from core.server.registry import GraphRegistry
        from core.storage.sqlite import SQLiteGraphStorageManager

        registry = self.get_registry()
        graph_id = GraphRegistry.normalize_graph_id(graph_id)
        graph_dir = registry.graph_dir(graph_id)
        if ensure:
            graph_dir.mkdir(parents=True, exist_ok=True)
            registry.set_graph_metadata(graph_id)
        elif not graph_dir.is_dir():
            raise FileNotFoundError(f"Graph does not exist: {graph_id}")
        vector_dim = (self.config.get("storage") or {}).get("vector_dim", 1024)
        storage = SQLiteGraphStorageManager(
            storage_path=str(graph_dir),
            graph_id=graph_id,
            vector_dim=vector_dim,
        )
        try:
            yield storage
        finally:
            storage.close()

    # ------------------------------------------------------------------
    # Active graph
    # ------------------------------------------------------------------

    def get_active_graph(self, explicit: Optional[str] = None) -> str:
        """Return the active graph ID (always LIBRARY_ID in single-library mode)."""
        return _get_library_id()

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    @property
    def storage_root(self) -> Path:
        """Root directory for library storage."""
        return Path(self.config.get("storage_path") or ".")

    def graph_dir(self, graph_id: str) -> Path:
        """Resolve the on-disk directory for *graph_id*."""
        return self.get_registry().graph_dir(graph_id)
