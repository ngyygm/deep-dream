"""Load and fingerprint the canonical Deep-Dream runtime policy."""
from __future__ import annotations

from functools import lru_cache
import hashlib
from importlib import resources


RUNTIME_POLICY_VERSION = "1.0.0"


@lru_cache(maxsize=1)
def load_runtime_policy() -> str:
    """Return the packaged policy used by every autonomous memory agent."""
    return resources.files("core.agent").joinpath(
        "policies/deep_dream_runtime.md"
    ).read_text(encoding="utf-8")


def runtime_policy_metadata() -> dict[str, str]:
    text = load_runtime_policy()
    return {
        "name": "deep-dream-runtime-policy",
        "version": RUNTIME_POLICY_VERSION,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
