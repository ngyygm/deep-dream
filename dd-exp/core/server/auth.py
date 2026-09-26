"""
Authentication and Authorization Module for Deep-Dream API.

Provides:
- API key authentication for machine-to-machine access
- JWT token authentication for user sessions
- Role-based access control (RBAC)

Environment Variables:
    DEEPDREAM_SECRET_KEY: Secret key for JWT signing (required)
    DEEPDREAM_API_KEYS_FILE: Path to file containing valid API keys (optional)
    DEEPDREAM_DEFAULT_API_KEY: Default API key for development (optional)
"""
from __future__ import annotations

import hmac
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Set, Tuple

import jwt

logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.environ.get("DEEPDREAM_SECRET_KEY")
if not SECRET_KEY:
    logger.warning(
        "DEEPDREAM_SECRET_KEY not set - authentication will be DISABLED. "
        "Set this environment variable in production!"
    )

# JWT Configuration
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Default permissions for different authentication methods
DEFAULT_PERMISSIONS = {
    "api_key": {
        "read",
        "find:read",
        "remember:write",
        "concepts:read",
        "documents:read",
    },
    "jwt": {
        "read",
        "find:read",
        "remember:write",
        "concepts:read",
        "concepts:write",
        "documents:read",
        "documents:write",
        "graphs:write",
        "system:write",
    },
}

# In-memory API key store (in production, use a database)
_API_KEYS: Dict[str, Set[str]] = {}
_ALLOW_DEV_KEY: bool | None = None


def _strict_bool(value: Any, default: bool = False) -> bool:
    """Parse configuration booleans without ``bool('false')`` surprises."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return default


def authentication_configured(config: Dict[str, Any] | None = None) -> bool:
    """Return whether request authentication should be active.

    Local development has historically worked without a secret key.  Treating
    the absence of a secret as an authenticated deployment is misleading (and
    makes the UI appear to work while silently accepting unauthenticated
    writes).  An explicit ``auth.enabled`` value always wins; otherwise auth
    is enabled only when a signing key, API-key file, or strict mode is
    configured.
    """
    auth_config = (config or {}).get("auth") or {}
    if "enabled" in auth_config and auth_config.get("enabled") is not None:
        return _strict_bool(auth_config.get("enabled"))
    return bool(
        SECRET_KEY
        or auth_config.get("strict_mode", False)
        or auth_config.get("api_keys_file")
        or os.environ.get("DEEPDREAM_API_KEYS_FILE")
    )


def load_api_keys(file_path: str | None = None) -> None:
    """
    Load API keys from a JSON file.

    File format:
    {
        "api_key_name": {
            "key": "actual_api_key_hash",
            "permissions": ["permission1", "permission2"]
        }
    }

    Args:
        file_path: Path to API keys file (default: from env var)
    """
    # Reloading auth must not leave keys from a previous configuration valid.
    # This matters for long-running workers and for tests that construct more
    # than one application in the same Python process.
    _API_KEYS.clear()

    if file_path is None:
        file_path = os.environ.get("DEEPDREAM_API_KEYS_FILE")

    if not file_path or not os.path.exists(file_path):
        # A predictable key is useful for an explicitly local development
        # server, but must never silently remain valid in strict mode.
        if _ALLOW_DEV_KEY is False:
            logger.error("No API-key file configured; refusing the development fallback key")
            return
        default_key = os.environ.get("DEEPDREAM_DEFAULT_API_KEY", "dev-key-insecure")
        _API_KEYS[default_key] = DEFAULT_PERMISSIONS["api_key"]
        logger.warning("Using default development API key - NOT FOR PRODUCTION")
        return

    try:
        with open(file_path) as f:
            data = json.load(f)

        for name, config in data.items():
            key = config.get("key", "")
            permissions = set(config.get("permissions", []))
            if key:
                _API_KEYS[key] = permissions
                logger.info(f"Loaded API key: {name}")

    except Exception as e:
        logger.error(f"Failed to load API keys from {file_path}: {e}")


def init_auth(config: Dict[str, Any] | None = None) -> None:
    """
    Initialize authentication module. Call during app startup.

    Args:
        config: Optional configuration dictionary with auth settings:
            - auth.enabled: Enable/disable authentication (default: True if SECRET_KEY set)
            -.auth.api_keys_file: Path to API keys file
            - auth.strict_mode: Require auth even if SECRET_KEY not set (default: False)
    """
    global _ALLOW_DEV_KEY
    config = config or {}
    auth_config = config.get("auth", {})
    if not isinstance(auth_config, dict):
        auth_config = {}
    strict_mode = _strict_bool(auth_config.get("strict_mode"), bool(SECRET_KEY))
    host = str(config.get("host", "127.0.0.1")).strip().lower()
    loopback = host in {"127.0.0.1", "localhost", "::1"}
    # A predictable development credential is only available through an
    # explicit opt-in on a loopback listener.  Merely setting a production
    # SECRET_KEY must never silently create a second known credential.
    _ALLOW_DEV_KEY = (
        (
            _strict_bool(auth_config.get("allow_dev_key", False))
            or (
                not SECRET_KEY
                and not (auth_config.get("api_keys_file") or os.environ.get("DEEPDREAM_API_KEYS_FILE"))
                and ("enabled" not in auth_config or auth_config.get("enabled") is None)
            )
        )
        and loopback and not strict_mode
    )

    # Check if authentication is explicitly disabled
    if not _strict_bool(auth_config.get("enabled"), True):
        # Keep the legacy helper behaviour available to local unit tests, but
        # no request is authenticated when the app explicitly disables auth.
        _ALLOW_DEV_KEY = True
        _API_KEYS.clear()
        logger.info("Authentication explicitly disabled via config")
        return

    # Load API keys from config or environment
    api_keys_file = auth_config.get("api_keys_file") or os.environ.get("DEEPDREAM_API_KEYS_FILE")
    load_api_keys(api_keys_file)

    # Log authentication status
    if SECRET_KEY:
        logger.info("Authentication enabled with SECRET_KEY")
    else:
        if auth_config.get("strict_mode", False):
            logger.error(
                "STRICT MODE: DEEPDREAM_SECRET_KEY not set but strict_mode=True. "
                "Authentication will FAIL all requests!"
            )
        else:
            logger.warning(
                "DEEPDREAM_SECRET_KEY not set - authentication will be DISABLED. "
                "Set this environment variable or enable strict_mode for production!"
            )


def _validate_api_key(api_key: str) -> Tuple[bool, Set[str]]:
    """
    Validate an API key and return associated permissions.

    Uses constant-time comparison to prevent timing attacks.

    Args:
        api_key: The API key to validate

    Returns:
        (is_valid, permissions_set) tuple
    """
    if not _API_KEYS:
        if _ALLOW_DEV_KEY is False:
            return False, set()
        default_key = os.environ.get("DEEPDREAM_DEFAULT_API_KEY", "dev-key-insecure")
        if hmac.compare_digest(api_key, default_key):
            return True, DEFAULT_PERMISSIONS["api_key"]
        return False, set()

    # Constant-time comparison for all keys to prevent timing attacks
    for stored_key, perms in _API_KEYS.items():
        if hmac.compare_digest(api_key, stored_key):
            return True, perms
    return False, set()


def _validate_jwt_token(token: str) -> Tuple[bool, Set[str], dict | None]:
    """
    Validate a JWT token and return permissions and payload.

    Args:
        token: The JWT token to validate

    Returns:
        (is_valid, permissions_set, payload) tuple
    """
    if not SECRET_KEY:
        logger.warning("JWT validation attempted but SECRET_KEY not set")
        return False, set(), None

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["exp", "user_id"]}
        )

        permissions = set(payload.get("permissions", DEFAULT_PERMISSIONS["jwt"]))
        return True, permissions, payload

    except jwt.ExpiredSignatureError:
        return False, set(), None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT token: {e}")
        return False, set(), None


def create_jwt_token(user_id: str, permissions: List[str] | None = None) -> str:
    """
    Create a JWT token for a user.

    Args:
        user_id: Unique user identifier
        permissions: List of permissions (uses defaults if None)

    Returns:
        JWT token string
    """
    if not SECRET_KEY:
        raise RuntimeError("Cannot create JWT token: DEEPDREAM_SECRET_KEY not set")

    payload = {
        "user_id": user_id,
        "permissions": permissions or list(DEFAULT_PERMISSIONS["jwt"]),
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
    }

    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)


# POST endpoints that only calculate/read data.  They are intentionally kept
# explicit so a newly-added mutating endpoint is denied by default.
_READ_ONLY_POST_PATHS = frozenset({
    "/api/v1/find",
    "/api/v1/traverse",
    "/api/v1/concepts/batch-neighbors",
    "/api/v1/concepts/search",
    "/api/v1/concepts/suggest",
    "/api/v1/concepts/traverse",
    "/api/v1/documents/graph",
    "/api/v1/documents/graph/chunk",
    "/api/v1/documents/graph/outline",
})


def required_permission(method: str, path: str) -> str | None:
    """Map an API request to the smallest permission it needs.

    ``None`` is reserved for CORS preflight and framework-only requests.  A
    read permission is still required for ordinary GETs when auth is active.
    """
    method = (method or "GET").upper()
    path = path or "/"
    if method == "OPTIONS":
        return None

    is_read = method in {"GET", "HEAD"} or path in _READ_ONLY_POST_PATHS
    if is_read:
        if path.startswith("/api/v1/find") or path == "/api/v1/traverse":
            return "find:read"
        if path.startswith("/api/v1/concepts"):
            return "concepts:read"
        if path.startswith("/api/v1/documents") or path.startswith("/api/v1/episodes"):
            return "documents:read"
        if path.startswith("/api/v1/remember"):
            return "read"
        return "read"

    if path.startswith("/api/v1/remember"):
        return "remember:write"
    if path.startswith("/api/v1/concepts"):
        return "concepts:write"
    if path.startswith("/api/v1/documents") or path.startswith("/api/v1/vaults"):
        return "documents:write"
    if path.startswith("/api/v1/graphs"):
        return "graphs:write"
    if path.startswith("/api/v1/system/config"):
        return "system:write"
    return "write"


def has_permission(permissions: Set[str] | None, required: str | None) -> bool:
    """Check a permission, supporting explicit admin/write wildcards."""
    if required is None:
        return True
    perms = permissions or set()
    if "admin" in perms or "*" in perms or required in perms:
        return True
    if required == "read" and "read" in perms:
        return True
    if required.endswith(":read") and "read" in perms:
        return True
    if required.endswith(":write") and "write" in perms:
        return True
    return False
