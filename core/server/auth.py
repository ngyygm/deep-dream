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
    },
}

# In-memory API key store (in production, use a database)
_API_KEYS: Dict[str, Set[str]] = {}


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
    if file_path is None:
        file_path = os.environ.get("DEEPDREAM_API_KEYS_FILE")

    if not file_path or not os.path.exists(file_path):
        # Load default development key
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
    config = config or {}
    auth_config = config.get("auth", {})

    # Check if authentication is explicitly disabled
    if not auth_config.get("enabled", True):
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
