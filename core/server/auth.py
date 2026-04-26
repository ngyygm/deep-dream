"""
Authentication and Authorization Module for Deep-Dream API.

Provides:
- API key authentication for machine-to-machine access
- JWT token authentication for user sessions
- Role-based access control (RBAC)
- Permission decorators for Flask routes

Environment Variables:
    DEEPDREAM_SECRET_KEY: Secret key for JWT signing (required)
    DEEPDREAM_API_KEYS_FILE: Path to file containing valid API keys (optional)
    DEEPDREAM_DEFAULT_API_KEY: Default API key for development (optional)

Usage:
    from core.server.auth import require_auth, require_permission

    @app.route("/api/v1/remember", methods=["POST"])
    @require_auth
    @require_permission("remember:write")
    def remember():
        # ... endpoint code
"""
from __future__ import annotations

import hmac
import json
import logging
import os
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from flask import request, g, jsonify
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
        "entities:read",
        "relations:read",
        "episodes:read",
    },
    "jwt": {
        "read",
        "find:read",
        "remember:write",
        "entities:read",
        "entities:write",
        "relations:read",
        "relations:write",
        "episodes:read",
        "dream:read",
        "dream:write",
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

    Args:
        api_key: The API key to validate

    Returns:
        (is_valid, permissions_set) tuple
    """
    if not _API_KEYS:
        # If no keys loaded, allow the default development key
        default_key = os.environ.get("DEEPDREAM_DEFAULT_API_KEY", "dev-key-insecure")
        if hmac.compare_digest(api_key, default_key):
            return True, DEFAULT_PERMISSIONS["api_key"]
        return False, set()

    return api_key in _API_KEYS, _API_KEYS.get(api_key, set())


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

        # Check expiration
        exp = payload.get("exp")
        if exp and datetime.fromtimestamp(exp) < datetime.now():
            return False, set(), None

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
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
    }

    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)


def is_authenticated() -> bool:
    """Check if the current request is authenticated."""
    return hasattr(g, "authenticated") and g.authenticated


def get_user_id() -> str | None:
    """Get the authenticated user's ID from the current request."""
    return getattr(g, "user_id", None)


def get_permissions() -> Set[str]:
    """Get the authenticated user's permissions."""
    return getattr(g, "permissions", set())


def require_auth(f: Callable | None = None, /, optional: bool = False) -> Callable:
    """
    Authentication decorator for protected endpoints.

    Args:
        f: The function to decorate (if used without arguments)
        optional: If True, allow unauthenticated access (sets g.authenticated=False)

    Usage:
        @require_auth
        def protected_endpoint():
            # ...

        @require_auth(optional=True)
        def optional_auth_endpoint():
            # ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def decorated_function(*args, **kwargs):
            # If SECRET_KEY not set, allow all requests (development mode)
            if not SECRET_KEY:
                g.authenticated = True
                g.auth_method = "none"
                g.user_id = "dev"
                g.permissions = {"read", "write", "admin"}
                return func(*args, **kwargs)

            # Try API key authentication
            api_key = request.headers.get("X-API-Key", "")
            if api_key:
                is_valid, permissions = _validate_api_key(api_key)
                if is_valid:
                    g.authenticated = True
                    g.auth_method = "api_key"
                    g.user_id = f"api_key:{api_key[:8]}"
                    g.permissions = permissions
                    return func(*args, **kwargs)
                else:
                    return jsonify({
                        "success": False,
                        "error": "Invalid API key"
                    }), 401

            # Try JWT authentication
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                is_valid, permissions, payload = _validate_jwt_token(token)
                if is_valid:
                    g.authenticated = True
                    g.auth_method = "jwt"
                    g.user_id = payload.get("user_id") if payload else None
                    g.permissions = permissions
                    return func(*args, **kwargs)
                else:
                    return jsonify({
                        "success": False,
                        "error": "Invalid or expired token"
                    }), 401

            # No authentication provided
            if optional:
                g.authenticated = False
                g.permissions = set()
                return func(*args, **kwargs)

            return jsonify({
                "success": False,
                "error": "Authentication required. Provide X-API-Key or Authorization: Bearer <token> header."
            }), 401

        return decorated_function

    # Support both @require_auth and @require_auth(optional=True) syntax
    if f is None:
        # Called with arguments: @require_auth(optional=True)
        return decorator
    else:
        # Called without arguments: @require_auth
        return decorator(f)


def require_permission(*required_permissions: str) -> Callable:
    """
    Authorization decorator requiring specific permissions.

    Args:
        *required_permissions: One or more permission strings required
            (user needs at least one of them)

    Usage:
        @require_permission("entities:write")
        def update_entity():
            # ...

        @require_permission("entities:write", "admin")
        def update_entity_or_admin():
            # ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def decorated_function(*args, **kwargs):
            # Check if user is authenticated
            if not is_authenticated():
                return jsonify({
                    "success": False,
                    "error": "Authentication required"
                }), 401

            user_permissions = get_permissions()

            # Check if user has any of the required permissions
            if not any(perm in user_permissions for perm in required_permissions):
                return jsonify({
                    "success": False,
                    "error": f"Insufficient permissions. Required: {required_permissions}",
                    "your_permissions": list(user_permissions)
                }), 403

            return func(*args, **kwargs)

        return decorated_function

    return decorator


def require_admin(f: Callable) -> Callable:
    """
    Shortcut decorator requiring admin permission.

    Usage:
        @require_admin
        def admin_endpoint():
            # ...
    """
    return require_permission("admin")(f)


# Public endpoints that don't require authentication
PUBLIC_ENDPOINTS = {
    "/api/v1/health",
    "/api/v1/routes",
}

# Read-only endpoints that only need read permission
READ_ONLY_ENDPOINTS = {
    "/api/v1/find",
    "/api/v1/find/entities",
    "/api/v1/find/relations",
    "/api/v1/find/episodes",
}


def is_public_endpoint(path: str) -> bool:
    """Check if a path is a public endpoint."""
    return any(path.startswith(prefix) for prefix in PUBLIC_ENDPOINTS)


def is_read_only_endpoint(path: str) -> bool:
    """Check if a path is a read-only endpoint."""
    return any(path.startswith(prefix) for prefix in READ_ONLY_ENDPOINTS)
