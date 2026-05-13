"""
Unified console logging for Deep-Dream.

All runtime log output (server, pipeline phases, chat) flows through
log.info / warn / error / debug, which routes to _emit_log_line for
serialized, thread-safe console output.

Pipeline code continues using wprint* (with window-label context);
server/storage code uses log.info("Source", msg).

Level controlled by DEEPDREAM_LOG_LEVEL env var (default: INFO).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime

from core.utils import _emit_log_line

# ---------------------------------------------------------------------------
# Log level from env
# ---------------------------------------------------------------------------

_LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
}

_DEFAULT_LEVEL = logging.INFO
_env_level = os.environ.get("DEEPDREAM_LOG_LEVEL", "").strip().upper()
_LOG_LEVEL = _LOG_LEVEL_MAP.get(_env_level, _DEFAULT_LEVEL)


# ---------------------------------------------------------------------------
# Custom handler: routes through _emit_log_line
# ---------------------------------------------------------------------------

class _ServerLogHandler(logging.Handler):
    """Logging handler that formats and routes through _emit_log_line."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            ts = datetime.now().strftime("%H:%M:%S")
            source = getattr(record, "source", "----")
            msg = self.format(record)
            line = f"{ts} {source:>10} | {msg}"
            _emit_log_line(line)
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

_server_logger = logging.getLogger("tmg.server")
if not _server_logger.handlers:
    _handler = _ServerLogHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _server_logger.addHandler(_handler)
    _server_logger.setLevel(_LOG_LEVEL)
    _server_logger.propagate = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def info(source: str, msg: str) -> None:
    _server_logger.info(msg, extra={"source": source})


def warn(source: str, msg: str) -> None:
    _server_logger.warning(msg, extra={"source": source})


def error(source: str, msg: str) -> None:
    _server_logger.error(msg, extra={"source": source})


def debug(source: str, msg: str) -> None:
    _server_logger.debug(msg, extra={"source": source})
