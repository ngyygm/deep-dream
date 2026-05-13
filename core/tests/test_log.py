"""Tests for core/log.py unified logging module."""
import logging
import os
from unittest import mock

import pytest

from core.log import debug, error, info, warn, _server_logger, _ServerLogHandler


class TestServerLogHandler:
    def test_format_includes_timestamp_source_message(self):
        handler = _ServerLogHandler()
        record = logging.LogRecord(
            "tmg.server", logging.INFO, "", 0, "hello world", (), None
        )
        record.source = "System"
        lines = []
        with mock.patch("core.log._emit_log_line", side_effect=lambda l: lines.append(l)):
            handler.emit(record)
        assert len(lines) == 1
        line = lines[0]
        assert "System" in line
        assert "hello world" in line
        # Timestamp format HH:MM:SS
        assert len(line.split()[0]) == 8  # HH:MM:SS

    def test_source_right_aligned(self):
        handler = _ServerLogHandler()
        record = logging.LogRecord(
            "tmg.server", logging.INFO, "", 0, "msg", (), None
        )
        record.source = "Neo4j"
        lines = []
        with mock.patch("core.log._emit_log_line", side_effect=lambda l: lines.append(l)):
            handler.emit(record)
        line = lines[0]
        parts = line.split("|")
        assert "     Neo4j" in parts[0]

    def test_exception_in_emit_does_not_crash(self):
        handler = _ServerLogHandler()
        record = logging.LogRecord(
            "tmg.server", logging.INFO, "", 0, "msg", (), None
        )
        record.source = "Test"
        with mock.patch("core.log._emit_log_line", side_effect=RuntimeError("boom")):
            handler.emit(record)  # should not raise


class TestPublicAPI:
    def test_info_routes_through_logger(self):
        with mock.patch.object(_server_logger, "info") as mock_info:
            info("System", "Server started")
            mock_info.assert_called_once_with("Server started", extra={"source": "System"})

    def test_warn_routes_through_logger(self):
        with mock.patch.object(_server_logger, "warning") as mock_warn:
            warn("LLM", "Rate limited")
            mock_warn.assert_called_once_with("Rate limited", extra={"source": "LLM"})

    def test_error_routes_through_logger(self):
        with mock.patch.object(_server_logger, "error") as mock_error:
            error("Neo4j", "Connection lost")
            mock_error.assert_called_once_with("Connection lost", extra={"source": "Neo4j"})

    def test_debug_routes_through_logger(self):
        with mock.patch.object(_server_logger, "debug") as mock_debug:
            debug("Storage", "Cache hit")
            mock_debug.assert_called_once_with("Cache hit", extra={"source": "Storage"})

    def test_end_to_end_info(self):
        lines = []
        with mock.patch("core.log._emit_log_line", side_effect=lambda l: lines.append(l)):
            info("Test", "integration check")
        assert len(lines) == 1
        assert "Test" in lines[0]
        assert "integration check" in lines[0]


class TestLevelFiltering:
    def test_debug_suppressed_at_info_level(self):
        _server_logger.setLevel(logging.INFO)
        lines = []
        with mock.patch("core.log._emit_log_line", side_effect=lambda l: lines.append(l)):
            debug("Test", "should not appear")
        assert len(lines) == 0

    def test_debug_visible_at_debug_level(self):
        original = _server_logger.level
        try:
            _server_logger.setLevel(logging.DEBUG)
            lines = []
            with mock.patch("core.log._emit_log_line", side_effect=lambda l: lines.append(l)):
                debug("Test", "should appear")
            assert len(lines) == 1
        finally:
            _server_logger.setLevel(original)
