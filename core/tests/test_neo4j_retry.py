"""
Tests for Neo4j retry logic in Neo4jBaseMixin.

Tests the _run_with_retry() method and _is_transient_neo4j_error() function
from core.storage.neo4j._base.
"""
import time
from unittest.mock import Mock, MagicMock, patch
import pytest

# Import the functions to test
from core.storage.neo4j._base import (
    _is_transient_neo4j_error,
    _MAX_RETRIES,
    _RETRY_BASE_DELAY,
    _MAX_RETRY_DELAY,
    Neo4jBaseMixin,
)


class TestIsTransientNeo4jError:
    """Tests for _is_transient_neo4j_error function."""

    def test_connection_refused_is_transient(self):
        """Connection refused should be identified as transient."""
        exc = Exception("connection refused")
        assert _is_transient_neo4j_error(exc) is True

    def test_connection_error_is_transient(self):
        """ConnectionError should be identified as transient."""
        exc = Exception("newconnectionerror: failed to establish connection")
        assert _is_transient_neo4j_error(exc) is True

    def test_temporarily_unreachable_is_transient(self):
        """Temporarily unreachable should be identified as transient."""
        exc = Exception("server temporarily unreachable")
        assert _is_transient_neo4j_error(exc) is True

    def test_server_unavailable_is_transient(self):
        """Server unavailable should be identified as transient."""
        exc = Exception("database unavailable")
        assert _is_transient_neo4j_error(exc) is True

    def test_no_write_leader_is_transient(self):
        """No write leader error should be identified as transient."""
        exc = Exception("no write leader available")
        assert _is_transient_neo4j_error(exc) is True

    def test_case_insensitive_matching(self):
        """Error detection should be case-insensitive."""
        exc = Exception("Connection Refused")
        assert _is_transient_neo4j_error(exc) is True

    def test_syntax_error_is_not_transient(self):
        """Cypher syntax errors should not be retried."""
        exc = Exception("Invalid Cypher syntax")
        assert _is_transient_neo4j_error(exc) is False

    def test_constraint_violation_is_not_transient(self):
        """Constraint violations should not be retried."""
        exc = Exception("Node already exists")
        assert _is_transient_neo4j_error(exc) is False

    def test_generic_error_is_not_transient(self):
        """Generic errors without transient keywords should not be retried."""
        exc = Exception("Some random error")
        assert _is_transient_neo4j_error(exc) is False

    def test_empty_exception_message(self):
        """Empty exception message should not be considered transient."""
        exc = Exception("")
        assert _is_transient_neo4j_error(exc) is False


class TestRunWithRetry:
    """Tests for _run_with_retry method."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock mixin instance with required dependencies
        self.mixin = Neo4jBaseMixin()
        self.mixin._run = Mock()
        self.mixin._graph_id = "test_graph"

    def test_success_on_first_attempt(self):
        """Should return immediately on successful execution."""
        mock_session = Mock()
        mock_result = {"data": "test"}
        self.mixin._run.return_value = mock_result

        result = self.mixin._run_with_retry(mock_session, "MATCH (n) RETURN n")

        assert result == mock_result
        assert self.mixin._run.call_count == 1

    def test_non_transient_error_raises_immediately(self):
        """Non-transient errors should raise immediately without retry."""
        mock_session = Mock()
        test_error = ValueError("Invalid syntax")
        self.mixin._run.side_effect = test_error

        with pytest.raises(ValueError) as exc_info:
            self.mixin._run_with_retry(mock_session, "INVALID CYPHER")

        assert exc_info.value == test_error
        assert self.mixin._run.call_count == 1

    def test_transient_error_retries_until_success(self):
        """Transient errors should retry until success."""
        mock_session = Mock()
        test_error = Exception("connection refused")
        mock_result = {"data": "success"}

        # Fail twice, then succeed
        self.mixin._run.side_effect = [test_error, test_error, mock_result]

        result = self.mixin._run_with_retry(mock_session, "MATCH (n) RETURN n")

        assert result == mock_result
        assert self.mixin._run.call_count == 3

    def test_max_retries_respected(self):
        """Should stop retrying after max attempts and raise the last error."""
        mock_session = Mock()
        test_error = Exception("connection refused")
        self.mixin._run.side_effect = test_error

        with pytest.raises(Exception) as exc_info:
            self.mixin._run_with_retry(mock_session, "MATCH (n) RETURN n")

        assert "connection refused" in str(exc_info.value)
        assert self.mixin._run.call_count == _MAX_RETRIES

    def test_exponential_backoff(self):
        """Should use exponential backoff between retries."""
        mock_session = Mock()
        test_error = Exception("connection refused")
        self.mixin._run.side_effect = [test_error, test_error, {"data": "success"}]

        delays = []
        original_sleep = time.sleep

        def capture_sleep(delay):
            delays.append(delay)

        with patch("time.sleep", side_effect=capture_sleep):
            result = self.mixin._run_with_retry(mock_session, "MATCH (n) RETURN n")

        assert result == {"data": "success"}
        # First retry: min(_RETRY_BASE_DELAY * 2^0, _MAX_RETRY_DELAY) = 1.0
        # Second retry: min(_RETRY_BASE_DELAY * 2^1, _MAX_RETRY_DELAY) = 2.0
        assert len(delays) == 2
        assert delays[0] == pytest.approx(_RETRY_BASE_DELAY)
        assert delays[1] == pytest.approx(_RETRY_BASE_DELAY * 2)

    def test_max_delay_cap(self):
        """Exponential backoff should be capped at _MAX_RETRY_DELAY."""
        mock_session = Mock()
        test_error = Exception("temporarily unreachable")
        # Make it fail many times to trigger max delay
        self.mixin._run.side_effect = [test_error] * (_MAX_RETRIES - 1) + [{"data": "success"}]

        delays = []

        def capture_sleep(delay):
            delays.append(delay)

        with patch("time.sleep", side_effect=capture_sleep):
            result = self.mixin._run_with_retry(mock_session, "MATCH (n) RETURN n")

        assert result == {"data": "success"}
        # All delays should be <= _MAX_RETRY_DELAY
        for delay in delays:
            assert delay <= _MAX_RETRY_DELAY

    def test_kwargs_passed_to_run(self):
        """Query parameters should be passed through to _run."""
        mock_session = Mock()
        mock_result = {"data": "test"}
        self.mixin._run.return_value = mock_result

        params = {"name": "test", "limit": 10}
        result = self.mixin._run_with_retry(
            mock_session, "MATCH (n) WHERE n.name = $name RETURN n LIMIT $limit", **params
        )

        assert result == mock_result
        self.mixin._run.assert_called_once_with(mock_session, "MATCH (n) WHERE n.name = $name RETURN n LIMIT $limit", **params)

    def test_different_transient_errors_all_retry(self):
        """All types of transient errors should trigger retries."""
        mock_session = Mock()
        mock_result = {"data": "success"}

        transient_errors = [
            Exception("connection refused"),
            Exception("temporarily unreachable"),
            Exception("database unavailable"),
            Exception("no write leader"),
        ]

        for error in transient_errors:
            self.mixin._run.reset_mock()
            self.mixin._run.side_effect = [error, mock_result]

            result = self.mixin._run_with_retry(mock_session, "MATCH (n) RETURN n")

            assert result == mock_result
            assert self.mixin._run.call_count == 2


class TestRunWithRetryEdgeCases:
    """Edge case tests for _run_with_retry."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mixin = Neo4jBaseMixin()
        self.mixin._run = Mock()
        self.mixin._graph_id = "test_graph"
        self.mock_session = Mock()

    def test_session_object_preserved(self):
        """Session object should be passed correctly to _run."""
        mock_result = {"data": "test"}
        self.mixin._run.return_value = mock_result

        result = self.mixin._run_with_retry(self.mock_session, "MATCH (n) RETURN n")

        assert result == mock_result
        self.mixin._run.assert_called_once()
        # Verify first argument is the session
        call_args = self.mixin._run.call_args
        assert call_args[0][0] is self.mock_session

    def test_cypher_query_preserved(self):
        """Cypher query string should be passed correctly."""
        mock_result = {"data": "test"}
        self.mixin._run.return_value = mock_result
        cypher = "MATCH (n:Entity {family_id: $fid}) RETURN n"

        result = self.mixin._run_with_retry(self.mock_session, cypher, fid="test123")

        assert result == mock_result
        call_args = self.mixin._run.call_args
        assert call_args[0][1] == cypher
