"""
Tests for API validation helpers from core.server.api.

Tests the validation helper functions:
- _validate_graph_id
- _validate_text_input
- _validate_positive_int
- _validate_float_range
"""
import pytest
from core.server.api import (
    _validate_graph_id,
    _validate_text_input,
    _validate_positive_int,
    _validate_float_range,
)


class TestValidateGraphId:
    """Tests for _validate_graph_id function."""

    def test_valid_graph_id(self):
        """Valid graph_id should pass validation."""
        is_valid, error_msg, status = _validate_graph_id("test_graph")
        assert is_valid is True
        assert error_msg is None
        assert status == 200

    def test_empty_graph_id(self):
        """Empty graph_id should fail validation."""
        is_valid, error_msg, status = _validate_graph_id("")
        assert is_valid is False
        assert "required" in error_msg.lower()
        assert status == 400

    def test_none_graph_id(self):
        """None graph_id should fail validation."""
        is_valid, error_msg, status = _validate_graph_id(None)
        assert is_valid is False
        assert "required" in error_msg.lower()
        assert status == 400

    def test_non_string_graph_id(self):
        """Non-string graph_id should fail validation."""
        is_valid, error_msg, status = _validate_graph_id(123)
        assert is_valid is False
        assert "string" in error_msg.lower()
        assert status == 400

    def test_invalid_graph_id_format(self):
        """Graph_id with invalid characters should fail validation."""
        # GraphRegistry.validate_graph_id rejects graph_ids with spaces
        is_valid, error_msg, status = _validate_graph_id("test graph")
        assert is_valid is False
        assert status == 400


class TestValidateTextInput:
    """Tests for _validate_text_input function."""

    def test_valid_text_input(self):
        """Valid text input should pass validation."""
        is_valid, error_msg, status = _validate_text_input("Sample text content")
        assert is_valid is True
        assert error_msg is None
        assert status == 200

    def test_none_text_input(self):
        """None text input should fail validation."""
        is_valid, error_msg, status = _validate_text_input(None)
        assert is_valid is False
        assert "required" in error_msg.lower()
        assert status == 400

    def test_non_string_text_input(self):
        """Non-string text input should fail validation."""
        is_valid, error_msg, status = _validate_text_input(12345)
        assert is_valid is False
        assert "string" in error_msg.lower()
        assert status == 400

    def test_empty_string_text_input(self):
        """Empty string should fail default min_length validation."""
        is_valid, error_msg, status = _validate_text_input("")
        assert is_valid is False
        assert "at least 1" in error_msg.lower()
        assert status == 400

    def test_whitespace_only_text_input(self):
        """Whitespace-only string should fail min_length validation after strip."""
        is_valid, error_msg, status = _validate_text_input("   ")
        assert is_valid is False
        assert status == 400

    def test_custom_min_length(self):
        """Custom min_length should be enforced."""
        is_valid, error_msg, status = _validate_text_input("ab", min_length=3)
        assert is_valid is False
        assert "at least 3" in error_msg.lower()
        assert status == 400

    def test_custom_max_length(self):
        """Custom max_length should be enforced."""
        long_text = "a" * 1001
        is_valid, error_msg, status = _validate_text_input(long_text, max_length=1000)
        assert is_valid is False
        assert "exceed" in error_msg.lower() or "must not exceed" in error_msg.lower()
        assert status == 400

    def test_custom_field_name(self):
        """Custom field name should appear in error message."""
        is_valid, error_msg, status = _validate_text_input(None, field_name="content")
        assert is_valid is False
        assert "content" in error_msg.lower()

    def test_text_within_bounds(self):
        """Text within min and max bounds should pass."""
        text = "a" * 50
        is_valid, error_msg, status = _validate_text_input(
            text, min_length=10, max_length=100
        )
        assert is_valid is True
        assert error_msg is None


class TestValidatePositiveInt:
    """Tests for _validate_positive_int function."""

    def test_valid_positive_int(self):
        """Valid positive integer should pass validation."""
        is_valid, error_msg, status = _validate_positive_int(42)
        assert is_valid is True
        assert error_msg is None
        assert status == 200

    def test_none_value(self):
        """None value should pass (optional parameter)."""
        is_valid, error_msg, status = _validate_positive_int(None)
        assert is_valid is True
        assert error_msg is None
        assert status == 200

    def test_string_int_conversion(self):
        """String representation of integer should be converted."""
        is_valid, error_msg, status = _validate_positive_int("42")
        assert is_valid is True
        assert error_msg is None

    def test_float_conversion(self):
        """Float should be converted to int."""
        is_valid, error_msg, status = _validate_positive_int(42.7)
        assert is_valid is True
        assert error_msg is None

    def test_below_min_value(self):
        """Value below min_val should fail validation."""
        is_valid, error_msg, status = _validate_positive_int(0, min_val=1)
        assert is_valid is False
        assert "at least 1" in error_msg.lower()
        assert status == 400

    def test_above_max_value(self):
        """Value above max_val should fail validation."""
        is_valid, error_msg, status = _validate_positive_int(101, max_val=100)
        assert is_valid is False
        assert "not exceed" in error_msg.lower() or "must not exceed" in error_msg.lower()
        assert status == 400

    def test_custom_field_name(self):
        """Custom field name should appear in error message."""
        is_valid, error_msg, status = _validate_positive_int(
            -1, field_name="limit", min_val=0
        )
        assert is_valid is False
        assert "limit" in error_msg.lower()

    def test_invalid_string_conversion(self):
        """Non-numeric string should fail validation."""
        is_valid, error_msg, status = _validate_positive_int("abc")
        assert is_valid is False
        assert status == 400

    def test_custom_min_max(self):
        """Custom min and max values should be enforced."""
        # Test lower bound
        is_valid, error_msg, status = _validate_positive_int(4, min_val=5, max_val=10)
        assert is_valid is False

        # Test upper bound
        is_valid, error_msg, status = _validate_positive_int(11, min_val=5, max_val=10)
        assert is_valid is False

        # Test within bounds
        is_valid, error_msg, status = _validate_positive_int(7, min_val=5, max_val=10)
        assert is_valid is True


class TestValidateFloatRange:
    """Tests for _validate_float_range function."""

    def test_valid_float_in_range(self):
        """Valid float within range should pass validation."""
        is_valid, error_msg, status = _validate_float_range(0.5)
        assert is_valid is True
        assert error_msg is None
        assert status == 200

    def test_none_value(self):
        """None value should pass (optional parameter)."""
        is_valid, error_msg, status = _validate_float_range(None)
        assert is_valid is True
        assert error_msg is None

    def test_at_lower_bound(self):
        """Value at lower bound should pass."""
        is_valid, error_msg, status = _validate_float_range(0.0, min_val=0.0, max_val=1.0)
        assert is_valid is True

    def test_at_upper_bound(self):
        """Value at upper bound should pass."""
        is_valid, error_msg, status = _validate_float_range(1.0, min_val=0.0, max_val=1.0)
        assert is_valid is True

    def test_below_lower_bound(self):
        """Value below lower bound should fail."""
        is_valid, error_msg, status = _validate_float_range(-0.1)
        assert is_valid is False
        assert "between 0.0 and 1.0" in error_msg.lower()
        assert status == 400

    def test_above_upper_bound(self):
        """Value above upper bound should fail."""
        is_valid, error_msg, status = _validate_float_range(1.5)
        assert is_valid is False
        assert "between 0.0 and 1.0" in error_msg.lower()
        assert status == 400

    def test_custom_field_name(self):
        """Custom field name should appear in error message."""
        is_valid, error_msg, status = _validate_float_range(
            2.0, field_name="threshold", min_val=0.0, max_val=1.0
        )
        assert is_valid is False
        assert "threshold" in error_msg.lower()

    def test_custom_range(self):
        """Custom min and max values should be enforced."""
        # Below range
        is_valid, error_msg, status = _validate_float_range(
            -5.0, min_val=-10.0, max_val=10.0
        )
        assert is_valid is True  # -5.0 is within -10 to 10

        # Above range
        is_valid, error_msg, status = _validate_float_range(
            15.0, min_val=-10.0, max_val=10.0
        )
        assert is_valid is False

        # Within range
        is_valid, error_msg, status = _validate_float_range(
            5.0, min_val=-10.0, max_val=10.0
        )
        assert is_valid is True

    def test_string_conversion(self):
        """String representation of float should be converted."""
        is_valid, error_msg, status = _validate_float_range("0.75")
        assert is_valid is True

    def test_int_conversion(self):
        """Integer should be converted to float."""
        is_valid, error_msg, status = _validate_float_range(1)
        assert is_valid is True

    def test_invalid_string_conversion(self):
        """Non-numeric string should fail validation."""
        is_valid, error_msg, status = _validate_float_range("abc")
        assert is_valid is False
        assert status == 400


class TestValidationHelpersIntegration:
    """Integration tests for validation helpers working together."""

    def test_multiple_validations(self):
        """Multiple validation helpers should work correctly together."""
        # Validate graph_id
        graph_valid, graph_err, _ = _validate_graph_id("test_graph")
        assert graph_valid

        # Validate text input
        text_valid, text_err, _ = _validate_text_input("Sample content", max_length=1000)
        assert text_valid

        # Validate positive int
        int_valid, int_err, _ = _validate_positive_int(10, min_val=1, max_val=100)
        assert int_valid

        # Validate float range
        float_valid, float_err, _ = _validate_float_range(0.85)
        assert float_valid

    def test_error_accumulation_pattern(self):
        """Demonstrate pattern for accumulating validation errors."""
        errors = []

        # Simulate validating multiple fields
        valid, msg, _ = _validate_text_input(None)
        if not valid:
            errors.append(msg)

        valid, msg, _ = _validate_positive_int(-1, min_val=1)
        if not valid:
            errors.append(msg)

        valid, msg, _ = _validate_float_range(1.5, max_val=1.0)
        if not valid:
            errors.append(msg)

        assert len(errors) == 3
        assert any("required" in e.lower() for e in errors)
        assert any("at least 1" in e.lower() for e in errors)
        assert any("between 0.0 and 1.0" in e.lower() for e in errors)
