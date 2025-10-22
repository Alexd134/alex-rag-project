"""Pytest tests for input validation.

Run with: pytest test/test_validation.py -v
"""

import pytest
from pydantic import ValidationError
import sys
sys.path.insert(0, 'src')

from api_handler import SubmitQueryRequest


class TestValidQueries:
    """Test that valid queries are accepted."""

    @pytest.mark.parametrize("query_text", [
        "What is the maximum speed?",
        "How does this work?",
        "Can you explain the features?",
        "What are the specifications?",
        "Tell me about the system.",
        "What is the price?",
        "How much does it cost?",
        "Can I use this for production?",
    ])
    def test_valid_query_accepted(self, query_text):
        """Test that valid queries are accepted without errors."""
        request = SubmitQueryRequest(query_text=query_text)
        assert request.query_text == query_text.strip()

    def test_whitespace_stripped(self):
        """Test that leading/trailing whitespace is stripped."""
        request = SubmitQueryRequest(query_text="  What is this?  ")
        assert request.query_text == "What is this?"

    def test_query_with_punctuation(self):
        """Test queries with various punctuation marks."""
        query = "What's the difference between A, B, and C?"
        request = SubmitQueryRequest(query_text=query)
        assert request.query_text == query

    def test_query_with_parentheses(self):
        """Test query with parentheses."""
        query = "What is the speed? (in mph)"
        request = SubmitQueryRequest(query_text=query)
        assert request.query_text == query

    def test_query_with_division(self):
        """Test query with division operator."""
        query = "How much is 1/2 of 100?"
        request = SubmitQueryRequest(query_text=query)
        assert request.query_text == query

    def test_query_with_colon(self):
        """Test query with colons."""
        query = "Test: one, two, three."
        request = SubmitQueryRequest(query_text=query)
        assert request.query_text == query


class TestInvalidQueries:
    """Test that invalid queries are rejected."""

    def test_empty_query_rejected(self):
        """Test that empty queries are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SubmitQueryRequest(query_text="")

        errors = exc_info.value.errors()
        assert any("at least 1 character" in str(error['msg']).lower() for error in errors)

    def test_whitespace_only_rejected(self):
        """Test that whitespace-only queries are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SubmitQueryRequest(query_text="   ")

        errors = exc_info.value.errors()
        assert any("empty" in str(error['msg']).lower() for error in errors)

    def test_too_long_query_rejected(self):
        """Test that queries exceeding 2000 characters are rejected."""
        long_query = "a" * 2001

        with pytest.raises(ValidationError) as exc_info:
            SubmitQueryRequest(query_text=long_query)

        errors = exc_info.value.errors()
        assert any("2000" in str(error['msg']) for error in errors)

    def test_max_length_allowed(self):
        """Test that exactly 2000 characters is allowed."""
        # Use varied characters
        query = "What is this thing? " * 100  # ~2000 characters
        query = query[:2000]

        request = SubmitQueryRequest(query_text=query)
        assert len(request.query_text) <= 2000


class TestSecurityValidation:
    """Test that malicious inputs are blocked."""

    @pytest.mark.parametrize("malicious_query", [
        "SELECT * FROM users",
        "<script>alert('xss')</script>",
        "<img src=x onerror=alert(1)>",
        "onclick=alert(1)",
        "onerror=alert(1)",
    ])
    def test_injection_attacks_blocked(self, malicious_query):
        """Test that common injection attacks are blocked by character whitelist."""
        with pytest.raises(ValidationError) as exc_info:
            SubmitQueryRequest(query_text=malicious_query)

        errors = exc_info.value.errors()
        assert len(errors) > 0

    @pytest.mark.parametrize("invalid_char_query", [
        "What is this? 😊",  
        "Hello\x00World", 
        "price is $500", 
        "temp = 50°C", 
        "What about @mentions?",
        "Check #hashtags",  
        "100% correct", 
        "this & that", 
        "less < more", 
        "more > less",  
    ])
    def test_invalid_characters_blocked(self, invalid_char_query):
        """Test that queries with invalid characters are blocked."""
        with pytest.raises(ValidationError) as exc_info:
            SubmitQueryRequest(query_text=invalid_char_query)

        errors = exc_info.value.errors()
        assert any("invalid character" in str(error['msg']).lower() for error in errors)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_character_allowed(self):
        """Test that a single character query is allowed."""
        request = SubmitQueryRequest(query_text="a")
        assert request.query_text == "a"

    def test_numbers_only_allowed(self):
        """Test that numeric-only queries are allowed."""
        request = SubmitQueryRequest(query_text="12345")
        assert request.query_text == "12345"

    def test_question_mark_only(self):
        """Test query with only punctuation."""
        request = SubmitQueryRequest(query_text="?")
        assert request.query_text == "?"

    def test_mixed_case_preserved(self):
        """Test that mixed case is preserved."""
        query = "What is the IoT Device?"
        request = SubmitQueryRequest(query_text=query)
        assert request.query_text == query

    def test_multiple_spaces_allowed(self):
        """Test that multiple spaces are allowed."""
        query = "What  is  this?"  # Double spaces
        request = SubmitQueryRequest(query_text=query)
        assert request.query_text == query

    @pytest.mark.parametrize("boundary_query", [
        "a",  # Minimum length (1 char)
        "?" * 10,  # Multiple punctuation
        "ABC123",  # Mixed alphanumeric
        "How's it going?",  # Apostrophe
        "Test: one, two, three.",  # Colons and commas
        "What is 1/2 or 1/3?",  # Division operator
        "Price (approx.)",  # Parentheses and period
    ])
    def test_boundary_cases(self, boundary_query):
        """Test various boundary cases."""
        request = SubmitQueryRequest(query_text=boundary_query)
        assert request.query_text == boundary_query.strip()


class TestValidationMessages:
    """Test that validation error messages are helpful."""

    def test_empty_query_error_message(self):
        """Test error message for empty query is descriptive."""
        with pytest.raises(ValidationError) as exc_info:
            SubmitQueryRequest(query_text="")

        error_msg = str(exc_info.value)
        assert "1 character" in error_msg or "at least" in error_msg

    def test_invalid_character_error_message(self):
        """Test error message for invalid characters is descriptive."""
        with pytest.raises(ValidationError) as exc_info:
            SubmitQueryRequest(query_text="test $$$")

        error_msg = str(exc_info.value)
        assert "invalid character" in error_msg.lower()

    def test_too_long_error_message(self):
        """Test error message for too-long query is descriptive."""
        with pytest.raises(ValidationError) as exc_info:
            SubmitQueryRequest(query_text="x" * 2001)

        error_msg = str(exc_info.value)
        assert "2000" in error_msg


class TestValidationIntegration:
    """Integration tests for validation in the full request flow."""

    def test_validation_runs_before_processing(self):
        """Test that validation happens before any processing."""
        # This would fail early due to validation, not during query processing
        with pytest.raises(ValidationError):
            SubmitQueryRequest(query_text="<script>alert(1)</script>")

    def test_valid_query_creates_proper_request(self):
        """Test that valid input creates a proper request object."""
        request = SubmitQueryRequest(query_text="What is this?")

        assert hasattr(request, 'query_text')
        assert isinstance(request.query_text, str)
        assert len(request.query_text) > 0

    def test_validation_is_case_insensitive_for_content(self):
        """Test that case is preserved in queries."""
        request = SubmitQueryRequest(query_text="What is AWS?")
        assert request.query_text == "What is AWS?"


if __name__ == "__main__":
    # Allow running as a script for quick testing
    pytest.main([__file__, "-v", "--tb=short"])
