"""Integration tests for API endpoints.

Tests the actual HTTP API behavior using FastAPI TestClient.
Run with: pytest test/test_api_endpoints.py -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import sys
sys.path.insert(0, 'src')

from api_handler import app
from rag_app.query import QueryResponse


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_query_response():
    """Create a mock QueryResponse for testing."""
    return QueryResponse(
        query_text="What is the maximum speed?",
        response_text="The maximum speed is 100 mph.",
        sources=["doc1.pdf:1:0", "doc2.pdf:3:1"]
    )


class TestRootEndpoint:
    """Test the root endpoint that serves the HTML interface."""

    def test_get_root_returns_200(self, client):
        """Test that GET / returns 200 OK."""
        response = client.get("/")
        assert response.status_code == 200

    def test_get_root_returns_html(self, client):
        """Test that GET / returns HTML content."""
        response = client.get("/")
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        assert "<!DOCTYPE html>" in response.text
        assert "<title>RAG Query Interface</title>" in response.text

    def test_root_contains_submit_query_form(self, client):
        """Test that the HTML interface contains the query form elements."""
        response = client.get("/")
        assert "submit_query" in response.text.lower()
        assert "query" in response.text.lower()


class TestSubmitQueryEndpoint:
    """Test the POST /submit_query endpoint."""

    @patch('api_handler.query_rag')
    def test_submit_query_success(self, mock_query_rag, client, mock_query_response):
        """Test successful query submission returns 200 with correct structure."""
        mock_query_rag.return_value = mock_query_response

        response = client.post(
            "/submit_query",
            json={"query_text": "What is the maximum speed?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "query_text" in data
        assert "response_text" in data
        assert "sources" in data

    @patch('api_handler.query_rag')
    def test_submit_query_returns_correct_data(self, mock_query_rag, client, mock_query_response):
        """Test that response contains the expected data from RAG."""
        mock_query_rag.return_value = mock_query_response

        response = client.post(
            "/submit_query",
            json={"query_text": "What is the maximum speed?"}
        )

        data = response.json()
        assert data["query_text"] == "What is the maximum speed?"
        assert data["response_text"] == "The maximum speed is 100 mph."
        assert data["sources"] == ["doc1.pdf:1:0", "doc2.pdf:3:1"]

    @patch('api_handler.query_rag')
    def test_submit_query_calls_query_rag(self, mock_query_rag, client, mock_query_response):
        """Test that the endpoint actually calls query_rag with the input."""
        mock_query_rag.return_value = mock_query_response

        client.post(
            "/submit_query",
            json={"query_text": "Test query"}
        )

        mock_query_rag.assert_called_once_with("Test query")

    def test_submit_query_empty_text_returns_422(self, client):
        """Test that empty query text returns 422 validation error."""
        response = client.post(
            "/submit_query",
            json={"query_text": ""}
        )

        assert response.status_code == 422

    def test_submit_query_too_long_returns_422(self, client):
        """Test that queries over 2000 characters return 422."""
        long_query = "a" * 2001

        response = client.post(
            "/submit_query",
            json={"query_text": long_query}
        )

        assert response.status_code == 422

    def test_submit_query_invalid_characters_returns_422(self, client):
        """Test that invalid characters return 422 validation error."""
        response = client.post(
            "/submit_query",
            json={"query_text": "test $$$"}
        )

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_submit_query_missing_field_returns_422(self, client):
        """Test that missing query_text field returns 422."""
        response = client.post(
            "/submit_query",
            json={}
        )

        assert response.status_code == 422

    def test_submit_query_invalid_json_returns_422(self, client):
        """Test that malformed JSON returns 422."""
        response = client.post(
            "/submit_query",
            data="not json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422


class TestErrorHandling:
    """Test error handling in the API."""

    @patch('api_handler.query_rag')
    def test_submit_query_processing_error_returns_500(self, mock_query_rag, client):
        """Test that processing errors return 500 with generic message."""
        mock_query_rag.side_effect = Exception("Database connection failed")

        response = client.post(
            "/submit_query",
            json={"query_text": "What is this?"}
        )

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        # Should not expose internal error details
        assert "Database connection failed" not in data["detail"]
        assert "error occurred" in data["detail"].lower()

    @patch('api_handler.query_rag')
    def test_submit_query_value_error_returns_400(self, mock_query_rag, client):
        """Test that ValueError returns 400 bad request."""
        mock_query_rag.side_effect = ValueError("Invalid query format")

        response = client.post(
            "/submit_query",
            json={"query_text": "What is this?"}
        )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "Invalid query" in data["detail"]

    @patch('api_handler.query_rag')
    def test_error_response_has_detail_field(self, mock_query_rag, client):
        """Test that error responses include detail field."""
        mock_query_rag.side_effect = Exception("Test error")

        response = client.post(
            "/submit_query",
            json={"query_text": "What is this?"}
        )

        assert response.status_code == 500
        data = response.json()
        assert isinstance(data, dict)
        assert "detail" in data
        assert isinstance(data["detail"], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
