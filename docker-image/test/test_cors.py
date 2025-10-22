"""Test CORS configuration.

Run with: pytest test/test_cors.py -v
"""

import pytest
import os
import sys
sys.path.insert(0, 'src')


def test_cors_configured_from_environment():
    """Test that CORS origins are read from environment variable."""
    # Set environment variable
    os.environ["ALLOWED_ORIGINS"] = "https://example.com,https://app.example.com"

    # Reload the module to pick up the environment variable
    import importlib
    import api_handler
    importlib.reload(api_handler)

    # Check that ALLOWED_ORIGINS is set correctly
    assert api_handler.ALLOWED_ORIGINS == ["https://example.com", "https://app.example.com"]

    # Clean up
    del os.environ["ALLOWED_ORIGINS"]


def test_cors_defaults_to_wildcard():
    """Test that CORS defaults to wildcard when no environment variable is set."""
    # Make sure environment variable is not set
    if "ALLOWED_ORIGINS" in os.environ:
        del os.environ["ALLOWED_ORIGINS"]

    # Reload the module
    import importlib
    import api_handler
    importlib.reload(api_handler)

    # Should default to wildcard
    assert api_handler.ALLOWED_ORIGINS == ["*"]


def test_cors_middleware_configured():
    """Test that CORS middleware is properly configured on the app."""
    import importlib
    import api_handler
    importlib.reload(api_handler)

    # Check that middleware is added
    app = api_handler.app

    # FastAPI stores middleware in app.user_middleware
    assert len(app.user_middleware) > 0

    # Find the CORS middleware
    cors_middleware = None
    for middleware in app.user_middleware:
        if "CORSMiddleware" in str(middleware):
            cors_middleware = middleware
            break

    assert cors_middleware is not None, "CORS middleware should be configured"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
