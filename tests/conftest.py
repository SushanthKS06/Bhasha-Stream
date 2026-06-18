# tests/conftest.py
"""
pytest configuration for Bhasha-Stream test suite.
- Sets asyncio_mode = auto so all async tests run without explicit markers
  (also set in pytest.ini, this provides the programmatic equivalent)
- Provides shared fixtures used across multiple test modules
"""
import asyncio
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Event loop policy
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop_policy():
    """Use the default asyncio event loop policy."""
    return asyncio.DefaultEventLoopPolicy()
