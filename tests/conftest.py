"""
Shared pytest fixtures for the test suite.

Fixtures provide reusable test doubles and configuration for both
unit and integration tests.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
import numpy as np


# -----------------------------------------------------------------------------
# Mock Redis Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_redis():
    """
    Mock Redis client for unit tests.
    Simulates basic Redis operations without network calls.
    """
    redis_mock = MagicMock()
    redis_mock.get = MagicMock(return_value=None)
    redis_mock.set = MagicMock(return_value=True)
    redis_mock.exists = MagicMock(return_value=False)
    redis_mock.setex = MagicMock(return_value=True)
    redis_mock.zadd = MagicMock(return_value=1)
    redis_mock.zremrangebyrank = MagicMock(return_value=0)
    redis_mock.expire = MagicMock(return_value=True)
    redis_mock.close = MagicMock()
    return redis_mock


@pytest.fixture
def mock_async_redis():
    """
    Async mock Redis client for integration tests with FastAPI.
    """
    redis_mock = AsyncMock()
    redis_mock.ping = AsyncMock(return_value=True)
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.set = AsyncMock(return_value=True)
    redis_mock.zrange = AsyncMock(return_value=[])
    redis_mock.pipeline = MagicMock(return_value=AsyncMock())
    redis_mock.aclose = AsyncMock()
    return redis_mock


# -----------------------------------------------------------------------------
# Mock Qdrant Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_qdrant():
    """
    Mock Qdrant client for unit tests.
    """
    qdrant_mock = MagicMock()
    qdrant_mock.retrieve = MagicMock(return_value=[])
    return qdrant_mock


@pytest.fixture
def mock_async_qdrant():
    """
    Async mock Qdrant client for integration tests.
    """
    qdrant_mock = AsyncMock()
    qdrant_mock.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
    qdrant_mock.query_points = AsyncMock(return_value=MagicMock(points=[]))
    qdrant_mock.close = AsyncMock()
    return qdrant_mock


# -----------------------------------------------------------------------------
# Sample Data Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_user_vector():
    """Sample 32-dimensional user vector (matching ALS rank)."""
    return np.random.rand(32).astype(np.float32)


@pytest.fixture
def sample_item_vector():
    """Sample 32-dimensional item vector (matching ALS rank)."""
    return np.random.rand(32).astype(np.float32)


@pytest.fixture
def sample_event():
    """Sample Kafka event as produced by producer.py."""
    return {
        "user_id": "abc123hash",
        "user_idx": 42,
        "item_id": "0123456789",
        "item_idx": 100,
        "event_type": "click",
        "timestamp": 1705250000.0,
        "quantity": 1,
    }


@pytest.fixture
def sample_popular_items():
    """Sample trending items as stored in Redis."""
    return [
        {
            "item_idx": 1,
            "score": 100.0,
            "metadata": {
                "article_id": "0123456789",
                "name": "Test Product",
                "group": "Garment Upper body",
                "type": "T-shirt",
            },
            "source": "trending_now",
        },
        {
            "item_idx": 2,
            "score": 90.0,
            "metadata": {
                "article_id": "0987654321",
                "name": "Another Product",
                "group": "Garment Lower body",
                "type": "Trousers",
            },
            "source": "trending_now",
        },
    ]
