"""
Integration tests for the FastAPI recommendation API.

Tests the HTTP endpoints with mocked database dependencies to verify
request handling, response formatting, and fallback behavior.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport

from src.serve.api import app


@pytest.fixture
def mock_redis_pipeline():
    """Create a mock Redis pipeline for batched operations."""
    pipeline = AsyncMock()
    pipeline.get = MagicMock(return_value=pipeline)
    pipeline.zrange = MagicMock(return_value=pipeline)
    pipeline.execute = AsyncMock(return_value=[None, []])
    pipeline.__aenter__ = AsyncMock(return_value=pipeline)
    pipeline.__aexit__ = AsyncMock(return_value=None)
    return pipeline


@pytest.fixture
def mock_qdrant_point():
    """Create a mock Qdrant search result point."""
    point = MagicMock()
    point.id = 123
    point.score = 0.95
    point.payload = {"article_id": "0123456789", "prod_name": "Test Product"}
    return point


# -----------------------------------------------------------------------------
# Health Check Tests
# -----------------------------------------------------------------------------


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    @pytest.fixture(autouse=True)
    async def setup_app(self, mock_async_redis, mock_async_qdrant):
        """Set up app state with mocked clients before each test."""
        app.state.redis_client = mock_async_redis
        app.state.qdrant_client = mock_async_qdrant
        yield

    async def test_health_live_returns_ok(self):
        """GET /health/live should always return 200 OK."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health/live")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    async def test_health_ready_with_healthy_dependencies(
        self, mock_async_redis, mock_async_qdrant
    ):
        """GET /health/ready should return 200 when all deps are healthy."""
        # Mocks already configured to return successfully
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["components"]["redis"]["status"] == "up"
        assert data["components"]["qdrant"]["status"] == "up"

    async def test_health_ready_with_redis_down(
        self, mock_async_redis, mock_async_qdrant
    ):
        """GET /health/ready should return 503 when Redis is down."""
        mock_async_redis.ping.side_effect = ConnectionError("Redis unavailable")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health/ready")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "error"
        assert data["components"]["redis"]["status"] == "down"

    async def test_health_ready_with_qdrant_down(
        self, mock_async_redis, mock_async_qdrant
    ):
        """GET /health/ready should return 503 when Qdrant is down."""
        mock_async_qdrant.get_collections.side_effect = Exception("Qdrant unavailable")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health/ready")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "error"
        assert data["components"]["qdrant"]["status"] == "down"


# -----------------------------------------------------------------------------
# Recommendation Endpoint Tests
# -----------------------------------------------------------------------------


class TestRecommendEndpoint:
    """Tests for the /recommend/{user_id} endpoint."""

    @pytest.fixture(autouse=True)
    async def setup_app(self, mock_async_redis, mock_async_qdrant, mock_redis_pipeline):
        """Set up app state with mocked clients."""
        mock_async_redis.pipeline.return_value = mock_redis_pipeline
        app.state.redis_client = mock_async_redis
        app.state.qdrant_client = mock_async_qdrant
        self.redis = mock_async_redis
        self.qdrant = mock_async_qdrant
        self.pipeline = mock_redis_pipeline
        yield

    async def test_recommend_personalized_success(self, mock_qdrant_point):
        """Should return personalized recommendations when user vector exists."""
        # User has a vector in Redis
        user_vector = [0.1] * 32
        self.pipeline.execute.return_value = [json.dumps(user_vector), []]

        # Qdrant returns recommendations
        self.qdrant.query_points.return_value = MagicMock(points=[mock_qdrant_point])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/recommend/42")

        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "personalized"
        assert len(data["recommendations"]) == 1
        assert data["recommendations"][0]["item_id"] == 123

    async def test_recommend_fallback_to_trending(self, sample_popular_items):
        """Should fall back to trending items when user has no vector."""
        # User has no vector
        self.pipeline.execute.return_value = [None, []]

        # Trending items exist in Redis
        self.redis.get.return_value = json.dumps(sample_popular_items)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/recommend/9999")

        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "trending_now"
        assert len(data["recommendations"]) == 2

    async def test_recommend_returns_503_when_both_fail(self):
        """Should return 503 when both personalized and fallback strategies fail."""
        # No user vector
        self.pipeline.execute.return_value = [None, []]

        # No trending items either
        self.redis.get.return_value = None

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/recommend/9999")

        assert response.status_code == 503

    async def test_recommend_respects_k_parameter(self, sample_popular_items):
        """Should respect the k parameter for number of recommendations."""
        self.pipeline.execute.return_value = [None, []]
        self.redis.get.return_value = json.dumps(sample_popular_items)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/recommend/1?k=1")

        assert response.status_code == 200
        data = response.json()
        # Should only return 1 item, not 2
        assert len(data["recommendations"]) == 1

    async def test_recommend_k_max_limit(self):
        """Should reject k values > 50."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/recommend/1?k=100")

        assert response.status_code == 422  # Validation error

    async def test_recommend_excludes_purchased_items(self, mock_qdrant_point):
        """Should filter out items the user has already purchased."""
        user_vector = [0.1] * 32
        purchased_items = ["100", "200"]  # Items already bought
        self.pipeline.execute.return_value = [json.dumps(user_vector), purchased_items]

        self.qdrant.query_points.return_value = MagicMock(points=[mock_qdrant_point])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/recommend/42")

        assert response.status_code == 200

        # Verify Qdrant was called with filter to exclude purchased items
        call_args = self.qdrant.query_points.call_args
        assert call_args.kwargs.get("query_filter") is not None
