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
    point.vector = [0.1] * 32  # Mock vector for diversity calculation
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
    """Tests for the /recommend/{user_idx} endpoint."""

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
        assert data["recommendations"][0]["item_idx"] == 123

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


# -----------------------------------------------------------------------------
# Purchase Event Endpoint Tests
# -----------------------------------------------------------------------------


class TestPurchaseEventEndpoint:
    """Tests for the POST /events/purchase endpoint."""

    @pytest.fixture(autouse=True)
    async def setup_app(self, mock_async_redis, mock_async_qdrant):
        """Set up app state with mocked clients."""
        app.state.redis_client = mock_async_redis
        app.state.qdrant_client = mock_async_qdrant
        app.state.kafka_producer = MagicMock()
        self.redis = mock_async_redis
        self.qdrant = mock_async_qdrant
        self.kafka = app.state.kafka_producer
        yield

    async def test_purchase_event_success(self):
        """Should accept valid purchase event and send to Kafka."""
        # User mapping exists
        self.redis.get.return_value = "customer_abc123"

        # Item exists in Qdrant with article_id
        item_point = MagicMock()
        item_point.payload = {"article_id": "0123456789", "prod_name": "Test Product"}
        self.qdrant.retrieve.return_value = [item_point]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/events/purchase",
                json={"user_idx": 42, "item_idx": 100},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"
        assert data["event"]["user_idx"] == 42
        assert data["event"]["item_idx"] == 100
        assert data["event"]["user_id"] == "customer_abc123"
        assert data["event"]["item_id"] == "0123456789"
        assert data["event"]["event_type"] == "purchase"

        # Verify Kafka send was called
        self.kafka.send.assert_called_once()

    async def test_purchase_event_user_not_found(self):
        """Should return 404 when user mapping doesn't exist."""
        self.redis.get.return_value = None  # No user mapping

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/events/purchase",
                json={"user_idx": 99999, "item_idx": 100},
            )

        assert response.status_code == 404
        assert "user mapping not found" in response.json()["detail"].lower()

    async def test_purchase_event_item_not_found(self):
        """Should return 404 when item doesn't exist in Qdrant."""
        self.redis.get.return_value = "customer_abc123"
        self.qdrant.retrieve.return_value = []  # No item found

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/events/purchase",
                json={"user_idx": 42, "item_idx": 99999},
            )

        assert response.status_code == 404
        assert "item not found" in response.json()["detail"].lower()

    async def test_purchase_event_item_missing_article_id(self):
        """Should return 404 when item has no article_id in payload."""
        self.redis.get.return_value = "customer_abc123"

        item_point = MagicMock()
        item_point.payload = {"prod_name": "Test Product"}  # No article_id
        self.qdrant.retrieve.return_value = [item_point]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/events/purchase",
                json={"user_idx": 42, "item_idx": 100},
            )

        assert response.status_code == 404
        assert "article_id not found" in response.json()["detail"].lower()

    async def test_purchase_event_kafka_unavailable(self):
        """Should return 503 when Kafka producer is not available."""
        app.state.kafka_producer = None

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/events/purchase",
                json={"user_idx": 42, "item_idx": 100},
            )

        assert response.status_code == 503
        assert "kafka" in response.json()["detail"].lower()

    async def test_purchase_event_kafka_send_failure(self):
        """Should return 500 when Kafka send fails."""
        self.redis.get.return_value = "customer_abc123"

        item_point = MagicMock()
        item_point.payload = {"article_id": "0123456789"}
        self.qdrant.retrieve.return_value = [item_point]

        # Kafka send raises exception
        self.kafka.send.side_effect = Exception("Kafka connection lost")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/events/purchase",
                json={"user_idx": 42, "item_idx": 100},
            )

        assert response.status_code == 500
        assert "failed to record purchase" in response.json()["detail"].lower()

    async def test_purchase_event_qdrant_error(self):
        """Should return 500 when Qdrant retrieve fails."""
        self.redis.get.return_value = "customer_abc123"
        self.qdrant.retrieve.side_effect = Exception("Qdrant connection error")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/events/purchase",
                json={"user_idx": 42, "item_idx": 100},
            )

        assert response.status_code == 500
        assert "failed to look up item" in response.json()["detail"].lower()

    async def test_purchase_event_invalid_payload(self):
        """Should return 422 for invalid request payload."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/events/purchase",
                json={"user_idx": "not_an_int", "item_idx": 100},
            )

        assert response.status_code == 422

    async def test_purchase_event_missing_fields(self):
        """Should return 422 when required fields are missing."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/events/purchase",
                json={"user_idx": 42},  # Missing item_idx
            )

        assert response.status_code == 422


# -----------------------------------------------------------------------------
# Similar Items Endpoint Tests
# -----------------------------------------------------------------------------


class TestSimilarItemsEndpoint:
    """Tests for the /items/{item_idx}/similar endpoint."""

    @pytest.fixture(autouse=True)
    async def setup_app(self, mock_async_redis, mock_async_qdrant):
        """Set up app state with mocked clients."""
        app.state.redis_client = mock_async_redis
        app.state.qdrant_client = mock_async_qdrant
        self.qdrant = mock_async_qdrant
        yield

    async def test_similar_items_success(self, mock_qdrant_point):
        """Should return similar items when source item exists."""
        # Source item exists in Qdrant
        source_point = MagicMock()
        source_point.id = 100
        source_point.vector = [0.1] * 32
        source_point.payload = {"article_id": "1111111111", "prod_name": "Source Item"}
        self.qdrant.retrieve.return_value = [source_point]

        # Similar items returned by search
        self.qdrant.query_points.return_value = MagicMock(points=[mock_qdrant_point])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/items/100/similar")

        assert response.status_code == 200
        data = response.json()
        assert data["source_item_idx"] == 100
        assert data["source_metadata"]["prod_name"] == "Source Item"
        assert len(data["similar_items"]) == 1
        assert data["similar_items"][0]["item_idx"] == 123

    async def test_similar_items_not_found(self):
        """Should return 404 when source item doesn't exist."""
        self.qdrant.retrieve.return_value = []

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/items/99999/similar")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_similar_items_with_product_group_filter(self, mock_qdrant_point):
        """Should filter by product_group when specified."""
        source_point = MagicMock()
        source_point.id = 100
        source_point.vector = [0.1] * 32
        source_point.payload = {}
        self.qdrant.retrieve.return_value = [source_point]
        self.qdrant.query_points.return_value = MagicMock(points=[mock_qdrant_point])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/items/100/similar?product_group=Accessories")

        assert response.status_code == 200

        # Verify filter was applied
        call_args = self.qdrant.query_points.call_args
        query_filter = call_args.kwargs.get("query_filter")
        assert query_filter is not None
        assert query_filter.must is not None

    async def test_similar_items_respects_k_parameter(self, mock_qdrant_point):
        """Should respect the k parameter."""
        source_point = MagicMock()
        source_point.id = 100
        source_point.vector = [0.1] * 32
        source_point.payload = {}
        self.qdrant.retrieve.return_value = [source_point]
        self.qdrant.query_points.return_value = MagicMock(points=[mock_qdrant_point])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/items/100/similar?k=5")

        assert response.status_code == 200

        # Verify limit was passed to Qdrant
        call_args = self.qdrant.query_points.call_args
        assert call_args.kwargs.get("limit") == 5

    async def test_similar_items_k_max_limit(self):
        """Should reject k values > 50."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/items/100/similar?k=100")

        assert response.status_code == 422
