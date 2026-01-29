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

    async def test_recommend_with_product_group_filter(self, mock_qdrant_point):
        """Should apply product_group filter to recommendations."""
        user_vector = [0.1] * 32
        self.pipeline.execute.return_value = [json.dumps(user_vector), []]
        self.qdrant.query_points.return_value = MagicMock(points=[mock_qdrant_point])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/recommend/42?product_group=Garment%20Upper%20body"
            )

        assert response.status_code == 200

        # Verify filter was applied with product_group
        call_args = self.qdrant.query_points.call_args
        query_filter = call_args.kwargs.get("query_filter")
        assert query_filter is not None
        assert query_filter.must is not None

    async def test_recommend_with_product_type_filter(self, mock_qdrant_point):
        """Should apply product_type filter to recommendations."""
        user_vector = [0.1] * 32
        self.pipeline.execute.return_value = [json.dumps(user_vector), []]
        self.qdrant.query_points.return_value = MagicMock(points=[mock_qdrant_point])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/recommend/42?product_type=T-shirt")

        assert response.status_code == 200

        call_args = self.qdrant.query_points.call_args
        query_filter = call_args.kwargs.get("query_filter")
        assert query_filter is not None
        assert query_filter.must is not None

    async def test_recommend_with_exclude_groups(self, mock_qdrant_point):
        """Should apply exclude_groups filter to recommendations."""
        user_vector = [0.1] * 32
        self.pipeline.execute.return_value = [json.dumps(user_vector), []]
        self.qdrant.query_points.return_value = MagicMock(points=[mock_qdrant_point])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/recommend/42?exclude_groups=Accessories,Underwear"
            )

        assert response.status_code == 200

        call_args = self.qdrant.query_points.call_args
        query_filter = call_args.kwargs.get("query_filter")
        assert query_filter is not None
        assert query_filter.must_not is not None

    async def test_recommend_with_exclude_types(self, mock_qdrant_point):
        """Should apply exclude_types filter to recommendations."""
        user_vector = [0.1] * 32
        self.pipeline.execute.return_value = [json.dumps(user_vector), []]
        self.qdrant.query_points.return_value = MagicMock(points=[mock_qdrant_point])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/recommend/42?exclude_types=Socks,Belt")

        assert response.status_code == 200

        call_args = self.qdrant.query_points.call_args
        query_filter = call_args.kwargs.get("query_filter")
        assert query_filter is not None
        assert query_filter.must_not is not None

    async def test_recommend_with_combined_filters(self, mock_qdrant_point):
        """Should combine multiple filters correctly."""
        user_vector = [0.1] * 32
        purchased_items = ["100"]
        self.pipeline.execute.return_value = [json.dumps(user_vector), purchased_items]
        self.qdrant.query_points.return_value = MagicMock(points=[mock_qdrant_point])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/recommend/42?product_group=Garment%20Upper%20body&exclude_types=Socks"
            )

        assert response.status_code == 200

        call_args = self.qdrant.query_points.call_args
        query_filter = call_args.kwargs.get("query_filter")
        assert query_filter is not None
        # Should have must (product_group) and must_not (purchased items + exclude_types)
        assert query_filter.must is not None
        assert query_filter.must_not is not None

    async def test_recommend_with_exclude_ids(self, mock_qdrant_point):
        """Should exclude explicitly specified item IDs."""
        user_vector = [0.1] * 32
        self.pipeline.execute.return_value = [json.dumps(user_vector), []]
        self.qdrant.query_points.return_value = MagicMock(points=[mock_qdrant_point])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/recommend/42?exclude_ids=100,200,300")

        assert response.status_code == 200

        call_args = self.qdrant.query_points.call_args
        query_filter = call_args.kwargs.get("query_filter")
        assert query_filter is not None
        assert query_filter.must_not is not None
        # Verify the HasIdCondition contains the excluded IDs
        has_id_condition = query_filter.must_not[0]
        assert hasattr(has_id_condition, "has_id")
        assert set(has_id_condition.has_id) == {100, 200, 300}

    async def test_recommend_with_exclude_ids_combined_with_history(
        self, mock_qdrant_point
    ):
        """Should combine explicit exclude_ids with purchase history exclusions."""
        user_vector = [0.1] * 32
        purchased_items = ["100", "200"]  # Already purchased
        self.pipeline.execute.return_value = [json.dumps(user_vector), purchased_items]
        self.qdrant.query_points.return_value = MagicMock(points=[mock_qdrant_point])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/recommend/42?exclude_ids=300,400")

        assert response.status_code == 200

        call_args = self.qdrant.query_points.call_args
        query_filter = call_args.kwargs.get("query_filter")
        assert query_filter is not None
        assert query_filter.must_not is not None
        # Verify both purchase history and explicit exclusions are combined
        has_id_condition = query_filter.must_not[0]
        assert hasattr(has_id_condition, "has_id")
        # Should have 100, 200 (history) + 300, 400 (explicit) = 4 items
        assert set(has_id_condition.has_id) == {100, 200, 300, 400}

    async def test_recommend_explain_false_by_default(self, mock_qdrant_point):
        """Should not include explanations by default."""
        user_vector = [0.1] * 32
        purchased_items = ["100"]
        self.pipeline.execute.return_value = [json.dumps(user_vector), purchased_items]
        self.qdrant.query_points.return_value = MagicMock(points=[mock_qdrant_point])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/recommend/42")

        assert response.status_code == 200
        data = response.json()
        assert data["recommendations"][0].get("explanation") is None

    async def test_recommend_with_explain_returns_explanations(self, mock_qdrant_point):
        """Should return explanations when explain=true."""
        user_vector = [0.1] * 32
        purchased_items = ["100", "200"]
        self.pipeline.execute.return_value = [json.dumps(user_vector), purchased_items]
        self.qdrant.query_points.return_value = MagicMock(points=[mock_qdrant_point])

        # Mock history item retrieval for explain
        history_point1 = MagicMock()
        history_point1.id = 100
        history_point1.vector = [0.2] * 32
        history_point1.payload = {"prod_name": "History Item 1"}
        history_point2 = MagicMock()
        history_point2.id = 200
        history_point2.vector = [0.15] * 32
        history_point2.payload = {"prod_name": "History Item 2"}
        self.qdrant.retrieve.return_value = [history_point1, history_point2]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/recommend/42?explain=true")

        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "personalized"
        rec = data["recommendations"][0]
        assert rec["explanation"] is not None
        assert len(rec["explanation"]) > 0
        # Check explanation structure
        exp = rec["explanation"][0]
        assert "item_idx" in exp
        assert "item_name" in exp
        assert "similarity" in exp
        assert "contribution_pct" in exp

    async def test_recommend_explain_respects_top_k(self, mock_qdrant_point):
        """Should respect explain_top_k parameter."""
        user_vector = [0.1] * 32
        purchased_items = ["100", "200", "300"]
        self.pipeline.execute.return_value = [json.dumps(user_vector), purchased_items]
        self.qdrant.query_points.return_value = MagicMock(points=[mock_qdrant_point])

        # Mock 3 history items
        history_points = []
        for i, item_id in enumerate([100, 200, 300]):
            hp = MagicMock()
            hp.id = item_id
            hp.vector = [0.1 + i * 0.05] * 32
            hp.payload = {"prod_name": f"History Item {i + 1}"}
            history_points.append(hp)
        self.qdrant.retrieve.return_value = history_points

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/recommend/42?explain=true&explain_top_k=2")

        assert response.status_code == 200
        data = response.json()
        rec = data["recommendations"][0]
        assert rec["explanation"] is not None
        assert len(rec["explanation"]) <= 2

    async def test_recommend_explain_only_for_personalized(self, sample_popular_items):
        """Should not include explanations for trending recommendations."""
        # User has no vector - will fall back to trending
        self.pipeline.execute.return_value = [None, []]
        self.redis.get.return_value = json.dumps(sample_popular_items)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/recommend/9999?explain=true")

        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "trending_now"
        # Trending items should not have explanations
        for rec in data["recommendations"]:
            assert rec.get("explanation") is None

    async def test_recommend_explain_top_k_validation(self):
        """Should reject explain_top_k values > 10."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/recommend/1?explain=true&explain_top_k=15")

        assert response.status_code == 422


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

    async def test_similar_items_excludes_source_item(self, mock_qdrant_point):
        """Should always exclude the source item from results."""
        source_point = MagicMock()
        source_point.id = 100
        source_point.vector = [0.1] * 32
        source_point.payload = {}
        self.qdrant.retrieve.return_value = [source_point]
        self.qdrant.query_points.return_value = MagicMock(points=[mock_qdrant_point])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/items/100/similar")

        assert response.status_code == 200

        call_args = self.qdrant.query_points.call_args
        query_filter = call_args.kwargs.get("query_filter")
        assert query_filter is not None
        assert query_filter.must_not is not None
        has_id_condition = query_filter.must_not[0]
        assert hasattr(has_id_condition, "has_id")
        assert 100 in has_id_condition.has_id

    async def test_similar_items_with_exclude_ids(self, mock_qdrant_point):
        """Should exclude explicitly specified item IDs."""
        source_point = MagicMock()
        source_point.id = 100
        source_point.vector = [0.1] * 32
        source_point.payload = {}
        self.qdrant.retrieve.return_value = [source_point]
        self.qdrant.query_points.return_value = MagicMock(points=[mock_qdrant_point])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/items/100/similar?exclude_ids=200,300")

        assert response.status_code == 200

        call_args = self.qdrant.query_points.call_args
        query_filter = call_args.kwargs.get("query_filter")
        assert query_filter is not None
        assert query_filter.must_not is not None
        has_id_condition = query_filter.must_not[0]
        assert hasattr(has_id_condition, "has_id")
        # Should have source item (100) + explicit exclusions (200, 300)
        assert set(has_id_condition.has_id) == {100, 200, 300}

    async def test_similar_items_with_exclude_groups(self, mock_qdrant_point):
        """Should exclude specified product groups."""
        source_point = MagicMock()
        source_point.id = 100
        source_point.vector = [0.1] * 32
        source_point.payload = {}
        self.qdrant.retrieve.return_value = [source_point]
        self.qdrant.query_points.return_value = MagicMock(points=[mock_qdrant_point])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/items/100/similar?exclude_groups=Accessories,Underwear"
            )

        assert response.status_code == 200

        call_args = self.qdrant.query_points.call_args
        query_filter = call_args.kwargs.get("query_filter")
        assert query_filter is not None
        assert query_filter.must_not is not None
        # Should have at least 2 conditions: HasIdCondition (source item) and FieldCondition (exclude_groups)
        assert len(query_filter.must_not) >= 2

    async def test_similar_items_with_exclude_types(self, mock_qdrant_point):
        """Should exclude specified product types."""
        source_point = MagicMock()
        source_point.id = 100
        source_point.vector = [0.1] * 32
        source_point.payload = {}
        self.qdrant.retrieve.return_value = [source_point]
        self.qdrant.query_points.return_value = MagicMock(points=[mock_qdrant_point])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/items/100/similar?exclude_types=Socks,Belt")

        assert response.status_code == 200

        call_args = self.qdrant.query_points.call_args
        query_filter = call_args.kwargs.get("query_filter")
        assert query_filter is not None
        assert query_filter.must_not is not None
        # Should have at least 2 conditions: HasIdCondition (source item) and FieldCondition (exclude_types)
        assert len(query_filter.must_not) >= 2

    async def test_similar_items_with_product_type_filter(self, mock_qdrant_point):
        """Should filter by product_type when specified."""
        source_point = MagicMock()
        source_point.id = 100
        source_point.vector = [0.1] * 32
        source_point.payload = {}
        self.qdrant.retrieve.return_value = [source_point]
        self.qdrant.query_points.return_value = MagicMock(points=[mock_qdrant_point])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/items/100/similar?product_type=T-shirt")

        assert response.status_code == 200

        call_args = self.qdrant.query_points.call_args
        query_filter = call_args.kwargs.get("query_filter")
        assert query_filter is not None
        assert query_filter.must is not None

    async def test_similar_items_with_combined_filters(self, mock_qdrant_point):
        """Should combine multiple filters correctly."""
        source_point = MagicMock()
        source_point.id = 100
        source_point.vector = [0.1] * 32
        source_point.payload = {}
        self.qdrant.retrieve.return_value = [source_point]
        self.qdrant.query_points.return_value = MagicMock(points=[mock_qdrant_point])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/items/100/similar?product_group=Garment%20Upper%20body&exclude_ids=200&exclude_types=Socks"
            )

        assert response.status_code == 200

        call_args = self.qdrant.query_points.call_args
        query_filter = call_args.kwargs.get("query_filter")
        assert query_filter is not None
        # Should have must (product_group) and must_not (source item + exclude_ids + exclude_types)
        assert query_filter.must is not None
        assert query_filter.must_not is not None
        # Must have at least 2 must_not conditions: HasIdCondition and FieldCondition
        assert len(query_filter.must_not) >= 2


# -----------------------------------------------------------------------------
# User Profile Endpoint Tests
# -----------------------------------------------------------------------------


class TestUserProfileEndpoint:
    """Tests for the /users/{user_idx}/profile endpoint."""

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

    async def test_user_profile_active_user(self):
        """Should return full profile for user with vector and history."""
        user_vector = [0.1] * 32
        # History: (item_idx, timestamp) pairs
        history_with_scores = [("100", 1700000000.0), ("200", 1700001000.0)]
        customer_id = "customer_abc123"
        self.pipeline.execute.return_value = [
            json.dumps(user_vector),
            history_with_scores,
            customer_id,
        ]

        # Mock Qdrant retrieve for item metadata
        point1 = MagicMock()
        point1.id = 100
        point1.payload = {
            "article_id": "1111111111",
            "prod_name": "T-shirt",
            "product_type_name": "T-shirt",
            "product_group_name": "Garment Upper body",
        }
        point2 = MagicMock()
        point2.id = 200
        point2.payload = {
            "article_id": "2222222222",
            "prod_name": "Jeans",
            "product_type_name": "Trousers",
            "product_group_name": "Garment Lower body",
        }
        self.qdrant.retrieve.return_value = [point1, point2]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/users/42/profile")

        assert response.status_code == 200
        data = response.json()
        assert data["user_idx"] == 42
        assert data["customer_id"] == "customer_abc123"
        assert data["total_purchases"] == 2
        assert data["first_purchase_at"] == 1700000000.0
        assert data["last_purchase_at"] == 1700001000.0
        assert len(data["recent_purchases"]) == 2
        assert len(data["top_product_groups"]) == 2
        assert len(data["top_product_types"]) == 2

    async def test_user_profile_no_vector_returns_404(self):
        """Should return 404 for user without vector (even if history exists)."""
        # No vector, but has history - still returns 404
        history_with_scores = [("100", 1700000000.0)]
        customer_id = "customer_xyz789"
        self.pipeline.execute.return_value = [None, history_with_scores, customer_id]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/users/42/profile")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_user_profile_not_found(self):
        """Should return 404 for unknown user."""
        # No vector, no history, no customer_id
        self.pipeline.execute.return_value = [None, [], None]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/users/99999/profile")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_user_profile_respects_recent_limit(self):
        """Should respect recent_limit parameter."""
        user_vector = [0.1] * 32
        # History with 5 items
        history_with_scores = [
            ("100", 1700000000.0),
            ("200", 1700001000.0),
            ("300", 1700002000.0),
            ("400", 1700003000.0),
            ("500", 1700004000.0),
        ]
        self.pipeline.execute.return_value = [
            json.dumps(user_vector),
            history_with_scores,
            "customer_123",
        ]

        points = []
        for i in range(5):
            p = MagicMock()
            p.id = (i + 1) * 100
            p.payload = {"prod_name": f"Product {i + 1}"}
            points.append(p)
        self.qdrant.retrieve.return_value = points

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/users/42/profile?recent_limit=2")

        assert response.status_code == 200
        data = response.json()
        assert data["total_purchases"] == 5  # all items in history
        assert len(data["recent_purchases"]) == 2  # only 2 returned

    async def test_user_profile_respects_top_categories(self):
        """Should respect top_categories parameter."""
        user_vector = [0.1] * 32
        history_with_scores = [
            ("100", 1700000000.0),
            ("200", 1700001000.0),
            ("300", 1700002000.0),
        ]
        self.pipeline.execute.return_value = [
            json.dumps(user_vector),
            history_with_scores,
            "customer_123",
        ]

        points = []
        groups = ["Accessories", "Garment Upper body", "Garment Lower body"]
        for i, group in enumerate(groups):
            p = MagicMock()
            p.id = (i + 1) * 100
            p.payload = {"product_group_name": group, "product_type_name": f"Type{i}"}
            points.append(p)
        self.qdrant.retrieve.return_value = points

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/users/42/profile?top_categories=2")

        assert response.status_code == 200
        data = response.json()
        assert len(data["top_product_groups"]) == 2
        assert len(data["top_product_types"]) == 2

    async def test_user_profile_category_aggregation(self):
        """Should correctly aggregate and calculate category percentages."""
        user_vector = [0.1] * 32
        history_with_scores = [
            ("100", 1700000000.0),
            ("200", 1700001000.0),
            ("300", 1700002000.0),
            ("400", 1700003000.0),
        ]
        self.pipeline.execute.return_value = [
            json.dumps(user_vector),
            history_with_scores,
            "customer_123",
        ]

        # 3 items from "Garment Upper body", 1 from "Accessories"
        points = []
        for i in range(3):
            p = MagicMock()
            p.id = (i + 1) * 100
            p.payload = {
                "product_group_name": "Garment Upper body",
                "product_type_name": "T-shirt",
            }
            points.append(p)
        p4 = MagicMock()
        p4.id = 400
        p4.payload = {
            "product_group_name": "Accessories",
            "product_type_name": "Hat",
        }
        points.append(p4)
        self.qdrant.retrieve.return_value = points

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/users/42/profile")

        assert response.status_code == 200
        data = response.json()

        # Check aggregation
        groups = {g["name"]: g for g in data["top_product_groups"]}
        assert "Garment Upper body" in groups
        assert groups["Garment Upper body"]["count"] == 3
        assert groups["Garment Upper body"]["percentage"] == 75.0
        assert "Accessories" in groups
        assert groups["Accessories"]["count"] == 1
        assert groups["Accessories"]["percentage"] == 25.0

    async def test_user_profile_no_history(self):
        """Should return profile with empty history for user with vector but no purchases."""
        user_vector = [0.1] * 32
        self.pipeline.execute.return_value = [
            json.dumps(user_vector),
            [],
            "customer_123",
        ]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/users/42/profile")

        assert response.status_code == 200
        data = response.json()
        assert data["total_purchases"] == 0
        assert data["recent_purchases"] == []
        assert data["top_product_groups"] == []
        assert data["top_product_types"] == []
        assert data["first_purchase_at"] is None
        assert data["last_purchase_at"] is None
