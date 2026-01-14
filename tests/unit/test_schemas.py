"""
Unit tests for Pydantic schemas used in the API layer.

Validates serialization, validation, and edge cases for
request/response models.
"""

from src.serve.schemas import RecommendationItem, RecommendationResponse


class TestRecommendationItem:
    """Tests for the RecommendationItem model."""

    def test_valid_item_with_int_id(self):
        """Item with integer ID should serialize correctly."""
        item = RecommendationItem(
            item_id=123,
            score=0.95,
            metadata={"name": "Test Product"},
        )

        assert item.item_id == 123
        assert item.score == 0.95
        assert item.metadata == {"name": "Test Product"}

    def test_valid_item_with_string_id(self):
        """Item with string ID should serialize correctly."""
        item = RecommendationItem(
            item_id="abc123",
            score=0.85,
            metadata={},
        )

        assert item.item_id == "abc123"

    def test_default_empty_metadata(self):
        """Metadata should default to empty dict if not provided."""
        item = RecommendationItem(item_id=1, score=0.5)

        assert item.metadata == {}

    def test_score_can_be_zero(self):
        """Zero score should be valid."""
        item = RecommendationItem(item_id=1, score=0.0)

        assert item.score == 0.0

    def test_score_can_be_negative(self):
        """Negative scores should be allowed (some similarity metrics use them)."""
        item = RecommendationItem(item_id=1, score=-0.5)

        assert item.score == -0.5

    def test_serialization_to_dict(self):
        """Model should serialize to dict for JSON responses."""
        item = RecommendationItem(
            item_id=42,
            score=0.99,
            metadata={"category": "Electronics"},
        )

        data = item.model_dump()

        assert data == {
            "item_id": 42,
            "score": 0.99,
            "metadata": {"category": "Electronics"},
        }


class TestRecommendationResponse:
    """Tests for the RecommendationResponse model."""

    def test_valid_response_with_items(self):
        """Response with recommendations should be valid."""
        response = RecommendationResponse(
            user_id=123,
            source="personalized",
            recommendations=[
                RecommendationItem(item_id=1, score=0.9),
                RecommendationItem(item_id=2, score=0.8),
            ],
        )

        assert response.user_id == 123
        assert response.source == "personalized"
        assert len(response.recommendations) == 2

    def test_empty_recommendations_list(self):
        """Empty recommendations list should be valid."""
        response = RecommendationResponse(
            user_id=1,
            source="trending_now",
            recommendations=[],
        )

        assert response.recommendations == []

    def test_source_values(self):
        """Both source values used by the API should work."""
        personalized = RecommendationResponse(
            user_id=1, source="personalized", recommendations=[]
        )
        trending = RecommendationResponse(
            user_id=1, source="trending_now", recommendations=[]
        )

        assert personalized.source == "personalized"
        assert trending.source == "trending_now"

    def test_serialization_to_dict(self):
        """Full response should serialize correctly."""
        response = RecommendationResponse(
            user_id=42,
            source="personalized",
            recommendations=[
                RecommendationItem(
                    item_id=100, score=0.95, metadata={"name": "Product A"}
                ),
            ],
        )

        data = response.model_dump()

        assert data["user_id"] == 42
        assert data["source"] == "personalized"
        assert len(data["recommendations"]) == 1
        assert data["recommendations"][0]["item_id"] == 100
