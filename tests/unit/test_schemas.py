"""
Unit tests for Pydantic schemas used in the API layer.

Validates serialization, validation, and edge cases for
request/response models.
"""

from src.serve.schemas import (
    ContributionItem,
    RecommendationItem,
    RecommendationResponse,
    PurchaseEvent,
    SimilarItemsResponse,
    CategoryCount,
    UserProfile,
)


class TestRecommendationItem:
    """Tests for the RecommendationItem model."""

    def test_valid_item_with_int_idx(self):
        """Item with integer index should serialize correctly."""
        item = RecommendationItem(
            item_idx=123,
            score=0.95,
            metadata={"name": "Test Product"},
        )

        assert item.item_idx == 123
        assert item.score == 0.95
        assert item.metadata == {"name": "Test Product"}

    def test_default_empty_metadata(self):
        """Metadata should default to empty dict if not provided."""
        item = RecommendationItem(item_idx=1, score=0.5)

        assert item.metadata == {}

    def test_score_can_be_zero(self):
        """Zero score should be valid."""
        item = RecommendationItem(item_idx=1, score=0.0)

        assert item.score == 0.0

    def test_score_can_be_negative(self):
        """Negative scores should be allowed (some similarity metrics use them)."""
        item = RecommendationItem(item_idx=1, score=-0.5)

        assert item.score == -0.5

    def test_serialization_to_dict(self):
        """Model should serialize to dict for JSON responses."""
        item = RecommendationItem(
            item_idx=42,
            score=0.99,
            metadata={"category": "Electronics"},
        )

        data = item.model_dump()

        assert data == {
            "item_idx": 42,
            "score": 0.99,
            "metadata": {"category": "Electronics"},
            "explanation": None,
        }


class TestRecommendationResponse:
    """Tests for the RecommendationResponse model."""

    def test_valid_response_with_items(self):
        """Response with recommendations should be valid."""
        response = RecommendationResponse(
            user_idx=123,
            source="personalized",
            recommendations=[
                RecommendationItem(item_idx=1, score=0.9),
                RecommendationItem(item_idx=2, score=0.8),
            ],
        )

        assert response.user_idx == 123
        assert response.source == "personalized"
        assert len(response.recommendations) == 2

    def test_empty_recommendations_list(self):
        """Empty recommendations list should be valid."""
        response = RecommendationResponse(
            user_idx=1,
            source="trending_now",
            recommendations=[],
        )

        assert response.recommendations == []

    def test_source_values(self):
        """Both source values used by the API should work."""
        personalized = RecommendationResponse(
            user_idx=1, source="personalized", recommendations=[]
        )
        trending = RecommendationResponse(
            user_idx=1, source="trending_now", recommendations=[]
        )

        assert personalized.source == "personalized"
        assert trending.source == "trending_now"

    def test_serialization_to_dict(self):
        """Full response should serialize correctly."""
        response = RecommendationResponse(
            user_idx=42,
            source="personalized",
            recommendations=[
                RecommendationItem(
                    item_idx=100, score=0.95, metadata={"name": "Product A"}
                ),
            ],
        )

        data = response.model_dump()

        assert data["user_idx"] == 42
        assert data["source"] == "personalized"
        assert len(data["recommendations"]) == 1
        assert data["recommendations"][0]["item_idx"] == 100


class TestPurchaseEvent:
    """Tests for the PurchaseEvent model."""

    def test_valid_purchase_event(self):
        """Purchase event with valid indices should work."""
        event = PurchaseEvent(user_idx=42, item_idx=100)

        assert event.user_idx == 42
        assert event.item_idx == 100

    def test_serialization_to_dict(self):
        """Purchase event should serialize correctly."""
        event = PurchaseEvent(user_idx=1, item_idx=2)

        data = event.model_dump()

        assert data == {"user_idx": 1, "item_idx": 2}


class TestContributionItem:
    """Tests for the ContributionItem model (explain feature)."""

    def test_valid_contribution_item(self):
        """Contribution item with all fields should be valid."""
        item = ContributionItem(
            item_idx=100,
            item_name="Blue T-Shirt",
            similarity=0.8523,
            contribution_pct=45.2,
        )

        assert item.item_idx == 100
        assert item.item_name == "Blue T-Shirt"
        assert item.similarity == 0.8523
        assert item.contribution_pct == 45.2

    def test_serialization_to_dict(self):
        """Contribution item should serialize correctly."""
        item = ContributionItem(
            item_idx=42,
            item_name="Test Product",
            similarity=0.95,
            contribution_pct=33.3,
        )

        data = item.model_dump()

        assert data == {
            "item_idx": 42,
            "item_name": "Test Product",
            "similarity": 0.95,
            "contribution_pct": 33.3,
        }


class TestSimilarItemsResponse:
    """Tests for the SimilarItemsResponse model."""

    def test_valid_response_with_items(self):
        """Response with similar items should be valid."""
        response = SimilarItemsResponse(
            source_item_idx=100,
            source_metadata={
                "prod_name": "Source Product",
                "product_group_name": "Accessories",
            },
            similar_items=[
                RecommendationItem(item_idx=101, score=0.95),
                RecommendationItem(item_idx=102, score=0.89),
            ],
        )

        assert response.source_item_idx == 100
        assert response.source_metadata["prod_name"] == "Source Product"
        assert len(response.similar_items) == 2

    def test_empty_similar_items_list(self):
        """Empty similar items list should be valid."""
        response = SimilarItemsResponse(
            source_item_idx=100,
            source_metadata={},
            similar_items=[],
        )

        assert response.similar_items == []

    def test_default_empty_source_metadata(self):
        """Source metadata should default to empty dict if not provided."""
        response = SimilarItemsResponse(
            source_item_idx=100,
            similar_items=[],
        )

        assert response.source_metadata == {}

    def test_serialization_to_dict(self):
        """Similar items response should serialize correctly."""
        response = SimilarItemsResponse(
            source_item_idx=50,
            source_metadata={"prod_name": "Test"},
            similar_items=[
                RecommendationItem(
                    item_idx=51, score=0.9, metadata={"prod_name": "Similar"}
                ),
            ],
        )

        data = response.model_dump()

        assert data["source_item_idx"] == 50
        assert data["source_metadata"] == {"prod_name": "Test"}
        assert len(data["similar_items"]) == 1
        assert data["similar_items"][0]["item_idx"] == 51


class TestCategoryCount:
    """Tests for the CategoryCount model."""

    def test_valid_category_count(self):
        """Category count with all fields should be valid."""
        category = CategoryCount(
            name="Garment Upper body",
            count=15,
            percentage=75.0,
        )

        assert category.name == "Garment Upper body"
        assert category.count == 15
        assert category.percentage == 75.0

    def test_zero_count(self):
        """Zero count should be valid."""
        category = CategoryCount(name="Accessories", count=0, percentage=0.0)

        assert category.count == 0
        assert category.percentage == 0.0

    def test_serialization_to_dict(self):
        """Category count should serialize correctly."""
        category = CategoryCount(
            name="Swimwear",
            count=5,
            percentage=25.0,
        )

        data = category.model_dump()

        assert data == {
            "name": "Swimwear",
            "count": 5,
            "percentage": 25.0,
        }


class TestUserProfile:
    """Tests for the UserProfile model."""

    def test_valid_profile_with_history(self):
        """User profile with full history should be valid."""
        profile = UserProfile(
            user_idx=42,
            customer_id="customer_abc123",
            total_purchases=10,
            first_purchase_at=1700000000.0,
            last_purchase_at=1700500000.0,
            recent_purchases=[
                RecommendationItem(
                    item_idx=100, score=0.0, metadata={"prod_name": "T-shirt"}
                ),
            ],
            top_product_groups=[
                CategoryCount(name="Garment Upper body", count=7, percentage=70.0),
            ],
            top_product_types=[
                CategoryCount(name="T-shirt", count=5, percentage=50.0),
            ],
        )

        assert profile.user_idx == 42
        assert profile.customer_id == "customer_abc123"
        assert profile.total_purchases == 10
        assert profile.first_purchase_at == 1700000000.0
        assert profile.last_purchase_at == 1700500000.0
        assert len(profile.recent_purchases) == 1
        assert len(profile.top_product_groups) == 1
        assert len(profile.top_product_types) == 1

    def test_profile_with_no_history(self):
        """User profile with no purchase history should be valid."""
        profile = UserProfile(
            user_idx=99,
            customer_id="new_customer",
            total_purchases=0,
            first_purchase_at=None,
            last_purchase_at=None,
            recent_purchases=[],
            top_product_groups=[],
            top_product_types=[],
        )

        assert profile.total_purchases == 0
        assert profile.recent_purchases == []
        assert profile.top_product_groups == []
        assert profile.top_product_types == []

    def test_optional_fields_can_be_none(self):
        """Optional fields should accept None values."""
        profile = UserProfile(
            user_idx=1,
            customer_id=None,
            total_purchases=0,
            first_purchase_at=None,
            last_purchase_at=None,
            recent_purchases=[],
            top_product_groups=[],
            top_product_types=[],
        )

        assert profile.customer_id is None
        assert profile.first_purchase_at is None
        assert profile.last_purchase_at is None

    def test_serialization_to_dict(self):
        """User profile should serialize correctly."""
        profile = UserProfile(
            user_idx=10,
            customer_id="cust_123",
            total_purchases=2,
            first_purchase_at=1700000000.0,
            last_purchase_at=1700001000.0,
            recent_purchases=[
                RecommendationItem(item_idx=1, score=0.0),
            ],
            top_product_groups=[
                CategoryCount(name="Accessories", count=2, percentage=100.0),
            ],
            top_product_types=[
                CategoryCount(name="Hat", count=1, percentage=50.0),
            ],
        )

        data = profile.model_dump()

        assert data["user_idx"] == 10
        assert data["customer_id"] == "cust_123"
        assert data["total_purchases"] == 2
        assert data["first_purchase_at"] == 1700000000.0
        assert data["last_purchase_at"] == 1700001000.0
        assert len(data["recent_purchases"]) == 1
        assert len(data["top_product_groups"]) == 1
        assert len(data["top_product_types"]) == 1
