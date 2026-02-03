"""Unit tests for API utility functions."""

import math
import pytest
from qdrant_client import models

from src.serve.api import (
    build_qdrant_filter,
    compute_cosine_similarity,
    compute_diversity,
    mmr_rerank,
)


class TestComputeCosineSimilarity:
    """Tests for compute_cosine_similarity function."""

    def test_identical_vectors_return_one(self):
        """Identical vectors should have cosine similarity of 1.0."""
        vec = [1.0, 2.0, 3.0]
        result = compute_cosine_similarity(vec, vec)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors_return_zero(self):
        """Orthogonal vectors should have cosine similarity of 0.0."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        result = compute_cosine_similarity(vec_a, vec_b)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors_return_negative_one(self):
        """Opposite vectors should have cosine similarity of -1.0."""
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [-1.0, -2.0, -3.0]
        result = compute_cosine_similarity(vec_a, vec_b)
        assert result == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector_a_returns_zero(self):
        """Zero vector should return 0.0 similarity."""
        vec_a = [0.0, 0.0, 0.0]
        vec_b = [1.0, 2.0, 3.0]
        result = compute_cosine_similarity(vec_a, vec_b)
        assert result == 0.0

    def test_zero_vector_b_returns_zero(self):
        """Zero vector should return 0.0 similarity."""
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [0.0, 0.0, 0.0]
        result = compute_cosine_similarity(vec_a, vec_b)
        assert result == 0.0

    def test_both_zero_vectors_return_zero(self):
        """Both zero vectors should return 0.0 similarity."""
        vec_a = [0.0, 0.0, 0.0]
        vec_b = [0.0, 0.0, 0.0]
        result = compute_cosine_similarity(vec_a, vec_b)
        assert result == 0.0

    def test_scaled_vectors_return_one(self):
        """Scaled versions of same vector should have similarity 1.0."""
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [2.0, 4.0, 6.0]
        result = compute_cosine_similarity(vec_a, vec_b)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_known_angle_vectors(self):
        """Test vectors at 45 degrees (cos(45) = sqrt(2)/2 ~ 0.707)."""
        vec_a = [1.0, 0.0]
        vec_b = [1.0, 1.0]
        result = compute_cosine_similarity(vec_a, vec_b)
        expected = math.sqrt(2) / 2
        assert result == pytest.approx(expected, abs=1e-6)


class TestComputeDiversity:
    """Tests for compute_diversity function."""

    def test_single_vector_returns_zero(self):
        """Single vector should return diversity of 0."""
        vectors = [[1.0, 2.0, 3.0]]
        result = compute_diversity(vectors)
        assert result == 0.0

    def test_empty_list_returns_zero(self):
        """Empty list should return diversity of 0."""
        vectors = []
        result = compute_diversity(vectors)
        assert result == 0.0

    def test_identical_vectors_return_zero(self):
        """Identical vectors should have diversity of 0."""
        vectors = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
        result = compute_diversity(vectors)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_scaled_identical_vectors_return_zero(self):
        """Scaled versions of same vector should have diversity of 0."""
        vectors = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]]
        result = compute_diversity(vectors)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_vectors_return_one(self):
        """Orthogonal vectors should have diversity of 1 (cosine distance)."""
        vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        result = compute_diversity(vectors)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors_return_two(self):
        """Opposite vectors should have diversity of 2 (max cosine distance)."""
        vectors = [[1.0, 0.0], [-1.0, 0.0]]
        result = compute_diversity(vectors)
        assert result == pytest.approx(2.0, abs=1e-6)

    def test_diversity_between_zero_and_two(self):
        """Diversity should always be between 0 and 2."""
        # Random-ish vectors
        vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        result = compute_diversity(vectors)
        assert 0.0 <= result <= 2.0

    def test_more_diverse_vectors_have_higher_diversity(self):
        """More diverse vectors should have higher diversity score."""
        # Similar vectors (all positive, similar direction)
        similar = [[1.0, 1.0], [1.1, 1.0], [1.0, 1.1]]
        # Diverse vectors (different directions)
        diverse = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]

        similar_diversity = compute_diversity(similar)
        diverse_diversity = compute_diversity(diverse)

        assert diverse_diversity > similar_diversity

    def test_handles_zero_vectors(self):
        """Should handle zero vectors without error."""
        vectors = [[0.0, 0.0], [1.0, 0.0]]
        result = compute_diversity(vectors)
        # Zero vector normalized stays zero, similarity with anything is 0
        # So distance is 1
        assert result == pytest.approx(1.0, abs=1e-6)


class TestMMRRerank:
    """Tests for mmr_rerank function."""

    def test_empty_input_returns_empty_list(self):
        """Empty input should return empty list."""
        result = mmr_rerank([], [], lambda_param=0.5, k=5)
        assert result == []

    def test_returns_correct_number_of_items(self):
        """Should return exactly k items."""
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        vectors = [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]]
        result = mmr_rerank(scores, vectors, lambda_param=0.5, k=3)
        assert len(result) == 3

    def test_k_larger_than_input_returns_all(self):
        """When k > n, should return all items."""
        scores = [0.9, 0.8, 0.7]
        vectors = [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]
        result = mmr_rerank(scores, vectors, lambda_param=0.5, k=10)
        assert len(result) == 3
        assert set(result) == {0, 1, 2}

    def test_single_item_returns_that_item(self):
        """Single item should return that item."""
        scores = [0.9]
        vectors = [[1.0, 2.0, 3.0]]
        result = mmr_rerank(scores, vectors, lambda_param=0.5, k=1)
        assert result == [0]

    def test_lambda_one_returns_score_order(self):
        """lambda=1.0 (pure relevance) should return items in score order."""
        scores = [0.5, 0.9, 0.7, 0.3, 0.8]
        # Use diverse vectors so diversity penalty wouldn't affect order
        vectors = [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
        result = mmr_rerank(scores, vectors, lambda_param=1.0, k=5)
        # Should be ordered by score: indices [1, 4, 2, 0, 3] (scores 0.9, 0.8, 0.7, 0.5, 0.3)
        assert result == [1, 4, 2, 0, 3]

    def test_lambda_zero_maximizes_diversity(self):
        """lambda=0.0 (pure diversity) should select diverse items."""
        scores = [0.9, 0.85, 0.8]  # Similar high scores
        # Two similar vectors and one different
        vectors = [
            [1.0, 0.0],  # index 0
            [0.99, 0.1],  # index 1 - very similar to 0
            [0.0, 1.0],  # index 2 - orthogonal to 0 and 1
        ]
        result = mmr_rerank(scores, vectors, lambda_param=0.0, k=3)
        # First pick: all have max_sim=0, so pick highest score (index 0)
        assert result[0] == 0
        # Second pick: index 2 has lower max_sim to selected (orthogonal)
        # so with lambda=0, it should be picked over index 1
        assert result[1] == 2

    def test_lambda_half_balances_relevance_and_diversity(self):
        """lambda=0.5 should balance relevance and diversity."""
        scores = [0.9, 0.85, 0.3]
        vectors = [
            [1.0, 0.0],  # index 0 - highest score
            [0.99, 0.1],  # index 1 - similar to 0, high score
            [0.0, 1.0],  # index 2 - diverse but low score
        ]
        result = mmr_rerank(scores, vectors, lambda_param=0.5, k=3)
        # First item should be highest score (index 0)
        assert result[0] == 0
        # The balance between 0.85 relevance + similarity penalty vs 0.3 relevance + no penalty
        # will determine second pick - depends on exact normalization

    def test_no_duplicate_indices(self):
        """Result should not contain duplicate indices."""
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        vectors = [[float(i)] * 3 for i in range(5)]
        result = mmr_rerank(scores, vectors, lambda_param=0.5, k=5)
        assert len(result) == len(set(result))

    def test_all_indices_valid(self):
        """All returned indices should be valid."""
        scores = [0.9, 0.8, 0.7]
        vectors = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        result = mmr_rerank(scores, vectors, lambda_param=0.5, k=3)
        for idx in result:
            assert 0 <= idx < len(scores)

    def test_same_scores_different_vectors(self):
        """When scores are equal, diversity should determine order."""
        scores = [1.0, 1.0, 1.0]
        vectors = [
            [1.0, 0.0],  # index 0
            [1.0, 0.0],  # index 1 - same as 0
            [0.0, 1.0],  # index 2 - orthogonal
        ]
        result = mmr_rerank(scores, vectors, lambda_param=0.5, k=3)
        # First pick is arbitrary (equal scores, no prior selection)
        first = result[0]
        # If first is 0 or 1, second should be 2 (orthogonal)
        if first in [0, 1]:
            assert result[1] == 2


class TestBuildQdrantFilter:
    """Tests for build_qdrant_filter function."""

    def test_no_filters_returns_none(self):
        """No filter parameters should return None."""
        result = build_qdrant_filter()
        assert result is None

    def test_empty_filters_returns_none(self):
        """Empty filter lists should return None."""
        result = build_qdrant_filter(
            excluded_ids=[],
            product_group=None,
            product_type=None,
            exclude_groups=[],
            exclude_types=[],
        )
        assert result is None

    def test_product_group_creates_must_condition(self):
        """product_group should create a must condition."""
        result = build_qdrant_filter(product_group="Garment Upper body")
        assert result is not None
        assert result.must is not None
        assert len(result.must) == 1
        condition = result.must[0]
        assert isinstance(condition, models.FieldCondition)
        assert condition.key == "product_group_name"
        assert condition.match.value == "Garment Upper body"

    def test_product_type_creates_must_condition(self):
        """product_type should create a must condition."""
        result = build_qdrant_filter(product_type="T-shirt")
        assert result is not None
        assert result.must is not None
        assert len(result.must) == 1
        condition = result.must[0]
        assert isinstance(condition, models.FieldCondition)
        assert condition.key == "product_type_name"
        assert condition.match.value == "T-shirt"

    def test_excluded_ids_creates_must_not_condition(self):
        """excluded_ids should create a must_not HasIdCondition."""
        result = build_qdrant_filter(excluded_ids=[1, 2, 3])
        assert result is not None
        assert result.must_not is not None
        assert len(result.must_not) == 1
        condition = result.must_not[0]
        assert isinstance(condition, models.HasIdCondition)
        assert condition.has_id == [1, 2, 3]

    def test_exclude_groups_creates_must_not_condition(self):
        """exclude_groups should create a must_not MatchAny condition."""
        result = build_qdrant_filter(exclude_groups=["Accessories", "Shoes"])
        assert result is not None
        assert result.must_not is not None
        assert len(result.must_not) == 1
        condition = result.must_not[0]
        assert isinstance(condition, models.FieldCondition)
        assert condition.key == "product_group_name"
        assert condition.match.any == ["Accessories", "Shoes"]

    def test_exclude_types_creates_must_not_condition(self):
        """exclude_types should create a must_not MatchAny condition."""
        result = build_qdrant_filter(exclude_types=["Socks", "Underwear"])
        assert result is not None
        assert result.must_not is not None
        assert len(result.must_not) == 1
        condition = result.must_not[0]
        assert isinstance(condition, models.FieldCondition)
        assert condition.key == "product_type_name"
        assert condition.match.any == ["Socks", "Underwear"]

    def test_combines_multiple_must_conditions(self):
        """Multiple must filters should be combined."""
        result = build_qdrant_filter(
            product_group="Garment Upper body", product_type="T-shirt"
        )
        assert result is not None
        assert result.must is not None
        assert len(result.must) == 2
        keys = {c.key for c in result.must}
        assert keys == {"product_group_name", "product_type_name"}

    def test_combines_multiple_must_not_conditions(self):
        """Multiple must_not filters should be combined."""
        result = build_qdrant_filter(
            excluded_ids=[1, 2],
            exclude_groups=["Accessories"],
            exclude_types=["Socks"],
        )
        assert result is not None
        assert result.must_not is not None
        assert len(result.must_not) == 3

    def test_combines_must_and_must_not(self):
        """Should combine must and must_not conditions."""
        result = build_qdrant_filter(
            product_group="Garment Upper body",
            excluded_ids=[1, 2, 3],
            exclude_types=["Socks"],
        )
        assert result is not None
        assert result.must is not None
        assert result.must_not is not None
        assert len(result.must) == 1
        assert len(result.must_not) == 2

    def test_all_filters_combined(self):
        """All filter types should work together."""
        result = build_qdrant_filter(
            excluded_ids=[1, 2],
            product_group="Garment Upper body",
            product_type="T-shirt",
            exclude_groups=["Accessories"],
            exclude_types=["Socks", "Underwear"],
        )
        assert result is not None
        assert result.must is not None
        assert result.must_not is not None
        assert len(result.must) == 2  # product_group + product_type
        assert (
            len(result.must_not) == 3
        )  # excluded_ids + exclude_groups + exclude_types
