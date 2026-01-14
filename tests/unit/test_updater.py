"""
Unit tests for the online learning logic in updater.py.

Tests the core vector calculation that drifts user profiles
toward items they interact with.
"""

import pytest
import numpy as np

from src.stream.updater import calculate_new_vector
from src.config import LEARNING_RATE


class TestCalculateNewVector:
    """Tests for the calculate_new_vector function."""

    def test_cold_start_returns_item_vector(self, sample_item_vector):
        """
        When user has no existing vector (cold start), the item vector
        should be used as the starting point for their profile.
        """
        user_vector = None
        weight_multiplier = 1.0

        result = calculate_new_vector(user_vector, sample_item_vector, weight_multiplier)

        # Should return exactly the item vector for cold start
        np.testing.assert_array_equal(result, sample_item_vector)

    def test_update_applies_exponential_moving_average(self):
        """
        When user has an existing vector, the new vector should be
        a weighted combination (EMA) of old and new.
        """
        user_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        item_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        weight_multiplier = 1.0

        result = calculate_new_vector(user_vector, item_vector, weight_multiplier)

        # Expected: user_vector * (1 - lr) + item_vector * lr
        # With LEARNING_RATE = 0.2 and weight_multiplier = 1.0:
        # [1, 0, 0] * 0.8 + [0, 1, 0] * 0.2 = [0.8, 0.2, 0]
        expected = user_vector * (1 - LEARNING_RATE) + item_vector * LEARNING_RATE
        np.testing.assert_array_almost_equal(result, expected)

    def test_weight_multiplier_scales_learning_rate(self):
        """
        Different event types should have different impact on the user vector.
        A purchase (weight=1.0) should move the vector more than a click (weight=0.1).
        """
        user_vector = np.array([1.0, 0.0], dtype=np.float32)
        item_vector = np.array([0.0, 1.0], dtype=np.float32)

        # Simulate a click (low weight)
        click_result = calculate_new_vector(user_vector, item_vector, weight_multiplier=0.1)

        # Simulate a purchase (high weight)
        purchase_result = calculate_new_vector(user_vector, item_vector, weight_multiplier=1.0)

        # Purchase should move the vector more toward the item
        # Check that purchase result is closer to item_vector than click result
        click_distance_to_item = np.linalg.norm(click_result - item_vector)
        purchase_distance_to_item = np.linalg.norm(purchase_result - item_vector)

        assert purchase_distance_to_item < click_distance_to_item, (
            "Purchase should move user vector closer to item than click"
        )

    def test_zero_weight_multiplier_no_change(self):
        """
        With weight_multiplier=0, the user vector should not change.
        """
        user_vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        item_vector = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        result = calculate_new_vector(user_vector, item_vector, weight_multiplier=0.0)

        np.testing.assert_array_equal(result, user_vector)

    def test_result_is_float32(self, sample_user_vector, sample_item_vector):
        """
        Output should maintain float32 dtype for memory efficiency
        and Qdrant compatibility.
        """
        result = calculate_new_vector(sample_user_vector, sample_item_vector, 1.0)

        assert result.dtype == np.float32

    def test_vector_dimension_preserved(self):
        """
        Output vector should have same dimension as inputs.
        """
        dim = 32
        user_vector = np.random.rand(dim).astype(np.float32)
        item_vector = np.random.rand(dim).astype(np.float32)

        result = calculate_new_vector(user_vector, item_vector, 1.0)

        assert result.shape == (dim,)
