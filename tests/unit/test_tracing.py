"""
Unit tests for the OpenTelemetry tracing utilities.

Tests the qdrant_span context manager for creating spans
around Qdrant operations.
"""

from unittest.mock import MagicMock, patch

from src.observability.tracing import qdrant_span


class TestQdrantSpan:
    """Tests for the qdrant_span context manager."""

    def test_creates_span_with_correct_name(self):
        """Should create a span named 'qdrant.<operation>'."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=None
        )

        with patch(
            "src.observability.tracing.trace.get_tracer", return_value=mock_tracer
        ):
            with qdrant_span("query_points", "items"):
                pass

        mock_tracer.start_as_current_span.assert_called_once()
        call_args = mock_tracer.start_as_current_span.call_args
        assert call_args[0][0] == "qdrant.query_points"

    def test_sets_db_system_attribute(self):
        """Should set db.system attribute to 'qdrant'."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=None
        )

        with patch(
            "src.observability.tracing.trace.get_tracer", return_value=mock_tracer
        ):
            with qdrant_span("query_points", "items"):
                pass

        mock_span.set_attribute.assert_any_call("db.system", "qdrant")

    def test_sets_db_operation_attribute(self):
        """Should set db.operation attribute to the operation name."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=None
        )

        with patch(
            "src.observability.tracing.trace.get_tracer", return_value=mock_tracer
        ):
            with qdrant_span("retrieve", "items"):
                pass

        mock_span.set_attribute.assert_any_call("db.operation", "retrieve")

    def test_sets_db_collection_attribute(self):
        """Should set db.collection attribute to the collection name."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=None
        )

        with patch(
            "src.observability.tracing.trace.get_tracer", return_value=mock_tracer
        ):
            with qdrant_span("query_points", "my_collection"):
                pass

        mock_span.set_attribute.assert_any_call("db.collection", "my_collection")

    def test_sets_custom_attributes(self):
        """Should set custom attributes with db.qdrant prefix."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=None
        )

        with patch(
            "src.observability.tracing.trace.get_tracer", return_value=mock_tracer
        ):
            with qdrant_span("query_points", "items", limit=10, with_vectors=True):
                pass

        mock_span.set_attribute.assert_any_call("db.qdrant.limit", 10)
        mock_span.set_attribute.assert_any_call("db.qdrant.with_vectors", True)

    def test_skips_none_attributes(self):
        """Should not set attributes with None values."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=None
        )

        with patch(
            "src.observability.tracing.trace.get_tracer", return_value=mock_tracer
        ):
            with qdrant_span("query_points", "items", limit=10, filter=None):
                pass

        # Check that db.qdrant.filter was NOT called
        calls = [call for call in mock_span.set_attribute.call_args_list]
        filter_calls = [c for c in calls if c[0][0] == "db.qdrant.filter"]
        assert len(filter_calls) == 0

    def test_yields_span_for_additional_attributes(self):
        """Should yield the span so caller can set additional attributes."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=None
        )

        with patch(
            "src.observability.tracing.trace.get_tracer", return_value=mock_tracer
        ):
            with qdrant_span("query_points", "items") as span:
                span.set_attribute("db.qdrant.results_count", 42)

        mock_span.set_attribute.assert_any_call("db.qdrant.results_count", 42)

    def test_records_exception_on_error(self):
        """Should record exception and set error status when exception occurs."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=None
        )

        with patch(
            "src.observability.tracing.trace.get_tracer", return_value=mock_tracer
        ):
            try:
                with qdrant_span("query_points", "items"):
                    raise ValueError("Connection failed")
            except ValueError:
                pass

        mock_span.record_exception.assert_called_once()
        mock_span.set_status.assert_called_once()

    def test_reraises_exception(self):
        """Should re-raise exceptions after recording them."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=None
        )

        with patch(
            "src.observability.tracing.trace.get_tracer", return_value=mock_tracer
        ):
            exception_raised = False
            try:
                with qdrant_span("query_points", "items"):
                    raise ValueError("Connection failed")
            except ValueError:
                exception_raised = True

        assert exception_raised, "Exception should be re-raised"
