import os
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor


@contextmanager
def qdrant_span(
    operation: str,
    collection: str,
    **attributes: Any,
):
    """
    Create an OpenTelemetry span for Qdrant operations.

    Usage:
        with qdrant_span("query_points", "items", limit=10) as span:
            result = await qdrant_client.query_points(...)
            span.set_attribute("db.qdrant.results_count", len(result.points))

    Args:
        operation: Qdrant operation name (e.g., "query_points", "retrieve")
        collection: Collection name being queried
        **attributes: Additional span attributes (e.g., limit, with_vectors)

    Yields:
        The active span, allowing additional attributes to be set after the call
    """
    tracer = trace.get_tracer("recommendation-api")
    with tracer.start_as_current_span(
        f"qdrant.{operation}",
        kind=trace.SpanKind.CLIENT,
    ) as span:
        span.set_attribute("db.system", "qdrant")
        span.set_attribute("db.operation", operation)
        span.set_attribute("db.collection", collection)
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(f"db.qdrant.{key}", value)
        try:
            yield span
        except Exception as e:
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def setup_tracing(app, service_name: str = "recommendation-api"):
    """
    setup opentelemetry tracing with jaeger exporter.
    """
    jaeger_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        exporter = OTLPSpanExporter(endpoint=jaeger_endpoint, insecure=True)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
    except Exception as e:
        # if jaeger isn't running, just log and continue without tracing export
        print(f"Tracing export disabled (Jaeger not available): {e}")

    trace.set_tracer_provider(provider)

    # auto-instrument fastapi
    FastAPIInstrumentor.instrument_app(app)

    # auto-instrument redis
    RedisInstrumentor().instrument()

    return trace.get_tracer(service_name)
