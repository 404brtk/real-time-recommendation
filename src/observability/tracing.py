import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor


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
