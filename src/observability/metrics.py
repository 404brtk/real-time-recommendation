from prometheus_client import Counter, Histogram, Gauge


class Metrics:
    def __init__(self):
        # api metrics
        self.recommendation_requests = Counter(
            "recommendation_requests_total",
            "Total recommendation requests",
            ["source", "user_type"],
        )

        self.recommendation_fallback = Counter(
            "recommendation_fallback_total",
            "Fallback to trending recommendations",
            ["reason"],
        )

        self.items_excluded_history = Counter(
            "items_excluded_history_total",
            "Items excluded from recommendations due to purchase history",
        )

        self.items_excluded_explicit = Counter(
            "items_excluded_explicit_total",
            "Items explicitly excluded via exclude_ids parameter",
        )

        self.filter_applied = Counter(
            "filter_applied_total",
            "Filters applied to recommendations",
            [
                "filter_type"
            ],  # "product_group", "product_type", "exclude_ids", "exclude_groups", "exclude_types"
        )

        self.redis_operation_duration = Histogram(
            "redis_operation_duration_seconds",
            "Redis operation latency",
            ["operation"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        )

        self.qdrant_search_duration = Histogram(
            "qdrant_search_duration_seconds",
            "Qdrant vector search latency",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        )

        # stream processor metrics
        self.events_processed = Counter(
            "events_processed_total",
            "Events processed by updater",
            ["event_type"],
        )

        self.event_processing_duration = Histogram(
            "event_processing_duration_seconds",
            "Event processing latency",
            ["event_type"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
        )

        self.debounce_hits = Counter(
            "debounce_hits_total",
            "Duplicate events skipped due to debounce",
        )

        self.user_vector_updates = Counter(
            "user_vector_updates_total",
            "User vector updates",
            ["type"],  # "created" or "updated"
        )

        # gauge for active connections/state
        self.active_user_vectors = Gauge(
            "active_user_vectors",
            "Number of user vectors in Redis (approximate)",
        )

        # recommendation quality metrics
        self.recommendation_diversity = Histogram(
            "recommendation_diversity_score",
            "Intra-list diversity of recommendations (avg pairwise cosine distance)",
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )

        # purchase endpoint metrics
        self.purchase_events_accepted = Counter(
            "purchase_events_accepted_total",
            "Purchase events successfully sent to Kafka",
        )

        self.purchase_lookup_errors = Counter(
            "purchase_lookup_errors_total",
            "Errors during purchase event ID lookups",
            [
                "error_type"
            ],  # "user_not_found", "item_not_found", "qdrant_error", "kafka_error"
        )

        # similar items endpoint metrics
        self.similar_items_requests = Counter(
            "similar_items_requests_total",
            "Total similar items requests",
            ["status"],  # "success", "not_found", "error"
        )

        self.qdrant_retrieve_duration = Histogram(
            "qdrant_retrieve_duration_seconds",
            "Qdrant point retrieve latency",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        )


# singleton instance
metrics = Metrics()


def setup_metrics(app):
    """
    setup prometheus metrics instrumentation for fastapi.
    it auto-instruments all http endpoints with request count/latency.
    """
    from prometheus_fastapi_instrumentator import Instrumentator

    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        excluded_handlers=["/health/live", "/health/ready", "/metrics"],
    )

    instrumentator.instrument(app).expose(app, include_in_schema=False)
