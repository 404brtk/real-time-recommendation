from contextlib import asynccontextmanager
from typing import Any, Optional
from fastapi import FastAPI, Depends, Response, Query, HTTPException, status
import redis.asyncio as redis
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
import json
import time
import numpy as np
from kafka import KafkaProducer

from src.logging import setup_logging
from src.observability import setup_metrics, setup_tracing, metrics

from src.serve.dependencies import get_redis_client, get_qdrant_client

from src.config import (
    REDIS_HOST,
    REDIS_PORT,
    REDIS_USER_VECTOR_PREFIX,
    REDIS_USER_HISTORY_PREFIX,
    REDIS_USER_MAP_PREFIX,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_NAME,
    REDIS_POPULAR_KEY,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_EVENTS,
)

from src.serve.schemas import (
    RecommendationItem,
    RecommendationResponse,
    PurchaseEvent,
    SimilarItemsResponse,
)

logger = setup_logging("api.log")


def compute_diversity(vectors: list[Any]) -> float:
    """
    compute intra-list diversity as average pairwise cosine distance.

    returns a value between 0 (all identical) and 2 (all opposite).
    higher values indicate more diverse recommendations.
    """
    if len(vectors) < 2:
        return 0.0

    vectors_np = np.array(vectors, dtype=np.float32)
    # normalize vectors for cosine similarity
    norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    normalized = vectors_np / norms

    # compute pairwise cosine similarities
    similarity_matrix = np.dot(normalized, normalized.T)

    # extract upper triangle (excluding diagonal) and convert to distances
    n = len(vectors)
    total_distance = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_distance += 1 - similarity_matrix[i, j]  # cosine distance
            count += 1

    return total_distance / count if count > 0 else 0.0


@asynccontextmanager  # thanks to this the app knows when to execute code before and after yield
async def lifespan(app: FastAPI):
    # during startup - before handling requests
    logger.info("Initializing connections...")
    app.state.redis_client = redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
    )
    app.state.qdrant_client = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # initialize kafka producer for purchase events
    try:
        app.state.kafka_producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda x: json.dumps(x).encode("utf-8"),
        )
        logger.info("Kafka producer initialized")
    except Exception as e:
        logger.warning(f"Kafka producer failed to initialize: {e}")
        app.state.kafka_producer = None

    yield
    # when the app finishes handling requests - right before shutdown
    logger.info("Closing connections...")
    await app.state.redis_client.aclose()
    await app.state.qdrant_client.close()
    if app.state.kafka_producer:
        app.state.kafka_producer.close()


app = FastAPI(title="RecommendationSystemAPI", lifespan=lifespan)

setup_metrics(app)
setup_tracing(app)


@app.get("/health/live")
async def health_check_live():
    return {"status": "ok"}


@app.get("/health/ready")
async def health_check(
    response: Response,
    redis_conn: redis.Redis = Depends(get_redis_client),
    qdrant_conn: AsyncQdrantClient = Depends(get_qdrant_client),
):
    health_status = {
        "status": "ok",
        "components": {
            "redis": {"status": "unknown", "latency_ms": 0},
            "qdrant": {"status": "unknown", "latency_ms": 0},
        },
    }
    has_error = False

    # redis
    start_time = time.time()
    try:
        await redis_conn.ping()
        latency = (time.time() - start_time) * 1000
        health_status["components"]["redis"] = {
            "status": "up",
            "latency_ms": round(latency, 2),
        }
    except Exception as e:
        has_error = True
        health_status["components"]["redis"] = {"status": "down", "error": str(e)}

    # qdrant
    start_time = time.time()
    try:
        await qdrant_conn.get_collections()
        latency = (time.time() - start_time) * 1000
        health_status["components"]["qdrant"] = {
            "status": "up",
            "latency_ms": round(latency, 2),
        }
    except Exception as e:
        has_error = True
        health_status["components"]["qdrant"] = {"status": "down", "error": str(e)}

    if has_error:
        health_status["status"] = "error"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        response.status_code = status.HTTP_200_OK

    return health_status


def build_qdrant_filter(
    excluded_ids: Optional[list[int]] = None,
    product_group: Optional[str] = None,
    product_type: Optional[str] = None,
    exclude_groups: Optional[list[str]] = None,
    exclude_types: Optional[list[str]] = None,
) -> Optional[models.Filter]:
    """
    Args:
        excluded_ids: Item IDs to exclude (e.g., already purchased)
        product_group: Include only items in this product group
        product_type: Include only items of this product type
        exclude_groups: Exclude items in these product groups
        exclude_types: Exclude items of these product types

    Returns:
        Qdrant Filter object or None if no filters specified
    """
    must_conditions: list[models.Condition] = []
    must_not_conditions: list[models.Condition] = []

    if excluded_ids:
        must_not_conditions.append(models.HasIdCondition(has_id=excluded_ids))

    if product_group:
        must_conditions.append(
            models.FieldCondition(
                key="product_group_name",
                match=models.MatchValue(value=product_group),
            )
        )

    if product_type:
        must_conditions.append(
            models.FieldCondition(
                key="product_type_name",
                match=models.MatchValue(value=product_type),
            )
        )

    if exclude_groups:
        must_not_conditions.append(
            models.FieldCondition(
                key="product_group_name",
                match=models.MatchAny(any=exclude_groups),
            )
        )

    if exclude_types:
        must_not_conditions.append(
            models.FieldCondition(
                key="product_type_name",
                match=models.MatchAny(any=exclude_types),
            )
        )

    if not must_conditions and not must_not_conditions:
        return None

    return models.Filter(
        must=must_conditions if must_conditions else None,
        must_not=must_not_conditions if must_not_conditions else None,
    )


@app.get("/recommend/{user_idx}", response_model=RecommendationResponse)
async def recommend(
    user_idx: int,
    k: int = Query(10, gt=0, le=50, description="Number of recommendations"),
    product_group: Optional[str] = Query(
        None, description="Filter by product group (e.g., 'Garment Upper body')"
    ),
    product_type: Optional[str] = Query(
        None, description="Filter by product type (e.g., 'T-shirt')"
    ),
    exclude_ids: Optional[str] = Query(
        None,
        description="Comma-separated item IDs to exclude (in addition to purchase history)",
    ),
    exclude_groups: Optional[str] = Query(
        None, description="Comma-separated product groups to exclude"
    ),
    exclude_types: Optional[str] = Query(
        None, description="Comma-separated product types to exclude"
    ),
    redis_conn: redis.Redis = Depends(get_redis_client),
    qdrant_conn: AsyncQdrantClient = Depends(get_qdrant_client),
):
    parsed_exclude_ids = (
        [int(i.strip()) for i in exclude_ids.split(",") if i.strip()]
        if exclude_ids
        else []
    )
    parsed_exclude_groups = (
        [g.strip() for g in exclude_groups.split(",") if g.strip()]
        if exclude_groups
        else None
    )
    parsed_exclude_types = (
        [t.strip() for t in exclude_types.split(",") if t.strip()]
        if exclude_types
        else None
    )

    recommendations = []
    item_vectors = []  # for diversity calculation
    source = "personalized"
    user_type = "known"

    try:
        redis_vector_key = f"{REDIS_USER_VECTOR_PREFIX}{user_idx}"
        redis_history_key = f"{REDIS_USER_HISTORY_PREFIX}{user_idx}"

        # track redis operation timing
        redis_start = time.time()
        async with redis_conn.pipeline() as pipe:
            pipe.get(redis_vector_key)
            pipe.zrange(redis_history_key, 0, -1)
            results = await pipe.execute()
        metrics.redis_operation_duration.labels(operation="pipeline_get").observe(
            time.time() - redis_start
        )

        user_vector_json, purchased_items_ids = results

        if user_vector_json:
            user_vector = json.loads(user_vector_json)
            # combine purchase history with explicitly excluded IDs
            history_ids = (
                [int(i) for i in purchased_items_ids] if purchased_items_ids else []
            )
            all_excluded_ids = list(set(history_ids + parsed_exclude_ids))

            # track excluded items separately
            if history_ids:
                metrics.items_excluded_history.inc(len(history_ids))
            if parsed_exclude_ids:
                metrics.items_excluded_explicit.inc(len(parsed_exclude_ids))

            # track filter usage
            if product_group:
                metrics.filter_applied.labels(filter_type="product_group").inc()
            if product_type:
                metrics.filter_applied.labels(filter_type="product_type").inc()
            if parsed_exclude_ids:
                metrics.filter_applied.labels(filter_type="exclude_ids").inc()
            if parsed_exclude_groups:
                metrics.filter_applied.labels(filter_type="exclude_groups").inc()
            if parsed_exclude_types:
                metrics.filter_applied.labels(filter_type="exclude_types").inc()

            # build combined filter (excluded IDs + attribute filters)
            query_filter = build_qdrant_filter(
                excluded_ids=all_excluded_ids if all_excluded_ids else None,
                product_group=product_group.strip() if product_group else None,
                product_type=product_type.strip() if product_type else None,
                exclude_groups=parsed_exclude_groups,
                exclude_types=parsed_exclude_types,
            )

            # track qdrant search timing
            qdrant_start = time.time()
            search_result = await qdrant_conn.query_points(
                collection_name=QDRANT_COLLECTION_NAME,
                query=user_vector,
                limit=k,
                with_payload=True,
                with_vectors=True,  # need vectors for diversity calculation
                query_filter=query_filter,
            )
            metrics.qdrant_search_duration.observe(time.time() - qdrant_start)

            recommendations = [
                RecommendationItem(
                    item_idx=int(point.id),
                    score=point.score,
                    metadata=point.payload or {},
                )
                for point in search_result.points
            ]

            # collect vectors for diversity calculation
            item_vectors = [
                list(point.vector) for point in search_result.points if point.vector
            ]

        else:
            user_type = "cold_start"

    except Exception as e:
        logger.error(f"Personalized search failed for user {user_idx}. Reason: {e}")
        metrics.recommendation_fallback.labels(reason="personalized_error").inc()
        recommendations = []

    if not recommendations:
        source = "trending_now"
        if user_type == "cold_start":
            metrics.recommendation_fallback.labels(reason="cold_start").inc()
        elif not recommendations:
            metrics.recommendation_fallback.labels(reason="no_results").inc()

        logger.info(f"Using fallback strategy for user {user_idx}")

        try:
            redis_start = time.time()
            popular_items_json = await redis_conn.get(REDIS_POPULAR_KEY)
            metrics.redis_operation_duration.labels(operation="get_popular").observe(
                time.time() - redis_start
            )

            if popular_items_json:
                popular_items = json.loads(popular_items_json)
                top_k_items = popular_items[:k]

                recommendations = [
                    RecommendationItem(
                        item_idx=item["item_idx"],
                        score=item["score"],
                        metadata=item.get("metadata", {}),
                    )
                    for item in top_k_items
                ]
                # no vectors available for trending items, diversity will be skipped
            else:
                logger.warning(f"Fallback key '{REDIS_POPULAR_KEY}' is empty in Redis!")

        except Exception as e:
            logger.error(f"Fallback strategy failed: {e}")
            recommendations = []

    if not recommendations:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable: No recommendations available (both personalized and fallback failed).",
        )

    metrics.recommendation_requests.labels(source=source, user_type=user_type).inc()

    # compute and record diversity score (only for personalized with vectors)
    if item_vectors and len(item_vectors) >= 2:
        diversity = compute_diversity(item_vectors)
        metrics.recommendation_diversity.observe(diversity)

    return RecommendationResponse(
        user_idx=user_idx,
        source=source,
        recommendations=recommendations,
    )


@app.get("/items/{item_idx}/similar", response_model=SimilarItemsResponse)
async def get_similar_items(
    item_idx: int,
    k: int = Query(10, gt=0, le=50, description="Number of similar items to return"),
    product_group: Optional[str] = Query(
        None, description="Filter by product group (e.g., 'Garment Upper body')"
    ),
    product_type: Optional[str] = Query(
        None, description="Filter by product type (e.g., 'T-shirt')"
    ),
    exclude_ids: Optional[str] = Query(
        None, description="Comma-separated item IDs to exclude"
    ),
    exclude_groups: Optional[str] = Query(
        None, description="Comma-separated product groups to exclude"
    ),
    exclude_types: Optional[str] = Query(
        None, description="Comma-separated product types to exclude"
    ),
    qdrant_conn: AsyncQdrantClient = Depends(get_qdrant_client),
):
    # Parse comma-separated filter values
    parsed_exclude_ids = (
        [int(i.strip()) for i in exclude_ids.split(",") if i.strip()]
        if exclude_ids
        else []
    )
    parsed_exclude_groups = (
        [g.strip() for g in exclude_groups.split(",") if g.strip()]
        if exclude_groups
        else None
    )
    parsed_exclude_types = (
        [t.strip() for t in exclude_types.split(",") if t.strip()]
        if exclude_types
        else None
    )

    qdrant_retrieve_start = time.time()
    try:
        source_points = await qdrant_conn.retrieve(
            collection_name=QDRANT_COLLECTION_NAME,
            ids=[item_idx],
            with_vectors=True,
            with_payload=True,
        )
        metrics.qdrant_retrieve_duration.observe(time.time() - qdrant_retrieve_start)
    except Exception as e:
        metrics.qdrant_retrieve_duration.observe(time.time() - qdrant_retrieve_start)
        metrics.similar_items_requests.labels(status="error").inc()
        logger.error(f"Failed to retrieve item {item_idx} from Qdrant: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve item: {e}",
        )

    if not source_points:
        metrics.similar_items_requests.labels(status="not_found").inc()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item with item_idx={item_idx} not found",
        )

    source_point = source_points[0]
    source_vector: list[float] = [float(x) for x in source_point.vector]  # type: ignore[union-attr]
    source_metadata = source_point.payload or {}

    # Combine source item with explicitly excluded IDs
    all_excluded_ids = list(set([item_idx] + parsed_exclude_ids))

    # track explicit exclusions (not counting the source item which is always excluded)
    if parsed_exclude_ids:
        metrics.items_excluded_explicit.inc(len(parsed_exclude_ids))

    # track filter usage
    if product_group:
        metrics.filter_applied.labels(filter_type="product_group").inc()
    if product_type:
        metrics.filter_applied.labels(filter_type="product_type").inc()
    if parsed_exclude_ids:
        metrics.filter_applied.labels(filter_type="exclude_ids").inc()
    if parsed_exclude_groups:
        metrics.filter_applied.labels(filter_type="exclude_groups").inc()
    if parsed_exclude_types:
        metrics.filter_applied.labels(filter_type="exclude_types").inc()

    # Build filter: exclude source item + user-specified exclusions + attribute filters
    query_filter = build_qdrant_filter(
        excluded_ids=all_excluded_ids,
        product_group=product_group.strip() if product_group else None,
        product_type=product_type.strip() if product_type else None,
        exclude_groups=parsed_exclude_groups,
        exclude_types=parsed_exclude_types,
    )

    # search for similar items
    qdrant_start = time.time()
    try:
        search_result = await qdrant_conn.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=source_vector,
            limit=k,
            with_payload=True,
            query_filter=query_filter,
        )
    except Exception as e:
        metrics.qdrant_search_duration.observe(time.time() - qdrant_start)
        metrics.similar_items_requests.labels(status="error").inc()
        logger.error(f"Qdrant search failed for similar items: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {e}",
        )

    metrics.qdrant_search_duration.observe(time.time() - qdrant_start)

    similar_items = [
        RecommendationItem(
            item_idx=int(point.id),
            score=point.score,
            metadata=point.payload or {},
        )
        for point in search_result.points
    ]

    metrics.similar_items_requests.labels(status="success").inc()
    logger.info(f"Found {len(similar_items)} items similar to item {item_idx}")

    return SimilarItemsResponse(
        source_item_idx=item_idx,
        source_metadata=source_metadata,
        similar_items=similar_items,
    )


@app.post("/events/purchase")
async def record_purchase(
    event: PurchaseEvent,
    redis_conn: redis.Redis = Depends(get_redis_client),
    qdrant_conn: AsyncQdrantClient = Depends(get_qdrant_client),
):
    """
    record a purchase event. publishes to kafka for the stream updater to process.

    The updater will:
    1. Update the user's vector (moving it towards the purchased item)
    2. Add the item to the user's purchase history (excluded from future recommendations)

    The archiver will:
    1. Archive the full event to Delta Lake for batch retraining

    Note: Requires the stream updater to be running to see effect on recommendations.
    """
    if not app.state.kafka_producer:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Kafka producer not available. Cannot record purchase.",
        )

    # look up external IDs for archiver compatibility
    # get customer_id from Redis user mapping
    user_map_key = f"{REDIS_USER_MAP_PREFIX}{event.user_idx}"
    customer_id = await redis_conn.get(user_map_key)
    if not customer_id:
        metrics.purchase_lookup_errors.labels(error_type="user_not_found").inc()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User mapping not found for user_idx={event.user_idx}",
        )

    # get article_id from Qdrant payload
    try:
        points = await qdrant_conn.retrieve(
            collection_name=QDRANT_COLLECTION_NAME,
            ids=[event.item_idx],
            with_payload=True,
        )
        if not points:
            metrics.purchase_lookup_errors.labels(error_type="item_not_found").inc()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Item not found for item_idx={event.item_idx}",
            )
        payload = points[0].payload
        if not payload:
            metrics.purchase_lookup_errors.labels(error_type="item_not_found").inc()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Payload not found for item_idx={event.item_idx}",
            )
        article_id = payload.get("article_id")
        if not article_id:
            metrics.purchase_lookup_errors.labels(error_type="item_not_found").inc()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"article_id not found in payload for item_idx={event.item_idx}",
            )
    except HTTPException:
        raise
    except Exception as e:
        metrics.purchase_lookup_errors.labels(error_type="qdrant_error").inc()
        logger.error(f"Failed to retrieve item from Qdrant: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to look up item: {e}",
        )

    kafka_event = {
        "user_id": customer_id,  # external ID for archiver
        "user_idx": event.user_idx,  # internal ID
        "item_id": article_id,  # external ID for archiver
        "item_idx": event.item_idx,  # internal ID
        "event_type": "purchase",
        "timestamp": time.time(),
        "quantity": 1,
    }

    try:
        app.state.kafka_producer.send(KAFKA_TOPIC_EVENTS, value=kafka_event)
        metrics.purchase_events_accepted.inc()
        logger.info(
            f"Purchase event sent: user_idx={event.user_idx}, item_idx={event.item_idx}, "
            f"customer_id={customer_id}, article_id={article_id}"
        )
    except Exception as e:
        metrics.purchase_lookup_errors.labels(error_type="kafka_error").inc()
        logger.error(f"Failed to send purchase event: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record purchase: {e}",
        )

    return {
        "status": "accepted",
        "message": "Purchase event sent to processing queue",
        "event": kafka_event,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
