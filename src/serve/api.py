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
from src.observability import setup_metrics, setup_tracing, metrics, qdrant_span

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
    CategoryCount,
    UserProfile,
    ContributionItem,
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


def compute_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def mmr_rerank(
    scores: list[float],
    vectors: list[list[float]],
    lambda_param: float,
    k: int,
) -> list[int]:
    """
    Rerank items using Maximal Marginal Relevance (MMR).

    MMR balances relevance with diversity by penalizing items similar to
    already-selected items.

    Formula: MMR(i) = lambda * relevance(i) - (1-lambda) * max_sim(i, selected)

    Args:
        scores: Relevance scores for each item (from Qdrant search)
        vectors: Item vectors for similarity computation
        lambda_param: Trade-off parameter (1.0 = pure relevance, 0.0 = pure diversity)
        k: Number of items to select

    Returns:
        List of indices into the original lists, representing the reranked order
    """
    if len(scores) == 0:
        return []

    n = len(scores)
    k = min(k, n)

    # normalize scores to [0, 1] for fair comparison with similarities
    scores_np = np.array(scores, dtype=np.float32)
    score_min, score_max = scores_np.min(), scores_np.max()
    score_range = score_max - score_min
    if score_range > 0:
        normalized_scores = (scores_np - score_min) / score_range
    else:
        normalized_scores = np.ones(n, dtype=np.float32)

    # precompute normalized vectors and full similarity matrix
    vectors_np = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized_vectors = vectors_np / norms
    sim_matrix = normalized_vectors @ normalized_vectors.T

    # track max similarity to selected set for each candidate (updated incrementally)
    max_sims = np.zeros(n, dtype=np.float32)

    selected: list[int] = []
    remaining_mask = np.ones(n, dtype=bool)

    for _ in range(k):
        # compute MMR scores for all candidates (vectorized)
        mmr_scores = lambda_param * normalized_scores - (1 - lambda_param) * max_sims
        mmr_scores[~remaining_mask] = -np.inf  # exclude already selected

        # select best candidate
        best_idx = int(np.argmax(mmr_scores))
        selected.append(best_idx)
        remaining_mask[best_idx] = False

        # update max similarities incrementally (only against newly selected item)
        new_sims = sim_matrix[:, best_idx]
        max_sims = np.maximum(max_sims, new_sims)

    return selected


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
    explain: bool = Query(
        False, description="Include explanation for each recommendation"
    ),
    explain_top_k: int = Query(
        3, gt=0, le=10, description="Number of contributing items per recommendation"
    ),
    diversity_lambda: float = Query(
        0.8,
        ge=0.0,
        le=1.0,
        description="MMR diversity parameter (1.0 = pure relevance, 0.0 = pure diversity)",
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
    history_ids: list[int] = []  # track for explain feature

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

            # overfetch candidates for MMR reranking (if diversity enabled)
            fetch_limit = k if diversity_lambda >= 1.0 else min(k * 2, k + 20, 50)

            # track qdrant search timing
            qdrant_start = time.time()
            with qdrant_span(
                "query_points",
                QDRANT_COLLECTION_NAME,
                limit=fetch_limit,
                with_vectors=True,
            ) as span:
                search_result = await qdrant_conn.query_points(
                    collection_name=QDRANT_COLLECTION_NAME,
                    query=user_vector,
                    limit=fetch_limit,
                    with_payload=True,
                    with_vectors=True,  # need vectors for diversity calculation and MMR
                    query_filter=query_filter,
                )
                span.set_attribute("db.qdrant.results_count", len(search_result.points))
            metrics.qdrant_search_duration.observe(time.time() - qdrant_start)

            # extract points data
            points = search_result.points
            all_scores = [point.score for point in points]
            all_vectors = [list(point.vector) for point in points if point.vector]
            all_payloads = [point.payload or {} for point in points]
            all_ids = [int(point.id) for point in points]

            # apply MMR reranking if diversity is enabled and we have vectors
            if (
                diversity_lambda < 1.0
                and len(all_vectors) == len(points)
                and len(points) > 0
            ):
                mmr_start = time.time()
                reranked_indices = mmr_rerank(
                    scores=all_scores,
                    vectors=all_vectors,
                    lambda_param=diversity_lambda,
                    k=k,
                )
                metrics.mmr_rerank_duration.observe(time.time() - mmr_start)
                metrics.mmr_rerank_applied.inc()

                # reorder based on MMR results
                recommendations = [
                    RecommendationItem(
                        item_idx=all_ids[idx],
                        score=all_scores[idx],
                        metadata=all_payloads[idx],
                    )
                    for idx in reranked_indices
                ]
                item_vectors = [all_vectors[idx] for idx in reranked_indices]
            else:
                # no MMR, just take top k
                recommendations = [
                    RecommendationItem(
                        item_idx=all_ids[i],
                        score=all_scores[i],
                        metadata=all_payloads[i],
                    )
                    for i in range(min(k, len(points)))
                ]
                item_vectors = all_vectors[:k]

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

    # compute explanations if requested (only for personalized with history)
    # TODO: this explanation only reflects real-time purchase history (redis), not the
    # batch-trained ALS embeddings (Delta Lake). The user vector is shaped by both sources,
    # so these contributions are approximate - they show which recent purchases are similar
    # to recommendations, not the full picture of why the model made each recommendation
    # We could just load the full history from delta lake but that would make it pretty slow I guess
    # Anyway it's a nice-to-have feature even with this limitation
    if explain and source == "personalized" and history_ids and item_vectors:
        metrics.explain_requests.inc()
        explain_start = time.time()

        # fetch last 20 history item vectors (limit for performance)
        history_to_fetch = history_ids[:20]
        try:
            with qdrant_span(
                "retrieve",
                QDRANT_COLLECTION_NAME,
                ids_count=len(history_to_fetch),
                with_vectors=True,
                purpose="explain_history",
            ) as span:
                history_points = await qdrant_conn.retrieve(
                    collection_name=QDRANT_COLLECTION_NAME,
                    ids=history_to_fetch,
                    with_vectors=True,
                    with_payload=True,
                )
                span.set_attribute("db.qdrant.results_count", len(history_points))

            # build history lookup: {item_idx: (vector, name)}
            history_data = {
                int(p.id): (
                    list(p.vector),
                    p.payload.get("prod_name", "Unknown") if p.payload else "Unknown",
                )
                for p in history_points
                if p.vector
            }

            # for each recommendation, compute similarities to history items
            for i, rec in enumerate(recommendations):
                if i >= len(item_vectors):
                    break
                rec_vector = item_vectors[i]
                similarities = []

                for hist_idx, (hist_vec, hist_name) in history_data.items():
                    sim = compute_cosine_similarity(rec_vector, hist_vec)
                    similarities.append((hist_idx, hist_name, sim))

                # sort by similarity, take top k
                similarities.sort(key=lambda x: x[2], reverse=True)
                top_contributors = similarities[:explain_top_k]

                # normalize to percentages
                total_sim = sum(s[2] for s in top_contributors) or 1.0
                rec.explanation = [
                    ContributionItem(
                        item_idx=idx,
                        item_name=name,
                        similarity=round(sim, 4),
                        contribution_pct=round(sim / total_sim * 100, 1),
                    )
                    for idx, name, sim in top_contributors
                ]

            metrics.explain_computation_duration.observe(time.time() - explain_start)
        except Exception as e:
            logger.error(f"Failed to compute explanations: {e}")
            # don't fail the request, just skip explanations

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
        with qdrant_span(
            "retrieve",
            QDRANT_COLLECTION_NAME,
            ids_count=1,
            with_vectors=True,
            purpose="similar_items_source",
        ) as span:
            source_points = await qdrant_conn.retrieve(
                collection_name=QDRANT_COLLECTION_NAME,
                ids=[item_idx],
                with_vectors=True,
                with_payload=True,
            )
            span.set_attribute("db.qdrant.results_count", len(source_points))
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
        with qdrant_span(
            "query_points",
            QDRANT_COLLECTION_NAME,
            limit=k,
            purpose="similar_items_search",
        ) as span:
            search_result = await qdrant_conn.query_points(
                collection_name=QDRANT_COLLECTION_NAME,
                query=source_vector,
                limit=k,
                with_payload=True,
                query_filter=query_filter,
            )
            span.set_attribute("db.qdrant.results_count", len(search_result.points))
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


def aggregate_categories(
    items_metadata: list[dict], field: str, top_k: int
) -> list[CategoryCount]:
    from collections import Counter

    counts = Counter(
        item.get(field, "Unknown") for item in items_metadata if item.get(field)
    )
    total = sum(counts.values())
    if total == 0:
        return []

    return [
        CategoryCount(
            name=name,
            count=count,
            percentage=round(count / total * 100, 1),
        )
        for name, count in counts.most_common(top_k)
    ]


@app.get("/users/{user_idx}/profile", response_model=UserProfile)
async def get_user_profile(
    user_idx: int,
    recent_limit: int = Query(10, gt=0, le=50, description="Max recent purchases"),
    top_categories: int = Query(5, gt=0, le=10, description="Max categories per type"),
    redis_conn: redis.Redis = Depends(get_redis_client),
    qdrant_conn: AsyncQdrantClient = Depends(get_qdrant_client),
):
    redis_start = time.time()
    async with redis_conn.pipeline() as pipe:
        pipe.get(f"{REDIS_USER_VECTOR_PREFIX}{user_idx}")
        pipe.zrange(f"{REDIS_USER_HISTORY_PREFIX}{user_idx}", 0, -1, withscores=True)
        pipe.get(f"{REDIS_USER_MAP_PREFIX}{user_idx}")
        results = await pipe.execute()
    metrics.redis_operation_duration.labels(operation="profile_pipeline").observe(
        time.time() - redis_start
    )

    user_vector_json, history_with_scores, customer_id = results

    if user_vector_json is None:
        metrics.user_profile_requests.inc()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with user_idx={user_idx} not found",
        )

    metrics.user_profile_requests.inc()

    if not history_with_scores:
        return UserProfile(
            user_idx=user_idx,
            customer_id=customer_id,
            total_purchases=0,
            first_purchase_at=None,
            last_purchase_at=None,
            recent_purchases=[],
            top_product_groups=[],
            top_product_types=[],
        )

    # Parse history: list of (item_idx_str, timestamp)
    history_items = [(int(item_id), float(ts)) for item_id, ts in history_with_scores]
    total_purchases = len(history_items)

    timestamps = [ts for _, ts in history_items]
    first_purchase_at = min(timestamps)
    last_purchase_at = max(timestamps)

    # Sort by timestamp descending to get recent items first
    history_items.sort(key=lambda x: x[1], reverse=True)
    recent_item_ids = [item_id for item_id, _ in history_items[:recent_limit]]
    all_item_ids = [item_id for item_id, _ in history_items]

    # Batch retrieve metadata from Qdrant
    qdrant_start = time.time()
    try:
        with qdrant_span(
            "retrieve",
            QDRANT_COLLECTION_NAME,
            ids_count=len(all_item_ids),
            purpose="user_profile_history",
        ) as span:
            points = await qdrant_conn.retrieve(
                collection_name=QDRANT_COLLECTION_NAME,
                ids=all_item_ids,
                with_payload=True,
            )
            span.set_attribute("db.qdrant.results_count", len(points))
        metrics.qdrant_retrieve_duration.observe(time.time() - qdrant_start)
    except Exception as e:
        metrics.qdrant_retrieve_duration.observe(time.time() - qdrant_start)
        logger.error(f"Failed to retrieve items for user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve item metadata: {e}",
        )

    # Build lookup dict: item_idx -> payload
    item_metadata_map = {int(p.id): p.payload or {} for p in points}

    # Build recent purchases list (preserve order)
    recent_purchases = []
    for item_id in recent_item_ids:
        metadata = item_metadata_map.get(item_id, {})
        recent_purchases.append(
            RecommendationItem(
                item_idx=item_id,
                score=0.0,  # no score for history items
                metadata=metadata,
            )
        )

    # Aggregate categories from all history items
    all_metadata = [item_metadata_map.get(item_id, {}) for item_id in all_item_ids]
    top_product_groups = aggregate_categories(
        all_metadata, "product_group_name", top_categories
    )
    top_product_types = aggregate_categories(
        all_metadata, "product_type_name", top_categories
    )

    return UserProfile(
        user_idx=user_idx,
        customer_id=customer_id,
        total_purchases=total_purchases,
        first_purchase_at=first_purchase_at,
        last_purchase_at=last_purchase_at,
        recent_purchases=recent_purchases,
        top_product_groups=top_product_groups,
        top_product_types=top_product_types,
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
        with qdrant_span(
            "retrieve",
            QDRANT_COLLECTION_NAME,
            ids_count=1,
            purpose="purchase_item_lookup",
        ) as span:
            points = await qdrant_conn.retrieve(
                collection_name=QDRANT_COLLECTION_NAME,
                ids=[event.item_idx],
                with_payload=True,
            )
            span.set_attribute("db.qdrant.results_count", len(points))
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
