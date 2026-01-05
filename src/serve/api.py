from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Response, Query, HTTPException, status
import redis.asyncio as redis
from qdrant_client import AsyncQdrantClient
import json
import time

from src.logging import setup_logging

from src.serve.dependencies import get_redis_client, get_qdrant_client

from src.config import (
    REDIS_HOST,
    REDIS_PORT,
    REDIS_USER_VECTOR_PREFIX,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_NAME,
)

from src.serve.schemas import RecommendationItem, RecommendationResponse

logger = setup_logging("api.log")


@asynccontextmanager  # thanks to this the app knows when to execute code before and after yield
async def lifespan(app: FastAPI):
    # during startup - before handling requests
    logger.info("Initializing DB connections...")
    app.state.redis_client = redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
    )
    app.state.qdrant_client = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    yield
    # when the app finishes handling requests - right before shutdown
    logger.info("Closing DB connections...")
    await app.state.redis_client.aclose()
    await app.state.qdrant_client.close()


app = FastAPI(title="RecommendationSystemAPI", lifespan=lifespan)


@app.get("/health/live")
async def health_check_live():
    return {"status": "ok"}


@app.get("/health/ready")
async def health_check(
    response: Response,
    # dependency injection
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


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
async def recommend(
    user_id: str,
    k: int = Query(1, gt=0, le=50, description="Number of recommendations"),
    # dependency injection
    redis_conn: redis.Redis = Depends(get_redis_client),
    qdrant_conn: AsyncQdrantClient = Depends(get_qdrant_client),
):
    redis_key = f"{REDIS_USER_VECTOR_PREFIX}{user_id}"
    user_vector_json = await redis_conn.get(redis_key)

    if not (user_vector_json):
        # cold start
        raise HTTPException(  # TODO: return popular items instead
            status_code=404, detail=f"User vector not found for ID: {user_id}"
        )

    try:
        user_vector = json.loads(user_vector_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Corrupted vector data in Redis")

    try:
        search_result = await qdrant_conn.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=user_vector,
            limit=k,
            with_payload=True,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Vector search failed: {str(e)}")

    recommendations = [
        RecommendationItem(
            item_id=point.id, score=point.score, metadata=point.payload or {}
        )
        for point in search_result.points
    ]

    return RecommendationResponse(
        # TODO: handle source properly
        user_id=user_id,
        source="personalized",
        recommendations=recommendations,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
