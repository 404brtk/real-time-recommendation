from fastapi import Request
import redis.asyncio as redis
from qdrant_client import AsyncQdrantClient


async def get_redis_client(request: Request) -> redis.Redis:
    return request.app.state.redis_client


async def get_qdrant_client(request: Request) -> AsyncQdrantClient:
    return request.app.state.qdrant_client
