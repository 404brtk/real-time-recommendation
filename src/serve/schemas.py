from pydantic import BaseModel, Field
from typing import List, Dict, Any


class RecommendationItem(BaseModel):
    item_idx: int  # internal item index (maps to article_id via item_map)
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    user_idx: int  # internal user index (maps to customer_id via user_map)
    source: str  # "personalized" or "trending_now"
    recommendations: List[RecommendationItem]


class PurchaseEvent(BaseModel):
    user_idx: int  # internal user index
    item_idx: int  # internal item index


class SimilarItemsResponse(BaseModel):
    source_item_idx: int
    source_metadata: Dict[str, Any] = Field(default_factory=dict)
    similar_items: List[RecommendationItem]
