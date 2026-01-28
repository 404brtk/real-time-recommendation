from pydantic import BaseModel, Field
from typing import List, Dict, Any


class RecommendationItem(BaseModel):
    item_id: int | str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    user_id: int | str
    source: str
    recommendations: List[RecommendationItem]
