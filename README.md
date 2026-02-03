# Real-Time Recommendation System

A real-time product recommendation engine using collaborative filtering (ALS) with online learning capabilities. The system combines batch-trained embeddings with real-time user profile updates to deliver personalized recommendations.

## Data

This project uses the [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data) dataset from Kaggle.

## Architecture

![Architecture](docs/images/architecture.png)

The system consists of three main components:

**Batch Pipeline** - Apache Spark jobs that:
- Preprocess raw transaction data into Delta Lake format
- Train ALS collaborative filtering model to generate user/item embeddings
- Load embeddings to Redis (users) and Qdrant (items)
- Load user ID mappings to Redis for reverse lookups
- Calculate trending items as fallback recommendations

**Stream Pipeline** - Real-time processing that:
- Consumes user events (clicks, add-to-cart, purchases) from Kafka
- Updates user vectors in Redis using online learning (exponential moving average)
- Archives events to Delta Lake for batch retraining

**Serving Layer** - FastAPI endpoint that:
- Queries user vectors from Redis
- Performs ANN search in Qdrant to find similar items
- Falls back to trending items for cold-start users
- Accepts purchase events and publishes to Kafka

**Observability Stack** - Full monitoring and tracing:
- Prometheus for metrics collection
- Grafana for dashboards and visualization
- Jaeger for distributed tracing (OpenTelemetry)

## Tech Stack

| Component | Technology |
|-----------|------------|
| Batch Processing | Apache Spark, Delta Lake |
| ML Model | ALS (Alternating Least Squares) |
| Message Queue | Apache Kafka |
| Stream Processing | Spark Structured Streaming, Kafka Consumer |
| Vector Database | Qdrant |
| Feature Store | Redis |
| API | FastAPI |
| Metrics | Prometheus, Grafana |
| Tracing | OpenTelemetry, Jaeger |

## Project Structure

```
src/
├── batch/                 # Batch processing jobs
│   ├── csv_to_delta.py    # Convert raw CSV to Delta Lake
│   ├── preprocess.py      # ETL: raw data -> training data
│   ├── train_als.py       # Train ALS model
│   ├── loader.py          # Load embeddings to Redis/Qdrant
│   └── calc_popular.py    # Calculate trending items
├── stream/                # Real-time processing
│   ├── producer.py        # Event simulator
│   ├── updater.py         # Online learning consumer
│   └── archiver.py        # Archive events to Delta Lake
├── serve/                 # API layer
│   ├── api.py             # FastAPI recommendation endpoint
│   ├── dependencies.py    # Dependency injection
│   └── schemas.py         # Pydantic models
├── observability/         # Monitoring and tracing
│   ├── metrics.py         # Prometheus metrics definitions
│   └── tracing.py         # OpenTelemetry tracing setup
├── config.py              # Configuration
└── logging.py             # Logging setup

docker/
├── prometheus/            # Prometheus configuration
│   └── prometheus.yml
└── grafana/               # Grafana provisioning
    └── provisioning/
        ├── datasources/
        └── dashboards/
```

## Getting Started

### Prerequisites

- Python 3.13+
- Docker and Docker Compose
- Java 17+ (for Spark)

### Installation

```bash
# Clone and install dependencies
git clone <repository>
cd real-time-recommendation
uv sync --dev

# Start infrastructure
docker-compose up -d
```

### Running the Pipeline

1. **Prepare data** - Place raw CSV files in `data/raw/`

2. **Run batch pipeline**:
   ```bash
   uv run scripts/run_batch.py
   ```
   This runs all batch steps in order: CSV to Delta, preprocess, train ALS, load to stores, calculate popular items.

3. **Start stream processing**:
   ```bash
   # Terminal 1: Online learning
   uv run -m src.stream.updater

   # Terminal 2: Archive events to Delta Lake
   uv run -m src.stream.archiver
   ```

4. **Start API**:
   ```bash
   uv run -m src.serve.api
   ```

5. **Get recommendations**:
   ```bash
   curl http://localhost:8000/recommend/123?k=10
   ```

6. **Record a purchase** (triggers online learning):
   ```bash
   curl -X POST http://localhost:8000/events/purchase \
     -H "Content-Type: application/json" \
     -d '{"user_idx": 123, "item_idx": 456}'
   ```

### Optional: Event Simulator

For testing, you can run the event simulator to generate random user events:
```bash
uv run -m src.stream.producer
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/recommend/{user_idx}` | GET | Get personalized recommendations |
| `/items/{item_idx}/similar` | GET | Find items similar to a given item |
| `/users/{user_idx}/profile` | GET | Get user profile and purchase history |
| `/events/purchase` | POST | Record a purchase event |
| `/health/live` | GET | Liveness probe |
| `/health/ready` | GET | Readiness probe (checks Redis/Qdrant) |
| `/metrics` | GET | Prometheus metrics |

### Recommendation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | 10 | Number of recommendations (1-50) |
| `product_group` | str | - | Filter by product group |
| `product_type` | str | - | Filter by product type |
| `exclude_ids` | str | - | Comma-separated item IDs to exclude |
| `exclude_groups` | str | - | Comma-separated product groups to exclude |
| `exclude_types` | str | - | Comma-separated product types to exclude |
| `explain` | bool | false | Include explanation for each recommendation |
| `explain_top_k` | int | 3 | Number of contributing items per explanation (1-10) |
| `diversity_lambda` | float | 0.8 | MMR diversity (1.0=relevance, 0.0=diversity) |

### Similar Items Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | 10 | Number of similar items (1-50) |
| `product_group` | str | - | Filter by product group |
| `product_type` | str | - | Filter by product type |
| `exclude_ids` | str | - | Comma-separated item IDs to exclude |
| `exclude_groups` | str | - | Comma-separated product groups to exclude |
| `exclude_types` | str | - | Comma-separated product types to exclude |

### User Profile Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `recent_limit` | int | 10 | Max recent purchases to return (1-50) |
| `top_categories` | int | 5 | Max categories per type (1-10) |

## Features

### Recommendation Explanations

Request explanations for why each item was recommended by adding `?explain=true`. The system computes similarity between recommended items and the user's recent purchase history, showing which past purchases contributed most to each recommendation.

```bash
curl "http://localhost:8000/recommend/123?k=5&explain=true&explain_top_k=3"
```

Response includes an `explanation` field for each item:
```json
{
  "item_idx": 456,
  "score": 0.92,
  "metadata": {...},
  "explanation": [
    {"item_idx": 789, "item_name": "Slim Fit Jeans", "similarity": 0.85, "contribution_pct": 45.2},
    {"item_idx": 101, "item_name": "Cotton T-Shirt", "similarity": 0.62, "contribution_pct": 32.9},
    {"item_idx": 202, "item_name": "Denim Jacket", "similarity": 0.41, "contribution_pct": 21.9}
  ]
}
```

> **Note:** Explanations reflect real-time purchase history only, not batch-trained ALS embeddings.

### MMR Diversity Reranking

Control the diversity of recommendations using Maximal Marginal Relevance (MMR) via the `diversity_lambda` parameter:

- `λ = 1.0` - Pure relevance (no diversity, fastest)
- `λ = 0.8` - Default, slight diversity boost
- `λ = 0.5` - Balanced relevance and diversity
- `λ = 0.0` - Pure diversity (may sacrifice relevance)

```bash
# More diverse recommendations
curl "http://localhost:8000/recommend/123?k=10&diversity_lambda=0.5"
```

MMR works by penalizing items similar to already-selected items, promoting variety in product types, styles, and categories.

### User Profile

View a user's purchase history and category preferences:

```bash
curl "http://localhost:8000/users/123/profile?recent_limit=10&top_categories=5"
```

Returns:
- `total_purchases` - Total purchase count
- `first_purchase_at` / `last_purchase_at` - Purchase timestamps
- `recent_purchases` - List of recent items with metadata
- `top_product_groups` - Most purchased product groups with percentages
- `top_product_types` - Most purchased product types with percentages

### Similar Items

Find items similar to a given item (useful for "customers also viewed" features):

```bash
curl "http://localhost:8000/items/456/similar?k=10&product_group=Garment%20Upper%20body"
```

### Filtering

All recommendation endpoints support filtering:

```bash
# Include only specific category
curl "http://localhost:8000/recommend/123?product_group=Accessories"

# Exclude categories
curl "http://localhost:8000/recommend/123?exclude_groups=Swimwear,Nightwear"

# Exclude specific items
curl "http://localhost:8000/recommend/123?exclude_ids=100,200,300"
```

## Observability

### Dashboards

- **Grafana**: http://localhost:3000 (admin/admin)
  - Pre-configured dashboard: "Recommendation System"
  - API metrics: request rate, latency, fallback rate
  - Stream metrics: events processed, processing latency, debounce hits

### Tracing

- **Jaeger UI**: http://localhost:16686
  - Distributed tracing for API requests
  - View request flow through Redis and Qdrant

### Metrics

Key metrics exposed:

| Metric | Type | Description |
|--------|------|-------------|
| `recommendation_requests_total` | Counter | Total recommendations by source/user_type |
| `recommendation_fallback_total` | Counter | Fallback to trending by reason |
| `recommendation_diversity_score` | Histogram | Intra-list diversity of recommendations |
| `similar_items_requests_total` | Counter | Similar items requests by status |
| `user_profile_requests_total` | Counter | User profile endpoint requests |
| `filter_applied_total` | Counter | Filters applied by type |
| `items_excluded_history_total` | Counter | Items excluded due to purchase history |
| `items_excluded_explicit_total` | Counter | Items explicitly excluded via parameter |
| `explain_requests_total` | Counter | Requests with explain=true |
| `explain_computation_duration_seconds` | Histogram | Time to compute explanations |
| `mmr_rerank_applied_total` | Counter | Times MMR reranking was applied |
| `mmr_rerank_duration_seconds` | Histogram | MMR reranking computation time |
| `purchase_events_accepted_total` | Counter | Purchase events sent to Kafka |
| `purchase_lookup_errors_total` | Counter | Purchase lookup errors by type |
| `events_processed_total` | Counter | Events processed by updater |
| `event_processing_duration_seconds` | Histogram | Event processing latency |
| `redis_operation_duration_seconds` | Histogram | Redis operation latency |
| `qdrant_search_duration_seconds` | Histogram | Qdrant search latency |
| `qdrant_retrieve_duration_seconds` | Histogram | Qdrant point retrieve latency |
