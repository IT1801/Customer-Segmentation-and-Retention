# ─────────────────────────────────────────────────────────────────────────────
# Customer Segmentation & Retention — Makefile
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   make help         → list all targets
#   make setup        → full first-time setup
#   make all          → run entire pipeline end to end
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help setup install db-up db-down db-reset \
        etl features \
        train-segmentation train-churn train-clv train-basket train \
        serve dashboard mlflow \
        test lint format clean all

# ── Default target ────────────────────────────────────────────────────────────
.DEFAULT_GOAL := help

help:
	@echo ""
	@echo "  Customer Segmentation & Retention — available commands"
	@echo ""
	@echo "  SETUP"
	@echo "    make setup              Full first-time setup (install + db-up)"
	@echo "    make install            Install Python dependencies"
	@echo "    make install-dev        Install with dev extras (pytest, ruff, black)"
	@echo ""
	@echo "  DOCKER / DATABASE"
	@echo "    make db-up              Start PostgreSQL container"
	@echo "    make db-down            Stop PostgreSQL container"
	@echo "    make db-reset           Drop and recreate database"
	@echo "    make db-shell           Open psql shell in container"
	@echo "    make db-logs            Tail Postgres container logs"
	@echo ""
	@echo "  PIPELINE"
	@echo "    make etl                Extract → Transform → Load to Postgres"
	@echo "    make features           Build all customer features"
	@echo ""
	@echo "  MODELS"
	@echo "    make train-segmentation KMeans customer segmentation"
	@echo "    make train-churn        XGBoost churn prediction"
	@echo "    make train-clv          BG/NBD + Gamma-Gamma CLV"
	@echo "    make train-basket       FP-Growth market basket rules"
	@echo "    make train              Run all 4 models in order"
	@echo ""
	@echo "  SERVE"
	@echo "    make serve              Start FastAPI server (port 8000)"
	@echo "    make dashboard          Start Streamlit dashboard (port 8501)"
	@echo "    make mlflow             Start MLflow UI (port 5000)"
	@echo ""
	@echo "  QUALITY"
	@echo "    make test               Run pytest with coverage"
	@echo "    make lint               Run ruff linter"
	@echo "    make format             Run black formatter"
	@echo ""
	@echo "  OTHER"
	@echo "    make all                Full pipeline: etl + features + train"
	@echo "    make clean              Remove logs, artifacts, __pycache__"
	@echo ""


# ── Setup ─────────────────────────────────────────────────────────────────────
setup: install db-up
	@echo "✓ Setup complete — copy .env.example to .env and fill in credentials"

install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"


# ── Docker / Database ─────────────────────────────────────────────────────────
db-up:
	docker-compose up -d db
	@echo "Waiting for Postgres to be ready..."
	@sleep 3
	@docker-compose exec db pg_isready -U $${DB_USER:-postgres} && echo "✓ Postgres ready"

db-down:
	docker-compose down

db-reset:
	docker-compose down -v
	docker-compose up -d db
	@sleep 3
	@echo "✓ Database reset complete"

db-shell:
	docker-compose exec db psql -U $${DB_USER:-postgres} -d $${DB_NAME:-customer_segmentation}

db-logs:
	docker-compose logs -f db


# ── Pipeline ──────────────────────────────────────────────────────────────────
etl:
	@echo "── Running ETL pipeline ──────────────────────────"
	python -m src.csr.etl.pipeline
	@echo "✓ ETL complete"

features:
	@echo "── Building features ─────────────────────────────"
	python -m src.csr.features.build_features
	@echo "✓ Features complete"


# ── Models ────────────────────────────────────────────────────────────────────
train-segmentation:
	@echo "── Training segmentation model ───────────────────"
	python -m src.csr.models.segmentation
	@echo "✓ Segmentation complete"

train-churn:
	@echo "── Training churn model ──────────────────────────"
	python -m src.csr.models.churn
	@echo "✓ Churn model complete"

train-clv:
	@echo "── Training CLV model ────────────────────────────"
	python -m src.csr.models.clv
	@echo "✓ CLV model complete"

train-basket:
	@echo "── Running market basket analysis ────────────────"
	python -m src.csr.models.market_basket
	@echo "✓ Market basket complete"

train: train-segmentation train-churn train-clv train-basket
	@echo "✓ All models trained"


# ── Serve ─────────────────────────────────────────────────────────────────────
serve:
	uvicorn src.csr.api.main:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload

dashboard:
	streamlit run dashboard/app.py \
		--server.port 8501 \
		--server.address 0.0.0.0

mlflow:
	mlflow ui \
		--host 0.0.0.0 \
		--port 5000 \
		--backend-store-uri mlruns


# ── Quality ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ dashboard/ tests/

format:
	black src/ dashboard/ tests/


# ── Full pipeline ─────────────────────────────────────────────────────────────
all: etl features train
	@echo ""
	@echo "══════════════════════════════════════════"
	@echo "  Full pipeline complete ✓"
	@echo "  Run: make serve      → API  (port 8000)"
	@echo "  Run: make dashboard  → UI   (port 8501)"
	@echo "  Run: make mlflow     → MLflow (port 5000)"
	@echo "══════════════════════════════════════════"


# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf logs/*.log 2>/dev/null || true
	@echo "✓ Clean complete"