"""
FastAPI entrypoint — registers all routes, middleware, startup/shutdown hooks.
"""

from __future__ import annotations

import csv
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.database import init_db, AsyncSessionLocal
from app.models import (
    IngestRequest, IngestResponse,
    StoreMetrics, StoreFunnel, StoreHeatmap, StoreAnomalies, HealthResponse,
)

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("startup: initialising database")
    await init_db()
    await seed_pos_transactions()
    log.info("startup: complete")
    yield
    log.info("shutdown: complete")


async def seed_pos_transactions():
    """Load pos_transactions.csv into DB on startup (idempotent)."""
    from sqlalchemy import select
    from app.database import POSTransaction

    pos_path = os.getenv("POS_TRANSACTIONS_PATH", "./data/pos_transactions.csv")
    if not os.path.exists(pos_path):
        log.warning("pos_transactions.csv not found — skipping seed", path=pos_path)
        return

    async with AsyncSessionLocal() as db:
        # Check if already seeded
        result = await db.execute(select(POSTransaction).limit(1))
        if result.scalars().first():
            log.info("pos_transactions already seeded — skipping")
            return

        rows = []
        with open(pos_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(POSTransaction(
                    store_id=row["store_id"].strip(),
                    transaction_id=row["transaction_id"].strip(),
                    timestamp=datetime.fromisoformat(row["timestamp"].strip().replace("Z", "+00:00")),
                    basket_value_inr=float(row["basket_value_inr"].strip()),
                ))

        db.add_all(rows)
        await db.commit()
        log.info("pos_transactions seeded", count=len(rows))


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Store Intelligence API",
    version="1.0.0",
    description="Retail analytics from CCTV event streams — Apex Retail / Purplle challenge",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Structured logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    trace_id = str(uuid.uuid4())
    start = time.perf_counter()
    request.state.trace_id = trace_id

    response: Response = await call_next(request)

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    store_id = request.path_params.get("store_id", None)

    log.info(
        "request",
        trace_id=trace_id,
        method=request.method,
        path=request.url.path,
        store_id=store_id,
        status_code=response.status_code,
        latency_ms=latency_ms,
    )
    response.headers["X-Trace-Id"] = trace_id
    return response


# ---------------------------------------------------------------------------
# Global exception handler — no raw tracebacks in responses
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# Routes — import and register
# ---------------------------------------------------------------------------

from app.ingestion import router as ingest_router
from app.metrics import router as metrics_router
from app.funnel import router as funnel_router
from app.anomalies import router as anomalies_router
from app.heatmap import router as heatmap_router
from app.health import router as health_router

app.include_router(ingest_router)
app.include_router(metrics_router)
app.include_router(funnel_router)
app.include_router(anomalies_router)
app.include_router(heatmap_router)
app.include_router(health_router)
