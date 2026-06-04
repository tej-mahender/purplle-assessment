"""
GET /health
Returns service status, DB connectivity, and per-store feed freshness.
STALE_FEED warning if last event > 10 minutes ago.
This is what an on-call engineer checks first — must be accurate.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import select, func, distinct
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import EventRow, get_db, check_db_health
from app.models import HealthResponse, StoreFeedStatus

router = APIRouter()

STALE_FEED_MINUTES = 10


@router.get("/health", response_model=HealthResponse)
async def health(db: AsyncSession = Depends(get_db)):
    now = datetime.now(timezone.utc)
    db_ok = await check_db_health()

    store_statuses: list[StoreFeedStatus] = []

    if db_ok:
        # Get last event timestamp per store
        result = await db.execute(
            select(EventRow.store_id, func.max(EventRow.timestamp).label("last_ts"))
            .group_by(EventRow.store_id)
        )
        rows = result.all()

        for store_id, last_ts in rows:
            lag = None
            status = "NO_DATA"
            if last_ts:
                # DB stores naive UTC — make aware for comparison
                last_ts_aware = last_ts.replace(tzinfo=timezone.utc) if last_ts.tzinfo is None else last_ts
                lag = round((now - last_ts_aware).total_seconds() / 60, 2)
                status = "STALE_FEED" if lag > STALE_FEED_MINUTES else "OK"

            store_statuses.append(StoreFeedStatus(
                store_id=store_id,
                last_event_at=last_ts,
                lag_minutes=lag,
                status=status,
            ))

    overall = "ok" if db_ok else "degraded"

    if not db_ok:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "db_connected": False,
                "as_of": now.isoformat(),
                "stores": [],
                "error": "database_unavailable",
            },
        )

    return HealthResponse(
        status=overall,
        db_connected=db_ok,
        as_of=now,
        stores=store_statuses,
    )
