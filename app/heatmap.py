"""
GET /stores/{store_id}/heatmap
Zone visit frequency + avg dwell, normalised 0-100.
Flags low-confidence zones (< 20 sessions).
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import EventRow, get_db
from app.models import StoreHeatmap, ZoneHeatmap, EventType

router = APIRouter()

MIN_SESSIONS_FOR_CONFIDENCE = 20


@router.get("/stores/{store_id}/heatmap", response_model=StoreHeatmap)
async def get_heatmap(store_id: str, db: AsyncSession = Depends(get_db)):
    now = datetime.now(timezone.utc)

    # Use most recent event date as session day (handles past-date clips)
    latest_result = await db.execute(
        select(func.max(EventRow.timestamp))
        .where(EventRow.store_id == store_id)
    )
    latest_ts = latest_result.scalar()
    if latest_ts:
        today_start = latest_ts.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    result = await db.execute(
        select(
            EventRow.zone_id,
            func.count(EventRow.id).label("visit_count"),
            func.avg(EventRow.dwell_ms).label("avg_dwell"),
        )
        .where(
            EventRow.store_id == store_id,
            EventRow.event_type.in_([EventType.ZONE_ENTER, EventType.ZONE_DWELL]),
            EventRow.is_staff == False,
            EventRow.zone_id.isnot(None),
            EventRow.timestamp >= today_start,
        )
        .group_by(EventRow.zone_id)
    )
    rows = result.all()

    if not rows:
        return StoreHeatmap(store_id=store_id, as_of=now, zones=[])

    max_visits = max(r.visit_count for r in rows)

    zones = [
        ZoneHeatmap(
            zone_id=row.zone_id,
            visit_frequency=row.visit_count,
            avg_dwell_ms=round(row.avg_dwell or 0, 2),
            normalised_score=round((row.visit_count / max_visits) * 100, 2) if max_visits > 0 else 0.0,
            data_confidence=row.visit_count >= MIN_SESSIONS_FOR_CONFIDENCE,
        )
        for row in rows
    ]

    return StoreHeatmap(store_id=store_id, as_of=now, zones=zones)
