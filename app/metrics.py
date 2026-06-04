"""
GET /stores/{store_id}/metrics
Real-time store metrics: unique visitors, conversion rate,
avg dwell per zone, current queue depth, abandonment rate.
Staff excluded. Zero-traffic safe.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func, distinct
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import EventRow, SessionRow, POSTransaction, get_db
from app.models import StoreMetrics, ZoneDwell, EventType

router = APIRouter()

POS_WINDOW_MINUTES = 5


@router.get("/stores/{store_id}/metrics", response_model=StoreMetrics)
async def get_metrics(store_id: str, db: AsyncSession = Depends(get_db)):
    now = datetime.now(timezone.utc)

    # Use the most recent event date for this store as the "session day".
    # This handles the common case where clips are from a past date (e.g. 2026-04-10)
    # and the API is queried today — without this, all metrics return 0.
    latest_result = await db.execute(
        select(func.max(EventRow.timestamp))
        .where(EventRow.store_id == store_id)
    )
    latest_event_ts = latest_result.scalar()

    if latest_event_ts:
        # Use the date of the most recent event as "today" for this store
        session_date = latest_event_ts.replace(hour=0, minute=0, second=0, microsecond=0)
        session_end  = session_date + timedelta(hours=24)
    else:
        # No events yet — use real today (returns zero-traffic response)
        session_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        session_end  = now

    today_start = session_date

    # ── Unique visitors today (exclude staff, count by visitor_id) ──────────
    result = await db.execute(
        select(func.count(distinct(EventRow.visitor_id)))
        .where(
            EventRow.store_id == store_id,
            EventRow.event_type == EventType.ENTRY,
            EventRow.is_staff == False,
            EventRow.timestamp >= today_start,
        )
    )
    unique_visitors = result.scalar() or 0

    # ── Conversion rate (POS time-window correlation) ────────────────────────
    conversion_rate = await _compute_conversion_rate(db, store_id, today_start, session_end)

    # ── Avg dwell per zone ───────────────────────────────────────────────────
    dwell_result = await db.execute(
        select(
            EventRow.zone_id,
            func.avg(EventRow.dwell_ms).label("avg_dwell"),
            func.count(EventRow.id).label("visit_count"),
        )
        .where(
            EventRow.store_id == store_id,
            EventRow.event_type.in_([EventType.ZONE_DWELL, EventType.ZONE_EXIT]),
            EventRow.is_staff == False,
            EventRow.zone_id.isnot(None),
            EventRow.timestamp >= today_start,
        )
        .group_by(EventRow.zone_id)
    )
    avg_dwell_by_zone = [
        ZoneDwell(zone_id=row.zone_id, avg_dwell_ms=round(row.avg_dwell or 0, 2), visit_count=row.visit_count)
        for row in dwell_result.all()
    ]

    # ── Current queue depth (visitors in billing zone right now) ─────────────
    queue_depth = await _compute_queue_depth(db, store_id)

    # ── Abandonment rate ─────────────────────────────────────────────────────
    abandon_result = await db.execute(
        select(func.count(EventRow.id))
        .where(
            EventRow.store_id == store_id,
            EventRow.event_type == EventType.BILLING_QUEUE_ABANDON,
            EventRow.is_staff == False,
            EventRow.timestamp >= today_start,
        )
    )
    abandonments = abandon_result.scalar() or 0

    join_result = await db.execute(
        select(func.count(EventRow.id))
        .where(
            EventRow.store_id == store_id,
            EventRow.event_type == EventType.BILLING_QUEUE_JOIN,
            EventRow.is_staff == False,
            EventRow.timestamp >= today_start,
        )
    )
    queue_joins = join_result.scalar() or 0
    abandonment_rate = round(abandonments / queue_joins, 4) if queue_joins > 0 else 0.0

    return StoreMetrics(
        store_id=store_id,
        as_of=now,
        unique_visitors=unique_visitors,
        conversion_rate=round(conversion_rate, 4),
        avg_dwell_by_zone=avg_dwell_by_zone,
        current_queue_depth=queue_depth,
        abandonment_rate=abandonment_rate,
    )


async def _compute_conversion_rate(
    db: AsyncSession, store_id: str, start: datetime, end: datetime
) -> float:
    """
    A visitor counts as converted if they were in the billing zone
    in the 5-minute window before a POS transaction timestamp.
    Correlation is by time window + store only (no customer_id).
    """
    # Get all POS transactions for this store today
    pos_result = await db.execute(
        select(POSTransaction.timestamp)
        .where(
            POSTransaction.store_id == store_id,
            POSTransaction.timestamp >= start,
            POSTransaction.timestamp <= end,
        )
    )
    pos_timestamps = [row[0] for row in pos_result.all()]

    if not pos_timestamps:
        # No purchases today — check if there are any visitors
        visitor_result = await db.execute(
            select(func.count(distinct(EventRow.visitor_id)))
            .where(
                EventRow.store_id == store_id,
                EventRow.event_type == EventType.ENTRY,
                EventRow.is_staff == False,
                EventRow.timestamp >= start,
            )
        )
        total_visitors = visitor_result.scalar() or 0
        return 0.0

    # For each POS transaction, find visitors in billing zone in prior 5 min
    converted_visitors: set[str] = set()
    for pos_ts in pos_timestamps:
        window_start = pos_ts - timedelta(minutes=POS_WINDOW_MINUTES)
        billing_result = await db.execute(
            select(distinct(EventRow.visitor_id))
            .where(
                EventRow.store_id == store_id,
                EventRow.zone_id.ilike("%billing%"),
                EventRow.is_staff == False,
                EventRow.timestamp >= window_start,
                EventRow.timestamp <= pos_ts,
            )
        )
        for (visitor_id,) in billing_result.all():
            converted_visitors.add(visitor_id)

    # Total unique visitors today
    total_result = await db.execute(
        select(func.count(distinct(EventRow.visitor_id)))
        .where(
            EventRow.store_id == store_id,
            EventRow.event_type == EventType.ENTRY,
            EventRow.is_staff == False,
            EventRow.timestamp >= start,
        )
    )
    total_visitors = total_result.scalar() or 0

    if total_visitors == 0:
        return 0.0

    return len(converted_visitors) / total_visitors


async def _compute_queue_depth(db: AsyncSession, store_id: str) -> int:
    """
    Estimate current queue depth: count visitors who joined billing queue
    but haven't exited the billing zone yet (no ZONE_EXIT from billing after JOIN).
    Simplified: use latest BILLING_QUEUE_JOIN metadata.queue_depth value.
    """
    import json
    result = await db.execute(
        select(EventRow.metadata_json)
        .where(
            EventRow.store_id == store_id,
            EventRow.event_type == EventType.BILLING_QUEUE_JOIN,
        )
        .order_by(EventRow.timestamp.desc())
        .limit(1)
    )
    row = result.scalar()
    if row:
        try:
            meta = json.loads(row)
            return meta.get("queue_depth", 0) or 0
        except Exception:
            pass
    return 0
