"""
GET /stores/{store_id}/funnel
Conversion funnel: Entry → Zone Visit → Billing Queue → Purchase
Unit = session (not raw events). Re-entries do NOT double-count a visitor.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select, func, distinct
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import EventRow, POSTransaction, get_db
from app.models import StoreFunnel, FunnelStage, EventType

router = APIRouter()


@router.get("/stores/{store_id}/funnel", response_model=StoreFunnel)
async def get_funnel(store_id: str, db: AsyncSession = Depends(get_db)):
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

    # Stage 1: unique visitors (ENTRY events, exclude staff, deduplicate by visitor_id)
    entry_result = await db.execute(
        select(distinct(EventRow.visitor_id))
        .where(
            EventRow.store_id == store_id,
            EventRow.event_type == EventType.ENTRY,
            EventRow.is_staff == False,
            EventRow.timestamp >= today_start,
        )
    )
    unique_visitors: set[str] = {row[0] for row in entry_result.all()}
    stage1_count = len(unique_visitors)

    # Stage 2: visitors who entered at least one named zone
    zone_result = await db.execute(
        select(distinct(EventRow.visitor_id))
        .where(
            EventRow.store_id == store_id,
            EventRow.event_type.in_([EventType.ZONE_ENTER, EventType.ZONE_DWELL]),
            EventRow.is_staff == False,
            EventRow.zone_id.isnot(None),
            EventRow.timestamp >= today_start,
            EventRow.visitor_id.in_(unique_visitors),
        )
    )
    zone_visitors: set[str] = {row[0] for row in zone_result.all()}
    stage2_count = len(zone_visitors)

    # Stage 3: visitors who reached billing zone (joined queue or entered billing zone)
    billing_result = await db.execute(
        select(distinct(EventRow.visitor_id))
        .where(
            EventRow.store_id == store_id,
            EventRow.event_type.in_([
                EventType.BILLING_QUEUE_JOIN,
                EventType.ZONE_ENTER,
            ]),
            EventRow.zone_id.ilike("%billing%"),
            EventRow.is_staff == False,
            EventRow.timestamp >= today_start,
            EventRow.visitor_id.in_(unique_visitors),
        )
    )
    billing_visitors: set[str] = {row[0] for row in billing_result.all()}
    stage3_count = len(billing_visitors)

    # Stage 4: purchased (POS transactions today)
    pos_result = await db.execute(
        select(func.count(POSTransaction.id))
        .where(
            POSTransaction.store_id == store_id,
            POSTransaction.timestamp >= today_start,
        )
    )
    purchase_count = pos_result.scalar() or 0
    # Cap purchases at billing visitors (can't have more purchases than billing visitors)
    stage4_count = min(purchase_count, stage3_count)

    # Build funnel stages with drop-off %
    def drop_off(prev: int, curr: int) -> float:
        if prev == 0:
            return 0.0
        return round((prev - curr) / prev * 100, 2)

    stages = [
        FunnelStage(stage="Entry", count=stage1_count, drop_off_pct=0.0),
        FunnelStage(stage="Zone Visit", count=stage2_count, drop_off_pct=drop_off(stage1_count, stage2_count)),
        FunnelStage(stage="Billing Queue", count=stage3_count, drop_off_pct=drop_off(stage2_count, stage3_count)),
        FunnelStage(stage="Purchase", count=stage4_count, drop_off_pct=drop_off(stage3_count, stage4_count)),
    ]

    return StoreFunnel(store_id=store_id, as_of=now, stages=stages)
