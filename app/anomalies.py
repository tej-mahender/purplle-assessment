"""
GET /stores/{store_id}/anomalies
Detects: BILLING_QUEUE_SPIKE, CONVERSION_DROP, DEAD_ZONE
Severity: INFO / WARN / CRITICAL
Each anomaly includes a suggested_action string.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import select, func, distinct
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import EventRow, POSTransaction, get_db
from app.models import StoreAnomalies, Anomaly, Severity, EventType

router = APIRouter()

QUEUE_SPIKE_THRESHOLD = 5          # visitors in billing simultaneously
QUEUE_SPIKE_DURATION_MINUTES = 5   # sustained for this long
CONVERSION_DROP_THRESHOLD = 0.20   # 20% below 7-day average
DEAD_ZONE_MINUTES = 30             # no visits in this zone for N minutes


@router.get("/stores/{store_id}/anomalies", response_model=StoreAnomalies)
async def get_anomalies(store_id: str, db: AsyncSession = Depends(get_db)):
    now = datetime.now(timezone.utc)
    anomalies: list[Anomaly] = []

    anomalies += await _detect_queue_spike(db, store_id, now)
    anomalies += await _detect_conversion_drop(db, store_id, now)
    anomalies += await _detect_dead_zones(db, store_id, now)

    return StoreAnomalies(store_id=store_id, as_of=now, anomalies=anomalies)


async def _detect_queue_spike(
    db: AsyncSession, store_id: str, now: datetime
) -> list[Anomaly]:
    """BILLING_QUEUE_SPIKE: queue_depth > threshold for > duration."""
    import json
    window_start = now - timedelta(minutes=QUEUE_SPIKE_DURATION_MINUTES)

    result = await db.execute(
        select(EventRow.metadata_json, EventRow.timestamp)
        .where(
            EventRow.store_id == store_id,
            EventRow.event_type == EventType.BILLING_QUEUE_JOIN,
            EventRow.timestamp >= window_start,
        )
        .order_by(EventRow.timestamp)
    )
    rows = result.all()

    spike_depths = []
    for meta_json, ts in rows:
        try:
            meta = json.loads(meta_json or "{}")
            depth = meta.get("queue_depth", 0) or 0
            if depth >= QUEUE_SPIKE_THRESHOLD:
                spike_depths.append(depth)
        except Exception:
            pass

    if len(spike_depths) >= 2:  # sustained spike
        max_depth = max(spike_depths)
        severity = Severity.CRITICAL if max_depth >= QUEUE_SPIKE_THRESHOLD * 2 else Severity.WARN
        return [Anomaly(
            anomaly_type="BILLING_QUEUE_SPIKE",
            severity=severity,
            description=f"Billing queue depth reached {max_depth} over the last {QUEUE_SPIKE_DURATION_MINUTES} minutes",
            suggested_action="Open an additional billing counter or redirect customers to express checkout",
            detected_at=now,
            zone_id="BILLING",
            value=float(max_depth),
        )]
    return []


async def _detect_conversion_drop(
    db: AsyncSession, store_id: str, now: datetime
) -> list[Anomaly]:
    """CONVERSION_DROP: today's conversion rate < 7-day average by > threshold."""

    # Use most recent event date as session day (handles past-date clips)
    latest_result = await db.execute(
        select(func.max(EventRow.timestamp))
        .where(EventRow.store_id == store_id)
    )
    latest_ts = latest_result.scalar()
    if latest_ts:
        today_start = latest_ts.replace(hour=0, minute=0, second=0, microsecond=0)
        session_end  = today_start + timedelta(hours=24)
    else:
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        session_end  = now
    week_start = today_start - timedelta(days=7)

    # Today's conversion rate
    today_rate = await _conversion_rate_for_window(db, store_id, today_start, session_end)

    # 7-day average (excluding today)
    weekly_rate = await _conversion_rate_for_window(db, store_id, week_start, today_start)

    if weekly_rate is None or weekly_rate == 0:
        return []  # No baseline — can't detect drop

    drop = (weekly_rate - today_rate) / weekly_rate if weekly_rate > 0 else 0

    if drop >= CONVERSION_DROP_THRESHOLD:
        severity = Severity.CRITICAL if drop >= 0.40 else Severity.WARN
        return [Anomaly(
            anomaly_type="CONVERSION_DROP",
            severity=severity,
            description=(
                f"Conversion rate {today_rate:.1%} is {drop:.0%} below "
                f"7-day average of {weekly_rate:.1%}"
            ),
            suggested_action=(
                "Check staff availability at billing, review recent product placement changes, "
                "or check for pricing issues"
            ),
            detected_at=now,
            value=round(today_rate, 4),
        )]
    return []


async def _detect_dead_zones(
    db: AsyncSession, store_id: str, now: datetime
) -> list[Anomaly]:
    """DEAD_ZONE: no ZONE_ENTER events for any zone in the last 30 minutes."""
    window_start = now - timedelta(minutes=DEAD_ZONE_MINUTES)

    # Get all zones that have ever had activity
    all_zones_result = await db.execute(
        select(distinct(EventRow.zone_id))
        .where(
            EventRow.store_id == store_id,
            EventRow.zone_id.isnot(None),
        )
    )
    all_zones = {row[0] for row in all_zones_result.all()}

    # Get zones with recent activity
    recent_result = await db.execute(
        select(distinct(EventRow.zone_id))
        .where(
            EventRow.store_id == store_id,
            EventRow.event_type == EventType.ZONE_ENTER,
            EventRow.zone_id.isnot(None),
            EventRow.timestamp >= window_start,
        )
    )
    active_zones = {row[0] for row in recent_result.all()}

    dead_zones = all_zones - active_zones
    anomalies = []
    for zone_id in dead_zones:
        anomalies.append(Anomaly(
            anomaly_type="DEAD_ZONE",
            severity=Severity.INFO,
            description=f"Zone '{zone_id}' has had no customer visits in the last {DEAD_ZONE_MINUTES} minutes",
            suggested_action=f"Check if zone '{zone_id}' camera is functioning; consider re-merchandising to drive traffic",
            detected_at=now,
            zone_id=zone_id,
        ))

    return anomalies


async def _conversion_rate_for_window(
    db: AsyncSession, store_id: str, start: datetime, end: datetime
) -> float:
    """Compute conversion rate for an arbitrary time window."""
    from datetime import timedelta as td

    pos_result = await db.execute(
        select(POSTransaction.timestamp)
        .where(
            POSTransaction.store_id == store_id,
            POSTransaction.timestamp >= start,
            POSTransaction.timestamp <= end,
        )
    )
    pos_timestamps = [row[0] for row in pos_result.all()]

    visitor_result = await db.execute(
        select(func.count(distinct(EventRow.visitor_id)))
        .where(
            EventRow.store_id == store_id,
            EventRow.event_type == EventType.ENTRY,
            EventRow.is_staff == False,
            EventRow.timestamp >= start,
            EventRow.timestamp <= end,
        )
    )
    total_visitors = visitor_result.scalar() or 0
    if total_visitors == 0:
        return 0.0

    converted: set[str] = set()
    for pos_ts in pos_timestamps:
        billing_result = await db.execute(
            select(distinct(EventRow.visitor_id))
            .where(
                EventRow.store_id == store_id,
                EventRow.zone_id.ilike("%billing%"),
                EventRow.is_staff == False,
                EventRow.timestamp >= pos_ts - td(minutes=5),
                EventRow.timestamp <= pos_ts,
            )
        )
        for (vid,) in billing_result.all():
            converted.add(vid)

    return len(converted) / total_visitors
