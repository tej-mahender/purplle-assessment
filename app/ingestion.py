"""
POST /events/ingest
- Accepts batches of up to 500 events
- Idempotent by event_id (safe to call twice with same payload)
- Partial success: returns accepted + rejected counts with reasons
- Updates sessions table after each ingest
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import EventRow, SessionRow, get_db
from app.models import IngestRequest, IngestResponse, IngestRejection, EventType

log = structlog.get_logger()
router = APIRouter()


@router.post("/events/ingest", response_model=IngestResponse)
async def ingest_events(
    payload: IngestRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    accepted = 0
    rejected = 0
    rejections: list[IngestRejection] = []

    for event in payload.events:
        event_id_str = str(event.event_id)
        try:
            # Idempotency check — skip if already stored
            existing = await db.execute(
                select(EventRow).where(EventRow.event_id == event_id_str)
            )
            if existing.scalars().first():
                accepted += 1   # count as accepted (idempotent)
                continue

            row = EventRow(
                event_id=event_id_str,
                store_id=event.store_id,
                camera_id=event.camera_id,
                visitor_id=event.visitor_id,
                event_type=event.event_type,
                timestamp=event.timestamp.replace(tzinfo=None),  # store as naive UTC
                zone_id=event.zone_id,
                dwell_ms=event.dwell_ms,
                is_staff=event.is_staff,
                confidence=event.confidence,
                metadata_json=json.dumps(event.metadata.model_dump()),
                ingested_at=datetime.now(timezone.utc),
            )
            db.add(row)
            await db.flush()

            # Update session state
            await _upsert_session(db, event)

            accepted += 1

        except Exception as e:
            log.warning("event_rejected", event_id=event_id_str, reason=str(e))
            rejected += 1
            rejections.append(IngestRejection(event_id=event_id_str, reason=str(e)))

    await db.commit()

    log.info(
        "ingest_complete",
        trace_id=getattr(request.state, "trace_id", None),
        accepted=accepted,
        rejected=rejected,
        event_count=len(payload.events),
    )

    return IngestResponse(accepted=accepted, rejected=rejected, rejections=rejections)


async def _upsert_session(db: AsyncSession, event) -> None:
    """
    Maintain session rows based on ENTRY/EXIT events.
    Session key = store_id + visitor_id + entry_time bucket.
    Re-entries: a new ENTRY after an EXIT creates a new session row.
    """
    session_id = f"{event.store_id}_{event.visitor_id}"
    ts = event.timestamp.replace(tzinfo=None)

    result = await db.execute(
        select(SessionRow)
        .where(SessionRow.session_id == session_id)
        .order_by(SessionRow.id.desc())
        .limit(1)
    )
    session = result.scalars().first()

    if event.event_type == EventType.ENTRY:
        if session and session.exit_time is None:
            # Already open session — ignore duplicate entry
            return
        new_session = SessionRow(
            session_id=session_id,
            store_id=event.store_id,
            visitor_id=event.visitor_id,
            entry_time=ts,
            is_staff=event.is_staff,
            reentry=(session is not None),  # had a prior session → this is re-entry
        )
        db.add(new_session)

    elif event.event_type == EventType.EXIT:
        if session and session.exit_time is None:
            session.exit_time = ts

    elif event.event_type in (EventType.BILLING_QUEUE_JOIN, EventType.ZONE_ENTER):
        if session and event.zone_id and "BILLING" in event.zone_id.upper():
            session.visited_billing = True

    elif event.event_type == EventType.BILLING_QUEUE_ABANDON:
        # Not converted — no action needed (converted stays False)
        pass
