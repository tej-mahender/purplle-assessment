"""
Event emitter — the schema enforcer for the detection pipeline.
All events pass through emit_event() before being written to disk.
Validates required fields, assigns event_id, writes to JSONL output.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

# Output path (overridden by env var in run.sh)
OUTPUT_PATH = os.getenv("EVENTS_OUTPUT_PATH", "./data/events.jsonl")


def emit_event(
    *,
    store_id: str,
    camera_id: str,
    visitor_id: str,
    event_type: str,
    timestamp: datetime,
    zone_id: Optional[str] = None,
    dwell_ms: int = 0,
    is_staff: bool = False,
    confidence: float,
    queue_depth: Optional[int] = None,
    sku_zone: Optional[str] = None,
    session_seq: Optional[int] = None,
    is_partial_occlusion: Optional[bool] = None,
    reid_confidence: Optional[float] = None,
) -> dict:
    """
    Construct, validate, and emit a single event.
    Returns the event dict (also written to OUTPUT_PATH).
    """
    # Validate required fields
    assert store_id, "store_id is required"
    assert camera_id, "camera_id is required"
    assert visitor_id, "visitor_id is required"
    assert event_type, "event_type is required"
    assert 0.0 <= confidence <= 1.0, f"confidence must be 0-1, got {confidence}"

    zone_events = {
        "ZONE_ENTER", "ZONE_EXIT", "ZONE_DWELL",
        "BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON"
    }
    if event_type in zone_events:
        assert zone_id, f"zone_id required for event_type={event_type}"

    # Ensure UTC
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    event = {
        "event_id": str(uuid.uuid4()),
        "store_id": store_id,
        "camera_id": camera_id,
        "visitor_id": visitor_id,
        "event_type": event_type,
        "timestamp": timestamp.isoformat(),
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": is_staff,
        "confidence": round(confidence, 4),
        "metadata": {
            "queue_depth": queue_depth,
            "sku_zone": sku_zone,
            "session_seq": session_seq,
            "is_partial_occlusion": is_partial_occlusion,
            "reid_confidence": reid_confidence,
        },
    }

    # Write to JSONL
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True) if os.path.dirname(OUTPUT_PATH) else None
    with open(OUTPUT_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")

    return event


def make_visitor_id(track_id: int, store_id: str) -> str:
    """Generate a stable visitor_id from tracking ID + store."""
    short = hex(abs(hash(f"{store_id}_{track_id}")))[2:8]
    return f"VIS_{short}"
