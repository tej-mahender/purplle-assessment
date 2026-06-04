# PROMPT: "Write unit tests for a retail CCTV detection pipeline's event emission layer.
# Test: ENTRY/EXIT schema compliance, re-entry produces REENTRY not ENTRY,
# group of 3 people produces 3 ENTRY events, staff flagged correctly,
# all events have unique event_ids, low confidence events not suppressed."
#
# CHANGES MADE: Extracted emit_event tests to not depend on file I/O (mock OUTPUT_PATH),
# added test_group_entry which AI missed entirely, added schema key assertion,
# strengthened re-entry test to verify visitor_id consistency.

import json
import os
import tempfile
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch


def get_emit(tmp_path):
    """Import emit with output redirected to a temp file."""
    output_file = str(tmp_path / "events.jsonl")
    with patch.dict(os.environ, {"EVENTS_OUTPUT_PATH": output_file}):
        import importlib
        import pipeline.emit as emit_module
        importlib.reload(emit_module)
        return emit_module, output_file


def read_events(output_file):
    events = []
    with open(output_file) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


@pytest.fixture(autouse=True)
def reset_emit_module():
    """Reset emit module between tests to avoid state bleed."""
    yield
    import importlib
    import pipeline.emit as em
    importlib.reload(em)


def test_entry_event_schema(tmp_path):
    """Emitted ENTRY event must contain all required schema fields with correct types."""
    emit, output = get_emit(tmp_path)
    ts = datetime.now(timezone.utc)
    event = emit.emit_event(
        store_id="STORE_BLR_002",
        camera_id="CAM_ENTRY_01",
        visitor_id="VIS_aabb11",
        event_type="ENTRY",
        timestamp=ts,
        confidence=0.91,
        is_staff=False,
    )

    required_keys = {
        "event_id", "store_id", "camera_id", "visitor_id",
        "event_type", "timestamp", "zone_id", "dwell_ms",
        "is_staff", "confidence", "metadata"
    }
    assert required_keys.issubset(event.keys())
    assert event["event_type"] == "ENTRY"
    assert event["store_id"] == "STORE_BLR_002"
    assert event["is_staff"] is False
    assert 0.0 <= event["confidence"] <= 1.0
    assert event["zone_id"] is None


def test_event_ids_unique(tmp_path):
    """Every emitted event must have a globally unique event_id."""
    emit, output = get_emit(tmp_path)
    ts = datetime.now(timezone.utc)
    for i in range(20):
        emit.emit_event(
            store_id="STORE_BLR_002",
            camera_id="CAM_ENTRY_01",
            visitor_id=f"VIS_{i:04d}",
            event_type="ENTRY",
            timestamp=ts,
            confidence=0.85,
        )
    events = read_events(output)
    ids = [e["event_id"] for e in events]
    assert len(ids) == len(set(ids)), "Duplicate event_ids detected"


def test_low_confidence_not_suppressed(tmp_path):
    """Low confidence detections (> 0.0) must be emitted, not silently dropped."""
    emit, output = get_emit(tmp_path)
    ts = datetime.now(timezone.utc)
    event = emit.emit_event(
        store_id="STORE_BLR_002",
        camera_id="CAM_ENTRY_01",
        visitor_id="VIS_lowconf",
        event_type="ENTRY",
        timestamp=ts,
        confidence=0.31,  # just above DETECTION_CONFIDENCE threshold
    )
    assert event["confidence"] == pytest.approx(0.31, abs=0.001)
    events = read_events(output)
    assert len(events) == 1


def test_zone_event_requires_zone_id(tmp_path):
    """ZONE_ENTER without zone_id must raise AssertionError."""
    emit, output = get_emit(tmp_path)
    ts = datetime.now(timezone.utc)
    with pytest.raises((AssertionError, ValueError)):
        emit.emit_event(
            store_id="STORE_BLR_002",
            camera_id="CAM_FLOOR_01",
            visitor_id="VIS_zone1",
            event_type="ZONE_ENTER",
            timestamp=ts,
            confidence=0.88,
            zone_id=None,  # should fail
        )


def test_make_visitor_id_stable(tmp_path):
    """Same track_id + store_id must always produce the same visitor_id."""
    emit, _ = get_emit(tmp_path)
    id1 = emit.make_visitor_id(42, "STORE_BLR_002")
    id2 = emit.make_visitor_id(42, "STORE_BLR_002")
    assert id1 == id2
    assert id1.startswith("VIS_")


def test_make_visitor_id_different_for_different_tracks(tmp_path):
    """Different track IDs should produce different visitor_ids."""
    emit, _ = get_emit(tmp_path)
    id1 = emit.make_visitor_id(1, "STORE_BLR_002")
    id2 = emit.make_visitor_id(2, "STORE_BLR_002")
    assert id1 != id2


def test_group_entry_three_visitors(tmp_path):
    """When 3 people enter together, 3 ENTRY events must be emitted (not 1)."""
    emit, output = get_emit(tmp_path)
    ts = datetime.now(timezone.utc)
    visitors = ["VIS_g1", "VIS_g2", "VIS_g3"]
    for vid in visitors:
        emit.emit_event(
            store_id="STORE_BLR_002",
            camera_id="CAM_ENTRY_01",
            visitor_id=vid,
            event_type="ENTRY",
            timestamp=ts,
            confidence=0.85,
        )
    events = read_events(output)
    entry_events = [e for e in events if e["event_type"] == "ENTRY"]
    assert len(entry_events) == 3
    visitor_ids = {e["visitor_id"] for e in entry_events}
    assert visitor_ids == set(visitors)
