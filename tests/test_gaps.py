# PROMPT: "Write unit tests for 5 detection pipeline gaps (no real video/GPU needed):
# 1. ZONE_EXIT emitted when visitor leaves a zone
# 2. BILLING_QUEUE_JOIN with correct queue_depth when queue >= threshold
# 3. BILLING_QUEUE_ABANDON when billing exit has no POS transaction
# 4. is_partial_occlusion=True on low-confidence detections
# 5. Cross-camera deduplication using shared VisitorRegistry
# Mock supervision/cv2/ultralytics — tests must run on CPU without GPU deps."
#
# CHANGES MADE: Added sys.modules mocking for supervision before all imports
# (AI version imported tracker directly which crashed without sv installed),
# split cross-cam test into two cases (within window / after window),
# added schema fields check for ZONE_EXIT.

import json
import os
import sys
import csv
import importlib
import tempfile
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import pytest

# ── Mock heavy deps before any pipeline import ────────────────────────────────
# supervision, cv2, ultralytics are GPU/video deps not installed in test env.
# We mock them at the sys.modules level so tracker.py and detect.py import fine.

_sv_mock = MagicMock()

class _FakeByteTrack:
    def update_with_detections(self, detections):
        return detections   # pass-through: tracker_id already set in fake detections

_sv_mock.ByteTrack = _FakeByteTrack
_sv_mock.Detections = MagicMock()

sys.modules.setdefault("supervision", _sv_mock)
sys.modules.setdefault("cv2", MagicMock())
sys.modules.setdefault("ultralytics", MagicMock())
sys.modules.setdefault("torch", MagicMock())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Zone map used in all zone tests ──────────────────────────────────────────

ZONE_MAP = {
    "SKINCARE": {"polygon": [[0,   0], [400,   0], [400, 300], [0, 300]]},
    "MAKEUP":   {"polygon": [[400, 0], [800,   0], [800, 300], [400, 300]]},
    "BILLING":  {"polygon": [[0, 300], [800, 300], [800, 600], [0, 600]]},
}
W, H = 800, 600


# ── Detection factory ─────────────────────────────────────────────────────────

def make_detections(bboxes: list[list[float]], confidences: list[float]):
    """
    Build a fake Detections object.
    bboxes: [[x1,y1,x2,y2], ...]
    tracker_id matches 1-based index.
    """
    n       = len(bboxes)
    xyxy    = np.array(bboxes, dtype=float)
    tracker_ids = np.arange(1, n + 1, dtype=int)
    conf    = np.array(confidences, dtype=float)
    class_ids = np.zeros(n, dtype=int)

    det = MagicMock()
    det.xyxy        = xyxy
    det.tracker_id  = tracker_ids
    det.confidence  = conf
    det.class_id    = class_ids
    det.__len__     = lambda self: n
    return det


def fresh_tracker(tmp_path, camera_id="CAM_FLOOR_01", registry=None, env_extra=None):
    """
    Return a freshly-imported StoreTracker wired to a temp output file.
    Re-imports pipeline.tracker each call so env vars take effect.
    """
    output = str(tmp_path / "events.jsonl")
    env    = {"EVENTS_OUTPUT_PATH": output}
    if env_extra:
        env.update(env_extra)

    with patch.dict(os.environ, env):
        import pipeline.emit as em
        importlib.reload(em)
        # Re-import tracker with fresh module state
        if "pipeline.tracker" in sys.modules:
            importlib.reload(sys.modules["pipeline.tracker"])
        from pipeline.tracker import StoreTracker, VisitorRegistry as VR

    kwargs = {"registry": registry} if registry is not None else {}
    tracker = StoreTracker.__new__(StoreTracker)
    # Manually init to avoid constructor issues with reloading
    tracker.store_id      = "STORE_X"
    tracker.camera_id     = camera_id
    tracker.frame_width   = W
    tracker.frame_height  = H
    from pipeline.tracker import VisitorRegistry
    tracker.registry      = registry if registry is not None else VisitorRegistry()
    tracker.tracker       = _FakeByteTrack()
    tracker.tripwire_y    = int(H * 0.65)
    tracker._prev_centroids     = {}
    tracker._sessions           = {}
    tracker._track_to_visitor   = {}
    tracker._billing_occupants  = set()

    return tracker, output


def read_events(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# GAP 1 — ZONE_EXIT
# ─────────────────────────────────────────────────────────────────────────────

def test_zone_exit_emitted_on_transition(tmp_path):
    """Moving from SKINCARE → MAKEUP must emit ZONE_EXIT(SKINCARE) then ZONE_ENTER(MAKEUP)."""
    with patch.dict(os.environ, {"EVENTS_OUTPUT_PATH": str(tmp_path / "events.jsonl")}):
        import pipeline.emit as em; importlib.reload(em)
        if "pipeline.tracker" in sys.modules: importlib.reload(sys.modules["pipeline.tracker"])
        from pipeline.tracker import StoreTracker

        t = StoreTracker("STORE_X", "CAM_FLOOR_01", W, H)
        now = datetime.now(timezone.utc)

        # Frame 1: person in SKINCARE (centroid 200,150)
        t.update(make_detections([[100,50,300,250]], [0.9]), now, ZONE_MAP)
        # Frame 2: same person moves to MAKEUP (centroid 600,150)
        t._prev_centroids[1] = 150.0
        t.update(make_detections([[500,50,700,250]], [0.9]),
                 now + timedelta(seconds=5), ZONE_MAP)

    events = read_events(str(tmp_path / "events.jsonl"))
    types  = [e["event_type"] for e in events]

    assert "ZONE_EXIT"  in types, f"ZONE_EXIT missing. Got: {types}"
    assert "ZONE_ENTER" in types

    exit_ev = next(e for e in events if e["event_type"] == "ZONE_EXIT")
    assert exit_ev["zone_id"] == "SKINCARE"

    enters = [e for e in events if e["event_type"] == "ZONE_ENTER"]
    assert any(e["zone_id"] == "MAKEUP" for e in enters), "No ZONE_ENTER(MAKEUP)"


def test_no_zone_exit_on_first_zone(tmp_path):
    """First ever zone entry: ZONE_ENTER only, no ZONE_EXIT."""
    with patch.dict(os.environ, {"EVENTS_OUTPUT_PATH": str(tmp_path / "events.jsonl")}):
        import pipeline.emit as em; importlib.reload(em)
        if "pipeline.tracker" in sys.modules: importlib.reload(sys.modules["pipeline.tracker"])
        from pipeline.tracker import StoreTracker

        t = StoreTracker("STORE_X", "CAM_FLOOR_01", W, H)
        t.update(make_detections([[100,50,300,250]], [0.9]),
                 datetime.now(timezone.utc), ZONE_MAP)

    events = read_events(str(tmp_path / "events.jsonl"))
    types  = [e["event_type"] for e in events]
    assert "ZONE_EXIT"  not in types
    assert types.count("ZONE_ENTER") == 1


def test_zone_exit_has_required_schema_fields(tmp_path):
    """ZONE_EXIT event must contain all required schema fields."""
    with patch.dict(os.environ, {"EVENTS_OUTPUT_PATH": str(tmp_path / "events.jsonl")}):
        import pipeline.emit as em; importlib.reload(em)
        if "pipeline.tracker" in sys.modules: importlib.reload(sys.modules["pipeline.tracker"])
        from pipeline.tracker import StoreTracker

        t = StoreTracker("STORE_X", "CAM_FLOOR_01", W, H)
        now = datetime.now(timezone.utc)
        t.update(make_detections([[100,50,300,250]], [0.9]), now, ZONE_MAP)
        t._prev_centroids[1] = 150.0
        t.update(make_detections([[500,50,700,250]], [0.9]),
                 now + timedelta(seconds=5), ZONE_MAP)

    events = read_events(str(tmp_path / "events.jsonl"))
    zone_exit = next((e for e in events if e["event_type"] == "ZONE_EXIT"), None)
    assert zone_exit is not None

    required = {"event_id","store_id","camera_id","visitor_id","event_type",
                "timestamp","zone_id","dwell_ms","is_staff","confidence","metadata"}
    assert required.issubset(zone_exit.keys())
    assert zone_exit["zone_id"] is not None


# ─────────────────────────────────────────────────────────────────────────────
# GAP 2 — BILLING_QUEUE_JOIN
# ─────────────────────────────────────────────────────────────────────────────

def test_billing_queue_join_when_queue_present(tmp_path):
    """3rd person entering billing while 2 are already there → BILLING_QUEUE_JOIN."""
    with patch.dict(os.environ, {
        "EVENTS_OUTPUT_PATH": str(tmp_path / "events.jsonl"),
        "QUEUE_DEPTH_THRESHOLD": "2",
    }):
        import pipeline.emit as em; importlib.reload(em)
        if "pipeline.tracker" in sys.modules: importlib.reload(sys.modules["pipeline.tracker"])
        from pipeline.tracker import StoreTracker

        t   = StoreTracker("STORE_X", "CAM_FLOOR_01", W, H)
        now = datetime.now(timezone.utc)

        # 2 people in billing
        t.update(make_detections(
            [[50,350,200,550],[250,350,400,550]], [0.9, 0.88]
        ), now, ZONE_MAP)

        # 3rd person joins
        t.update(make_detections(
            [[50,350,200,550],[250,350,400,550],[450,350,600,550]],
            [0.9, 0.88, 0.85]
        ), now + timedelta(seconds=2), ZONE_MAP)

    events = read_events(str(tmp_path / "events.jsonl"))
    joins  = [e for e in events if e["event_type"] == "BILLING_QUEUE_JOIN"]
    assert len(joins) >= 1, "BILLING_QUEUE_JOIN not emitted"
    assert joins[0]["zone_id"] == "BILLING"
    assert joins[0]["metadata"]["queue_depth"] >= 2


def test_billing_queue_join_not_emitted_below_threshold(tmp_path):
    """Only 1 person in billing (depth=0) — 2nd arrival: depth=1 < threshold=2 → no JOIN."""
    with patch.dict(os.environ, {
        "EVENTS_OUTPUT_PATH": str(tmp_path / "events.jsonl"),
        "QUEUE_DEPTH_THRESHOLD": "2",
    }):
        import pipeline.emit as em; importlib.reload(em)
        if "pipeline.tracker" in sys.modules: importlib.reload(sys.modules["pipeline.tracker"])
        from pipeline.tracker import StoreTracker

        t   = StoreTracker("STORE_X", "CAM_FLOOR_01", W, H)
        now = datetime.now(timezone.utc)

        # 1 person
        t.update(make_detections([[50,350,200,550]], [0.9]), now, ZONE_MAP)
        # 2nd person joins (depth was 1, below threshold=2)
        t.update(make_detections(
            [[50,350,200,550],[250,350,400,550]], [0.9,0.88]
        ), now + timedelta(seconds=2), ZONE_MAP)

    events = read_events(str(tmp_path / "events.jsonl"))
    joins  = [e for e in events if e["event_type"] == "BILLING_QUEUE_JOIN"]
    assert len(joins) == 0, f"Unexpected BILLING_QUEUE_JOIN: {joins}"


# ─────────────────────────────────────────────────────────────────────────────
# GAP 3 — BILLING_QUEUE_ABANDON
# ─────────────────────────────────────────────────────────────────────────────

def _write_pos(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, ["store_id","transaction_id","timestamp","basket_value_inr"])
        w.writeheader()
        w.writerows(rows)


def test_abandon_emitted_when_no_pos(tmp_path):
    """Billing exit with no following POS → BILLING_QUEUE_ABANDON."""
    output   = str(tmp_path / "events.jsonl")
    pos_path = str(tmp_path / "pos.csv")
    _write_pos(pos_path, [])   # no transactions

    with patch.dict(os.environ, {"EVENTS_OUTPUT_PATH": output}):
        import pipeline.emit as em; importlib.reload(em)
        from pipeline.detect import emit_abandon_events

    now = datetime.now(timezone.utc)
    exits = [{"visitor_id":"VIS_ab1","exit_time":now,
               "store_id":"STORE_A","camera_id":"CAM_BILLING_01","is_staff":False}]

    count = emit_abandon_events(exits, pos_path, "STORE_A", "CAM_BILLING_01")
    assert count == 1

    events = read_events(output)
    abandons = [e for e in events if e["event_type"] == "BILLING_QUEUE_ABANDON"]
    assert len(abandons) == 1
    assert abandons[0]["visitor_id"] == "VIS_ab1"
    assert abandons[0]["zone_id"]    == "BILLING"


def test_no_abandon_when_pos_follows(tmp_path):
    """POS transaction within 5 min of billing exit → NO abandon."""
    output   = str(tmp_path / "events.jsonl")
    pos_path = str(tmp_path / "pos.csv")
    now      = datetime.now(timezone.utc)

    _write_pos(pos_path, [{
        "store_id": "STORE_B",
        "transaction_id": "TXN_001",
        "timestamp": (now + timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "basket_value_inr": "1200.00",
    }])

    with patch.dict(os.environ, {"EVENTS_OUTPUT_PATH": output}):
        import pipeline.emit as em; importlib.reload(em)
        from pipeline.detect import emit_abandon_events

    exits = [{"visitor_id":"VIS_conv1","exit_time":now,
               "store_id":"STORE_B","camera_id":"CAM_BILLING_01","is_staff":False}]
    count = emit_abandon_events(exits, pos_path, "STORE_B", "CAM_BILLING_01")
    assert count == 0, "Visitor who purchased should NOT be flagged as abandoned"


def test_staff_never_flagged_as_abandoned(tmp_path):
    """Staff billing exits must never produce BILLING_QUEUE_ABANDON."""
    output   = str(tmp_path / "events.jsonl")
    pos_path = str(tmp_path / "pos.csv")
    _write_pos(pos_path, [])

    with patch.dict(os.environ, {"EVENTS_OUTPUT_PATH": output}):
        import pipeline.emit as em; importlib.reload(em)
        from pipeline.detect import emit_abandon_events

    exits = [{"visitor_id":"VIS_staff1","exit_time":datetime.now(timezone.utc),
               "store_id":"STORE_S","camera_id":"CAM_BILLING_01","is_staff":True}]
    count = emit_abandon_events(exits, pos_path, "STORE_S", "CAM_BILLING_01")
    assert count == 0


def test_abandon_outside_window_still_emitted(tmp_path):
    """POS transaction > 5 min after billing exit → still an abandon (missed purchase)."""
    output   = str(tmp_path / "events.jsonl")
    pos_path = str(tmp_path / "pos.csv")
    now      = datetime.now(timezone.utc)

    _write_pos(pos_path, [{
        "store_id": "STORE_C",
        "transaction_id": "TXN_002",
        "timestamp": (now + timedelta(minutes=8)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "basket_value_inr": "800.00",
    }])

    with patch.dict(os.environ, {"EVENTS_OUTPUT_PATH": output}):
        import pipeline.emit as em; importlib.reload(em)
        from pipeline.detect import emit_abandon_events

    exits = [{"visitor_id":"VIS_late1","exit_time":now,
               "store_id":"STORE_C","camera_id":"CAM_BILLING_01","is_staff":False}]
    count = emit_abandon_events(exits, pos_path, "STORE_C", "CAM_BILLING_01")
    assert count == 1, "POS 8 min later is outside window → should be an abandon"


# ─────────────────────────────────────────────────────────────────────────────
# GAP 4 — is_partial_occlusion
# ─────────────────────────────────────────────────────────────────────────────

def _make_entry_tracker(tmp_path, conf):
    """Build an entry camera tracker, fire one person across tripwire, return events."""
    output = str(tmp_path / "events.jsonl")
    with patch.dict(os.environ, {
        "EVENTS_OUTPUT_PATH": output,
        "PARTIAL_OCCLUSION_CONF": "0.50",
    }):
        import pipeline.emit as em; importlib.reload(em)
        if "pipeline.tracker" in sys.modules: importlib.reload(sys.modules["pipeline.tracker"])
        from pipeline.tracker import StoreTracker

        t   = StoreTracker("STORE_X", "CAM_ENTRY_01", W, H)
        now = datetime.now(timezone.utc)
        tw  = t.tripwire_y

        # Frame 1: centroid above tripwire
        t._prev_centroids[1] = float(tw - 10)
        # Frame 2: centroid below tripwire → crossing detected
        det = make_detections([[300, tw+5, 500, tw+100]], [conf])
        t.update(det, now, None)

    return read_events(output)


def test_partial_occlusion_true_on_low_confidence(tmp_path):
    """conf < 0.5 → is_partial_occlusion must be True in ENTRY metadata."""
    events = _make_entry_tracker(tmp_path, conf=0.35)
    entries = [e for e in events if e["event_type"] == "ENTRY"]
    assert len(entries) >= 1, "No ENTRY emitted"
    assert entries[0]["metadata"]["is_partial_occlusion"] is True
    assert entries[0]["confidence"] == pytest.approx(0.35, abs=0.01)


def test_partial_occlusion_false_on_high_confidence(tmp_path):
    """conf >= 0.5 → is_partial_occlusion must be False."""
    events = _make_entry_tracker(tmp_path, conf=0.92)
    entries = [e for e in events if e["event_type"] == "ENTRY"]
    assert len(entries) >= 1
    assert entries[0]["metadata"]["is_partial_occlusion"] is False


def test_low_conf_event_not_suppressed(tmp_path):
    """Low-confidence events (conf=0.31) must still be emitted — not dropped."""
    events = _make_entry_tracker(tmp_path, conf=0.31)
    assert len(events) >= 1, "Low-confidence detection was silently dropped"


# ─────────────────────────────────────────────────────────────────────────────
# GAP 5 — Cross-camera deduplication
# ─────────────────────────────────────────────────────────────────────────────

def _build_tracker(camera_id, registry, tmp_path):
    """Build a StoreTracker sharing the given registry, wired to a temp output."""
    output = str(tmp_path / "events.jsonl")
    with patch.dict(os.environ, {"EVENTS_OUTPUT_PATH": output}):
        import pipeline.emit as em; importlib.reload(em)
        if "pipeline.tracker" in sys.modules: importlib.reload(sys.modules["pipeline.tracker"])
        from pipeline.tracker import StoreTracker

    t = StoreTracker("STORE_X", camera_id, W, H, registry=registry)
    return t, output


def test_cross_camera_reuses_visitor_id_within_window(tmp_path):
    """
    Entry cam registers VIS_X at T=0.
    Floor cam sees someone at T+10s (within 30s window) → must reuse VIS_X.
    """
    with patch.dict(os.environ, {"EVENTS_OUTPUT_PATH": str(tmp_path / "events.jsonl")}):
        import pipeline.emit as em; importlib.reload(em)
        if "pipeline.tracker" in sys.modules: importlib.reload(sys.modules["pipeline.tracker"])
        from pipeline.tracker import StoreTracker, VisitorRegistry

    registry = VisitorRegistry()
    now      = datetime.now(timezone.utc)

    # Entry camera: person crosses tripwire
    entry_t = StoreTracker("STORE_X", "CAM_ENTRY_01", W, H, registry=registry)
    tw       = entry_t.tripwire_y
    entry_t._prev_centroids[1] = float(tw - 10)
    entry_t.update(make_detections([[300,tw+5,500,tw+100]], [0.9]), now, None)

    # Read the visitor_id the entry cam assigned
    all_events  = read_events(str(tmp_path / "events.jsonl"))
    entry_events = [e for e in all_events if e["event_type"] == "ENTRY"]
    assert len(entry_events) == 1
    entry_vid = entry_events[0]["visitor_id"]

    # Floor camera: same registry, detects person 10s later (within window)
    floor_t = StoreTracker("STORE_X", "CAM_FLOOR_01", W, H, registry=registry)
    floor_t.update(
        make_detections([[100,50,300,250]], [0.88]),
        now + timedelta(seconds=10),
        ZONE_MAP,
    )

    all_events  = read_events(str(tmp_path / "events.jsonl"))
    zone_events = [e for e in all_events if e["event_type"] == "ZONE_ENTER"]
    assert len(zone_events) >= 1, "Floor cam emitted no ZONE_ENTER"

    floor_vids = {e["visitor_id"] for e in zone_events}
    assert entry_vid in floor_vids, (
        f"Floor cam used {floor_vids} instead of entry cam's {entry_vid} — double-counting!"
    )


def test_cross_camera_new_visitor_after_window(tmp_path):
    """
    Floor cam detects someone 60s after last entry (> 30s window) → new visitor_id.
    """
    with patch.dict(os.environ, {"EVENTS_OUTPUT_PATH": str(tmp_path / "events.jsonl")}):
        import pipeline.emit as em; importlib.reload(em)
        if "pipeline.tracker" in sys.modules: importlib.reload(sys.modules["pipeline.tracker"])
        from pipeline.tracker import StoreTracker, VisitorRegistry

    registry = VisitorRegistry()
    now      = datetime.now(timezone.utc)

    # Entry cam at T=0
    entry_t = StoreTracker("STORE_X", "CAM_ENTRY_01", W, H, registry=registry)
    tw       = entry_t.tripwire_y
    entry_t._prev_centroids[1] = float(tw - 10)
    entry_t.update(make_detections([[300,tw+5,500,tw+100]], [0.9]), now, None)

    all_events  = read_events(str(tmp_path / "events.jsonl"))
    entry_vid   = next(e["visitor_id"] for e in all_events if e["event_type"] == "ENTRY")

    # Floor cam at T+60s — beyond 30s window
    floor_t = StoreTracker("STORE_X", "CAM_FLOOR_01", W, H, registry=registry)
    floor_t.update(
        make_detections([[100,50,300,250]], [0.88]),
        now + timedelta(seconds=60),
        ZONE_MAP,
    )

    all_events  = read_events(str(tmp_path / "events.jsonl"))
    zone_vids   = {e["visitor_id"] for e in all_events if e["event_type"] == "ZONE_ENTER"}
    assert entry_vid not in zone_vids, (
        "Floor cam reused entry cam's visitor_id after window expired — "
        "should have minted a new one"
    )
