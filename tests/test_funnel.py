# PROMPT: "Write pytest tests for a retail store conversion funnel endpoint.
# Test: funnel stages in correct order, drop-off percentages computed correctly,
# re-entries don't double-count unique visitors in funnel,
# empty store returns 4 stages all with count=0, stage unit is session not raw events."
#
# CHANGES MADE: Added test_reentry_not_double_counted which AI omitted,
# fixed drop_off_pct assertion to use pytest.approx, added test for stage ordering.

import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from tests.conftest import make_event


def ts(minutes_ago=0):
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


@pytest.mark.asyncio
async def test_funnel_empty_store(client):
    """Empty store returns 4 funnel stages all with count=0."""
    resp = await client.get("/stores/STORE_EMPTY_FUNNEL/funnel")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["stages"]) == 4
    for stage in data["stages"]:
        assert stage["count"] == 0
    assert data["stages"][0]["drop_off_pct"] == 0.0


@pytest.mark.asyncio
async def test_funnel_stage_names_and_order(client):
    """Stages must be in order: Entry → Zone Visit → Billing Queue → Purchase."""
    resp = await client.get("/stores/STORE_ORDER_TEST/funnel")
    stages = resp.json()["stages"]
    names = [s["stage"] for s in stages]
    assert names == ["Entry", "Zone Visit", "Billing Queue", "Purchase"]


@pytest.mark.asyncio
async def test_funnel_drop_off_calculated(client):
    """drop_off_pct should reflect real loss between stages."""
    store = "STORE_FUNNEL_CALC"
    events = [
        # 4 visitors enter
        make_event(store, "VIS_f1", "ENTRY"),
        make_event(store, "VIS_f2", "ENTRY"),
        make_event(store, "VIS_f3", "ENTRY"),
        make_event(store, "VIS_f4", "ENTRY"),
        # 2 visit a zone
        make_event(store, "VIS_f1", "ZONE_ENTER", zone_id="SKINCARE"),
        make_event(store, "VIS_f2", "ZONE_ENTER", zone_id="SKINCARE"),
    ]
    await client.post("/events/ingest", json={"events": events})
    resp = await client.get(f"/stores/{store}/funnel")
    stages = {s["stage"]: s for s in resp.json()["stages"]}

    assert stages["Entry"]["count"] == 4
    assert stages["Zone Visit"]["count"] == 2
    # 4 → 2 = 50% drop-off
    assert stages["Zone Visit"]["drop_off_pct"] == pytest.approx(50.0, abs=0.1)


@pytest.mark.asyncio
async def test_funnel_reentry_not_double_counted(client):
    """A visitor who re-enters must count as 1 unique visitor in funnel, not 2."""
    store = "STORE_REENTRY_FUNNEL"
    vid = "VIS_reentry1"
    events = [
        {**make_event(store, vid, "ENTRY"), "timestamp": ts(60)},
        {**make_event(store, vid, "EXIT"), "timestamp": ts(50)},
        # Re-enters 30 min later
        {**make_event(store, vid, "REENTRY"), "timestamp": ts(20)},
    ]
    await client.post("/events/ingest", json={"events": events})
    resp = await client.get(f"/stores/{store}/funnel")
    stages = {s["stage"]: s for s in resp.json()["stages"]}
    # Only 1 unique visitor despite 2 entries
    assert stages["Entry"]["count"] == 1


@pytest.mark.asyncio
async def test_funnel_excludes_staff(client):
    """Staff must not appear in funnel visitor counts."""
    store = "STORE_STAFF_FUNNEL"
    events = [
        make_event(store, "VIS_cust1", "ENTRY", is_staff=False),
        make_event(store, "VIS_staff1", "ENTRY", is_staff=True),
        make_event(store, "VIS_staff2", "ENTRY", is_staff=True),
    ]
    await client.post("/events/ingest", json={"events": events})
    resp = await client.get(f"/stores/{store}/funnel")
    stages = {s["stage"]: s for s in resp.json()["stages"]}
    assert stages["Entry"]["count"] == 1


@pytest.mark.asyncio
async def test_funnel_full_conversion_path(client):
    """Visitor who enters → visits zone → joins billing → should appear in all stages."""
    store = "STORE_FULL_PATH"
    vid = "VIS_converted1"
    events = [
        make_event(store, vid, "ENTRY"),
        make_event(store, vid, "ZONE_ENTER", zone_id="LIPSTICK"),
        make_event(store, vid, "BILLING_QUEUE_JOIN", zone_id="BILLING", queue_depth=1),
    ]
    await client.post("/events/ingest", json={"events": events})
    resp = await client.get(f"/stores/{store}/funnel")
    stages = {s["stage"]: s for s in resp.json()["stages"]}
    assert stages["Entry"]["count"] == 1
    assert stages["Zone Visit"]["count"] == 1
    assert stages["Billing Queue"]["count"] == 1


@pytest.mark.asyncio
async def test_metrics_with_zone_dwell(client):
    """avg_dwell_by_zone should populate when ZONE_DWELL events are present."""
    store = "STORE_DWELL_TEST"
    events = [
        make_event(store, "VIS_d1", "ENTRY"),
        make_event(store, "VIS_d1", "ZONE_ENTER", zone_id="HAIRCARE"),
        {**make_event(store, "VIS_d1", "ZONE_DWELL", zone_id="HAIRCARE", dwell_ms=35000),
         "timestamp": (datetime.now(timezone.utc)).isoformat()},
    ]
    await client.post("/events/ingest", json={"events": events})
    resp = await client.get(f"/stores/{store}/metrics")
    assert resp.status_code == 200
    zones = {z["zone_id"]: z for z in resp.json()["avg_dwell_by_zone"]}
    assert "HAIRCARE" in zones
    assert zones["HAIRCARE"]["avg_dwell_ms"] > 0


@pytest.mark.asyncio
async def test_anomalies_conversion_drop(client):
    """CONVERSION_DROP fires when today's rate is 20%+ below 7-day average."""
    # We can't easily seed 7-day history in unit tests without time travel,
    # so test the happy-path: no anomaly when no 7-day baseline exists.
    store = "STORE_NO_BASELINE"
    events = [make_event(store, f"VIS_cb{i}", "ENTRY") for i in range(3)]
    await client.post("/events/ingest", json={"events": events})
    resp = await client.get(f"/stores/{store}/anomalies")
    assert resp.status_code == 200
    # No baseline → no CONVERSION_DROP anomaly (not enough data)
    types = [a["anomaly_type"] for a in resp.json()["anomalies"]]
    assert "CONVERSION_DROP" not in types


@pytest.mark.asyncio
async def test_funnel_heatmap_data_confidence_high(client):
    """Zone with 20+ visits should have data_confidence=True."""
    store = "STORE_HIGH_CONF"
    events = [
        make_event(store_id=store, visitor_id=f"VIS_hc{i}",
                   event_type="ZONE_ENTER", zone_id="FOUNDATION")
        for i in range(22)
    ]
    await client.post("/events/ingest", json={"events": events})
    resp = await client.get(f"/stores/{store}/heatmap")
    zone = next(z for z in resp.json()["zones"] if z["zone_id"] == "FOUNDATION")
    assert zone["data_confidence"] is True


@pytest.mark.asyncio
async def test_ingest_up_to_500_events(client):
    """API should accept a full batch of 500 events."""
    from uuid import uuid4
    store = "STORE_BATCH"
    events = [
        make_event(store_id=store, visitor_id=f"VIS_{i:04d}", event_type="ENTRY")
        for i in range(500)
    ]
    resp = await client.post("/events/ingest", json={"events": events})
    assert resp.status_code == 200
    data = resp.json()
    assert data["accepted"] == 500
    assert data["rejected"] == 0


# ── Direct async unit tests (bypass ASGI for coverage) ──────────────────────

@pytest.mark.asyncio
async def test_metrics_direct(db_session):
    """Call get_metrics function directly for coverage."""
    from app.metrics import get_metrics
    from app.database import EventRow
    from datetime import datetime, timezone
    import json

    # Insert events directly
    now = datetime.now(timezone.utc)
    for i in range(3):
        db_session.add(EventRow(
            event_id=f"direct-test-{i}",
            store_id="STORE_DIRECT",
            camera_id="CAM_ENTRY_01",
            visitor_id=f"VIS_dir{i}",
            event_type="ENTRY",
            timestamp=now,
            is_staff=False,
            confidence=0.9,
            metadata_json="{}",
        ))
    await db_session.commit()

    result = await get_metrics("STORE_DIRECT", db_session)
    assert result.unique_visitors == 3
    assert result.conversion_rate == 0.0
    assert result.store_id == "STORE_DIRECT"


@pytest.mark.asyncio
async def test_funnel_direct(db_session):
    """Call get_funnel function directly for coverage."""
    from app.funnel import get_funnel
    from app.database import EventRow
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    for i in range(2):
        db_session.add(EventRow(
            event_id=f"funnel-direct-{i}",
            store_id="STORE_FUNNEL_D",
            camera_id="CAM_ENTRY_01",
            visitor_id=f"VIS_fd{i}",
            event_type="ENTRY",
            timestamp=now,
            is_staff=False,
            confidence=0.9,
            metadata_json="{}",
        ))
    await db_session.commit()

    result = await get_funnel("STORE_FUNNEL_D", db_session)
    assert result.stages[0].stage == "Entry"
    assert result.stages[0].count == 2


@pytest.mark.asyncio
async def test_anomalies_direct(db_session):
    """Call get_anomalies function directly for coverage."""
    from app.anomalies import get_anomalies
    result = await get_anomalies("STORE_ANOM_D", db_session)
    assert result.anomalies == []


@pytest.mark.asyncio
async def test_health_direct(db_session):
    """Call health function directly for coverage."""
    from app.health import health
    from app.database import EventRow
    from datetime import datetime, timezone, timedelta

    # Insert a recent event
    db_session.add(EventRow(
        event_id="health-direct-1",
        store_id="STORE_HEALTH_D",
        camera_id="CAM_ENTRY_01",
        visitor_id="VIS_hd1",
        event_type="ENTRY",
        timestamp=datetime.now(timezone.utc),
        is_staff=False,
        confidence=0.9,
        metadata_json="{}",
    ))
    await db_session.commit()

    from fastapi import Request
    result = await health(db_session)
    assert result.status == "ok"
    assert result.db_connected is True
    store_status = next((s for s in result.stores if s.store_id == "STORE_HEALTH_D"), None)
    assert store_status is not None
    assert store_status.status == "OK"
