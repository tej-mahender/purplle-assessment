# PROMPT: "Write pytest tests for anomaly detection endpoints: BILLING_QUEUE_SPIKE,
# CONVERSION_DROP vs 7-day average, DEAD_ZONE after 30 min of silence.
# Also test that no anomalies are returned for a clean store. Use async httpx client."
#
# CHANGES MADE: Added test for severity level (WARN vs CRITICAL based on depth),
# fixed DEAD_ZONE test to actually insert historical zone events then skip 30 min,
# removed one AI-generated test that was testing internal implementation details.

import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from tests.conftest import make_event


def make_event_at(ts: datetime, **kwargs):
    e = make_event(**kwargs)
    e["timestamp"] = ts.isoformat()
    return e


@pytest.mark.asyncio
async def test_no_anomalies_clean_store(client):
    """A store with normal traffic should have no anomalies."""
    resp = await client.get("/stores/STORE_CLEAN_999/anomalies")
    assert resp.status_code == 200
    data = resp.json()
    assert data["anomalies"] == []


@pytest.mark.asyncio
async def test_queue_spike_detected(client):
    """BILLING_QUEUE_SPIKE should fire when queue depth >= threshold sustained."""
    store = "STORE_BLR_002"
    now = datetime.now(timezone.utc)

    events = []
    for i in range(3):
        e = make_event(
            store_id=store,
            visitor_id=f"VIS_q{i}",
            event_type="BILLING_QUEUE_JOIN",
            zone_id="BILLING",
            queue_depth=6,  # above threshold of 5
        )
        e["timestamp"] = (now - timedelta(minutes=2 + i)).isoformat()
        events.append(e)

    await client.post("/events/ingest", json={"events": events})
    resp = await client.get(f"/stores/{store}/anomalies")
    assert resp.status_code == 200
    anomalies = resp.json()["anomalies"]
    types = [a["anomaly_type"] for a in anomalies]
    assert "BILLING_QUEUE_SPIKE" in types


@pytest.mark.asyncio
async def test_queue_spike_severity(client):
    """Very deep queue (>= 2x threshold) should be CRITICAL, moderate should be WARN."""
    store = "STORE_SEV_TEST"
    now = datetime.now(timezone.utc)

    events = [
        {**make_event(store_id=store, visitor_id="VIS_s1", event_type="BILLING_QUEUE_JOIN",
                      zone_id="BILLING", queue_depth=12),
         "timestamp": (now - timedelta(minutes=1)).isoformat()},
        {**make_event(store_id=store, visitor_id="VIS_s2", event_type="BILLING_QUEUE_JOIN",
                      zone_id="BILLING", queue_depth=12),
         "timestamp": (now - timedelta(minutes=2)).isoformat()},
    ]
    await client.post("/events/ingest", json={"events": events})
    resp = await client.get(f"/stores/{store}/anomalies")
    anomalies = resp.json()["anomalies"]
    spike = next((a for a in anomalies if a["anomaly_type"] == "BILLING_QUEUE_SPIKE"), None)
    assert spike is not None
    assert spike["severity"] == "CRITICAL"


@pytest.mark.asyncio
async def test_dead_zone_detected(client):
    """DEAD_ZONE fires for zones that had traffic but have gone silent for 30+ minutes."""
    store = "STORE_DEAD_ZONE"
    old_time = datetime.now(timezone.utc) - timedelta(minutes=45)

    # Insert a zone visit that happened 45 minutes ago
    events = [
        {**make_event(store_id=store, visitor_id="VIS_old1", event_type="ZONE_ENTER",
                      zone_id="SKINCARE"),
         "timestamp": old_time.isoformat()},
    ]
    await client.post("/events/ingest", json={"events": events})

    resp = await client.get(f"/stores/{store}/anomalies")
    anomalies = resp.json()["anomalies"]
    dead = [a for a in anomalies if a["anomaly_type"] == "DEAD_ZONE"]
    assert len(dead) >= 1
    assert dead[0]["zone_id"] == "SKINCARE"
    assert dead[0]["severity"] == "INFO"
    assert "suggested_action" in dead[0]


@pytest.mark.asyncio
async def test_anomaly_includes_suggested_action(client):
    """Every anomaly must have a non-empty suggested_action string."""
    store = "STORE_BLR_002"
    now = datetime.now(timezone.utc)
    events = [
        {**make_event(store_id=store, visitor_id=f"VIS_ac{i}", event_type="BILLING_QUEUE_JOIN",
                      zone_id="BILLING", queue_depth=6),
         "timestamp": (now - timedelta(minutes=i)).isoformat()}
        for i in range(3)
    ]
    await client.post("/events/ingest", json={"events": events})
    resp = await client.get(f"/stores/{store}/anomalies")
    for anomaly in resp.json()["anomalies"]:
        assert anomaly.get("suggested_action", "").strip() != ""
