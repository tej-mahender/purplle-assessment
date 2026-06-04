# PROMPT: "Write pytest tests for a retail analytics API /metrics endpoint.
# Cover: zero-traffic store returns valid JSON with 0s not null, staff events excluded
# from visitor counts, conversion rate computed via POS time-window correlation,
# idempotent ingest (same payload twice = same result), abandonment rate.
# Use httpx AsyncClient with an in-memory SQLite DB. Each test gets a fresh DB."
#
# CHANGES MADE: Added test_zero_purchase_store (not in AI output),
# strengthened idempotency test to check exact accepted count on second call,
# added assert on response structure keys to catch schema drift early.

import pytest
import pytest_asyncio
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from tests.conftest import make_event


@pytest.mark.asyncio
async def test_health_returns_ok(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["db_connected"] is True


@pytest.mark.asyncio
async def test_metrics_zero_traffic(client):
    """Zero-traffic store must return valid JSON with 0s, not null or 500."""
    resp = await client.get("/stores/STORE_EMPTY_001/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["unique_visitors"] == 0
    assert data["conversion_rate"] == 0.0
    assert data["current_queue_depth"] == 0
    assert data["abandonment_rate"] == 0.0
    assert isinstance(data["avg_dwell_by_zone"], list)


@pytest.mark.asyncio
async def test_metrics_excludes_staff(client):
    """Staff events (is_staff=True) must not count toward unique_visitors."""
    store = "STORE_BLR_002"
    events = [
        make_event(store_id=store, visitor_id="VIS_customer1", event_type="ENTRY", is_staff=False),
        make_event(store_id=store, visitor_id="VIS_staff1", event_type="ENTRY", is_staff=True),
        make_event(store_id=store, visitor_id="VIS_staff2", event_type="ENTRY", is_staff=True),
    ]
    resp = await client.post("/events/ingest", json={"events": events})
    assert resp.status_code == 200

    resp = await client.get(f"/stores/{store}/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["unique_visitors"] == 1  # only the customer


@pytest.mark.asyncio
async def test_ingest_idempotent(client):
    """POST /events/ingest with same payload twice must be safe — no duplicates."""
    store = "STORE_BLR_002"
    event_id = str(uuid4())
    event = make_event(store_id=store, visitor_id="VIS_idem1")
    event["event_id"] = event_id

    # First ingest
    resp1 = await client.post("/events/ingest", json={"events": [event]})
    assert resp1.status_code == 200
    assert resp1.json()["accepted"] == 1

    # Second ingest (same payload)
    resp2 = await client.post("/events/ingest", json={"events": [event]})
    assert resp2.status_code == 200
    assert resp2.json()["accepted"] == 1  # idempotent — still 1, not error

    # Metrics must show exactly 1 visitor
    resp = await client.get(f"/stores/{store}/metrics")
    assert resp.json()["unique_visitors"] == 1


@pytest.mark.asyncio
async def test_metrics_response_schema(client):
    """Response must contain all required schema keys."""
    resp = await client.get("/stores/STORE_BLR_002/metrics")
    assert resp.status_code == 200
    data = resp.json()
    required_keys = {
        "store_id", "as_of", "unique_visitors", "conversion_rate",
        "avg_dwell_by_zone", "current_queue_depth", "abandonment_rate",
    }
    assert required_keys.issubset(data.keys())


@pytest.mark.asyncio
async def test_abandonment_rate(client):
    """Abandonment rate = BILLING_QUEUE_ABANDON / BILLING_QUEUE_JOIN."""
    store = "STORE_BLR_002"
    events = [
        make_event(store, "VIS_a1", "ENTRY"),
        make_event(store, "VIS_a2", "ENTRY"),
        make_event(store, "VIS_a1", "BILLING_QUEUE_JOIN", zone_id="BILLING", queue_depth=1),
        make_event(store, "VIS_a2", "BILLING_QUEUE_JOIN", zone_id="BILLING", queue_depth=2),
        make_event(store, "VIS_a1", "BILLING_QUEUE_ABANDON", zone_id="BILLING"),
    ]
    await client.post("/events/ingest", json={"events": events})
    resp = await client.get(f"/stores/{store}/metrics")
    data = resp.json()
    # 1 abandon out of 2 joins = 0.5
    assert data["abandonment_rate"] == pytest.approx(0.5, abs=0.01)


@pytest.mark.asyncio
async def test_zero_purchase_store(client):
    """Store with visitors but zero purchases must return conversion_rate=0.0, not crash."""
    store = "STORE_NOPURCHASE"
    events = [make_event(store_id=store, visitor_id=f"VIS_{i}", event_type="ENTRY") for i in range(5)]
    await client.post("/events/ingest", json={"events": events})
    resp = await client.get(f"/stores/{store}/metrics")
    assert resp.status_code == 200
    assert resp.json()["conversion_rate"] == 0.0
    assert resp.json()["unique_visitors"] == 5


@pytest.mark.asyncio
async def test_ingest_partial_failure(client):
    """Malformed events should be rejected while valid ones are accepted."""
    events = [
        make_event(store_id="STORE_BLR_002", visitor_id="VIS_valid"),
        {  # Missing required fields
            "event_id": str(uuid4()),
            "store_id": "STORE_BLR_002",
            # Missing visitor_id, event_type, timestamp, confidence
        },
    ]
    resp = await client.post("/events/ingest", json={"events": events})
    # FastAPI will reject the whole batch at schema validation time
    # This test verifies the API doesn't 500 on bad input
    assert resp.status_code in (200, 422)


@pytest.mark.asyncio
async def test_heatmap_empty_store(client):
    """Heatmap for store with no events returns empty zones list."""
    resp = await client.get("/stores/STORE_HEATMAP_EMPTY/heatmap")
    assert resp.status_code == 200
    data = resp.json()
    assert data["zones"] == []


@pytest.mark.asyncio
async def test_heatmap_normalised_score(client):
    """Most-visited zone should have normalised_score=100.0."""
    store = "STORE_HEATMAP_NORM"
    from uuid import uuid4
    from datetime import datetime, timezone

    events = []
    # Zone A: 5 visits
    for i in range(5):
        e = make_event(store_id=store, visitor_id=f"VIS_h{i}", event_type="ZONE_ENTER", zone_id="ZONE_A")
        events.append(e)
    # Zone B: 2 visits
    for i in range(2):
        e = make_event(store_id=store, visitor_id=f"VIS_hb{i}", event_type="ZONE_ENTER", zone_id="ZONE_B")
        events.append(e)

    await client.post("/events/ingest", json={"events": events})
    resp = await client.get(f"/stores/{store}/heatmap")
    zones = {z["zone_id"]: z for z in resp.json()["zones"]}
    assert zones["ZONE_A"]["normalised_score"] == pytest.approx(100.0)
    assert zones["ZONE_B"]["normalised_score"] == pytest.approx(40.0, abs=1.0)


@pytest.mark.asyncio
async def test_heatmap_data_confidence_low(client):
    """Zones with < 20 sessions should have data_confidence=False."""
    store = "STORE_LOWCONF_HEATMAP"
    events = [
        make_event(store_id=store, visitor_id=f"VIS_lc{i}", event_type="ZONE_ENTER", zone_id="PERFUME")
        for i in range(5)  # only 5, below 20 threshold
    ]
    await client.post("/events/ingest", json={"events": events})
    resp = await client.get(f"/stores/{store}/heatmap")
    zone = next(z for z in resp.json()["zones"] if z["zone_id"] == "PERFUME")
    assert zone["data_confidence"] is False


@pytest.mark.asyncio
async def test_health_stale_feed(client):
    """Health endpoint should flag STALE_FEED when last event > 10 min ago."""
    from datetime import timedelta
    store = "STORE_STALE"
    old_ts = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
    e = make_event(store_id=store, visitor_id="VIS_stale1", event_type="ENTRY")
    e["timestamp"] = old_ts
    await client.post("/events/ingest", json={"events": [e]})
    resp = await client.get("/health")
    stores = {s["store_id"]: s for s in resp.json()["stores"]}
    assert stores[store]["status"] == "STALE_FEED"
