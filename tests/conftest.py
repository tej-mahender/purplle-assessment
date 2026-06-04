"""
Shared pytest fixtures for all tests.
"""

import asyncio
import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.database import Base, engine, AsyncSessionLocal


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def db_session():
    """Fresh in-memory DB for each test."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with AsyncSessionLocal() as session:
        yield session
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def client():
    """Async test client with fresh DB."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


def make_event(
    store_id="STORE_BLR_002",
    visitor_id="VIS_aabbcc",
    event_type="ENTRY",
    zone_id=None,
    is_staff=False,
    confidence=0.91,
    dwell_ms=0,
    timestamp=None,
    queue_depth=None,
):
    """Helper to create a valid event dict for testing."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "event_id": str(uuid4()),
        "store_id": store_id,
        "camera_id": "CAM_ENTRY_01",
        "visitor_id": visitor_id,
        "event_type": event_type,
        "timestamp": timestamp,
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": is_staff,
        "confidence": confidence,
        "metadata": {
            "queue_depth": queue_depth,
            "sku_zone": zone_id,
            "session_seq": 1,
            "is_partial_occlusion": False,
            "reid_confidence": None,
        },
    }
