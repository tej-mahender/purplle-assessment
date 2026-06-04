"""
Database setup — SQLAlchemy async with SQLite (swappable to PostgreSQL via DATABASE_URL).
All tables defined here. DB is initialized on API startup.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text,
    UniqueConstraint, event, text
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./store_intelligence.db")

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Events table — one row per emitted event
# ---------------------------------------------------------------------------

class EventRow(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(36), nullable=False, unique=True, index=True)
    store_id = Column(String(64), nullable=False, index=True)
    camera_id = Column(String(64), nullable=False)
    visitor_id = Column(String(64), nullable=False, index=True)
    event_type = Column(String(32), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    zone_id = Column(String(64), nullable=True)
    dwell_ms = Column(Integer, default=0)
    is_staff = Column(Boolean, default=False)
    confidence = Column(Float, nullable=False)
    metadata_json = Column(Text, default="{}")
    ingested_at = Column(DateTime, default=datetime.utcnow)

    def metadata_dict(self):
        return json.loads(self.metadata_json or "{}")


# ---------------------------------------------------------------------------
# POS transactions table
# ---------------------------------------------------------------------------

class POSTransaction(Base):
    __tablename__ = "pos_transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    store_id = Column(String(64), nullable=False, index=True)
    transaction_id = Column(String(64), nullable=False, unique=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    basket_value_inr = Column(Float, nullable=False)


# ---------------------------------------------------------------------------
# Sessions table — computed from events, updated on ingest
# ---------------------------------------------------------------------------

class SessionRow(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, unique=True, index=True)
    store_id = Column(String(64), nullable=False, index=True)
    visitor_id = Column(String(64), nullable=False, index=True)
    entry_time = Column(DateTime, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    is_staff = Column(Boolean, default=False)
    visited_billing = Column(Boolean, default=False)
    converted = Column(Boolean, default=False)
    reentry = Column(Boolean, default=False)


# ---------------------------------------------------------------------------
# DB lifecycle
# ---------------------------------------------------------------------------

async def init_db():
    """Create all tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """FastAPI dependency — yields an async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_db_health() -> bool:
    """Returns True if DB is reachable."""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
