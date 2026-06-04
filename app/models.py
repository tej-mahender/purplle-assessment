"""
Pydantic models — the schema contract for the entire pipeline.
Every event emitted by the detection layer and every API response is validated here.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Event type catalogue (matches problem statement exactly)
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_DWELL = "ZONE_DWELL"
    BILLING_QUEUE_JOIN = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON = "BILLING_QUEUE_ABANDON"
    REENTRY = "REENTRY"


class Severity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Event metadata (nested inside StoreEvent)
# ---------------------------------------------------------------------------

class EventMetadata(BaseModel):
    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: Optional[int] = None
    is_partial_occlusion: Optional[bool] = None
    reid_confidence: Optional[float] = None


# ---------------------------------------------------------------------------
# Core event — emitted by pipeline, ingested by API
# ---------------------------------------------------------------------------

class StoreEvent(BaseModel):
    event_id: UUID
    store_id: str
    camera_id: str
    visitor_id: str
    event_type: EventType
    timestamp: datetime
    zone_id: Optional[str] = None
    dwell_ms: int = 0
    is_staff: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: EventMetadata = Field(default_factory=EventMetadata)

    @field_validator("zone_id")
    @classmethod
    def zone_required_for_zone_events(cls, v, info):
        event_type = info.data.get("event_type")
        zone_events = {
            EventType.ZONE_ENTER, EventType.ZONE_EXIT,
            EventType.ZONE_DWELL, EventType.BILLING_QUEUE_JOIN,
            EventType.BILLING_QUEUE_ABANDON,
        }
        if event_type in zone_events and v is None:
            raise ValueError(f"zone_id required for event_type={event_type}")
        return v

    model_config = {"use_enum_values": True}


# ---------------------------------------------------------------------------
# Ingest request / response
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    events: list[StoreEvent] = Field(max_length=500)


class IngestRejection(BaseModel):
    event_id: str
    reason: str


class IngestResponse(BaseModel):
    accepted: int
    rejected: int
    rejections: list[IngestRejection] = []


# ---------------------------------------------------------------------------
# /metrics response
# ---------------------------------------------------------------------------

class ZoneDwell(BaseModel):
    zone_id: str
    avg_dwell_ms: float
    visit_count: int


class StoreMetrics(BaseModel):
    store_id: str
    as_of: datetime
    unique_visitors: int
    conversion_rate: float
    avg_dwell_by_zone: list[ZoneDwell]
    current_queue_depth: int
    abandonment_rate: float


# ---------------------------------------------------------------------------
# /funnel response
# ---------------------------------------------------------------------------

class FunnelStage(BaseModel):
    stage: str
    count: int
    drop_off_pct: float


class StoreFunnel(BaseModel):
    store_id: str
    as_of: datetime
    stages: list[FunnelStage]


# ---------------------------------------------------------------------------
# /heatmap response
# ---------------------------------------------------------------------------

class ZoneHeatmap(BaseModel):
    zone_id: str
    visit_frequency: int
    avg_dwell_ms: float
    normalised_score: float
    data_confidence: bool


class StoreHeatmap(BaseModel):
    store_id: str
    as_of: datetime
    zones: list[ZoneHeatmap]


# ---------------------------------------------------------------------------
# /anomalies response
# ---------------------------------------------------------------------------

class Anomaly(BaseModel):
    anomaly_type: str
    severity: Severity
    description: str
    suggested_action: str
    detected_at: datetime
    zone_id: Optional[str] = None
    value: Optional[float] = None


class StoreAnomalies(BaseModel):
    store_id: str
    as_of: datetime
    anomalies: list[Anomaly]


# ---------------------------------------------------------------------------
# /health response
# ---------------------------------------------------------------------------

class StoreFeedStatus(BaseModel):
    store_id: str
    last_event_at: Optional[datetime]
    lag_minutes: Optional[float]
    status: str


class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    as_of: datetime
    stores: list[StoreFeedStatus]
