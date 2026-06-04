# DESIGN.md — Store Intelligence API

## Architecture Overview

The system is a three-stage pipeline: raw CCTV video → structured events → queryable REST API.

```
CCTV Clips (.mp4)
       │
       ▼
┌──────────────────┐
│  Detection Layer │  YOLOv8n + ByteTrack (via supervision)
│  pipeline/       │  Per-frame person detection → track assignment
│  detect.py       │  Tripwire crossing → ENTRY/EXIT direction
│  tracker.py      │  Zone mapping → ZONE_ENTER/ZONE_DWELL
│  emit.py         │  Re-ID → REENTRY detection
└──────┬───────────┘
       │ events.jsonl (JSONL stream)
       ▼
┌──────────────────┐
│  Intelligence    │  FastAPI + SQLAlchemy async
│  API             │  SQLite (dev) / PostgreSQL (prod)
│  app/            │  Idempotent ingest by event_id
└──────┬───────────┘
       │ HTTP REST
       ▼
  /metrics  /funnel  /heatmap  /anomalies  /health
```

### Detection Layer

**YOLOv8n** runs on every frame at 15fps, filtering to class 0 (person). Detections are passed to **ByteTrack** (via the `supervision` library) which assigns stable track IDs across frames.

Direction detection uses a **virtual tripwire**: a horizontal line drawn at 65% of frame height. A centroid crossing downward (away from the door) emits ENTRY; crossing upward emits EXIT. This position was chosen after manually inspecting the entry camera perspective — the threshold sits just past the door frame itself.

Zone mapping uses a **point-in-polygon test** (ray-casting) against zone polygons from `store_layout.json`. This is run on floor camera frames only — entry camera is exclusively used for entry/exit counting to avoid cross-camera double-counting.

### Event Stream

All events pass through `emit.py` which:
1. Validates required fields (fails loudly — not silently)
2. Assigns a uuid4 event_id
3. Writes to `events.jsonl` (append-only JSONL)

The schema was kept minimal: only fields the API actually queries are stored. `metadata` is a flexible JSON blob for event-type-specific data (queue_depth, session_seq).

### Intelligence API

**FastAPI** with async SQLAlchemy and SQLite. The DB has three tables:
- `events` — one row per event, indexed on (store_id, timestamp, visitor_id)
- `sessions` — computed session state, updated on every ingest
- `pos_transactions` — seeded from pos_transactions.csv on startup

Every API endpoint is **real-time**: queries hit the DB directly, no caching layer. This means accuracy over speed — acceptable at 40-store scale, revisitable if latency becomes an issue.

**POS correlation** is done by time-window join: a visitor who was in a billing zone in the 5-minute window before a transaction timestamp counts as converted. This is the only available signal given no customer_id in the POS data.

**Session deduplication**: the funnel unit is `visitor_id`, not `event_id`. A visitor who re-enters on the same day counts as one unique visitor in conversion math.

### Structured Logging

Every HTTP request emits a structured log line with: `trace_id`, `store_id`, `method`, `path`, `status_code`, `latency_ms`. Ingestion additionally logs `event_count`, `accepted`, `rejected`. Using `structlog` for consistent JSON output.

---

## AI-Assisted Decisions

### 1. ByteTrack vs DeepSORT for tracking

I asked Claude to compare ByteTrack, DeepSORT, and StrongSORT for this use case (15fps, 1080p, retail environment with partial occlusion). The AI recommended ByteTrack on grounds of speed and its robustness to low-confidence detections — it uses a two-stage association that keeps "lost" tracks alive for a few frames before discarding them, which helps in partial occlusion cases.

I agreed with this recommendation but added one override: ByteTrack's default `lost_track_buffer` is 30 frames (2 seconds at 15fps). For retail, where customers can stand still for 30+ seconds in front of a display, this would drop tracks prematurely. I increased the buffer to 150 frames (10 seconds).

### 2. POS correlation strategy

The AI initially suggested building a probabilistic model for visitor-to-transaction matching using dwell time distributions. I overrode this: the 5-minute time-window approach is specified in the problem statement and is both simpler and more auditable. The probabilistic model would require training data we don't have and would be harder to explain to a business stakeholder asking "why is my conversion rate X?"

### 3. Session storage: compute at ingest vs compute at query time

Claude suggested computing all session metrics at query time from raw events (simpler ingestion, no denormalisation). I chose a hybrid: sessions table updated at ingest, complex metrics (conversion rate, funnel) computed at query time from both sessions and events.

Reasoning: session state (entry_time, exit_time, visited_billing) is expensive to recompute on every `/metrics` call if there are many events. But I don't pre-compute the final metrics because the time window (today) changes and caching would require invalidation logic. The hybrid gives fast session lookups without stale metric risk.
