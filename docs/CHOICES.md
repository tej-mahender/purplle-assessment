# CHOICES.md — Three Key Decisions

## Decision 1: Detection Model — YOLOv8n

**Options considered:** YOLOv8n, YOLOv8s, YOLOv8m, RT-DETR-L, MediaPipe Pose

**What the AI suggested:** Claude initially suggested YOLOv8s (small) over nano for better accuracy on partial occlusions, noting that the billing queue scenario — where people are close together and partially behind each other — would benefit from the stronger backbone.

**What I chose and why:** YOLOv8n (nano) for the detection pipeline, with a plan to swap to YOLOv8s if accuracy on billing clips is inadequate.

The reasoning: YOLOv8n runs at ~60fps on a mid-range GPU for 1080p input, meaning it keeps up with the 15fps clips with headroom. YOLOv8s is 2–3x slower at the same resolution. Since the detection runs offline (batch processing clips, not live inference), speed matters less — but developing and iterating against 20-minute clips is much faster with nano, and the detection confidence threshold at 0.3 (not 0.5) compensates for some accuracy loss.

For the billing queue specifically, partial occlusion is handled by *not dropping* low-confidence detections — every detection above 0.3 is emitted with its confidence score in the event. The API consumer decides how to weight low-confidence events. This is the right place to make that decision, not in the detection layer.

RT-DETR was appealing (transformer-based, better at dense scenes) but the `ultralytics` integration adds a dependency not worth it for this challenge scope. MediaPipe was ruled out because it's optimised for single-person scenarios, not crowd detection.

---

## Decision 2: Event Schema Design

**Options considered:**
- Flat schema (all fields at top level)
- Typed schema per event_type (separate schemas for ENTRY vs ZONE_DWELL etc.)
- Hybrid: shared fields + `metadata` blob

**What the AI suggested:** Claude suggested a fully typed schema with separate Pydantic models per event type, arguing it gives better validation and clearer contracts. It generated six separate models.

**What I chose and why:** The hybrid approach from the problem spec — shared required fields + a flexible `metadata` JSON blob.

I disagreed with the typed-per-event approach for two reasons:

1. **Ingest simplicity**: The API receives batches of mixed event types. A flat schema with one validator is much simpler to ingest and store (one DB table, one Pydantic model) than dispatching to six different validators.

2. **Schema evolution**: The `metadata` blob lets the detection layer add fields (e.g. `reid_confidence`, `is_partial_occlusion`) without a DB migration. These fields are only needed for debugging and don't appear in API responses — they don't need first-class schema treatment.

The one concession to the typed approach: `zone_id` is validated as required for zone-related event types via a Pydantic field_validator. This catches the most common schema error (forgetting zone_id on ZONE_ENTER) without requiring separate models.

---

## Decision 3: Storage Engine — SQLite with async SQLAlchemy

**Options considered:** SQLite, PostgreSQL, Redis + PostgreSQL, DuckDB

**What the AI suggested:** PostgreSQL with a Redis cache for the `/metrics` endpoint, arguing that at 40 live stores the read load on `/metrics` would be high.

**What I chose and why:** SQLite with async SQLAlchemy (`aiosqlite`), designed to swap to PostgreSQL via `DATABASE_URL` environment variable.

For the challenge scope, SQLite is the right choice: zero infrastructure, `docker compose up` works on any machine without a Postgres install, and the test suite uses the same DB engine as production (no fixture-level mocking). The async SQLAlchemy setup means the connection string is the only thing that changes when switching to Postgres.

I specifically rejected the Redis cache suggestion because: (a) it adds operational complexity (another container, cache invalidation logic), (b) the `/metrics` endpoint is specified as "real-time — not cached from yesterday" which implies the evaluators will test that fresh events are immediately reflected, and (c) SQLite with proper indexes on (store_id, timestamp) is fast enough for the event volumes in these 20-minute clips.

The one place the AI was right: I added an index on `(store_id, event_type, timestamp)` to the events table after it pointed out that the conversion rate query does a full scan of billing-zone events. That index reduced the query from O(n) over all events to O(log n) over the relevant subset.
