# Store Intelligence API — Apex Retail / Purplle Challenge

End-to-end pipeline: CCTV footage → person detection → event stream → live analytics API.

---

## Architecture

```
[CCTV .mp4 clips]
       │
       ▼  (run on Google Colab T4 GPU — free)
┌─────────────────────────────────────┐
│  Detection Pipeline                 │
│  YOLOv8n + ByteTrack                │
│  Tripwire → ENTRY/EXIT direction    │
│  Zone mapping → ZONE_ENTER/DWELL    │
│  Re-ID → REENTRY detection          │
│  Staff heuristic → is_staff flag    │
└────────────────┬────────────────────┘
                 │ events.jsonl (download from Colab)
                 ▼
┌─────────────────────────────────────┐
│  Intelligence API  (run locally)    │
│  FastAPI + SQLite + Docker          │
│  POST /events/ingest  (idempotent)  │
│  GET  /stores/{id}/metrics          │
│  GET  /stores/{id}/funnel           │
│  GET  /stores/{id}/heatmap          │
│  GET  /stores/{id}/anomalies        │
│  GET  /health                       │
└────────────────┬────────────────────┘
                 │
                 ▼
      Terminal Live Dashboard (rich)
```

---

## Quick Start (5 commands)

```bash
git clone <your-repo-url> store-intelligence
cd store-intelligence
cp .env.example .env
docker compose up --build -d
curl http://localhost:8000/health
```

Then run detection on Colab (see below) and ingest the output:

```bash
python pipeline/ingest_events.py --events ./data/events.jsonl
```

---

## Full Setup Guide

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Docker Desktop | 24+ | Runs the API |
| Python | 3.10+ | Local scripts (ingest, dashboard) |
| Google Account | — | Free Colab GPU for detection |
| ffmpeg (optional) | any | Clip downsampling for local CPU |

---

### Part 1 — Place your data files

```
store-intelligence/
└── data/
    ├── clips/
    │   ├── ST1008_CAM_ENTRY_01.mp4
    │   ├── ST1008_CAM_FLOOR_01.mp4
    │   ├── ST1008_CAM_BILLING_01.mp4
    │   └── ... (other stores)
    ├── store_layout.json
    └── pos_transactions.csv
```

> The `data/` folder is git-ignored. Put your dataset ZIP contents here.

---

### Part 2 — Start the API (Docker)

```bash
# Build and start
docker compose up --build -d

# Verify it's running
curl http://localhost:8000/health
# Expected: {"status":"ok","db_connected":true,...}

# Watch live logs
docker compose logs -f api
```

API is now at **http://localhost:8000**
Swagger UI (interactive docs): **http://localhost:8000/docs**

---

### Part 3 — Run Detection on Google Colab (GPU)

Your machine (i5 + integrated GPU) would take 15–30 hours to process all clips.
Google Colab T4 GPU does it in **~10–20 minutes total**. It's free.

#### Step-by-step Colab setup:

**3.1 — Upload your data to Google Drive**

Create this folder structure in your Google Drive:
```
MyDrive/
└── purplle-challenge/
    ├── clips/
    │   └── *.mp4  ← all your clips here
    ├── store_layout.json
    └── pos_transactions.csv
```

**3.2 — Open the notebook**

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Upload `colab_detection.ipynb` from this project

**3.3 — Enable GPU**

1. Click **Runtime → Change runtime type**
2. Set Hardware accelerator to **T4 GPU**
3. Click **Save**

**3.4 — Run all cells in order**

| Cell | What it does | Time |
|------|-------------|------|
| Step 1 | Verify GPU | 5s |
| Step 2 | Install ultralytics + supervision | 60s |
| Step 3 | Mount Google Drive | 10s |
| Step 4 | Write pipeline code to Colab | 5s |
| Step 5 | **Run detection on all clips** | 10–20 min |
| Step 6 | Validate events.jsonl schema | 10s |
| Step 7 | Download events.jsonl | instant |

> **Edit Step 5 before running**: set `STORE_IDS` to your actual store IDs,
> e.g. `['ST1008', 'STORE_MUM_001']`

**3.5 — Download events.jsonl**

After Step 7, your browser will auto-download `events.jsonl`.
Place it at: `store-intelligence/data/events.jsonl`

---

### Part 4 — Ingest events into the API

```bash
# Install ingest script dependencies (lightweight, no torch needed)
pip install requests typer

# Ingest all events
python pipeline/ingest_events.py \
  --events ./data/events.jsonl \
  --api http://localhost:8000

# Expected output:
# Loaded 12483 events (0 parse errors)
# API health: ok
# Ingesting 12483 events in 25 batches...
#   Batch 1/25: ✓ 500 accepted
#   ...
# === Ingest complete in 4.2s ===
#   Accepted: 12483
#   Rejected: 0
```

---

### Part 5 — Query the API

```bash
STORE=ST1008

# Live metrics
curl http://localhost:8000/stores/$STORE/metrics | python -m json.tool

# Conversion funnel
curl http://localhost:8000/stores/$STORE/funnel | python -m json.tool

# Zone heatmap
curl http://localhost:8000/stores/$STORE/heatmap | python -m json.tool

# Active anomalies
curl http://localhost:8000/stores/$STORE/anomalies | python -m json.tool

# Health + feed freshness
curl http://localhost:8000/health | python -m json.tool
```

---

### Part 6 — Live Dashboard (Bonus +10 pts)

```bash
# Install dashboard dependencies
pip install rich requests typer

# Replay events.jsonl in simulated real-time
# (replays at 10x speed, shows metrics updating live)
python dashboard/live.py \
  --store ST1008 \
  --events ./data/events.jsonl \
  --speed 10
```

The terminal dashboard shows:
- Live visitor count (updates every 2s)
- Conversion rate
- Queue depth
- Zone dwell table (top 5 zones)
- Active anomalies with severity

---

### Part 7 — Run Tests

```bash
pip install -r requirements.txt

# All tests with coverage
pytest tests/ -v --cov=app --cov-report=term-missing

# Expected: 38 passed, ~79% coverage
```

---

## API Reference

### POST /events/ingest

Accepts batches of up to 500 events. Idempotent by `event_id`.

```bash
curl -X POST http://localhost:8000/events/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "events": [{
      "event_id": "550e8400-e29b-41d4-a716-446655440000",
      "store_id": "ST1008",
      "camera_id": "CAM_ENTRY_01",
      "visitor_id": "VIS_c8a2f1",
      "event_type": "ENTRY",
      "timestamp": "2026-03-03T14:22:10Z",
      "zone_id": null,
      "dwell_ms": 0,
      "is_staff": false,
      "confidence": 0.91,
      "metadata": {"queue_depth": null, "sku_zone": null, "session_seq": 1}
    }]
  }'
```

Response:
```json
{"accepted": 1, "rejected": 0, "rejections": []}
```

### GET /stores/{store_id}/metrics

```json
{
  "store_id": "ST1008",
  "as_of": "2026-03-03T15:00:00Z",
  "unique_visitors": 47,
  "conversion_rate": 0.3191,
  "avg_dwell_by_zone": [
    {"zone_id": "SKINCARE", "avg_dwell_ms": 42300, "visit_count": 23}
  ],
  "current_queue_depth": 3,
  "abandonment_rate": 0.12
}
```

### GET /stores/{store_id}/funnel

```json
{
  "store_id": "ST1008",
  "as_of": "2026-03-03T15:00:00Z",
  "stages": [
    {"stage": "Entry",         "count": 47, "drop_off_pct": 0.0},
    {"stage": "Zone Visit",    "count": 39, "drop_off_pct": 17.0},
    {"stage": "Billing Queue", "count": 18, "drop_off_pct": 53.8},
    {"stage": "Purchase",      "count": 15, "drop_off_pct": 16.7}
  ]
}
```

### GET /stores/{store_id}/anomalies

```json
{
  "anomalies": [
    {
      "anomaly_type": "BILLING_QUEUE_SPIKE",
      "severity": "WARN",
      "description": "Billing queue depth reached 7 over the last 5 minutes",
      "suggested_action": "Open an additional billing counter or redirect to express checkout",
      "detected_at": "2026-03-03T14:55:00Z",
      "zone_id": "BILLING",
      "value": 7.0
    }
  ]
}
```

---

## Project Structure

```
store-intelligence/
├── pipeline/
│   ├── detect.py          # YOLOv8 + ByteTrack detection (local CPU)
│   ├── tracker.py         # StoreTracker: tripwire, re-ID, zones
│   ├── emit.py            # Event schema + JSONL emission
│   ├── ingest_events.py   # Batch ingest events.jsonl into API
│   └── run.sh             # Local end-to-end (with ffmpeg downsampling)
├── app/
│   ├── main.py            # FastAPI + middleware + startup
│   ├── models.py          # Pydantic schema (API contract)
│   ├── database.py        # SQLAlchemy tables
│   ├── ingestion.py       # POST /events/ingest
│   ├── metrics.py         # GET /stores/{id}/metrics
│   ├── funnel.py          # GET /stores/{id}/funnel
│   ├── heatmap.py         # GET /stores/{id}/heatmap
│   ├── anomalies.py       # GET /stores/{id}/anomalies
│   └── health.py          # GET /health
├── tests/
│   ├── conftest.py        # Shared fixtures
│   ├── test_pipeline.py   # Event schema + emission tests
│   ├── test_metrics.py    # API endpoint tests
│   ├── test_funnel.py     # Funnel + heatmap tests
│   └── test_anomalies.py  # Anomaly detection tests
├── dashboard/
│   └── live.py            # Terminal live dashboard (rich)
├── docs/
│   ├── DESIGN.md          # Architecture + AI-assisted decisions
│   └── CHOICES.md         # 3 decisions with full reasoning
├── colab_detection.ipynb  # ← Google Colab GPU detection notebook
├── docker-compose.yml
├── Dockerfile
├── requirements.txt       # API + test deps (no torch)
├── requirements-pipeline.txt  # Detection deps (torch, ultralytics)
└── .env.example
```

---

## Troubleshooting

### API won't start
```bash
docker compose down -v && docker compose up --build
```

### events.jsonl is empty after Colab run
Check that `STORE_IDS` in Step 5 matches your actual clip filenames.
Print `clips` variable in Step 5 to see what was found.

### Metrics returns 0 visitors after ingest
Verify ingest worked: `curl http://localhost:8000/health` should show your store with `status: OK`
and a recent `last_event_at` timestamp.

### Colab disconnects mid-run
The events.jsonl is saved to Google Drive on every `emit_event()` call (append mode),
so partial results are preserved. Re-run Step 5 — it will append to the existing file.
To start fresh, delete the Drive file first.

### Port 8000 already in use
```bash
# Change port in docker-compose.yml: "8001:8000"
# Then use http://localhost:8001 everywhere
```

---

## Design Decisions

- [DESIGN.md](docs/DESIGN.md) — Architecture + 3 AI-assisted decisions
- [CHOICES.md](docs/CHOICES.md) — Model selection, schema design, storage choice
