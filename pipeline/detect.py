"""
detect.py — Main detection script.
Usage: python pipeline/detect.py --store STORE_BLR_002 --clips-dir ./data/clips

Processes all 3 camera clips for a store.
Shares a VisitorRegistry across clips so cross-camera deduplication works.
After all clips are processed, runs a post-processing pass to emit
BILLING_QUEUE_ABANDON events (requires POS correlation).           [GAP 3]
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import typer

try:
    import cv2
    import numpy as np
    import supervision as sv
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    cv2 = sv = YOLO = np = None  # type: ignore

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.tracker import StoreTracker, VisitorRegistry
from pipeline.emit import emit_event, make_visitor_id

app_cli = typer.Typer()

YOLO_MODEL             = os.getenv("YOLO_MODEL", "yolov8n.pt")
CONFIDENCE             = float(os.getenv("DETECTION_CONFIDENCE", "0.3"))
DWELL_INTERVAL_SECONDS = int(os.getenv("ZONE_DWELL_INTERVAL_SECONDS", "30"))
SAMPLE_EVERY_N         = int(os.getenv("SAMPLE_EVERY_N_FRAMES", "1"))  # 1 = every frame
FPS                    = 15
POS_WINDOW_MINUTES     = 5    # abandon detection: no POS within this window after billing exit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_store_layout(layout_path: str, store_id: str) -> dict:
    with open(layout_path) as f:
        layout = json.load(f)
    return layout.get(store_id, layout)


def get_clip_paths(clips_dir: str, store_id: str) -> dict[str, str]:
    """Returns {camera_id: clip_path} sorted by camera type: ENTRY first.
    
    Supports two naming conventions:
    1. Prefixed:   ST1008_CAM_ENTRY_01.mp4
    2. Bare name:  entry.mp4 / floor.mp4 / billing.mp4 / 1.mp4 / 2.mp4
    """
    clips = {}
    clips_path = Path(clips_dir)

    # Convention 1: store_id prefix
    for clip in clips_path.glob(f"{store_id}*.mp4"):
        name = clip.stem.replace(store_id + "_", "").replace(store_id, "")
        camera_id = name if name.startswith("CAM_") else f"CAM_{name.upper()}"
        clips[camera_id] = str(clip)

    # Convention 2: any .mp4 — auto-assign camera_id from filename keywords
    if not clips:
        for clip in sorted(clips_path.glob("*.mp4")):
            stem_upper = clip.stem.upper()
            if "ENTRY" in stem_upper or clip.stem in ("1","01"):
                cam = "CAM_ENTRY_01"
            elif "BILLING" in stem_upper or "CASH" in stem_upper or clip.stem in ("3","03"):
                cam = "CAM_BILLING_01"
            else:
                cam = "CAM_FLOOR_01"
            # avoid overwriting if already assigned
            if cam in clips:
                cam = f"CAM_{clip.stem.upper()}"
            clips[cam] = str(clip)

    # Sort: ENTRY first (so registry is populated before floor/billing cams run)
    def sort_key(cam_id: str) -> int:
        if "ENTRY" in cam_id:   return 0
        if "FLOOR" in cam_id:   return 1
        if "BILLING" in cam_id: return 2
        return 3

    return dict(sorted(clips.items(), key=lambda x: sort_key(x[0])))


def frame_to_timestamp(frame_idx: int, clip_start: datetime, fps: int = FPS) -> datetime:
    return clip_start + timedelta(seconds=frame_idx / fps)


# ---------------------------------------------------------------------------
# Process a single clip
# ---------------------------------------------------------------------------

def process_clip(
    clip_path: str,
    store_id: str,
    camera_id: str,
    clip_start: datetime,
    zone_map: dict,
    model: YOLO,
    registry: VisitorRegistry,
    debug: bool = False,
    sample_every: int = 1,
) -> tuple[int, list[dict]]:
    """
    Process one camera clip.
    Returns (event_count, billing_exit_records) where billing_exit_records
    is used by the abandon post-processor.
    """
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        typer.echo(f"  ERROR: Cannot open {clip_path}", err=True)
        return 0, []

    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    typer.echo(f"  {camera_id}: {total_frames} frames  {width}x{height}  sample=1/{sample_every}")

    # Pass shared registry so cross-camera dedup works
    tracker    = StoreTracker(store_id, camera_id, width, height, registry=registry)
    event_count = 0
    frame_idx   = 0

    # Zone dwell tracking: visitor_id → {zone_id: last_dwell_emit_time}
    dwell_tracker: dict[str, dict[str, datetime]] = {}
    # billing exit records for abandon post-processing
    billing_exits: list[dict] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Frame sampling for CPU speed (set SAMPLE_EVERY_N_FRAMES > 1)
        if frame_idx % sample_every != 0:
            frame_idx += 1
            continue

        frame_time = frame_to_timestamp(frame_idx, clip_start)

        results    = model(frame, classes=[0], conf=CONFIDENCE, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        if len(detections) > 0:
            detections = detections[detections.class_id == 0]

        # Zone map only for floor/billing cameras
        active_zone_map = zone_map if (
            "FLOOR" in camera_id.upper() or "BILLING" in camera_id.upper()
        ) else None

        events = tracker.update(
            detections=detections,
            frame_time=frame_time,
            zone_map=active_zone_map,
        )
        event_count += len(events)

        # ZONE_DWELL — emit every 30s of continuous zone presence
        for vid, session in tracker._sessions.items():
            if not session.zone_history:
                continue
            current_zone = session.zone_history[-1]
            dwell_tracker.setdefault(vid, {})
            last = dwell_tracker[vid].get(current_zone)
            if last is None:
                dwell_tracker[vid][current_zone] = frame_time
            elif (frame_time - last).total_seconds() >= DWELL_INTERVAL_SECONDS:
                dwell_ms = int((frame_time - last).total_seconds() * 1000)
                session.session_seq += 1
                emit_event(
                    store_id=store_id, camera_id=camera_id,
                    visitor_id=vid, event_type="ZONE_DWELL",
                    timestamp=frame_time, zone_id=current_zone,
                    dwell_ms=dwell_ms, is_staff=session.is_staff,
                    confidence=0.9, session_seq=session.session_seq,
                    sku_zone=current_zone,
                )
                dwell_tracker[vid][current_zone] = frame_time
                event_count += 1

            # Track billing exits for abandon detection
            if session.in_billing_zone and current_zone != session.zone_history[-1]:
                billing_exits.append({
                    "visitor_id": vid,
                    "exit_time": frame_time,
                    "store_id": store_id,
                    "camera_id": camera_id,
                    "is_staff": session.is_staff,
                })
                session.in_billing_zone = False

        if debug and frame_idx % (FPS * 30) == 0:
            minute = frame_idx // (FPS * 60)
            typer.echo(f"    [{camera_id}] min={minute} dets={len(detections)} events={event_count}")

        frame_idx += 1

    cap.release()
    typer.echo(f"  ✓ {camera_id}: {event_count} events")
    return event_count, billing_exits


# ---------------------------------------------------------------------------
# GAP 3 — BILLING_QUEUE_ABANDON post-processor
# ---------------------------------------------------------------------------

def emit_abandon_events(
    billing_exits: list[dict],
    pos_path: str,
    store_id: str,
    camera_id: str,
) -> int:
    """
    Post-processing pass after all clips are done.
    For each visitor who exited the billing zone, check if a POS transaction
    occurred within POS_WINDOW_MINUTES. If not → BILLING_QUEUE_ABANDON.

    Uses the same time-window correlation as the API's conversion rate logic.
    """
    if not billing_exits:
        return 0

    # Load POS transactions for this store
    pos_timestamps: list[datetime] = []
    if Path(pos_path).exists():
        import csv
        with open(pos_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["store_id"].strip() == store_id:
                    ts = datetime.fromisoformat(
                        row["timestamp"].strip().replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                    pos_timestamps.append(ts)

    abandon_count = 0
    for record in billing_exits:
        if record.get("is_staff"):
            continue

        exit_time = record["exit_time"].replace(tzinfo=None)
        window_end = exit_time + timedelta(minutes=POS_WINDOW_MINUTES)

        # Check if any POS transaction happened within window
        converted = any(exit_time <= ts <= window_end for ts in pos_timestamps)

        if not converted:
            emit_event(
                store_id=record["store_id"],
                camera_id=record["camera_id"],
                visitor_id=record["visitor_id"],
                event_type="BILLING_QUEUE_ABANDON",
                timestamp=record["exit_time"],
                zone_id="BILLING",
                confidence=0.80,
                is_staff=False,
            )
            abandon_count += 1

    return abandon_count


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

@app_cli.command()
def main(
    store: str      = typer.Option(...,                          help="Store ID e.g. ST1008"),
    clips_dir: str  = typer.Option("./data/clips",               help="Directory with .mp4 clips"),
    layout: str     = typer.Option("./data/store_layout.json",   help="Path to store_layout.json"),
    pos: str        = typer.Option("./data/pos_transactions.csv",help="Path to pos_transactions.csv"),
    output: str     = typer.Option("./data/events.jsonl",        help="Output JSONL path"),
    clip_start: str = typer.Option("2026-04-10T12:00:00Z",       help="ISO-8601 UTC clip start time"),
    sample: int     = typer.Option(1,                            help="Process 1 in N frames (1=all, 3=5fps)"),
    debug: bool     = typer.Option(False,                        help="Print per-minute progress"),
):
    """Process all CCTV clips for a store → structured events.jsonl"""
    os.environ["EVENTS_OUTPUT_PATH"] = output
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    typer.echo(f"\n=== Store Intelligence Pipeline ===")
    typer.echo(f"Store: {store}  |  Output: {output}  |  Sample: 1/{sample}")

    model = YOLO(YOLO_MODEL)
    typer.echo(f"Model: {YOLO_MODEL}  |  Confidence: {CONFIDENCE}")

    # Load layout
    store_layout = {}
    if Path(layout).exists():
        store_layout = load_store_layout(layout, store)
    else:
        typer.echo(f"WARNING: {layout} not found — zone detection disabled")
    zone_map = store_layout.get("zones", {})

    start_dt   = datetime.fromisoformat(clip_start.replace("Z", "+00:00"))
    clip_paths = get_clip_paths(clips_dir, store)

    if not clip_paths:
        typer.echo(f"ERROR: no clips found for {store} in {clips_dir}", err=True)
        raise typer.Exit(1)

    typer.echo(f"\nClips ({len(clip_paths)}) — processed in this order:")
    for cam_id, path in clip_paths.items():
        typer.echo(f"  {cam_id}: {path}")

    # GAP 5: one shared registry across ALL cameras for this store
    registry = VisitorRegistry()

    total_events  = 0
    all_billing_exits: list[dict] = []

    for camera_id, clip_path in clip_paths.items():
        typer.echo(f"\nProcessing {camera_id}...")
        count, billing_exits = process_clip(
            clip_path=clip_path,
            store_id=store,
            camera_id=camera_id,
            clip_start=start_dt,
            zone_map=zone_map,
            model=model,
            registry=registry,          # shared!
            debug=debug,
            sample_every=sample,
        )
        total_events      += count
        all_billing_exits += billing_exits

    # GAP 3: abandon post-processing after all clips
    typer.echo(f"\nRunning abandon detection ({len(all_billing_exits)} billing exits)...")
    abandons = emit_abandon_events(all_billing_exits, pos, store, "CAM_BILLING_01")
    total_events += abandons
    typer.echo(f"  BILLING_QUEUE_ABANDON events emitted: {abandons}")

    typer.echo(f"\n=== Complete: {total_events} events → {output} ===")


if __name__ == "__main__":
    app_cli()
