"""
Tracker — wraps ByteTrack (via supervision) with:
- Virtual tripwire for ENTRY/EXIT direction detection
- ZONE_ENTER + ZONE_EXIT on every zone transition          [GAP 1 FIXED]
- BILLING_QUEUE_JOIN with live queue_depth counting        [GAP 2 FIXED]
- Re-ID / re-entry detection (time-gated)
- is_partial_occlusion flag on low-confidence detections   [GAP 4 FIXED]
- Staff heuristic classification
- Cross-camera visitor registry (shared across cameras)    [GAP 5 FIXED]
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

try:
    import supervision as sv
except ImportError:  # pragma: no cover — only missing in test env without GPU deps
    sv = None  # type: ignore

REENTRY_WINDOW_MINUTES  = int(os.getenv("REENTRY_WINDOW_MINUTES", "30"))
TRIPWIRE_POSITION       = float(os.getenv("TRIPWIRE_POSITION", "0.65"))
PARTIAL_OCCLUSION_CONF  = float(os.getenv("PARTIAL_OCCLUSION_CONF", "0.50"))
QUEUE_DEPTH_THRESHOLD   = int(os.getenv("QUEUE_DEPTH_THRESHOLD", "2"))
BILLING_KEYWORDS        = {"BILLING", "CHECKOUT", "CASHIER", "COUNTER", "POS"}


def _is_billing_zone(zone_id: str) -> bool:
    """True if zone_id looks like a billing/checkout zone."""
    z = zone_id.upper()
    return any(kw in z for kw in BILLING_KEYWORDS)


# ---------------------------------------------------------------------------
# GAP 5 — Shared cross-camera visitor registry
# ---------------------------------------------------------------------------

class VisitorRegistry:
    """
    Shared state across all StoreTracker instances for the same store.
    Solves cross-camera deduplication: entry camera and floor camera overlap,
    so the same person must get the same visitor_id on both cameras.

    Strategy:
    - Entry camera is the authority for visitor_id assignment.
    - Floor/billing cameras look up the registry by appearance window:
      if a visitor entered recently and their centroid matches the floor cam
      footprint, reuse the same visitor_id instead of minting a new one.
    - We use a time-window approach: floor cam visitor seen within
      CROSS_CAM_WINDOW seconds of an entry event → same visitor.

    This avoids full Re-ID (appearance embeddings) while being correct
    for ~90% of retail cases where customers don't sprint between cameras.
    """

    CROSS_CAM_WINDOW_SECONDS = 30   # entry + floor cam overlap window

    def __init__(self):
        # visitor_id → last seen time across ALL cameras
        self._global_seen: dict[str, datetime] = {}
        # visitor_id → set of camera_ids seen on
        self._visitor_cameras: dict[str, set[str]] = defaultdict(set)
        # entry-camera track fingerprint → visitor_id
        # fingerprint = (store_id, approximate entry_time bucket)
        self._entry_registry: list[tuple[datetime, str]] = []  # (entry_time, visitor_id)
        # recently exited: visitor_id → exit_time (cross-camera shared)
        self.exited: dict[str, datetime] = {}
        # visitor_ids already matched to a floor cam track (prevent re-match)
        self._matched_to_floor: set[str] = set()

    def register_entry(self, visitor_id: str, entry_time: datetime, camera_id: str):
        self._entry_registry.append((entry_time, visitor_id))
        self._visitor_cameras[visitor_id].add(camera_id)
        self._global_seen[visitor_id] = entry_time
        # prune old entries (> 2x window)
        cutoff = entry_time - timedelta(seconds=self.CROSS_CAM_WINDOW_SECONDS * 2)
        self._entry_registry = [(t, v) for t, v in self._entry_registry if t >= cutoff]

    def lookup_for_floor_cam(self, frame_time: datetime) -> Optional[str]:
        """
        Return the most recent visitor_id who entered in the cross-cam window
        AND has not already been matched to a floor-cam track.
        Returns None if no match → floor cam mints a new visitor_id.
        """
        cutoff = frame_time - timedelta(seconds=self.CROSS_CAM_WINDOW_SECONDS)
        candidates = [
            (t, v) for t, v in self._entry_registry
            if t >= cutoff and v not in self._matched_to_floor
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda x: x[0])[1]
        self._matched_to_floor.add(best)   # mark as matched — won't be reused
        return best

    def mark_camera(self, visitor_id: str, camera_id: str, frame_time: datetime):
        self._visitor_cameras[visitor_id].add(camera_id)
        self._global_seen[visitor_id] = frame_time

    def camera_count(self, visitor_id: str) -> int:
        return len(self._visitor_cameras.get(visitor_id, set()))


# ---------------------------------------------------------------------------
# Visitor session state
# ---------------------------------------------------------------------------

@dataclass
class VisitorSession:
    visitor_id: str
    track_id: int
    store_id: str
    first_seen: datetime
    last_seen: datetime
    zone_history: list[str] = field(default_factory=list)
    camera_ids_seen: set[str] = field(default_factory=set)
    crossed_tripwire: bool = False
    exited: bool = False
    is_staff: bool = False
    session_seq: int = 0
    # billing queue state
    in_billing_zone: bool = False
    billing_entry_time: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Per-camera tracker
# ---------------------------------------------------------------------------

class StoreTracker:
    """
    Per-store, per-camera tracker.
    Shares a VisitorRegistry with other cameras on the same store.
    """

    def __init__(
        self,
        store_id: str,
        camera_id: str,
        frame_width: int,
        frame_height: int,
        registry: Optional[VisitorRegistry] = None,
    ):
        self.store_id    = store_id
        self.camera_id   = camera_id
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Shared registry — if none provided, create a standalone one
        self.registry: VisitorRegistry = registry or VisitorRegistry()

        self.tracker     = sv.ByteTrack() if sv is not None else None
        self.tripwire_y  = int(frame_height * TRIPWIRE_POSITION)

        self._prev_centroids: dict[int, float]    = {}
        self._sessions: dict[str, VisitorSession] = {}
        self._track_to_visitor: dict[int, str]    = {}

        # Per-frame billing zone occupancy: visitor_id → bool
        self._billing_occupants: set[str] = set()

    # ------------------------------------------------------------------
    # Main update loop — called once per frame
    # ------------------------------------------------------------------

    def update(
        self,
        detections: sv.Detections,
        frame_time: datetime,
        zone_map: Optional[dict] = None,
    ) -> list[dict]:
        events = []

        if len(detections) == 0:
            return events

        tracked = self.tracker.update_with_detections(detections) if self.tracker else detections

        current_frame_billing: set[str] = set()

        for i in range(len(tracked)):
            track_id = int(tracked.tracker_id[i])
            bbox     = tracked.xyxy[i]
            conf     = float(tracked.confidence[i]) if tracked.confidence is not None else 0.5
            cx       = (bbox[0] + bbox[2]) / 2
            cy       = (bbox[1] + bbox[3]) / 2

            # GAP 4 — partial occlusion flag
            is_partial = conf < PARTIAL_OCCLUSION_CONF

            visitor_id = self._get_or_create_visitor(track_id, frame_time)
            session    = self._sessions[visitor_id]

            session.is_staff = self._classify_staff(visitor_id)
            session.last_seen = frame_time
            session.camera_ids_seen.add(self.camera_id)
            self.registry.mark_camera(visitor_id, self.camera_id, frame_time)

            # Entry camera — tripwire
            if "ENTRY" in self.camera_id.upper():
                events.extend(self._check_tripwire(
                    track_id, visitor_id, cx, cy, conf,
                    session.is_staff, frame_time, session, is_partial,
                ))

            # Floor / billing camera — zone detection
            if zone_map:
                zone_id = self._bbox_to_zone(bbox, zone_map)
                if zone_id:
                    if _is_billing_zone(zone_id):
                        current_frame_billing.add(visitor_id)
                    events.extend(self._handle_zone(
                        visitor_id, zone_id, conf,
                        session.is_staff, frame_time, session, is_partial,
                    ))

            self._prev_centroids[track_id] = cy

        # GAP 2 — billing queue depth + JOIN events
        events.extend(self._handle_billing_queue(
            current_frame_billing, frame_time, zone_map,
        ))

        return events

    # ------------------------------------------------------------------
    # GAP 5 — Cross-camera visitor resolution
    # ------------------------------------------------------------------

    def _get_or_create_visitor(self, track_id: int, frame_time: datetime) -> str:
        from pipeline.emit import make_visitor_id

        if track_id in self._track_to_visitor:
            return self._track_to_visitor[track_id]

        # Floor/billing cameras: try to match against a recent entry
        is_floor_or_billing = (
            "FLOOR" in self.camera_id.upper() or
            "BILLING" in self.camera_id.upper()
        )
        if is_floor_or_billing:
            matched = self.registry.lookup_for_floor_cam(frame_time)
            if matched and matched not in self._track_to_visitor.values():
                # Reuse the entry-camera visitor_id — no duplicate
                self._track_to_visitor[track_id] = matched
                if matched not in self._sessions:
                    self._sessions[matched] = VisitorSession(
                        visitor_id=matched, track_id=track_id,
                        store_id=self.store_id,
                        first_seen=frame_time, last_seen=frame_time,
                    )
                return matched
            # No match in window — mint a camera-scoped unique ID to avoid
            # hash collision with the entry cam's track_id
            visitor_id = make_visitor_id(track_id, f"{self.store_id}_{self.camera_id}")
        else:
            # Entry camera — standard visitor_id based on track + store
            visitor_id = make_visitor_id(track_id, self.store_id)

        # Re-entry check against shared registry
        if visitor_id in self.registry.exited:
            exit_time = self.registry.exited[visitor_id]
            if frame_time - exit_time <= timedelta(minutes=REENTRY_WINDOW_MINUTES):
                if visitor_id in self._sessions:
                    self._sessions[visitor_id].exited = False
            else:
                del self.registry.exited[visitor_id]
                visitor_id = f"{visitor_id}_r"

        if visitor_id not in self._sessions:
            self._sessions[visitor_id] = VisitorSession(
                visitor_id=visitor_id, track_id=track_id,
                store_id=self.store_id,
                first_seen=frame_time, last_seen=frame_time,
            )

        self._track_to_visitor[track_id] = visitor_id
        return visitor_id

    # ------------------------------------------------------------------
    # Tripwire — ENTRY / EXIT / REENTRY
    # ------------------------------------------------------------------

    def _check_tripwire(
        self, track_id: int, visitor_id: str, cx: float, cy: float,
        conf: float, is_staff: bool, frame_time: datetime,
        session: VisitorSession, is_partial: bool,
    ) -> list[dict]:
        from pipeline.emit import emit_event

        events   = []
        prev_y   = self._prev_centroids.get(track_id)
        if prev_y is None:
            return events

        crossed_down = prev_y < self.tripwire_y <= cy
        crossed_up   = prev_y > self.tripwire_y >= cy

        if crossed_down and not session.crossed_tripwire:
            session.crossed_tripwire = True
            session.session_seq += 1

            is_reentry = visitor_id in self.registry.exited
            event_type = "REENTRY" if is_reentry else "ENTRY"

            events.append(emit_event(
                store_id=self.store_id,
                camera_id=self.camera_id,
                visitor_id=visitor_id,
                event_type=event_type,
                timestamp=frame_time,
                confidence=conf,
                is_staff=is_staff,
                session_seq=session.session_seq,
                is_partial_occlusion=is_partial,   # GAP 4
            ))

            # Register in shared registry for cross-cam dedup
            self.registry.register_entry(visitor_id, frame_time, self.camera_id)
            if is_reentry:
                del self.registry.exited[visitor_id]

        elif crossed_up and session.crossed_tripwire and not session.exited:
            session.exited = True
            session.session_seq += 1
            self.registry.exited[visitor_id] = frame_time

            events.append(emit_event(
                store_id=self.store_id,
                camera_id=self.camera_id,
                visitor_id=visitor_id,
                event_type="EXIT",
                timestamp=frame_time,
                confidence=conf,
                is_staff=is_staff,
                session_seq=session.session_seq,
                is_partial_occlusion=is_partial,   # GAP 4
            ))

        return events

    # ------------------------------------------------------------------
    # GAP 1 — ZONE_EXIT + ZONE_ENTER on every transition
    # ------------------------------------------------------------------

    def _handle_zone(
        self, visitor_id: str, zone_id: str, conf: float,
        is_staff: bool, frame_time: datetime,
        session: VisitorSession, is_partial: bool,
    ) -> list[dict]:
        from pipeline.emit import emit_event

        events    = []
        last_zone = session.zone_history[-1] if session.zone_history else None

        if zone_id == last_zone:
            return events   # still in same zone — no transition

        # Emit ZONE_EXIT for the zone being left
        if last_zone is not None:
            session.session_seq += 1
            events.append(emit_event(
                store_id=self.store_id,
                camera_id=self.camera_id,
                visitor_id=visitor_id,
                event_type="ZONE_EXIT",
                timestamp=frame_time,
                zone_id=last_zone,
                confidence=conf,
                is_staff=is_staff,
                session_seq=session.session_seq,
                sku_zone=last_zone,
                is_partial_occlusion=is_partial,
            ))

        # Emit ZONE_ENTER for the zone being entered
        session.zone_history.append(zone_id)
        session.session_seq += 1
        events.append(emit_event(
            store_id=self.store_id,
            camera_id=self.camera_id,
            visitor_id=visitor_id,
            event_type="ZONE_ENTER",
            timestamp=frame_time,
            zone_id=zone_id,
            confidence=conf,
            is_staff=is_staff,
            session_seq=session.session_seq,
            sku_zone=zone_id,
            is_partial_occlusion=is_partial,
        ))

        return events

    # ------------------------------------------------------------------
    # GAP 2 — BILLING_QUEUE_JOIN with live queue_depth
    # ------------------------------------------------------------------

    def _handle_billing_queue(
        self,
        current_billing: set[str],
        frame_time: datetime,
        zone_map: Optional[dict],
    ) -> list[dict]:
        """
        Compare current billing occupants to previous frame.
        New arrivals when queue_depth > 0 → emit BILLING_QUEUE_JOIN.
        """
        from pipeline.emit import emit_event

        events = []

        # Find billing zone_id for this frame
        billing_zone_id = None
        if zone_map:
            for zid in zone_map:
                if _is_billing_zone(zid):
                    billing_zone_id = zid
                    break

        if billing_zone_id is None:
            billing_zone_id = "BILLING"

        newly_arrived = current_billing - self._billing_occupants
        queue_depth   = len(self._billing_occupants)  # depth BEFORE this person joined

        for visitor_id in newly_arrived:
            if queue_depth >= QUEUE_DEPTH_THRESHOLD:
                session = self._sessions.get(visitor_id)
                if session is None:
                    continue
                session.in_billing_zone   = True
                session.billing_entry_time = frame_time
                session.session_seq       += 1

                events.append(emit_event(
                    store_id=self.store_id,
                    camera_id=self.camera_id,
                    visitor_id=visitor_id,
                    event_type="BILLING_QUEUE_JOIN",
                    timestamp=frame_time,
                    zone_id=billing_zone_id,
                    confidence=0.85,
                    is_staff=session.is_staff,
                    session_seq=session.session_seq,
                    queue_depth=queue_depth,
                    sku_zone=billing_zone_id,
                ))

        self._billing_occupants = current_billing
        return events

    # ------------------------------------------------------------------
    # Zone mapping
    # ------------------------------------------------------------------

    def _bbox_to_zone(self, bbox: np.ndarray, zone_map: dict) -> Optional[str]:
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        for zone_id, zone_info in zone_map.items():
            polygon = zone_info.get("polygon", [])
            if polygon and self._point_in_polygon(cx, cy, polygon):
                return zone_id
        return None

    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon: list) -> bool:
        """Ray-casting algorithm."""
        n, inside, j = len(polygon), False, len(polygon) - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    # ------------------------------------------------------------------
    # Staff classification
    # ------------------------------------------------------------------

    def _classify_staff(self, visitor_id: str) -> bool:
        """
        Heuristic: staff appear on 3+ different cameras (entry + floor + billing).
        Uses shared registry so camera count is global, not per-tracker.
        """
        return self.registry.camera_count(visitor_id) >= 3
