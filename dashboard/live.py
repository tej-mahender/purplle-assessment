"""
Live dashboard — replays events.jsonl in simulated real-time
and shows metrics updating in the terminal using `rich`.

Usage: python dashboard/live.py --store STORE_BLR_002 --events ./data/events.jsonl
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import typer
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()
app_cli = typer.Typer()

API_URL = "http://localhost:8000"
REPLAY_SPEED = 10          # replay at 10x real time
BATCH_SIZE = 20            # events per ingest call
REFRESH_SECONDS = 2


def ingest_batch(events: list[dict]) -> tuple[int, int]:
    try:
        resp = requests.post(f"{API_URL}/events/ingest", json={"events": events}, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return data["accepted"], data["rejected"]
    except Exception:
        pass
    return 0, 0


def fetch_metrics(store_id: str) -> dict:
    try:
        resp = requests.get(f"{API_URL}/stores/{store_id}/metrics", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def fetch_anomalies(store_id: str) -> list:
    try:
        resp = requests.get(f"{API_URL}/stores/{store_id}/anomalies", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("anomalies", [])
    except Exception:
        pass
    return []


def build_dashboard(store_id: str, metrics: dict, anomalies: list, stats: dict) -> Panel:
    # Header
    title = Text(f"⚡ Store Intelligence — {store_id}", style="bold white")

    # Metrics panel
    m_table = Table(show_header=False, box=None, padding=(0, 2))
    m_table.add_column("Metric", style="dim")
    m_table.add_column("Value", style="bold cyan")

    m_table.add_row("Unique Visitors", str(metrics.get("unique_visitors", "—")))
    conv = metrics.get("conversion_rate")
    m_table.add_row("Conversion Rate", f"{conv:.1%}" if conv is not None else "—")
    m_table.add_row("Queue Depth", str(metrics.get("current_queue_depth", "—")))
    aband = metrics.get("abandonment_rate")
    m_table.add_row("Abandonment Rate", f"{aband:.1%}" if aband is not None else "—")

    metrics_panel = Panel(m_table, title="[bold]Live Metrics", border_style="cyan")

    # Zone dwell panel
    zones = metrics.get("avg_dwell_by_zone", [])
    z_table = Table(show_header=True, box=None, padding=(0, 1))
    z_table.add_column("Zone", style="dim")
    z_table.add_column("Visits", justify="right")
    z_table.add_column("Avg Dwell", justify="right", style="green")
    for z in sorted(zones, key=lambda x: x.get("visit_count", 0), reverse=True)[:5]:
        dwell_s = z.get("avg_dwell_ms", 0) / 1000
        z_table.add_row(z["zone_id"], str(z["visit_count"]), f"{dwell_s:.0f}s")
    if not zones:
        z_table.add_row("—", "—", "—")

    zone_panel = Panel(z_table, title="[bold]Zone Dwell", border_style="blue")

    # Anomalies panel
    a_lines = []
    severity_colors = {"CRITICAL": "red", "WARN": "yellow", "INFO": "white"}
    for a in anomalies[:4]:
        color = severity_colors.get(a.get("severity", "INFO"), "white")
        a_lines.append(f"[{color}]● {a['anomaly_type']}[/{color}]: {a['description'][:60]}")
    if not a_lines:
        a_lines = ["[green]✓ No active anomalies[/green]"]

    from rich.markup import render as render_markup
    anomaly_panel = Panel(
        "\n".join(a_lines),
        title="[bold]Anomalies",
        border_style="yellow" if anomalies else "green",
    )

    # Pipeline stats
    s_table = Table(show_header=False, box=None, padding=(0, 2))
    s_table.add_column("", style="dim")
    s_table.add_column("", style="bold")
    s_table.add_row("Events replayed", str(stats.get("replayed", 0)))
    s_table.add_row("Ingested", str(stats.get("accepted", 0)))
    s_table.add_row("Rejected", str(stats.get("rejected", 0)))
    s_table.add_row("Updated", datetime.now().strftime("%H:%M:%S"))
    stats_panel = Panel(s_table, title="[bold]Pipeline", border_style="dim")

    from rich.columns import Columns
    top = Columns([metrics_panel, zone_panel], equal=True)

    from rich import box as rbox
    from rich.table import Table as RTable
    outer = Table(show_header=False, box=None, padding=(0, 0))
    outer.add_column()
    outer.add_row(top)
    outer.add_row(anomaly_panel)
    outer.add_row(stats_panel)

    return Panel(outer, title=title, border_style="bright_white")


@app_cli.command()
def main(
    store: str = typer.Option("STORE_BLR_002", help="Store ID to monitor"),
    events: str = typer.Option("./data/events.jsonl", help="Path to events.jsonl"),
    speed: float = typer.Option(10.0, help="Replay speed multiplier"),
):
    """Replay events.jsonl in simulated real-time and show live metrics."""
    events_path = Path(events)
    if not events_path.exists():
        console.print(f"[red]Events file not found: {events}[/red]")
        raise typer.Exit(1)

    all_events = []
    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    console.print(f"[green]Loaded {len(all_events)} events. Replaying at {speed}x speed...[/green]")
    time.sleep(1)

    stats = {"replayed": 0, "accepted": 0, "rejected": 0}
    metrics: dict = {}
    anomalies: list = []

    batch: list = []
    last_refresh = 0.0

    with Live(console=console, refresh_per_second=2, screen=True) as live:
        for i, event in enumerate(all_events):
            batch.append(event)
            stats["replayed"] += 1

            if len(batch) >= BATCH_SIZE or i == len(all_events) - 1:
                acc, rej = ingest_batch(batch)
                stats["accepted"] += acc
                stats["rejected"] += rej
                batch = []

                now = time.time()
                if now - last_refresh >= REFRESH_SECONDS:
                    metrics = fetch_metrics(store)
                    anomalies = fetch_anomalies(store)
                    last_refresh = now

                live.update(build_dashboard(store, metrics, anomalies, stats))
                time.sleep(BATCH_SIZE / (speed * 15))  # simulate real-time pacing

        # Final update
        metrics = fetch_metrics(store)
        anomalies = fetch_anomalies(store)
        live.update(build_dashboard(store, metrics, anomalies, stats))
        time.sleep(2)

    console.print(f"\n[bold green]✓ Replay complete.[/bold green] {stats['replayed']} events ingested.")


if __name__ == "__main__":
    app_cli()
