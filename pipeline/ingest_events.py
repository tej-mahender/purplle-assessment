"""
ingest_events.py — Standalone script to batch-ingest events.jsonl into the API.
Run this after detection (local or Colab) to populate the database.

Usage:
  python pipeline/ingest_events.py --events ./data/events.jsonl --api http://localhost:8000
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests
import typer

app_cli = typer.Typer()


@app_cli.command()
def main(
    events: str = typer.Option("./data/events.jsonl", help="Path to events.jsonl"),
    api: str = typer.Option("http://localhost:8000", help="API base URL"),
    batch_size: int = typer.Option(500, help="Events per ingest call"),
    dry_run: bool = typer.Option(False, help="Validate only, don't POST"),
):
    events_path = Path(events)
    if not events_path.exists():
        typer.echo(f"ERROR: {events} not found", err=True)
        raise typer.Exit(1)

    # Load all events
    all_events = []
    errors = 0
    with open(events_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                all_events.append(json.loads(line))
            except json.JSONDecodeError as e:
                typer.echo(f"  Line {i}: JSON error — {e}", err=True)
                errors += 1

    typer.echo(f"Loaded {len(all_events)} events ({errors} parse errors)")

    if dry_run:
        typer.echo("Dry run — skipping POST")
        return

    # Check API is up
    try:
        resp = requests.get(f"{api}/health", timeout=5)
        typer.echo(f"API health: {resp.json().get('status', 'unknown')}")
    except Exception as e:
        typer.echo(f"ERROR: API not reachable at {api} — {e}", err=True)
        raise typer.Exit(1)

    # Batch ingest
    total_accepted = total_rejected = 0
    num_batches = (len(all_events) + batch_size - 1) // batch_size

    typer.echo(f"\nIngesting {len(all_events)} events in {num_batches} batches...")
    t0 = time.time()

    for i in range(0, len(all_events), batch_size):
        batch = all_events[i : i + batch_size]
        batch_num = i // batch_size + 1

        try:
            resp = requests.post(
                f"{api}/events/ingest",
                json={"events": batch},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                total_accepted += data["accepted"]
                total_rejected += data["rejected"]
                if data["rejected"] > 0:
                    typer.echo(f"  Batch {batch_num}/{num_batches}: ✓ {data['accepted']} accepted, ⚠ {data['rejected']} rejected")
                    for r in data.get("rejections", [])[:3]:
                        typer.echo(f"    Rejected {r['event_id']}: {r['reason']}")
                else:
                    typer.echo(f"  Batch {batch_num}/{num_batches}: ✓ {data['accepted']} accepted")
            else:
                typer.echo(f"  Batch {batch_num}: ERROR {resp.status_code} — {resp.text[:200]}", err=True)
                total_rejected += len(batch)
        except Exception as e:
            typer.echo(f"  Batch {batch_num}: REQUEST FAILED — {e}", err=True)
            sys.exit(1)

    elapsed = time.time() - t0
    typer.echo(f"\n=== Ingest complete in {elapsed:.1f}s ===")
    typer.echo(f"  Accepted: {total_accepted}")
    typer.echo(f"  Rejected: {total_rejected}")

    # Quick verification
    try:
        stores = {e["store_id"] for e in all_events}
        for store_id in list(stores)[:2]:
            resp = requests.get(f"{api}/stores/{store_id}/metrics", timeout=10)
            if resp.status_code == 200:
                m = resp.json()
                typer.echo(f"\n  {store_id} metrics:")
                typer.echo(f"    unique_visitors:  {m['unique_visitors']}")
                typer.echo(f"    conversion_rate:  {m['conversion_rate']:.1%}")
                typer.echo(f"    queue_depth:      {m['current_queue_depth']}")
    except Exception:
        pass


if __name__ == "__main__":
    app_cli()
