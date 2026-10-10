"""Case events in timeline order, and the cleanup that deleting an event needs.

The timeline routes use these, and so can anything else that lists a case's events
(the court report's timeline, the reconstruction stages).
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.records import delete_links_to
from app.models import Event, Reconstruction, ReconstructionStage


def utc_times(values: dict[str, Any]) -> dict[str, Any]:
    """Returns request values with every datetime as aware UTC (naive ones are taken
    as UTC, as the columns do), so the response and the audit diff match what the
    database reads back."""
    return {
        field: (value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC))
        if isinstance(value, datetime)
        else value
        for field, value in values.items()
    }


def case_events(db: Session, case_id: int, category: str | None = None) -> list[Event]:
    """A case's events by time_start, with untimed events last and ties by id."""
    query = select(Event).where(Event.case_id == case_id)
    if category is not None:
        query = query.where(Event.category == category)
    query = query.order_by(Event.time_start.is_(None), Event.time_start, Event.id)
    return list(db.scalars(query))


def remove_event(db: Session, event: Event) -> None:
    """Deletes an event, takes it out of every stage of its case's reconstructions
    and deletes the evidence links to it. The caller audits and commits."""
    stages = db.scalars(
        select(ReconstructionStage)
        .join(Reconstruction, ReconstructionStage.reconstruction_id == Reconstruction.id)
        .where(Reconstruction.case_id == event.case_id)
    )
    for stage in stages:
        if event.id in stage.event_ids:
            stage.event_ids = [event_id for event_id in stage.event_ids if event_id != event.id]
    delete_links_to(db, "event", event.id)
    db.delete(event)
