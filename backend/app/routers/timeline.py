"""Timeline routes (task B3 in docs/BACKEND_TASKS.md):

- GET, POST /api/cases/{id}/events?category=
- PATCH, DELETE /api/events/{id}
"""

from fastapi import APIRouter, status

from app.core.deps import ActorId, DbSession
from app.core.records import apply_update, get_or_404, record_audit
from app.models import Case, Event, Location
from app.schemas.common import EventCategory
from app.schemas.timeline import CaseEvent, CaseEventCreate, CaseEventUpdate
from app.services.timeline.events import case_events, remove_event, utc_times
from app.services.timeline.validation import invalid, reject_nulls

router = APIRouter(tags=["timeline"])


def _require_location(db: DbSession, case_id: int, location_id: int) -> None:
    location = db.get(Location, location_id)
    if location is None or location.case_id != case_id:
        raise invalid("location_id", f"Location {location_id} is not in this case")


@router.get("/cases/{case_id}/events", response_model=list[CaseEvent])
def list_events(case_id: int, db: DbSession, category: EventCategory | None = None):
    case = get_or_404(db, Case, case_id)
    return case_events(db, case.id, category)


@router.post("/cases/{case_id}/events", response_model=CaseEvent, status_code=status.HTTP_201_CREATED)
def create_event(case_id: int, body: CaseEventCreate, db: DbSession, actor_id: ActorId):
    case = get_or_404(db, Case, case_id)
    if body.location_id is not None:
        _require_location(db, case.id, body.location_id)
    event = Event(case_id=case.id, **utc_times(body.model_dump()))
    db.add(event)
    db.flush()
    record_audit(db, actor_id, "modified", "event", event.id, case_id=case.id, detail={"created": True})
    db.commit()
    return event


@router.patch("/events/{event_id}", response_model=CaseEvent)
def update_event(event_id: int, body: CaseEventUpdate, db: DbSession, actor_id: ActorId):
    event = get_or_404(db, Event, event_id)
    changes = utc_times(body.model_dump(exclude_unset=True))
    reject_nulls(event, changes)
    if changes.get("location_id") is not None:
        _require_location(db, event.case_id, changes["location_id"])
    apply_update(db, event, changes, record_type="event", actor_id=actor_id)
    db.commit()
    return event


@router.delete("/events/{event_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_event(event_id: int, db: DbSession, actor_id: ActorId) -> None:
    event = get_or_404(db, Event, event_id)
    record_audit(db, actor_id, "modified", "event", event.id, case_id=event.case_id, detail={"deleted": True})
    remove_event(db, event)
    db.commit()
