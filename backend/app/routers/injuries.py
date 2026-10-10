"""Injuries routes (task B4 in docs/BACKEND_TASKS.md): injury markers for the autopsy
view and the anatomy explorer. The AI can propose injuries, so they are reviewed
through PATCH like the other case records.

- GET, POST /api/cases/{id}/injuries
- PATCH /api/injuries/{id}
"""

from fastapi import APIRouter, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.deps import ActorId, DbSession
from app.core.records import apply_update, get_or_404, record_audit
from app.models import Case, Injury, Person
from app.schemas.injuries import Injury as InjuryOut, InjuryCreate, InjuryUpdate
from app.services.reconstruction.validation import ids_in_case, invalid

router = APIRouter(tags=["injuries"])

# Columns a PATCH may not set to null; the rest of InjuryUpdate's fields are nullable.
REQUIRED_FIELDS = ("body_region", "type", "severity", "description", "review_status")


def check_person(db: Session, person_id: int | None, case_id: int) -> None:
    if person_id is not None and not ids_in_case(db, Person, [person_id], case_id):
        raise invalid(("person_id",), f"Person {person_id} is not a person of this case")


@router.get("/cases/{case_id}/injuries", response_model=list[InjuryOut])
def list_injuries(case_id: int, db: DbSession):
    case = get_or_404(db, Case, case_id)
    return db.scalars(select(Injury).where(Injury.case_id == case.id).order_by(Injury.id)).all()


@router.post("/cases/{case_id}/injuries", response_model=InjuryOut, status_code=status.HTTP_201_CREATED)
def create_injury(case_id: int, body: InjuryCreate, db: DbSession, actor_id: ActorId):
    case = get_or_404(db, Case, case_id)
    check_person(db, body.person_id, case.id)
    injury = Injury(case_id=case.id, **body.model_dump())
    db.add(injury)
    db.flush()
    record_audit(db, actor_id, "modified", "injury", injury.id, case_id=case.id, detail={"created": True})
    db.commit()
    # Reload so estimated_time comes back as UTC, whatever offset was sent.
    db.refresh(injury)
    return injury


@router.patch("/injuries/{injury_id}", response_model=InjuryOut)
def update_injury(injury_id: int, body: InjuryUpdate, db: DbSession, actor_id: ActorId):
    injury = get_or_404(db, Injury, injury_id)
    changes = body.model_dump(exclude_unset=True)
    for field in REQUIRED_FIELDS:
        if field in changes and changes[field] is None:
            raise invalid((field,), "Field may not be null")
    if "person_id" in changes:
        check_person(db, changes["person_id"], injury.case_id)
    apply_update(db, injury, changes, record_type="injury", actor_id=actor_id)
    db.commit()
    db.refresh(injury)
    return injury
