"""Cases routes (task B1 in docs/BACKEND_TASKS.md):

- GET /api/investigators
- GET, POST /api/cases?q=&status=&priority=
- GET, PATCH /api/cases/{id}
- GET /api/cases/{id}/workspace
"""

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import exists, func, or_, select
from sqlalchemy.orm import Session

from app.core.deps import ActorId, DbSession
from app.core.records import apply_update, case_counts, get_or_404, record_audit
from app.models import (
    Base,
    Case,
    CaseObject,
    Event,
    Evidence,
    EvidenceLink,
    Injury,
    Investigator,
    Location,
    Person,
    Relationship,
)
from app.models.base import utcnow
from app.schemas.cases import Case as CaseOut
from app.schemas.cases import CaseCreate, CaseListItem, CaseUpdate
from app.schemas.cases import Investigator as InvestigatorOut
from app.schemas.common import CasePriority, CaseStatus
from app.schemas.workspace import CaseWorkspace

router = APIRouter(tags=["cases"])


def unprocessable(errors: list[dict[str, Any]]) -> HTTPException:
    """A 422 shaped like FastAPI's own, which the frontend shows as "field: message"."""
    return HTTPException(status_code=422, detail=errors)


def body_error(field: str, msg: str, value: Any) -> dict[str, Any]:
    return {"type": "value_error", "loc": ["body", field], "msg": msg, "input": value}


def reject_nulls(model: type[Base], changes: dict[str, Any]) -> None:
    """The update schemas allow null on every field; only nullable columns may take it."""
    columns = model.__table__.columns
    errors = [
        body_error(field, "Field can't be null", None)
        for field, value in changes.items()
        if value is None and not columns[field].nullable
    ]
    if errors:
        raise unprocessable(errors)


def check_investigators(db: Session, values: dict[str, Any]) -> None:
    """lead_investigator_id and investigator_ids must name existing investigators."""
    errors = []
    lead_id = values.get("lead_investigator_id")
    if lead_id is not None and db.get(Investigator, lead_id) is None:
        errors.append(body_error("lead_investigator_id", f"Investigator {lead_id} not found", lead_id))
    team_ids = values.get("investigator_ids") or []
    known = set(db.scalars(select(Investigator.id).where(Investigator.id.in_(team_ids))))
    unknown = [i for i in team_ids if i not in known]
    if unknown:
        listed = ", ".join(str(i) for i in unknown)
        errors.append(body_error("investigator_ids", f"Investigator(s) {listed} not found", team_ids))
    if errors:
        raise unprocessable(errors)


def contains(column, q: str):
    """Case-insensitive substring match, with q's LIKE wildcards taken literally."""
    escaped = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return column.ilike(f"%{escaped}%", escape="\\")


@router.get("/investigators", response_model=list[InvestigatorOut])
def list_investigators(db: DbSession):
    return db.scalars(select(Investigator).order_by(Investigator.id)).all()


@router.get("/cases", response_model=list[CaseListItem])
def list_cases(
    db: DbSession,
    q: str = "",
    status: CaseStatus | None = None,
    priority: CasePriority | None = None,
):
    evidence_count = (
        select(func.count(Evidence.id)).where(Evidence.case_id == Case.id).correlate(Case).scalar_subquery()
    )
    query = select(Case, evidence_count)
    if status is not None:
        query = query.where(Case.status == status)
    if priority is not None:
        query = query.where(Case.priority == priority)
    q = q.strip()
    if q:
        query = query.where(
            or_(
                contains(Case.case_number, q),
                contains(Case.title, q),
                contains(Case.description, q),
                exists().where(Investigator.id == Case.lead_investigator_id, contains(Investigator.name, q)),
                exists().where(Location.case_id == Case.id, contains(Location.address, q)),
                exists().where(Person.case_id == Case.id, contains(Person.name, q)),
            )
        )
    rows = db.execute(query.order_by(Case.created_at.desc(), Case.id.desc())).all()
    return [CaseListItem(**CaseOut.model_validate(case).model_dump(), evidence_count=count) for case, count in rows]


@router.post("/cases", response_model=CaseOut, status_code=status.HTTP_201_CREATED)
def create_case(body: CaseCreate, db: DbSession, actor_id: ActorId):
    values = body.model_dump()
    check_investigators(db, values)
    # The number needs the id, so insert under a unique placeholder first.
    case = Case(case_number=uuid4().hex, **values)
    db.add(case)
    db.flush()
    case.case_number = f"FX-{case.created_at.year}-{case.id:04d}"
    # Setting the number isn't an edit, so keep updated_at equal to created_at.
    case.updated_at = case.created_at
    record_audit(db, actor_id, "modified", "case", case.id, case_id=case.id, detail={"created": True})
    db.commit()
    # Reload, so times come back as stored (UTC) rather than as sent.
    db.refresh(case)
    return case


@router.get("/cases/{case_id}", response_model=CaseOut)
def get_case(case_id: int, db: DbSession):
    return get_or_404(db, Case, case_id)


@router.patch("/cases/{case_id}", response_model=CaseOut)
def update_case(case_id: int, body: CaseUpdate, db: DbSession, actor_id: ActorId):
    case = get_or_404(db, Case, case_id)
    changes = body.model_dump(exclude_unset=True)
    reject_nulls(Case, changes)
    check_investigators(db, changes)
    if apply_update(db, case, changes, record_type="case", actor_id=actor_id, case_id=case.id):
        case.updated_at = utcnow()
    db.commit()
    db.refresh(case)
    return case


def rows_of(db: Session, model: type[Base], case_id: int) -> list:
    return list(db.scalars(select(model).where(model.case_id == case_id).order_by(model.id)))


@router.get("/cases/{case_id}/workspace", response_model=CaseWorkspace)
def get_workspace(case_id: int, db: DbSession):
    case = get_or_404(db, Case, case_id)
    links = db.scalars(
        select(EvidenceLink)
        .join(Evidence, Evidence.id == EvidenceLink.evidence_id)
        .where(Evidence.case_id == case.id)
        .order_by(EvidenceLink.id)
    ).all()
    return {
        "case": case,
        "persons": rows_of(db, Person, case.id),
        "objects": rows_of(db, CaseObject, case.id),
        "locations": rows_of(db, Location, case.id),
        "evidence": rows_of(db, Evidence, case.id),
        "events": rows_of(db, Event, case.id),
        "relationships": rows_of(db, Relationship, case.id),
        "injuries": rows_of(db, Injury, case.id),
        "links": links,
        "counts": case_counts(db, case.id),
    }
