"""Review routes (task B3 in docs/BACKEND_TASKS.md):

- GET /api/cases/{id}/observations?status=
- PATCH /api/observations/{id}
- GET /api/cases/{id}/conflicts?status=
- PATCH /api/conflicts/{id}
"""

from fastapi import APIRouter
from sqlalchemy import select

from app.core.deps import ActorId, DbSession
from app.core.records import apply_update, get_or_404
from app.models import Case, Conflict, Observation
from app.schemas.common import ConflictStatus, ReviewStatus
from app.schemas.review import Conflict as ConflictOut
from app.schemas.review import ConflictUpdate, ObservationUpdate
from app.schemas.review import Observation as ObservationOut
from app.services.timeline.validation import reject_nulls

router = APIRouter(tags=["review"])


@router.get("/cases/{case_id}/observations", response_model=list[ObservationOut])
def list_observations(case_id: int, db: DbSession, status: ReviewStatus | None = None):
    case = get_or_404(db, Case, case_id)
    query = select(Observation).where(Observation.case_id == case.id)
    if status is not None:
        query = query.where(Observation.review_status == status)
    return db.scalars(query.order_by(Observation.id)).all()


@router.patch("/observations/{observation_id}", response_model=ObservationOut)
def update_observation(observation_id: int, body: ObservationUpdate, db: DbSession, actor_id: ActorId):
    observation = get_or_404(db, Observation, observation_id)
    changes = body.model_dump(exclude_unset=True)
    reject_nulls(observation, changes)
    apply_update(db, observation, changes, record_type="observation", actor_id=actor_id)
    db.commit()
    return observation


@router.get("/cases/{case_id}/conflicts", response_model=list[ConflictOut])
def list_conflicts(case_id: int, db: DbSession, status: ConflictStatus | None = None):
    case = get_or_404(db, Case, case_id)
    query = select(Conflict).where(Conflict.case_id == case.id)
    if status is not None:
        query = query.where(Conflict.status == status)
    return db.scalars(query.order_by(Conflict.id)).all()


@router.patch("/conflicts/{conflict_id}", response_model=ConflictOut)
def update_conflict(conflict_id: int, body: ConflictUpdate, db: DbSession, actor_id: ActorId):
    conflict = get_or_404(db, Conflict, conflict_id)
    apply_update(db, conflict, body.model_dump(exclude_unset=True), record_type="conflict", actor_id=actor_id)
    db.commit()
    return conflict
