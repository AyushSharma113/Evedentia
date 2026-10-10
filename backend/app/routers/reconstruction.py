"""Reconstruction routes (task B4 in docs/BACKEND_TASKS.md): competing scenarios of a
case, their stages, the scene objects placed at each stage, and annotations.

- GET, POST /api/cases/{id}/reconstructions
- GET /api/reconstructions/{id}
- POST /api/reconstructions/{id}/stages
- PUT /api/stages/{id}/scene-objects
- GET, POST /api/reconstructions/{id}/annotations
"""

from fastapi import APIRouter, status
from sqlalchemy import delete, select

from app.core.deps import ActorId, DbSession
from app.core.records import delete_links_to, get_or_404, record_audit
from app.models import Case, CaseObject, Event, Reconstruction, ReconstructionStage, SceneAnnotation, SceneObject
from app.schemas.reconstruction import (
    Reconstruction as ReconstructionOut,
    ReconstructionCreate,
    ReconstructionDetail,
    ReconstructionStage as StageOut,
    SceneAnnotation as AnnotationOut,
    SceneAnnotationCreate,
    SceneObject as SceneObjectOut,
    SceneObjectInput,
    StageCreate,
)
from app.services.reconstruction.scenes import annotations_of, next_stage_order, reconstruction_detail
from app.services.reconstruction.validation import ids_in_case, invalid

router = APIRouter(tags=["reconstruction"])


@router.get("/cases/{case_id}/reconstructions", response_model=list[ReconstructionOut])
def list_reconstructions(case_id: int, db: DbSession):
    case = get_or_404(db, Case, case_id)
    return db.scalars(
        select(Reconstruction).where(Reconstruction.case_id == case.id).order_by(Reconstruction.id)
    ).all()


@router.post(
    "/cases/{case_id}/reconstructions", response_model=ReconstructionOut, status_code=status.HTTP_201_CREATED
)
def create_reconstruction(case_id: int, body: ReconstructionCreate, db: DbSession, actor_id: ActorId):
    case = get_or_404(db, Case, case_id)
    reconstruction = Reconstruction(case_id=case.id, version=1, **body.model_dump())
    db.add(reconstruction)
    db.flush()
    record_audit(
        db, actor_id, "modified", "reconstruction", reconstruction.id, case_id=case.id, detail={"created": True}
    )
    db.commit()
    return reconstruction


@router.get("/reconstructions/{reconstruction_id}", response_model=ReconstructionDetail)
def get_reconstruction(reconstruction_id: int, db: DbSession):
    return reconstruction_detail(db, get_or_404(db, Reconstruction, reconstruction_id))


@router.post(
    "/reconstructions/{reconstruction_id}/stages", response_model=StageOut, status_code=status.HTTP_201_CREATED
)
def create_stage(reconstruction_id: int, body: StageCreate, db: DbSession, actor_id: ActorId):
    reconstruction = get_or_404(db, Reconstruction, reconstruction_id)
    # Each event once, in the order sent.
    event_ids = list(dict.fromkeys(body.event_ids))
    known = ids_in_case(db, Event, event_ids, reconstruction.case_id)
    for index, event_id in enumerate(body.event_ids):
        if event_id not in known:
            raise invalid(("event_ids", index), f"Event {event_id} is not an event of this case")
    stage = ReconstructionStage(
        reconstruction_id=reconstruction.id,
        stage_order=body.stage_order if body.stage_order is not None else next_stage_order(db, reconstruction.id),
        label=body.label,
        event_ids=event_ids,
    )
    db.add(stage)
    db.flush()
    record_audit(db, actor_id, "modified", "stage", stage.id, case_id=reconstruction.case_id, detail={"created": True})
    db.commit()
    return stage


@router.put("/stages/{stage_id}/scene-objects", response_model=list[SceneObjectOut])
def replace_scene_objects(stage_id: int, body: list[SceneObjectInput], db: DbSession, actor_id: ActorId):
    """Replaces the stage's placements with the list sent. The old rows are deleted,
    with any evidence links to them, and the new ones get new IDs."""
    stage = get_or_404(db, ReconstructionStage, stage_id, "Stage")
    case_id = db.get(Reconstruction, stage.reconstruction_id).case_id
    known = ids_in_case(db, CaseObject, (o.object_id for o in body if o.object_id is not None), case_id)
    for index, placement in enumerate(body):
        if placement.object_id is not None and placement.object_id not in known:
            raise invalid((index, "object_id"), f"Object {placement.object_id} is not an object of this case")

    old_ids = db.scalars(select(SceneObject.id).where(SceneObject.stage_id == stage.id)).all()
    for old_id in old_ids:
        delete_links_to(db, "scene_object", old_id)
    db.execute(delete(SceneObject).where(SceneObject.stage_id == stage.id))

    placed = [
        SceneObject(
            stage_id=stage.id,
            model_id=placement.model_id,
            object_id=placement.object_id,
            label=placement.label,
            position=list(placement.position),
            rotation=list(placement.rotation),
            scale=list(placement.scale),
        )
        for placement in body
    ]
    db.add_all(placed)
    db.flush()
    record_audit(
        db,
        actor_id,
        "modified",
        "stage",
        stage.id,
        case_id=case_id,
        detail={"scene_objects": {"before": len(old_ids), "after": len(placed)}},
    )
    db.commit()
    return placed


@router.get("/reconstructions/{reconstruction_id}/annotations", response_model=list[AnnotationOut])
def list_annotations(reconstruction_id: int, db: DbSession):
    reconstruction = get_or_404(db, Reconstruction, reconstruction_id)
    return annotations_of(db, reconstruction.id)


@router.post(
    "/reconstructions/{reconstruction_id}/annotations",
    response_model=AnnotationOut,
    status_code=status.HTTP_201_CREATED,
)
def create_annotation(reconstruction_id: int, body: SceneAnnotationCreate, db: DbSession, actor_id: ActorId):
    reconstruction = get_or_404(db, Reconstruction, reconstruction_id)
    if body.stage_id is not None:
        stage = db.get(ReconstructionStage, body.stage_id)
        if stage is None or stage.reconstruction_id != reconstruction.id:
            raise invalid(("stage_id",), f"Stage {body.stage_id} is not a stage of this reconstruction")
    annotation = SceneAnnotation(reconstruction_id=reconstruction.id, **body.model_dump())
    db.add(annotation)
    db.flush()
    record_audit(
        db, actor_id, "modified", "annotation", annotation.id, case_id=reconstruction.case_id, detail={"created": True}
    )
    db.commit()
    return annotation
