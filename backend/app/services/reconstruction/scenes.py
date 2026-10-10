"""Assembles a scenario's stages, scene objects and annotations for the 3D scene
and the reconstruction module."""

from collections import defaultdict

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Reconstruction, ReconstructionStage, SceneAnnotation, SceneObject
from app.schemas.reconstruction import (
    Reconstruction as ReconstructionOut,
    ReconstructionDetail,
    ReconstructionStage as StageOut,
    SceneAnnotation as AnnotationOut,
    SceneObject as SceneObjectOut,
    StageDetail,
)


def annotations_of(db: Session, reconstruction_id: int) -> list[SceneAnnotation]:
    return list(
        db.scalars(
            select(SceneAnnotation)
            .where(SceneAnnotation.reconstruction_id == reconstruction_id)
            .order_by(SceneAnnotation.id)
        )
    )


def next_stage_order(db: Session, reconstruction_id: int) -> int:
    """One after the scenario's last stage, or 1 for its first."""
    last = db.scalar(
        select(func.max(ReconstructionStage.stage_order)).where(
            ReconstructionStage.reconstruction_id == reconstruction_id
        )
    )
    return 1 if last is None else last + 1


def reconstruction_detail(db: Session, reconstruction: Reconstruction) -> ReconstructionDetail:
    """GET /reconstructions/{id}: stages in stage_order, each with its scene objects,
    plus every annotation of the scenario."""
    stages = list(
        db.scalars(
            select(ReconstructionStage)
            .where(ReconstructionStage.reconstruction_id == reconstruction.id)
            .order_by(ReconstructionStage.stage_order, ReconstructionStage.id)
        )
    )
    objects_by_stage: dict[int, list[SceneObject]] = defaultdict(list)
    if stages:
        objects = db.scalars(
            select(SceneObject).where(SceneObject.stage_id.in_([s.id for s in stages])).order_by(SceneObject.id)
        )
        for scene_object in objects:
            objects_by_stage[scene_object.stage_id].append(scene_object)
    return ReconstructionDetail(
        **ReconstructionOut.model_validate(reconstruction).model_dump(),
        stages=[
            StageDetail(
                **StageOut.model_validate(stage).model_dump(),
                scene_objects=[SceneObjectOut.model_validate(o) for o in objects_by_stage[stage.id]],
            )
            for stage in stages
        ],
        annotations=[AnnotationOut.model_validate(a) for a in annotations_of(db, reconstruction.id)],
    )
