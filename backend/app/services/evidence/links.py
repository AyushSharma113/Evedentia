"""Checks for evidence links: which case a link target belongs to."""

from sqlalchemy.orm import Session

from app.models import Base, Reconstruction, ReconstructionStage


def case_id_of(db: Session, record_type: str, row: Base) -> int | None:
    """The case a record belongs to. Most tables carry case_id; reconstruction
    stages, scene objects and annotations reach it through their reconstruction."""
    if record_type == "case":
        return row.id
    if record_type == "scene_object":
        row = db.get(ReconstructionStage, row.stage_id)
        if row is None:
            return None
        record_type = "stage"
    if record_type in ("stage", "annotation"):
        reconstruction = db.get(Reconstruction, row.reconstruction_id)
        return reconstruction.case_id if reconstruction else None
    return getattr(row, "case_id", None)
