"""Evidence routes (task B2 in docs/BACKEND_TASKS.md):

- GET /api/evidence?q=&type=
- POST, GET /api/cases/{id}/evidence
- GET, PATCH /api/evidence/{id}
- GET /api/evidence/{id}/file
- POST /api/evidence/{id}/reprocess
- POST /api/evidence-links
- DELETE /api/evidence-links/{id}
- POST /api/cases/{id}/verify

An evidence item's chain of custody is its audit rows: uploaded, viewed and
exported through the file route, modified by edits and reprocessing.
"""

import re
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import or_, select

from app.core.deps import ActorId, DbSession
from app.core.records import RECORD_MODELS, RECORD_NAMES, apply_update, get_or_404, record_audit
from app.core.storage import absolute_path
from app.models import AuditLog, Case, Evidence, EvidenceChunk, EvidenceLink
from app.models.base import utcnow
from app.schemas.common import EvidenceType
from app.schemas.evidence import AuditEntry, EvidenceDetail, EvidenceLinkCreate, EvidenceListItem, EvidenceUpdate
from app.schemas.evidence import Evidence as EvidenceOut
from app.schemas.evidence import EvidenceLink as EvidenceLinkOut
from app.schemas.evidence import HashMismatch, VerifyResult
from app.services.ai.pipeline import process_evidence
from app.services.evidence.ingest import ingest_upload
from app.services.evidence.integrity import verify_case_files
from app.services.evidence.links import case_id_of

router = APIRouter(tags=["evidence"])

# A Range request that doesn't start at byte 0 continues a view already logged,
# such as a <video> seeking, so it isn't logged again.
FIRST_RANGE = re.compile(r"^\s*bytes\s*=\s*0\s*-")


def _list_item(evidence: Evidence, case: Case) -> EvidenceListItem:
    return EvidenceListItem(
        **EvidenceOut.model_validate(evidence).model_dump(),
        case_number=case.case_number,
        case_title=case.title,
    )


def _file_name(evidence: Evidence) -> str:
    """The evidence title as a file name, keeping the original's extension if an
    edited title dropped it."""
    title = evidence.title.strip() or "evidence"
    suffix = Path(evidence.file_path or "").suffix
    if suffix and not title.lower().endswith(suffix.lower()):
        return title + suffix
    return title


@router.get("/evidence", response_model=list[EvidenceListItem])
def list_evidence(
    db: DbSession,
    q: str = "",
    type_: Annotated[EvidenceType | None, Query(alias="type")] = None,
):
    query = select(Evidence, Case).join(Case, Evidence.case_id == Case.id)
    if type_:
        query = query.where(Evidence.type == type_)
    if q := q.strip():
        searched = (Evidence.title, Evidence.description, Case.case_number, Case.title, Evidence.summary)
        query = query.where(or_(*(column.icontains(q, autoescape=True) for column in searched)))
    rows = db.execute(query.order_by(Evidence.created_at.desc(), Evidence.id.desc())).all()
    return [_list_item(evidence, case) for evidence, case in rows]


@router.post("/cases/{case_id}/evidence", response_model=list[EvidenceOut], status_code=status.HTTP_201_CREATED)
def upload_evidence(
    case_id: int,
    files: Annotated[list[UploadFile], File()],
    background_tasks: BackgroundTasks,
    db: DbSession,
    actor_id: ActorId,
):
    case = get_or_404(db, Case, case_id)
    created = [ingest_upload(db, case.id, f.filename, f.content_type, f.file, actor_id) for f in files]
    db.commit()
    for evidence in created:
        background_tasks.add_task(process_evidence, evidence.id)
    return created


@router.get("/cases/{case_id}/evidence", response_model=list[EvidenceOut])
def list_case_evidence(case_id: int, db: DbSession):
    case = get_or_404(db, Case, case_id)
    return db.scalars(
        select(Evidence).where(Evidence.case_id == case.id).order_by(Evidence.created_at.desc(), Evidence.id.desc())
    ).all()


@router.get("/evidence/{evidence_id}", response_model=EvidenceDetail)
def get_evidence(evidence_id: int, db: DbSession):
    evidence = get_or_404(db, Evidence, evidence_id)
    case = db.get(Case, evidence.case_id)
    links = db.scalars(select(EvidenceLink).where(EvidenceLink.evidence_id == evidence.id).order_by(EvidenceLink.id))
    custody = db.scalars(
        select(AuditLog)
        .where(AuditLog.record_type == "evidence", AuditLog.record_id == evidence.id)
        .order_by(AuditLog.at, AuditLog.id)
    )
    return EvidenceDetail(
        **_list_item(evidence, case).model_dump(),
        extracted_text=evidence.extracted_text,
        links=[EvidenceLinkOut.model_validate(link) for link in links],
        custody=[AuditEntry.model_validate(entry) for entry in custody],
    )


@router.patch("/evidence/{evidence_id}", response_model=EvidenceOut)
def update_evidence(evidence_id: int, body: EvidenceUpdate, db: DbSession, actor_id: ActorId):
    evidence = get_or_404(db, Evidence, evidence_id)
    # Every editable column is required, so an explicit null leaves it as it is.
    changes = {field: value for field, value in body.model_dump(exclude_unset=True).items() if value is not None}
    apply_update(db, evidence, changes, record_type="evidence", actor_id=actor_id)
    db.commit()
    return evidence


@router.get("/evidence/{evidence_id}/file", response_class=FileResponse)
def get_evidence_file(
    evidence_id: int,
    request: Request,
    db: DbSession,
    actor_id: ActorId,
    download: bool = False,
):
    """The original file, inline for previews or, with ?download=1, as an attachment.
    Logs "viewed" or "exported" against the investigator in ?investigator_id=."""
    evidence = get_or_404(db, Evidence, evidence_id)
    path = absolute_path(evidence.file_path) if evidence.file_path else None
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    range_header = request.headers.get("range")
    if range_header is None or FIRST_RANGE.match(range_header):
        action = "exported" if download else "viewed"
        record_audit(db, actor_id, action, "evidence", evidence.id, case_id=evidence.case_id)
        db.commit()
    return FileResponse(
        path,
        media_type=evidence.mime_type,
        filename=_file_name(evidence),
        content_disposition_type="attachment" if download else "inline",
    )


@router.post("/evidence/{evidence_id}/reprocess", response_model=EvidenceOut)
def reprocess_evidence(evidence_id: int, background_tasks: BackgroundTasks, db: DbSession, actor_id: ActorId):
    evidence = get_or_404(db, Evidence, evidence_id)
    evidence.processing_status = "queued"
    record_audit(
        db, actor_id, "modified", "evidence", evidence.id, case_id=evidence.case_id, detail={"reprocess": True}
    )
    db.commit()
    background_tasks.add_task(process_evidence, evidence.id)
    return evidence


@router.post("/evidence-links", response_model=EvidenceLinkOut, status_code=status.HTTP_201_CREATED)
def create_evidence_link(body: EvidenceLinkCreate, db: DbSession, actor_id: ActorId):
    evidence = get_or_404(db, Evidence, body.evidence_id)
    name = RECORD_NAMES[body.target_type]
    target = get_or_404(db, RECORD_MODELS[body.target_type], body.target_id, name)
    if case_id_of(db, body.target_type, target) != evidence.case_id:
        raise HTTPException(status_code=422, detail=f"{name} is not in the same case as the evidence")
    if body.chunk_id is not None:
        chunk = db.get(EvidenceChunk, body.chunk_id)
        if chunk is None or chunk.evidence_id != evidence.id:
            raise HTTPException(status_code=422, detail="Chunk is not part of this evidence")
    link = EvidenceLink(**body.model_dump())
    db.add(link)
    db.flush()
    record_audit(
        db,
        actor_id,
        "modified",
        body.target_type,
        body.target_id,
        case_id=evidence.case_id,
        detail={"linked_evidence": evidence.id},
    )
    db.commit()
    return link


@router.delete("/evidence-links/{link_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_evidence_link(link_id: int, db: DbSession, actor_id: ActorId) -> None:
    link = get_or_404(db, EvidenceLink, link_id, "Evidence link")
    case_id = db.scalar(select(Evidence.case_id).where(Evidence.id == link.evidence_id))
    record_audit(
        db,
        actor_id,
        "modified",
        link.target_type,
        link.target_id,
        case_id=case_id,
        detail={"unlinked_evidence": link.evidence_id},
    )
    db.delete(link)
    db.commit()


@router.post("/cases/{case_id}/verify", response_model=VerifyResult)
def verify_case(case_id: int, db: DbSession, actor_id: ActorId):
    case = get_or_404(db, Case, case_id)
    report = verify_case_files(db, case.id)
    verified_at = utcnow()
    record_audit(
        db,
        actor_id,
        "reviewed",
        "case",
        case.id,
        case_id=case.id,
        detail={"integrity_check": {"checked": report.checked, "mismatches": len(report.mismatches)}},
    )
    db.commit()
    return VerifyResult(
        checked=report.checked,
        mismatches=[HashMismatch.model_validate(m) for m in report.mismatches],
        verified_at=verified_at,
    )
