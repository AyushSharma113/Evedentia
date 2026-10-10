"""The evidence pipeline (steps 2 to 4 in docs/PLAN.md), run as a FastAPI
background task after an upload or a reprocess:

    background_tasks.add_task(process_evidence, evidence.id)

It opens its own session, because the request's session is closed by then, and
moves the evidence from queued to processing, then to done or failed:

1. Extract the stored file's text (extract.py), with Gemini OCR and transcription
   for what can't be read locally.
2. Index the text (services/search/indexing.py) before linking anything, so
   evidence links can carry chunk IDs.
3. Structure it with one Gemini call (structure.py) and save the summary and the
   proposed records as pending AI rows linked to their passages (proposals.py).

Every Gemini call is logged to ai_runs. Without GEMINI_API_KEY, step 1 reads what
it can locally, step 2 still runs and step 3 is skipped, so the evidence ends up
done with no summary. Any error marks it failed, with the reason in its summary;
what earlier steps committed (extracted text, chunks, AI run logs) stays.
"""

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import SessionLocal
from app.core.storage import absolute_path
from app.models import Case, CaseObject, Evidence, Person
from app.models.base import utcnow
from app.services.ai.calls import decode_json, log_run
from app.services.ai.client import get_client
from app.services.ai.extract import extract
from app.services.ai.proposals import PassageIndex, clear_pending_proposals, relink_passages, save_proposals
from app.services.ai.structure import parse_output, request_structure
from app.services.search.indexing import index_evidence

logger = logging.getLogger(__name__)

MAX_REASON_LENGTH = 300


def process_evidence(evidence_id: int) -> None:
    with SessionLocal() as db:
        evidence = db.get(Evidence, evidence_id)
        if evidence is None:
            return
        evidence.processing_status = "processing"
        evidence.summary = None
        db.commit()
        try:
            run_pipeline(db, evidence)
        except Exception as exc:  # noqa: BLE001 - every failure is reported on the evidence
            logger.exception("Processing evidence %s failed", evidence_id)
            db.rollback()
            evidence = db.get(Evidence, evidence_id)
            if evidence is None:
                return
            evidence.processing_status = "failed"
            evidence.summary = f"Processing failed: {failure_reason(exc)}"
        else:
            evidence.processing_status = "done"
        evidence.processed_at = utcnow()
        db.commit()


def run_pipeline(db: Session, evidence: Evidence) -> None:
    """Extract, index, structure. Commits after each step; raises on failure."""
    if not evidence.file_path:
        return
    path = absolute_path(evidence.file_path)
    if not path.is_file():
        raise FileNotFoundError("the stored file is missing")

    client = get_client()
    extraction = extract(path, evidence.mime_type, client)
    for task, output in extraction.runs:
        log_run(db, task, output, evidence_id=evidence.id, case_id=evidence.case_id)
    text = extraction.text
    evidence.extracted_text = text or None
    db.commit()

    chunks = index_evidence(db, evidence.id, extraction.pages if text else [], paged=extraction.paged)
    passages = PassageIndex(chunks)
    relink_passages(db, evidence.id, passages)
    db.commit()

    if client is None or not text:
        return
    case = db.get(Case, evidence.case_id)
    persons = list(db.scalars(select(Person).where(Person.case_id == case.id).order_by(Person.id)))
    objects = list(db.scalars(select(CaseObject).where(CaseObject.case_id == case.id).order_by(CaseObject.id)))
    raw = request_structure(client, case, evidence, text, persons, objects)
    decoded = decode_json(raw)
    log_run(db, "structure", raw if decoded is None else decoded, evidence_id=evidence.id, case_id=evidence.case_id)
    db.commit()
    if decoded is None:
        raise ValueError("Gemini returned no valid JSON")

    output = parse_output(decoded)
    evidence.summary = output.summary or None
    clear_pending_proposals(db, evidence)
    save_proposals(db, evidence, output, passages)
    db.commit()


def failure_reason(exc: Exception) -> str:
    reason = str(exc).strip() or type(exc).__name__
    if len(reason) > MAX_REASON_LENGTH:
        reason = reason[: MAX_REASON_LENGTH - 1] + "…"
    return reason
