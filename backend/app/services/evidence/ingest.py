"""Evidence intake: store an uploaded original, record it and start its custody.

The router calls `ingest_upload` once per file in a multipart upload, commits, and
then queues the AI pipeline for each new row.
"""

import mimetypes
from typing import BinaryIO

from sqlalchemy.orm import Session

from app.core.records import record_audit
from app.core.storage import save_original
from app.models import Evidence

DEFAULT_MIME = "application/octet-stream"


def evidence_type_for(mime: str) -> str:
    """Guesses the evidence type from a MIME type, as the mock's evidenceTypeFor does."""
    if mime.startswith("image/"):
        return "photo"
    if mime.startswith("video/"):
        return "video"
    if mime.startswith("audio/"):
        return "audio"
    if mime == "application/pdf" or "word" in mime or mime.startswith("text/"):
        return "document"
    return "other"


def mime_type_for(content_type: str | None, filename: str) -> str:
    """The upload's MIME type without parameters (`text/plain; charset=utf-8` becomes
    `text/plain`). Clients that send no type, or the generic octet-stream, get one
    guessed from the file name."""
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if not mime or mime == DEFAULT_MIME:
        mime = mimetypes.guess_type(filename)[0] or DEFAULT_MIME
    return mime


def ingest_upload(
    db: Session,
    case_id: int,
    filename: str | None,
    content_type: str | None,
    source: BinaryIO,
    actor_id: int | None,
) -> Evidence:
    """Saves one uploaded file under storage/<case_id>/, adds its evidence row
    (queued for processing) and the "uploaded" audit entry that opens its chain of
    custody. Flushes so the row has an id; the caller commits."""
    name = (filename or "").strip() or "Untitled file"
    mime = mime_type_for(content_type, name)
    stored = save_original(case_id, name, source)
    evidence = Evidence(
        case_id=case_id,
        type=evidence_type_for(mime),
        title=name[:300],
        file_path=stored.file_path,
        mime_type=mime,
        size_bytes=stored.size_bytes,
        sha256=stored.sha256,
        processing_status="queued",
    )
    db.add(evidence)
    db.flush()
    record_audit(
        db,
        actor_id,
        "uploaded",
        "evidence",
        evidence.id,
        case_id=case_id,
        detail={"file_name": name, "sha256": stored.sha256},
    )
    return evidence
