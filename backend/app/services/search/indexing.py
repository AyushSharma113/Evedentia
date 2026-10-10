"""Chunking and embedding (step 4 in docs/PLAN.md).

The pipeline (app/services/ai/pipeline.py) calls index_evidence once it has the
text, with one string per page for PDFs or a single string otherwise. The seed
calls it too, so the seed case can be searched.

With a Gemini key the chunks are embedded here. Without one, or if the call
fails, they are stored without embeddings: search ranks them by keyword, and
embeds them on first use once a key is set (see ranking.py).
"""

import logging
import re

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.models import EvidenceChunk
from app.services.ai import client as ai_client
from app.services.search.embeddings import embed_chunks

logger = logging.getLogger(__name__)


def index_evidence(db: Session, evidence_id: int, pages: list[str], *, paged: bool = False) -> list[EvidenceChunk]:
    """Replaces the evidence's chunks. `paged` says whether `pages` are real pages
    (PDF), so chunks keep their page number; otherwise `page` stays null.
    Adds the rows to the session and flushes; the caller commits."""
    db.execute(delete(EvidenceChunk).where(EvidenceChunk.evidence_id == evidence_id))
    chunks: list[EvidenceChunk] = []
    for page_number, text in enumerate(pages, start=1):
        for paragraph in re.split(r"\n\s*\n", text):
            paragraph = paragraph.strip()
            if paragraph:
                chunks.append(
                    EvidenceChunk(
                        evidence_id=evidence_id,
                        ordinal=len(chunks),
                        page=page_number if paged else None,
                        text=paragraph,
                    )
                )
    client = ai_client.get_client()
    if client is not None and chunks:
        try:
            embed_chunks(client, chunks)
        except Exception:
            # Indexing must not fail the pipeline; search embeds these on first use.
            logger.warning("Embedding evidence %s failed; search embeds it on first use", evidence_id, exc_info=True)
    db.add_all(chunks)
    db.flush()
    return chunks
