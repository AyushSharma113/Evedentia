"""Chunking and embedding (step 4 in docs/PLAN.md).

The pipeline (app/services/ai/pipeline.py) calls index_evidence once it has the
text, with one string per page for PDFs or a single string otherwise. The seed
calls it too, so the seed case can be searched.

Chunks are about a page of text, split on paragraph boundaries. Each chunk is an
exact slice of the text it came from, so a passage quoted from the evidence is
found in its chunk, and a chunk's text is found in the evidence's extracted text.

With a Gemini key the chunks are embedded here. Without one, or if the call
fails, they are stored without embeddings: search ranks them by keyword, and
embeds them on first use once a key is set (see ranking.py).
"""

import logging
import re
from collections.abc import Iterator

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.models import EvidenceChunk
from app.services.ai import client as ai_client
from app.services.search.embeddings import embed_chunks

logger = logging.getLogger(__name__)

# About one page of text (some 350 words), well under the embedding model's input limit.
CHUNK_CHARS = 2000

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")

Span = tuple[int, int]


def _trim(text: str, start: int, end: int) -> Span:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _pieces(text: str, start: int, end: int, separator: re.Pattern[str]) -> list[Span]:
    """The non-blank parts of text[start:end] between separators, trimmed."""
    bounds: list[Span] = []
    position = start
    for match in separator.finditer(text, start, end):
        bounds.append((position, match.start()))
        position = match.end()
    bounds.append((position, end))
    return [span for span in (_trim(text, s, e) for s, e in bounds) if span[0] < span[1]]


def _units(text: str, max_chars: int) -> Iterator[Span]:
    """Paragraphs, with a paragraph longer than max_chars cut into sentences and a
    sentence still too long cut at the last space that fits."""
    for p_start, p_end in _pieces(text, 0, len(text), _PARAGRAPH_BREAK):
        if p_end - p_start <= max_chars:
            yield p_start, p_end
            continue
        for start, end in _pieces(text, p_start, p_end, _SENTENCE_BREAK):
            while end - start > max_chars:
                cut = text.rfind(" ", start + 1, start + max_chars + 1)
                if cut == -1:
                    cut = start + max_chars
                yield _trim(text, start, cut)
                start, _ = _trim(text, cut, end)
            yield start, end


def split_into_chunks(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Packs whole paragraphs into chunks of at most max_chars, in order."""
    chunks: list[str] = []
    current: Span | None = None
    for start, end in _units(text, max_chars):
        if current is not None and end - current[0] <= max_chars:
            current = (current[0], end)
            continue
        if current is not None:
            chunks.append(text[current[0] : current[1]])
        current = (start, end)
    if current is not None:
        chunks.append(text[current[0] : current[1]])
    return chunks


def index_evidence(db: Session, evidence_id: int, pages: list[str], *, paged: bool = False) -> list[EvidenceChunk]:
    """Replaces the evidence's chunks. `paged` says whether `pages` are real pages
    (PDF), so chunks keep their page number; otherwise `page` stays null. Chunks
    never cross from one string of `pages` to the next.
    Adds the rows to the session and flushes; the caller commits."""
    db.execute(delete(EvidenceChunk).where(EvidenceChunk.evidence_id == evidence_id))
    chunks: list[EvidenceChunk] = []
    for page_number, text in enumerate(pages, start=1):
        for piece in split_into_chunks(text):
            chunks.append(
                EvidenceChunk(
                    evidence_id=evidence_id,
                    ordinal=len(chunks),
                    page=page_number if paged else None,
                    text=piece,
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
