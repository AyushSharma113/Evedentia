"""Ranks a case's chunks against a query (steps 4 and 5 in docs/PLAN.md).

With a Gemini client: cosine similarity between the query's embedding and every
chunk's, with numpy. Chunks without an embedding (seed data, or indexed before
a key was set) are embedded first and keep it. If an embedding call fails, the
ranking falls back to keywords for that request.

Without a client: keyword overlap, as rankChunks in frontend/src/mocks/handlers.ts.
"""

import logging
import re
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Evidence, EvidenceChunk
from app.schemas.search import Citation, SearchHit
from app.services.search.embeddings import embed_chunks, embed_query, from_bytes, has_embedding

logger = logging.getLogger(__name__)

STOPWORDS = frozenset(
    "the a an and or of to in on at for by with what when who whom which was were is are be been did does do it "
    "its this that from as about any there their they he she his her i my me we our you your how why where after "
    "before into than then so not no has have had".split()
)
_NON_WORD = re.compile(r"[^a-z0-9.:\s-]")


@dataclass
class RankedChunk:
    chunk: EvidenceChunk
    evidence_title: str
    score: float

    def citation(self) -> Citation:
        return Citation(
            chunk_id=self.chunk.id,
            evidence_id=self.chunk.evidence_id,
            evidence_title=self.evidence_title,
            page=self.chunk.page,
            passage=self.chunk.text,
        )

    def hit(self) -> SearchHit:
        return SearchHit(**self.citation().model_dump(), score=self.score)


def tokens(text: str) -> list[str]:
    """Lower-case words without stopwords, keeping times (16:10) and plates (KA-01)."""
    words = (word.strip(".:-") for word in _NON_WORD.sub(" ", text.lower()).split())
    return [word for word in words if len(word) > 1 and word not in STOPWORDS]


def case_chunks(db: Session, case_id: int) -> list[RankedChunk]:
    """Every chunk of the case's evidence, unscored, in the order they were indexed."""
    rows = db.execute(
        select(EvidenceChunk, Evidence.title)
        .join(Evidence, Evidence.id == EvidenceChunk.evidence_id)
        .where(Evidence.case_id == case_id)
        .order_by(EvidenceChunk.id)
    ).all()
    return [RankedChunk(chunk, title, 0.0) for chunk, title in rows]


def keyword_rank(candidates: list[RankedChunk], query: str) -> list[RankedChunk]:
    """The share of query terms a chunk contains (prefix matches count), plus 0.5
    if it contains the whole query. Chunks with no match are dropped."""
    terms = list(dict.fromkeys(tokens(query)))
    if not terms:
        return []
    phrase = query.strip().lower()
    ranked: list[RankedChunk] = []
    for candidate in candidates:
        words = set(tokens(candidate.chunk.text))
        hits = sum(1 for t in terms if any(w.startswith(t) or (t.startswith(w) and len(w) > 3) for w in words))
        score = hits / len(terms) + (0.5 if phrase in candidate.chunk.text.lower() else 0)
        if score > 0:
            ranked.append(RankedChunk(candidate.chunk, candidate.evidence_title, round(score, 2)))
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked


def semantic_rank(db: Session, client, candidates: list[RankedChunk], query: str) -> list[RankedChunk]:
    """Cosine similarity of every chunk to the query, highest first. Embeds the
    chunks that have no embedding yet and flushes them; the caller commits."""
    missing = [c.chunk for c in candidates if not has_embedding(c.chunk)]
    if missing:
        try:
            embed_chunks(client, missing)
        finally:
            # Keep the batches that were embedded, even if a later one failed.
            db.flush()
    query_vector = embed_query(client, query)
    matrix = np.vstack([from_bytes(c.chunk.embedding) for c in candidates])
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector)
    scores = (matrix @ query_vector) / np.where(norms == 0, 1, norms)
    order = np.argsort(-scores, kind="stable")
    return [
        RankedChunk(candidates[i].chunk, candidates[i].evidence_title, round(float(scores[i]), 4)) for i in order
    ]


def rank_chunks(db: Session, case_id: int, query: str, *, client, embed_text: str | None = None) -> list[RankedChunk]:
    """The case's chunks ranked against `query`, best first. `embed_text` is what
    to embed instead of the query, such as the question with earlier turns; the
    keyword fallback always uses the query alone."""
    if not query.strip():
        return []
    candidates = case_chunks(db, case_id)
    if not candidates:
        return []
    if client is not None:
        try:
            return semantic_rank(db, client, candidates, embed_text or query)
        except Exception:
            logger.warning("Semantic search failed for case %s; ranking by keyword", case_id, exc_info=True)
    return keyword_rank(candidates, query)
