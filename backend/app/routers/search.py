"""Search routes (task A2 in docs/BACKEND_TASKS.md):

- POST /api/cases/{id}/ask
- GET /api/cases/{id}/search?q=

Both commit, because ranking embeds chunks that had no embedding yet.
"""

from fastapi import APIRouter, HTTPException

from app.core.deps import DbSession
from app.core.records import get_or_404
from app.models import Case
from app.schemas.search import AskRequest, AskResponse, SearchHit
from app.services.ai import client as ai_client
from app.services.search.answers import answer_question
from app.services.search.ranking import rank_chunks

router = APIRouter(tags=["search"])

SEARCH_LIMIT = 20


@router.get("/cases/{case_id}/search", response_model=list[SearchHit])
def search_case(case_id: int, db: DbSession, q: str = ""):
    case = get_or_404(db, Case, case_id)
    ranked = rank_chunks(db, case.id, q, client=ai_client.get_client())[:SEARCH_LIMIT]
    db.commit()
    return [r.hit() for r in ranked]


@router.post("/cases/{case_id}/ask", response_model=AskResponse)
def ask_case(case_id: int, body: AskRequest, db: DbSession):
    case = get_or_404(db, Case, case_id)
    try:
        response = answer_question(db, case.id, body, client=ai_client.get_client())
    except HTTPException:
        # Keep the chunk embeddings made on the way, so a retry doesn't redo them.
        db.commit()
        raise
    db.commit()
    return response
