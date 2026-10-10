"""Cited answers (step 5 in docs/PLAN.md).

The question, with the last few turns of the conversation, is embedded and the
case's chunks ranked against it. Gemini answers from the top chunks alone and
returns the chunk IDs it used and up to three follow-up questions. Each call is
logged to ai_runs.

Without a Gemini client the answer quotes the best keyword matches, as
answerQuestion in frontend/src/mocks/handlers.ts.
"""

import logging
import re

from fastapi import HTTPException
from google.genai import types
from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import AiRun
from app.schemas.search import AskRequest, AskResponse, AskTurn
from app.services.search.ranking import RankedChunk, rank_chunks, tokens

logger = logging.getLogger(__name__)

# Chunks Gemini sees per question, and chunks quoted by the keyword answer.
CONTEXT_CHUNKS = 6
KEYWORD_CHUNKS = 3
# Earlier turns embedded with the question and shown to Gemini.
HISTORY_TURNS = 6
# Long assistant answers are cut so they don't drown the question.
TURN_CHARS = 1000
MAX_FOLLOW_UPS = 3

NOT_FOUND = "I couldn't find anything about that in this case's evidence. Try naming a person, vehicle, place or time."
DEFAULT_FOLLOW_UPS = ["Summarise the evidence in this case.", "Who are the witnesses?", "What happened first?"]

SYSTEM_INSTRUCTION = """You help police investigators question the evidence in one case.
Answer the investigator's latest question using only the evidence passages given. Each passage starts with its chunk ID and source file.

Rules:
- Use nothing but the passages: no outside knowledge, no guessing. If they don't answer the question, say so plainly and say what is missing.
- Be factual and brief. Say which source each fact comes from, and point out where sources disagree.
- chunk_ids lists the IDs of the passages your answer relies on, and only those.
- follow_ups lists up to three short questions the investigator could ask next that this evidence might answer. Don't repeat questions already asked."""


# The JSON Gemini returns. No docstring (it would be sent as the schema's
# description) and no defaults (the Gemini API rejects them in a response schema).
class GeminiAnswer(BaseModel):
    answer: str
    chunk_ids: list[int]
    follow_ups: list[str]


def _speaker(turn: AskTurn) -> str:
    return "Investigator" if turn.role == "user" else "Assistant"


def _conversation(history: list[AskTurn], question: str) -> str:
    lines = [f"{_speaker(turn)}: {turn.content.strip()[:TURN_CHARS]}" for turn in history[-HISTORY_TURNS:]]
    return "\n".join([*lines, f"Investigator: {question.strip()}"])


def _passage_heading(ranked: RankedChunk) -> str:
    page = f", page {ranked.chunk.page}" if ranked.chunk.page is not None else ""
    return f"[chunk {ranked.chunk.id}] {ranked.evidence_title}{page}"


def _prompt(ranked: list[RankedChunk], body: AskRequest) -> str:
    passages = "\n\n".join(f"{_passage_heading(r)}\n{r.chunk.text}" for r in ranked)
    return f"Evidence passages:\n\n{passages}\n\nConversation:\n{_conversation(body.history, body.question)}"


def _asked(body: AskRequest) -> set[str]:
    return {q.strip().lower() for q in [body.question, *(t.content for t in body.history if t.role == "user")]}


def _follow_ups(candidates: list[str], body: AskRequest) -> list[str]:
    asked = _asked(body)
    follow_ups: list[str] = []
    for question in candidates:
        question = question.strip()
        if question and question.lower() not in asked and question not in follow_ups:
            follow_ups.append(question)
    return follow_ups[:MAX_FOLLOW_UPS]


def _best_sentence(passage: str, question: str) -> str:
    """The sentence of the passage that contains the most question terms."""
    terms = tokens(question)
    sentences = re.split(r"(?<=[.!?])\s+", passage)
    best = max(sentences, key=lambda s: sum(1 for t in terms if t in s.lower()), default=passage)
    return best.strip()


def keyword_answer(db: Session, case_id: int, body: AskRequest) -> AskResponse:
    ranked = rank_chunks(db, case_id, body.question, client=None)[:KEYWORD_CHUNKS]
    follow_ups = _follow_ups(DEFAULT_FOLLOW_UPS, body)
    if not ranked:
        return AskResponse(answer=NOT_FOUND, citations=[], follow_ups=follow_ups)
    lines = [
        f'{i}. {r.evidence_title}: "{_best_sentence(r.chunk.text, body.question)}"' for i, r in enumerate(ranked, 1)
    ]
    answer = (
        "Gemini is not configured, so this answer is keyword matching, not AI. "
        "The most relevant passages in this case's evidence are:\n" + "\n".join(lines)
    )
    return AskResponse(answer=answer, citations=[r.citation() for r in ranked], follow_ups=follow_ups)


def answer_question(db: Session, case_id: int, body: AskRequest, *, client) -> AskResponse:
    """Answers from the case's evidence. Adds the ai_runs row to the session; the
    caller commits. Raises 503 if Gemini fails and 502 if its reply can't be read."""
    if client is None:
        return keyword_answer(db, case_id, body)

    embed_text = _conversation(body.history, body.question)
    ranked = rank_chunks(db, case_id, body.question, client=client, embed_text=embed_text)[:CONTEXT_CHUNKS]
    if not ranked:
        return AskResponse(answer=NOT_FOUND, citations=[], follow_ups=_follow_ups(DEFAULT_FOLLOW_UPS, body))

    settings = get_settings()
    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=_prompt(ranked, body),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=GeminiAnswer,
                temperature=0.2,
            ),
        )
    except Exception as error:
        logger.warning("Gemini answer failed for case %s", case_id, exc_info=True)
        raise HTTPException(status_code=503, detail="The AI service is unavailable. Try again.") from error
    try:
        reply = GeminiAnswer.model_validate_json(response.text or "")
    except ValidationError as error:
        logger.warning("Unreadable Gemini answer for case %s: %r", case_id, response.text)
        detail = "The AI service returned an answer that could not be read."
        raise HTTPException(status_code=502, detail=detail) from error

    by_id = {r.chunk.id: r for r in ranked}
    cited = [by_id[chunk_id] for chunk_id in dict.fromkeys(reply.chunk_ids) if chunk_id in by_id]
    result = AskResponse(
        answer=reply.answer.strip(),
        citations=[r.citation() for r in cited],
        follow_ups=_follow_ups(reply.follow_ups, body),
    )
    db.add(
        AiRun(
            case_id=case_id,
            task="answer",
            model=settings.gemini_model,
            output={
                "question": body.question,
                "history_turns": len(body.history),
                "context_chunk_ids": [r.chunk.id for r in ranked],
                "response": reply.model_dump(),
            },
        )
    )
    return result
