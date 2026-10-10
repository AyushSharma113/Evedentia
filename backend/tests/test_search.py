"""Tests for task A2: indexing, search and cited answers. Gemini is replaced by a
fake client whose embeddings count a few topic words, so cosine ranking is
predictable."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import select

from app.core.config import get_settings
from app.models import AiRun, EvidenceChunk
from app.services.ai import client as ai_client
from app.services.search import answers, embeddings
from app.services.search.embeddings import EMBED_DIMENSIONS, to_bytes
from app.services.search.indexing import index_evidence, split_into_chunks
from tests.factories import make_case, make_evidence

TOPICS = ["signal", "amber", "van", "hatchback", "ambulance", "injur", "fracture"]

STATEMENT_PAGES = [
    "I was waiting at the bus stop on Ring Road at about 4 pm.\n\n"
    "The hatchback's signal was amber when it entered the junction. The van came from the left.",
    "I called 112 straight away. The ambulance came at 16:25.",
]
CCTV_TEXT = "16:09:58 Van moves off from the stop line.\n\n16:10:03 Van strikes the hatchback on the near side."
MEDICAL_TEXT = "Rohan Kapoor: fractured left wrist and a minor head injury."


def fake_vector(text: str) -> list[float]:
    lower = text.lower()
    vector = np.zeros(EMBED_DIMENSIONS, dtype=np.float32)
    for i, topic in enumerate(TOPICS):
        vector[i] = lower.count(topic)
    vector[-1] = 0.1
    return vector.tolist()


class FakeModels:
    def __init__(self, answer=None, fail_embed=False, fail_generate=False):
        self.answer = answer
        self.fail_embed = fail_embed
        self.fail_generate = fail_generate
        self.embedded: list[SimpleNamespace] = []
        self.generated: list[SimpleNamespace] = []

    def embed_content(self, *, model, contents, config):
        if self.fail_embed:
            raise RuntimeError("embedding service down")
        self.embedded.append(
            SimpleNamespace(
                model=model, task_type=config.task_type, dims=config.output_dimensionality, texts=list(contents)
            )
        )
        return SimpleNamespace(embeddings=[SimpleNamespace(values=fake_vector(text)) for text in contents])

    def generate_content(self, *, model, contents, config):
        if self.fail_generate:
            raise RuntimeError("Gemini down")
        self.generated.append(SimpleNamespace(model=model, contents=contents, config=config))
        return SimpleNamespace(text=self.answer if isinstance(self.answer, str) else json.dumps(self.answer))

    def calls(self, task_type: str) -> list[SimpleNamespace]:
        return [call for call in self.embedded if call.task_type == task_type]


class FakeClient:
    def __init__(self, **kwargs):
        self.models = FakeModels(**kwargs)


@pytest.fixture
def use_gemini(monkeypatch):
    def install(**kwargs) -> FakeClient:
        fake = FakeClient(**kwargs)
        monkeypatch.setattr(ai_client, "get_client", lambda: fake)
        return fake

    return install


@pytest.fixture
def case_files(db):
    """A case with a two-page statement, a CCTV log and a medical report, indexed
    without Gemini, and a second case whose chunk must never show up."""
    case = make_case(db)
    statement = make_evidence(db, case, type="statement", title="Statement - Anita Desai.pdf")
    cctv = make_evidence(db, case, type="video", title="CCTV log.txt", mime_type="text/plain")
    medical = make_evidence(db, case, type="medical", title="Medical report.pdf")
    other_case = make_case(db, title="Other case")
    other = make_evidence(db, other_case, title="Other case.txt")
    index_evidence(db, statement.id, STATEMENT_PAGES, paged=True)
    index_evidence(db, cctv.id, [CCTV_TEXT])
    index_evidence(db, medical.id, [MEDICAL_TEXT])
    index_evidence(db, other.id, ["The signal was red."])
    db.commit()
    return SimpleNamespace(case=case, statement=statement, cctv=cctv, medical=medical, other_case=other_case)


def chunk_with(db, text: str) -> EvidenceChunk:
    return db.scalars(select(EvidenceChunk).where(EvidenceChunk.text.contains(text))).one()


# Indexing ---------------------------------------------------------------------


def test_chunks_pack_whole_paragraphs_up_to_the_size():
    text = "Para one is here.\n\nPara two.\n\n\nPara three is longer than that."
    assert split_into_chunks(text, max_chars=40) == [
        "Para one is here.\n\nPara two.",
        "Para three is longer than that.",
    ]
    assert split_into_chunks(text) == [text]


def test_long_paragraphs_split_on_sentences_then_spaces():
    sentences = "First sentence here. Second sentence here. Third one."
    assert split_into_chunks(sentences, max_chars=25) == [
        "First sentence here.",
        "Second sentence here.",
        "Third one.",
    ]
    assert split_into_chunks("aaaa bbbb cccc dddd", max_chars=10) == ["aaaa bbbb", "cccc dddd"]
    assert split_into_chunks("x" * 25, max_chars=10) == ["x" * 10, "x" * 10, "x" * 5]


def test_chunks_are_exact_slices_of_the_text():
    text = "  Heading\n \nA  line with   odd spacing.\nSame paragraph.\n\n\tLast one.  "
    chunks = split_into_chunks(text, max_chars=30)
    assert chunks == ["Heading", "A  line with   odd spacing.", "Same paragraph.\n\n\tLast one."]
    positions = [text.index(chunk) for chunk in chunks]
    assert positions == sorted(positions)
    assert split_into_chunks(" \n\n ") == []


def test_index_keeps_pages_apart_and_replaces_old_chunks(db):
    case = make_case(db)
    evidence = make_evidence(db, case)
    chunks = index_evidence(db, evidence.id, ["Page one.\n\nStill page one.", "", "Page three."], paged=True)
    db.commit()
    assert [(c.ordinal, c.page, c.text) for c in chunks] == [
        (0, 1, "Page one.\n\nStill page one."),
        (1, 3, "Page three."),
    ]
    again = index_evidence(db, evidence.id, ["Not paged."])
    db.commit()
    assert [(c.ordinal, c.page, c.text) for c in again] == [(0, None, "Not paged.")]
    assert db.scalars(select(EvidenceChunk.text)).all() == ["Not paged."]


def test_index_without_a_key_stores_chunks_without_embeddings(db, case_files):
    chunks = db.scalars(select(EvidenceChunk).where(EvidenceChunk.evidence_id == case_files.statement.id)).all()
    assert chunks
    assert all(chunk.embedding is None for chunk in chunks)


def test_index_embeds_chunks_in_batches_as_float32(db, monkeypatch, use_gemini):
    monkeypatch.setattr(embeddings, "EMBED_BATCH_SIZE", 2)
    fake = use_gemini()
    case = make_case(db)
    evidence = make_evidence(db, case)
    chunks = index_evidence(db, evidence.id, ["One.", "Two signal.", "Three van."], paged=True)
    db.commit()

    calls = fake.models.calls("RETRIEVAL_DOCUMENT")
    assert [call.texts for call in calls] == [["One.", "Two signal."], ["Three van."]]
    assert {call.model for call in calls} == {get_settings().gemini_embed_model}
    assert {call.dims for call in calls} == {EMBED_DIMENSIONS}
    db.expire_all()
    for chunk in chunks:
        stored = db.get(EvidenceChunk, chunk.id).embedding
        assert len(stored) == EMBED_DIMENSIONS * 4
        assert np.array_equal(np.frombuffer(stored, dtype=np.float32), np.float32(fake_vector(chunk.text)))


def test_index_keeps_the_chunks_when_embedding_fails(db, use_gemini):
    use_gemini(fail_embed=True)
    case = make_case(db)
    evidence = make_evidence(db, case)
    chunks = index_evidence(db, evidence.id, ["Some text."])
    db.commit()
    assert [(chunk.text, chunk.embedding) for chunk in chunks] == [("Some text.", None)]


# Search -----------------------------------------------------------------------


def test_search_ranks_by_keyword_without_a_key(client, db, case_files):
    response = client.get(f"/api/cases/{case_files.case.id}/search", params={"q": "van strikes"})
    assert response.status_code == 200
    hits = response.json()
    cctv = chunk_with(db, "Van strikes")
    assert hits[0] == {
        "chunk_id": cctv.id,
        "evidence_id": case_files.cctv.id,
        "evidence_title": "CCTV log.txt",
        "page": None,
        "passage": CCTV_TEXT,
        "score": 1.5,
    }
    assert [(hit["evidence_title"], hit["page"], hit["score"]) for hit in hits[1:]] == [
        ("Statement - Anita Desai.pdf", 1, 0.5)
    ]

    hits = client.get(f"/api/cases/{case_files.case.id}/search", params={"q": "van hatchback"}).json()
    assert [(hit["evidence_title"], hit["score"]) for hit in hits] == [
        ("Statement - Anita Desai.pdf", 1.0),
        ("CCTV log.txt", 1.0),
    ]


def test_keyword_search_matches_prefixes_and_the_whole_phrase(client, case_files):
    hits = client.get(f"/api/cases/{case_files.case.id}/search", params={"q": "fracture"}).json()
    assert [(hit["evidence_title"], hit["score"]) for hit in hits] == [("Medical report.pdf", 1.5)]


def test_search_stays_inside_the_case(client, case_files):
    hits = client.get(f"/api/cases/{case_files.case.id}/search", params={"q": "signal red"}).json()
    assert {hit["evidence_id"] for hit in hits} == {case_files.statement.id}


def test_search_with_no_terms_is_empty(client, case_files):
    for q in ["", "   ", "what was it"]:
        assert client.get(f"/api/cases/{case_files.case.id}/search", params={"q": q}).json() == []
    assert client.get(f"/api/cases/{case_files.case.id}/search").json() == []


def test_search_unknown_case_is_404(client):
    response = client.get("/api/cases/99/search", params={"q": "van"})
    assert response.status_code == 404
    assert response.json() == {"detail": "Case not found"}


def test_search_returns_at_most_20_hits(client, db):
    case = make_case(db)
    evidence = make_evidence(db, case)
    index_evidence(db, evidence.id, [f"Van sighting {i}." for i in range(25)], paged=True)
    db.commit()
    assert len(client.get(f"/api/cases/{case.id}/search", params={"q": "van"}).json()) == 20


def test_search_embeds_missing_chunks_once_and_ranks_by_cosine(client, db, case_files, use_gemini):
    fake = use_gemini()
    hits = client.get(f"/api/cases/{case_files.case.id}/search", params={"q": "What colour was the signal?"}).json()

    assert hits[0]["passage"] == STATEMENT_PAGES[0]
    assert len(hits) == 4  # every chunk of the case, none of the other case's
    assert [hit["score"] for hit in hits] == sorted((hit["score"] for hit in hits), reverse=True)
    assert 0 < hits[0]["score"] <= 1
    documents = fake.models.calls("RETRIEVAL_DOCUMENT")
    assert sorted(text for call in documents for text in call.texts) == sorted(hit["passage"] for hit in hits)
    assert [call.texts for call in fake.models.calls("RETRIEVAL_QUERY")] == [["What colour was the signal?"]]

    db.expire_all()
    case_chunks = [chunk_with(db, hit["passage"]) for hit in hits]
    assert all(len(chunk.embedding) == EMBED_DIMENSIONS * 4 for chunk in case_chunks)
    assert chunk_with(db, "The signal was red.").embedding is None

    client.get(f"/api/cases/{case_files.case.id}/search", params={"q": "ambulance"})
    assert len(fake.models.calls("RETRIEVAL_DOCUMENT")) == len(documents)
    assert len(fake.models.calls("RETRIEVAL_QUERY")) == 2


def test_search_reembeds_vectors_of_another_size(client, db, case_files, use_gemini):
    medical = chunk_with(db, "fractured")
    medical.embedding = to_bytes([1.0, 2.0, 3.0])
    db.commit()
    fake = use_gemini()
    client.get(f"/api/cases/{case_files.case.id}/search", params={"q": "fracture"})
    assert any(medical.text in call.texts for call in fake.models.calls("RETRIEVAL_DOCUMENT"))
    db.expire_all()
    assert len(chunk_with(db, "fractured").embedding) == EMBED_DIMENSIONS * 4


def test_search_falls_back_to_keywords_when_embedding_fails(client, case_files, use_gemini):
    use_gemini(fail_embed=True)
    hits = client.get(f"/api/cases/{case_files.case.id}/search", params={"q": "fracture"}).json()
    assert [(hit["evidence_title"], hit["score"]) for hit in hits] == [("Medical report.pdf", 1.5)]


# Answers ----------------------------------------------------------------------


def test_ask_without_a_key_quotes_the_best_keyword_matches(client, db, case_files):
    response = client.post(
        f"/api/cases/{case_files.case.id}/ask",
        json={
            "question": "When did the van strike the hatchback?",
            "history": [
                {"role": "user", "content": "Who are the witnesses?"},
                {"role": "assistant", "content": "Anita Desai."},
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == (
        "Gemini is not configured, so this answer is keyword matching, not AI. "
        "The most relevant passages in this case's evidence are:\n"
        '1. CCTV log.txt: "16:10:03 Van strikes the hatchback on the near side."\n'
        '2. Statement - Anita Desai.pdf: "The hatchback\'s signal was amber when it entered the junction."'
    )
    assert [c["chunk_id"] for c in body["citations"]] == [
        chunk_with(db, "Van strikes").id,
        chunk_with(db, "signal was amber").id,
    ]
    assert body["follow_ups"] == ["Summarise the evidence in this case.", "What happened first?"]
    assert db.scalars(select(AiRun)).all() == []


def test_ask_without_a_match_says_so(client, case_files):
    body = client.post(f"/api/cases/{case_files.case.id}/ask", json={"question": "Where is the knife?"}).json()
    assert body["answer"].startswith("I couldn't find anything about that")
    assert body["citations"] == []
    assert len(body["follow_ups"]) == 3


def test_ask_returns_a_cited_answer_from_gemini(client, db, case_files, use_gemini):
    ambulance = chunk_with(db, "ambulance")
    amber = chunk_with(db, "signal was amber")
    fake = use_gemini(
        answer={
            "answer": " The ambulance arrived at 16:25, according to Anita Desai's statement. ",
            "chunk_ids": [ambulance.id, 9999, ambulance.id, amber.id],
            "follow_ups": [
                "What colour was the signal?",
                "Who called 112?",
                "  ",
                "When did the van move off?",
                "Who called 112?",
                "What injuries were recorded?",
                "One too many?",
            ],
        }
    )
    response = client.post(
        f"/api/cases/{case_files.case.id}/ask",
        json={
            "question": "And when did the ambulance arrive?",
            "history": [
                {"role": "user", "content": "What colour was the signal?"},
                {"role": "assistant", "content": "Amber, according to the statement."},
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The ambulance arrived at 16:25, according to Anita Desai's statement."
    assert body["citations"] == [
        {
            "chunk_id": ambulance.id,
            "evidence_id": case_files.statement.id,
            "evidence_title": "Statement - Anita Desai.pdf",
            "page": 2,
            "passage": ambulance.text,
        },
        {
            "chunk_id": amber.id,
            "evidence_id": case_files.statement.id,
            "evidence_title": "Statement - Anita Desai.pdf",
            "page": 1,
            "passage": amber.text,
        },
    ]
    assert body["follow_ups"] == ["Who called 112?", "When did the van move off?", "What injuries were recorded?"]

    [query] = fake.models.calls("RETRIEVAL_QUERY")
    assert query.texts == [
        "Investigator: What colour was the signal?\n"
        "Assistant: Amber, according to the statement.\n"
        "Investigator: And when did the ambulance arrive?"
    ]
    [call] = fake.models.generated
    assert call.model == get_settings().gemini_model
    assert call.config.response_mime_type == "application/json"
    assert call.config.response_schema is answers.GeminiAnswer
    assert f"[chunk {ambulance.id}] Statement - Anita Desai.pdf, page 2\n{ambulance.text}" in call.contents
    assert "The signal was red." not in call.contents
    assert call.contents.endswith("Investigator: And when did the ambulance arrive?")

    [run] = db.scalars(select(AiRun)).all()
    assert (run.task, run.case_id, run.evidence_id, run.model) == (
        "answer",
        case_files.case.id,
        None,
        get_settings().gemini_model,
    )
    assert run.output["question"] == "And when did the ambulance arrive?"
    assert run.output["response"]["chunk_ids"] == [ambulance.id, 9999, ambulance.id, amber.id]
    assert len(run.output["context_chunk_ids"]) == 4


def test_ask_gives_gemini_only_the_top_chunks(client, db, monkeypatch, case_files, use_gemini):
    monkeypatch.setattr(answers, "CONTEXT_CHUNKS", 1)
    medical = chunk_with(db, "fractured")
    amber = chunk_with(db, "signal was amber")
    fake = use_gemini(answer={"answer": "A fractured wrist.", "chunk_ids": [amber.id, medical.id], "follow_ups": []})
    body = client.post(
        f"/api/cases/{case_files.case.id}/ask", json={"question": "Which fracture did the victim have?"}
    ).json()

    [call] = fake.models.generated
    assert f"[chunk {medical.id}]" in call.contents
    assert f"[chunk {amber.id}]" not in call.contents
    assert [c["chunk_id"] for c in body["citations"]] == [medical.id]
    assert body["follow_ups"] == []


def test_ask_returns_503_when_gemini_fails_and_keeps_new_embeddings(client, db, case_files, use_gemini):
    use_gemini(fail_generate=True)
    response = client.post(f"/api/cases/{case_files.case.id}/ask", json={"question": "Who called 112?"})
    assert response.status_code == 503
    assert response.json() == {"detail": "The AI service is unavailable. Try again."}
    assert db.scalars(select(AiRun)).all() == []
    db.expire_all()
    assert chunk_with(db, "ambulance").embedding is not None


def test_ask_returns_502_when_the_answer_is_unreadable(client, db, case_files, use_gemini):
    use_gemini(answer="Sorry, here is some prose instead of JSON.")
    response = client.post(f"/api/cases/{case_files.case.id}/ask", json={"question": "Who called 112?"})
    assert response.status_code == 502
    assert db.scalars(select(AiRun)).all() == []


def test_ask_without_chunks_does_not_call_gemini(client, db, use_gemini):
    fake = use_gemini(answer={"answer": "x", "chunk_ids": [], "follow_ups": []})
    case = make_case(db)
    body = client.post(f"/api/cases/{case.id}/ask", json={"question": "What happened?"}).json()
    assert body["answer"].startswith("I couldn't find anything about that")
    assert fake.models.generated == []


def test_ask_validates_the_request(client, case_files):
    url = f"/api/cases/{case_files.case.id}/ask"
    assert client.post(url, json={"question": ""}).status_code == 422
    assert client.post(url, json={"question": "Why?", "extra": 1}).status_code == 422
    bad_turn = {"role": "system", "content": "x"}
    assert client.post(url, json={"question": "Why?", "history": [bad_turn]}).status_code == 422
    assert client.post("/api/cases/99/ask", json={"question": "Why?"}).status_code == 404
