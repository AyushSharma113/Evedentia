"""Tests for conflict detection (task A3). Gemini is never called: tests patch
get_client in conflicts.py (and in pipeline.py for the end-to-end tests) with a
fake that answers the conflict call with set findings."""

import json
import re
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app.core.storage import save_bytes
from app.models import AiRun, Conflict, Evidence, EvidenceLink, Observation
from app.services.ai import conflicts, pipeline
from app.services.ai.conflicts import Findings, detect_conflicts
from app.services.ai.pipeline import process_evidence
from tests.factories import make_case, make_event, make_evidence, make_location, make_object, make_observation

AT = datetime(2026, 10, 3, 16, 10, 10, tzinfo=UTC)
AMBER = "Anita Desai says the hatchback's signal was amber when it entered the junction."
RED = "Harpreet Singh says the hatchback's signal was red before it reached the junction."
SIGNAL_CONFLICT = "Witnesses disagree on the hatchback's signal when it entered the junction."
NOTHING_STRUCTURED = {
    "summary": "Nothing new.",
    **{key: [] for key in ("persons", "objects", "locations", "events", "relationships", "observations")},
}


class FakeModels:
    def __init__(self, gemini):
        self.gemini = gemini

    def generate_content(self, *, model, contents, config=None):
        if config is not None and config.response_schema is Findings:
            self.gemini.prompts.append(contents[0])
            if self.gemini.error:
                raise self.gemini.error
            findings = self.gemini.findings
            output = findings(contents[0]) if callable(findings) else findings
        else:
            output = self.gemini.structured
        return SimpleNamespace(text=output if isinstance(output, str) else json.dumps(output))


class FakeGemini:
    """findings is the conflict call's JSON (a dict or raw text), or a function of
    the prompt that returns it; structured answers the pipeline's structure call."""

    def __init__(self, findings=None, error=None, structured=None):
        self.findings = findings if findings is not None else {"findings": []}
        self.error = error
        self.structured = structured
        self.prompts: list[str] = []
        self.models = FakeModels(self)


@pytest.fixture
def gemini(monkeypatch):
    """Turns Gemini on for conflict detection and the pipeline; returns a factory."""

    def use(**kwargs) -> FakeGemini:
        fake = FakeGemini(**kwargs)
        monkeypatch.setattr(conflicts, "get_client", lambda: fake)
        monkeypatch.setattr(pipeline, "get_client", lambda: fake)
        return fake

    return use


def conflict(ids: list[int], description: str = SIGNAL_CONFLICT, kind: str = "conflict") -> dict:
    return {"kind": kind, "observation_ids": ids, "description": description}


def signal_statements(db, case, **fields) -> tuple[Observation, Observation]:
    """Two witnesses' statements about the hatchback's signal, each from its own evidence."""
    hatchback = make_object(db, case)
    observations = []
    for title, statement in (("Desai statement.txt", AMBER), ("Singh statement.txt", RED)):
        evidence = make_evidence(db, case, type="statement", title=title, mime_type="text/plain")
        observation = make_observation(
            db,
            case,
            subject_type="object",
            subject_id=hatchback.id,
            statement=statement,
            observed_at=AT,
            origin="ai",
            review_status="pending",
            **fields,
        )
        db.add(EvidenceLink(evidence_id=evidence.id, target_type="observation", target_id=observation.id))
        db.commit()
        observations.append(observation)
    return observations[0], observations[1]


def conflicts_of(db, case) -> list[Conflict]:
    db.expire_all()
    return list(db.scalars(select(Conflict).where(Conflict.case_id == case.id).order_by(Conflict.id)))


def conflict_runs(db) -> list[AiRun]:
    return list(db.scalars(select(AiRun).where(AiRun.task == "conflicts").order_by(AiRun.id)))


# --- Without Gemini --------------------------------------------------------------


def test_without_a_key_nothing_happens(db, monkeypatch):
    case = make_case(db)
    signal_statements(db, case)
    monkeypatch.setattr(conflicts, "get_client", lambda: None)

    assert detect_conflicts(db, case.id) == []

    assert conflicts_of(db, case) == []
    assert conflict_runs(db) == []


# --- The signal-colour case ------------------------------------------------------


def test_two_witnesses_who_disagree_open_exactly_one_conflict(db, gemini):
    case = make_case(db)
    amber, red = signal_statements(db, case)
    fake = gemini(findings={"findings": [conflict([red.id, amber.id])]})

    created = detect_conflicts(db, case.id)

    rows = conflicts_of(db, case)
    assert [row.id for row in rows] == [row.id for row in created]
    assert len(rows) == 1
    assert rows[0].observation_ids == [amber.id, red.id]
    assert rows[0].description == SIGNAL_CONFLICT
    assert rows[0].status == "open"

    # One call, which saw both statements, whose account each is and why they were grouped.
    assert len(fake.prompts) == 1
    prompt = fake.prompts[0]
    assert (
        f'[{amber.id}] (about White hatchback; at 2026-10-03T16:10:10+00:00; from "Desai statement.txt") {AMBER}'
        in prompt
    )
    assert f'from "Singh statement.txt") {RED}' in prompt
    assert f"- Subject: White hatchback; Time: 2026-10-03 16:10 UTC: {amber.id}, {red.id}" in prompt

    [run] = conflict_runs(db)
    assert run.case_id == case.id
    assert run.evidence_id is None
    assert run.output["observation_ids"] == [amber.id, red.id]
    assert run.output["response"] == {"findings": [conflict([red.id, amber.id])]}


def test_a_new_statement_from_the_pipeline_opens_a_conflict_in_the_review_queue_and_alerts(db, client, gemini):
    case = make_case(db)
    hatchback = make_object(db, case)

    def statement_output(statement: str, passage: str) -> dict:
        return {
            "summary": "A witness describes the hatchback's signal.",
            "persons": [],
            "objects": [
                {
                    "ref": "O1",
                    "kind": "vehicle",
                    "label": "White hatchback",
                    "plate": "KA-01-AB-1234",
                    "number": "",
                    "make": "",
                    "colour": "White",
                    "passage": passage,
                }
            ],
            "locations": [],
            "events": [],
            "relationships": [],
            "observations": [
                {"subject_ref": "O1", "statement": statement, "observed_at": AT.isoformat(), "passage": passage}
            ],
        }

    def signal_findings(prompt: str) -> dict:
        ids = [int(i) for i in re.findall(r"^\[(\d+)\] .*signal", prompt, flags=re.MULTILINE)]
        return {"findings": [conflict(ids)] if len(ids) > 1 else []}

    fake = gemini(findings=signal_findings)

    desai_text = "Statement of Anita Desai. The hatchback KA 01 AB 1234 entered the junction on amber."
    desai = stored_statement(db, case, "Desai statement.txt", desai_text)
    fake.structured = statement_output(AMBER, "The hatchback KA 01 AB 1234 entered the junction on amber.")
    process_evidence(desai.id)
    # One statement has nothing to be compared with, so no call is made.
    assert fake.prompts == []
    assert conflicts_of(db, case) == []

    singh_text = "Statement of Harpreet Singh. The hatchback's light was already red before it reached the junction."
    singh = stored_statement(db, case, "Singh statement.txt", singh_text)
    fake.structured = statement_output(RED, "The hatchback's light was already red before it reached the junction.")
    process_evidence(singh.id)

    observations = list(db.scalars(select(Observation).where(Observation.case_id == case.id).order_by(Observation.id)))
    assert [(o.subject_type, o.subject_id, o.statement) for o in observations] == [
        ("object", hatchback.id, AMBER),
        ("object", hatchback.id, RED),
    ]
    [row] = conflicts_of(db, case)
    assert row.observation_ids == [o.id for o in observations]
    assert db.get(Evidence, singh.id).processing_status == "done"
    assert len(fake.prompts) == 1

    # Reprocessing finds the same statement, so nothing new is asked or opened.
    process_evidence(singh.id)
    assert len(fake.prompts) == 1
    assert len(conflicts_of(db, case)) == 1

    review = client.get(f"/api/cases/{case.id}/conflicts", params={"status": "open"}).json()
    assert [(c["id"], c["observation_ids"], c["description"]) for c in review] == [
        (row.id, row.observation_ids, SIGNAL_CONFLICT)
    ]
    alerts = client.get("/api/alerts").json()
    assert f"conflict-{row.id}" in [alert["id"] for alert in alerts]


def stored_statement(db, case, filename: str, text: str) -> Evidence:
    stored = save_bytes(case.id, filename, text.encode())
    return make_evidence(
        db,
        case,
        type="statement",
        title=filename,
        file_path=stored.file_path,
        size_bytes=stored.size_bytes,
        sha256=stored.sha256,
        mime_type="text/plain",
        processing_status="queued",
    )


def test_a_failed_conflict_call_does_not_fail_the_evidence(db, gemini):
    case = make_case(db)
    signal_statements(db, case)
    fake = gemini(error=RuntimeError("Gemini is down"), structured=NOTHING_STRUCTURED)
    stored = stored_statement(db, case, "note.txt", "A short note.")

    process_evidence(stored.id)

    assert len(fake.prompts) == 1
    db.expire_all()
    assert db.get(Evidence, stored.id).processing_status == "done"
    assert conflicts_of(db, case) == []


# --- Duplicates and re-runs -------------------------------------------------------


def test_a_finding_with_the_same_observations_as_an_existing_conflict_is_skipped(db, gemini):
    case = make_case(db)
    amber, red = signal_statements(db, case)
    third = make_observation(
        db,
        case,
        subject_type="object",
        subject_id=amber.subject_id,
        statement="The hatchback was speeding.",
        observed_at=AT,
    )
    existing = Conflict(
        case_id=case.id, observation_ids=[red.id, amber.id], description="Already flagged.", status="resolved"
    )
    db.add(existing)
    db.commit()
    gemini(
        findings={
            "findings": [
                conflict([amber.id, red.id]),
                conflict([amber.id, third.id], "Accounts disagree on the hatchback's speed."),
                # The same finding twice in one answer opens it once.
                conflict([third.id, amber.id], "The same again."),
            ]
        }
    )

    created = detect_conflicts(db, case.id)

    assert [(c.observation_ids, c.description) for c in created] == [
        ([amber.id, third.id], "Accounts disagree on the hatchback's speed.")
    ]
    assert len(conflicts_of(db, case)) == 2
    assert db.get(Conflict, existing.id).status == "resolved"


def test_a_group_that_already_has_a_conflict_is_not_sent(db, gemini):
    case = make_case(db)
    amber, red = signal_statements(db, case)
    db.add(Conflict(case_id=case.id, observation_ids=[amber.id, red.id], description=SIGNAL_CONFLICT))
    db.commit()
    fake = gemini(findings={"findings": [conflict([amber.id, red.id], "Again.")]})

    assert detect_conflicts(db, case.id) == []

    assert fake.prompts == []
    assert conflict_runs(db) == []
    assert len(conflicts_of(db, case)) == 1


def test_only_groups_with_a_new_observation_are_sent_again(db, gemini):
    case = make_case(db)
    amber, red = signal_statements(db, case)
    fake = gemini()  # finds nothing
    assert detect_conflicts(db, case.id) == []
    assert len(fake.prompts) == 1

    # Nothing new since: no second call.
    assert detect_conflicts(db, case.id) == []
    assert len(fake.prompts) == 1

    third = make_observation(
        db, case, subject_type="object", subject_id=amber.subject_id, statement="A third account.", observed_at=AT
    )
    fake.findings = {"findings": [conflict([amber.id, third.id], "The third account disagrees.")]}

    assert [c.observation_ids for c in detect_conflicts(db, case.id)] == [[amber.id, third.id]]
    assert len(fake.prompts) == 2
    assert f"[{third.id}]" in fake.prompts[1]
    assert f"[{red.id}]" in fake.prompts[1]
    assert len(conflict_runs(db)) == 2


def test_after_a_failed_or_unreadable_call_the_next_run_asks_again(db, gemini):
    case = make_case(db)
    amber, red = signal_statements(db, case)
    fake = gemini(error=RuntimeError("timeout"))

    assert detect_conflicts(db, case.id) == []
    assert conflict_runs(db) == []

    fake.error = None
    fake.findings = "not json"
    assert detect_conflicts(db, case.id) == []
    [run] = conflict_runs(db)
    assert run.output["response"] == "not json"

    fake.findings = {"findings": [conflict([amber.id, red.id])]}
    assert len(detect_conflicts(db, case.id)) == 1
    assert len(fake.prompts) == 3


# --- Findings ---------------------------------------------------------------------


def test_gaps_are_saved_as_missing_information(db, gemini):
    case = make_case(db)
    amber, red = signal_statements(db, case)
    gemini(
        findings={
            "findings": [
                conflict([amber.id], "No account says how fast the hatchback was going.", kind="gap"),
                conflict([amber.id, red.id], "Missing: the van's signal at the same moment.", kind="gap"),
            ]
        }
    )

    created = detect_conflicts(db, case.id)

    assert [(c.observation_ids, c.description, c.status) for c in created] == [
        ([amber.id], "Missing information: No account says how fast the hatchback was going.", "open"),
        ([amber.id, red.id], "Missing: the van's signal at the same moment.", "open"),
    ]


def test_findings_that_do_not_fit_are_dropped(db, gemini):
    case = make_case(db)
    amber, red = signal_statements(db, case)
    other_case = make_case(db, title="Another case")
    stranger = make_observation(db, other_case, statement="Elsewhere.")
    gemini(
        findings={
            "findings": [
                conflict([amber.id, stranger.id]),  # only one of its IDs was sent
                conflict([amber.id, red.id], "   "),  # no description
                {"kind": "contradiction", "observation_ids": [amber.id, red.id], "description": "Bad kind."},
                {"observation_ids": [amber.id, red.id]},
                conflict([amber.id, 9999, red.id], "Unknown IDs are dropped from the finding."),
            ]
        }
    )

    created = detect_conflicts(db, case.id)

    assert [(c.observation_ids, c.description) for c in created] == [
        ([amber.id, red.id], "Unknown IDs are dropped from the finding.")
    ]


def test_rejected_observations_are_left_out(db, gemini):
    case = make_case(db)
    amber, red = signal_statements(db, case)
    red.review_status = "rejected"
    db.commit()
    fake = gemini(findings={"findings": [conflict([amber.id, red.id])]})

    assert detect_conflicts(db, case.id) == []
    assert fake.prompts == []


# --- Grouping ---------------------------------------------------------------------


def test_observations_are_grouped_by_subject_place_and_time(db, gemini):
    case = make_case(db)
    junction = make_location(db, case, address="Ring Road and 5th Avenue junction")
    collision = make_event(db, case, location_id=junction.id)
    van = make_object(db, case, label="Blue van", details={})

    def observe(subject_type: str, subject_id: int, statement: str, at: datetime | None) -> Observation:
        return make_observation(
            db, case, subject_type=subject_type, subject_id=subject_id, statement=statement, observed_at=at
        )

    # The same place: the junction itself, and an event there.
    junction_note = observe("location", junction.id, "The junction signal was amber from 21:40:08.", None)
    collision_note = observe("event", collision.id, "The van struck the hatchback in the junction.", None)
    # The same moment, about different subjects; the last one is an hour later.
    van_early = observe("object", van.id, "The van moved off early.", AT)
    case_note = observe("case", case.id, "A bus passed through the junction.", AT + timedelta(minutes=9))
    van_later = observe("object", van.id, "The van was towed away.", AT + timedelta(hours=1))
    fake = gemini()

    detect_conflicts(db, case.id)

    [prompt] = fake.prompts
    groups = prompt.split("Groups to compare:\n")[1].splitlines()
    assert groups == [
        f"- Subject: Blue van: {van_early.id}, {van_later.id}",
        f"- Place: Ring Road and 5th Avenue junction: {junction_note.id}, {collision_note.id}",
        f"- Time: 2026-10-03 16:10 UTC to 2026-10-03 16:19 UTC: {van_early.id}, {case_note.id}",
    ]
    assert f"[{collision_note.id}] (about event: The van strikes the hatchback.)" in prompt
    assert f"[{case_note.id}] (about the case; at " in prompt
