"""Tests for the shared foundation: app wiring, time handling, storage and the
helpers in app/core that every router builds on."""

import io
from datetime import UTC, datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.deps import ActorId
from app.core.records import apply_update, case_counts, delete_links_to, delete_relationships_of, record_audit
from app.core.storage import absolute_path, save_original, sha256_of
from app.models import AuditLog, Conflict, EvidenceLink, Relationship
from app.schemas.cases import Case as CaseSchema
from app.services.ai.pipeline import process_evidence
from app.services.search.indexing import index_evidence
from tests.factories import (
    make_case,
    make_event,
    make_evidence,
    make_investigator,
    make_object,
    make_observation,
    make_person,
)


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "ai_enabled": False}


def test_openapi_is_served(client):
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "Forensix API"


def test_every_schema_builds_a_json_schema():
    import importlib
    import pkgutil

    import app.schemas
    from pydantic import BaseModel

    for module_info in pkgutil.iter_modules(app.schemas.__path__):
        module = importlib.import_module(f"app.schemas.{module_info.name}")
        for value in vars(module).values():
            if isinstance(value, type) and issubclass(value, BaseModel) and value.__module__ == module.__name__:
                value.model_json_schema()


def test_datetimes_round_trip_as_utc(db):
    ist = timezone(timedelta(hours=5, minutes=30))
    case = make_case(db, incident_at=datetime(2026, 10, 3, 21, 40, tzinfo=ist))
    db.expire_all()
    stored = db.get(type(case), case.id)
    assert stored.incident_at == datetime(2026, 10, 3, 16, 10, tzinfo=UTC)
    payload = CaseSchema.model_validate(stored).model_dump(mode="json")
    assert payload["incident_at"] == "2026-10-03T16:10:00Z"
    assert payload["created_at"].endswith("Z")


def test_storage_saves_and_hashes(db):
    stored = save_original(7, "../Police report (final).pdf", io.BytesIO(b"hello"))
    assert stored.file_path.startswith("7/")
    assert stored.file_path.endswith("_Police-report-final-.pdf")
    assert stored.size_bytes == 5
    assert stored.sha256 == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert sha256_of(stored.file_path) == stored.sha256
    absolute_path(stored.file_path).write_bytes(b"tampered")
    assert sha256_of(stored.file_path) != stored.sha256
    assert sha256_of("7/missing.pdf") is None


def test_apply_update_audits_before_and_after(db):
    investigator = make_investigator(db)
    case = make_case(db)
    person = make_person(db, case, origin="ai", review_status="pending")

    diff = apply_update(
        db, person, {"name": "Anita D.", "notes": ""}, record_type="person", actor_id=investigator.id
    )
    db.commit()
    assert diff == {"name": {"before": "Anita Desai", "after": "Anita D."}}

    apply_update(db, person, {"review_status": "confirmed"}, record_type="person", actor_id=investigator.id)
    db.commit()

    entries = db.scalars(select(AuditLog).order_by(AuditLog.id)).all()
    assert [e.action for e in entries] == ["modified", "reviewed"]
    assert entries[0].case_id == case.id
    assert entries[0].actor_id == investigator.id
    assert entries[1].detail == {"review_status": {"before": "pending", "after": "confirmed"}}


def test_apply_update_skips_unchanged_fields(db):
    case = make_case(db)
    person = make_person(db, case)
    assert apply_update(db, person, {"name": person.name}, record_type="person", actor_id=None) == {}
    db.commit()
    assert db.scalars(select(AuditLog)).all() == []


def test_record_audit_serialises_datetimes(db):
    case = make_case(db)
    record_audit(db, None, "modified", "case", case.id, case_id=case.id, detail={"at": datetime(2026, 1, 1, tzinfo=UTC)})
    db.commit()
    assert db.scalars(select(AuditLog)).one().detail == {"at": "2026-01-01T00:00:00Z"}


def test_case_counts_are_computed(db):
    case = make_case(db)
    other = make_case(db)
    make_evidence(db, case)
    make_evidence(db, case)
    make_person(db, case, origin="ai", review_status="pending")
    make_event(db, case, origin="ai", review_status="confirmed")
    make_observation(db, case, origin="ai", review_status="rejected")
    make_person(db, case)  # investigator rows don't count as AI items
    make_person(db, other, origin="ai", review_status="pending")
    db.add_all([Conflict(case_id=case.id, description="Signal colour", status="open"),
                Conflict(case_id=case.id, description="Done", status="resolved")])
    db.commit()

    assert case_counts(db, case.id) == {
        "evidence": 2,
        "pending_reviews": 1,
        "open_conflicts": 1,
        "ai_items_total": 3,
        "ai_items_reviewed": 2,
    }


def test_deleting_a_node_removes_its_relationships_and_links(db):
    case = make_case(db)
    evidence = make_evidence(db, case)
    driver = make_person(db, case, role="suspect", name="Suresh Nair")
    witness = make_person(db, case)
    van = make_object(db, case)
    drove = Relationship(case_id=case.id, from_type="person", from_id=driver.id, to_type="object", to_id=van.id, kind="drove")
    saw = Relationship(case_id=case.id, from_type="person", from_id=witness.id, to_type="object", to_id=van.id, kind="witnessed")
    db.add_all([drove, saw])
    db.commit()
    db.add_all([
        EvidenceLink(evidence_id=evidence.id, target_type="relationship", target_id=drove.id),
        EvidenceLink(evidence_id=evidence.id, target_type="person", target_id=driver.id),
    ])
    db.commit()

    delete_relationships_of(db, "person", driver.id)
    delete_links_to(db, "person", driver.id)
    db.commit()

    assert [r.id for r in db.scalars(select(Relationship))] == [saw.id]
    assert db.scalars(select(EvidenceLink)).all() == []


def test_actor_id_comes_from_header_or_query(db):
    investigator = make_investigator(db)
    probe = FastAPI()

    @probe.get("/whoami")
    def whoami(actor_id: ActorId) -> dict:
        return {"actor_id": actor_id}

    client = TestClient(probe)
    assert client.get("/whoami", headers={"X-Investigator-Id": str(investigator.id)}).json() == {"actor_id": investigator.id}
    assert client.get(f"/whoami?investigator_id={investigator.id}").json() == {"actor_id": investigator.id}
    assert client.get("/whoami", headers={"X-Investigator-Id": "999"}).json() == {"actor_id": None}
    assert client.get("/whoami").json() == {"actor_id": None}


def test_pipeline_stub_marks_evidence_done(db):
    case = make_case(db)
    evidence = make_evidence(db, case, processing_status="queued")
    process_evidence(evidence.id)
    db.expire_all()
    evidence = db.get(type(evidence), evidence.id)
    assert evidence.processing_status == "done"
    assert evidence.processed_at is not None


def test_index_chunks_by_page(db):
    case = make_case(db)
    evidence = make_evidence(db, case)
    chunks = index_evidence(db, evidence.id, ["First para.\n\nSecond para.", "Page two."], paged=True)
    db.commit()
    assert [(c.page, c.text) for c in chunks] == [(1, "First para.\n\nSecond para."), (2, "Page two.")]
    again = index_evidence(db, evidence.id, ["Only one."])
    db.commit()
    assert [(c.page, c.ordinal) for c in again] == [(None, 0)]


def test_audit_entry_schema_reads_actor_name(db):
    from app.schemas.evidence import AuditEntry

    investigator = make_investigator(db)
    case = make_case(db)
    record_audit(db, investigator.id, "viewed", "case", case.id, case_id=case.id)
    record_audit(db, None, "viewed", "case", case.id, case_id=case.id)
    db.commit()
    entries = [AuditEntry.model_validate(row) for row in db.scalars(select(AuditLog).order_by(AuditLog.id))]
    assert [e.actor_name for e in entries] == ["Insp. Meera Rao", None]


def test_gemini_client_is_off_without_a_key():
    from app.services.ai.client import get_client

    assert get_client() is None
