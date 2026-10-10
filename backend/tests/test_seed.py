"""Tests for the seed (app/seed.py): it loads frontend/src/mocks/seed.ts into the
database, with real evidence files whose hashes verify."""

import io
import wave

import pymupdf
import pytest
from sqlalchemy import func, select

from app import seed
from app.core.records import RECORD_MODELS, case_counts
from app.core.storage import absolute_path, sha256_of, storage_root
from app.models import (
    AuditLog,
    Case,
    CaseObject,
    Conflict,
    Event,
    Evidence,
    EvidenceChunk,
    EvidenceLink,
    Injury,
    Investigator,
    Location,
    Observation,
    Person,
    Reconstruction,
    ReconstructionStage,
    Relationship,
    SceneAnnotation,
    SceneObject,
)
from app.schemas.cases import Case as CaseOut
from app.schemas.cases import Investigator as InvestigatorOut
from app.schemas.evidence import AuditEntry, Evidence as EvidenceOut, EvidenceLink as EvidenceLinkOut
from app.schemas.graph import Relationship as RelationshipOut
from app.schemas.injuries import Injury as InjuryOut
from app.schemas.people import CaseObject as CaseObjectOut, Location as LocationOut, Person as PersonOut
from app.schemas.reconstruction import (
    Reconstruction as ReconstructionOut,
    ReconstructionStage as StageOut,
    SceneAnnotation as SceneAnnotationOut,
    SceneObject as SceneObjectOut,
)
from app.schemas.review import Conflict as ConflictOut, Observation as ObservationOut
from app.schemas.timeline import CaseEvent as CaseEventOut
from tests.factories import make_case

# Rows per table in frontend/src/mocks/seed.ts. Chunks are left out: their number
# depends on how app/services/search/indexing.py splits the text.
MOCK_COUNTS = {
    "investigators": 4,
    "cases": 4,
    "evidence": 12,
    "persons": 6,
    "objects": 4,
    "locations": 8,
    "events": 14,
    "relationships": 12,
    "observations": 8,
    "conflicts": 2,
    "injuries": 3,
    "reconstructions": 2,
    "reconstruction_stages": 5,
    "scene_objects": 10,
    "scene_annotations": 3,
    "evidence_links": 32,
    # One upload per evidence file, a view, two reviews and a status change.
    "audit_log": 16,
}

SCHEMAS = [
    (Investigator, InvestigatorOut),
    (Case, CaseOut),
    (Evidence, EvidenceOut),
    (Person, PersonOut),
    (CaseObject, CaseObjectOut),
    (Location, LocationOut),
    (Event, CaseEventOut),
    (Relationship, RelationshipOut),
    (Observation, ObservationOut),
    (Conflict, ConflictOut),
    (Injury, InjuryOut),
    (Reconstruction, ReconstructionOut),
    (ReconstructionStage, StageOut),
    (SceneObject, SceneObjectOut),
    (SceneAnnotation, SceneAnnotationOut),
    (EvidenceLink, EvidenceLinkOut),
    (AuditLog, AuditEntry),
]


@pytest.fixture
def seeded(db):
    seed.load_seed(db)
    return db


def table_counts(db) -> dict[str, int]:
    counts = seed.row_counts(db)
    counts.pop("evidence_chunks", None)
    return counts


def stored_files() -> list:
    return sorted(path for path in storage_root().rglob("*") if path.is_file())


def test_counts_match_the_mock(seeded):
    assert table_counts(seeded) == MOCK_COUNTS


def test_every_stored_file_matches_its_hash(seeded):
    evidence = seeded.scalars(select(Evidence)).all()
    assert len(stored_files()) == len(evidence) == 12
    for item in evidence:
        assert item.file_path and item.file_path.startswith(f"{item.case_id}/")
        assert sha256_of(item.file_path) == item.sha256
        assert absolute_path(item.file_path).stat().st_size == item.size_bytes


def test_files_are_real_and_match_their_type(seeded):
    for item in seeded.scalars(select(Evidence)):
        data = absolute_path(item.file_path).read_bytes()
        if item.mime_type == "application/pdf":
            seed_pages = next(s.pages for s in seed.evidence_seeds() if s.id == item.id)
            with pymupdf.open(stream=data, filetype="pdf") as document:
                assert document.page_count == len(seed_pages)
                first_line = seed_pages[0].splitlines()[0]
                assert first_line in document[0].get_text()
        elif item.mime_type == "text/plain":
            assert data.decode() == item.extracted_text
        elif item.mime_type == "audio/wav":
            with wave.open(io.BytesIO(data)) as recording:
                assert (recording.getnchannels(), recording.getframerate(), recording.getnframes()) == (1, 8000, 32000)
        elif item.mime_type == "image/svg+xml":
            assert data.startswith(b"<svg") and b"Mock evidence photo" in data
        elif item.mime_type == "video/mp4":
            assert data[4:8] == b"ftyp"
        else:
            pytest.fail(f"Unexpected type {item.mime_type}")


def test_text_is_extracted_and_indexed(seeded):
    for item in seeded.scalars(select(Evidence)):
        assert item.processing_status == "done"
        assert item.processed_at is None
        chunks = seeded.scalars(select(EvidenceChunk).where(EvidenceChunk.evidence_id == item.id)).all()
        if item.type == "photo":
            assert item.extracted_text is None
            assert chunks == []
        else:
            assert item.extracted_text
            assert chunks

    police_report = seeded.get(Evidence, 1)
    assert police_report.extracted_text == "\n\n".join(seed.POLICE_REPORT)
    pages = seeded.scalars(select(EvidenceChunk.page).where(EvidenceChunk.evidence_id == 1)).all()
    assert set(pages) == {1, 2}
    statement_pages = seeded.scalars(select(EvidenceChunk.page).where(EvidenceChunk.evidence_id == 2)).all()
    assert set(statement_pages) == {None}


def test_links_point_at_the_chunk_with_their_passage(seeded):
    links = seeded.scalars(select(EvidenceLink).order_by(EvidenceLink.id)).all()
    assert [link.id for link in links] == list(range(1, 33))
    for link in links:
        chunk = seeded.get(EvidenceChunk, link.chunk_id)
        assert chunk is not None and chunk.evidence_id == link.evidence_id
        assert link.passage in chunk.text
        evidence = seeded.get(Evidence, link.evidence_id)
        target = seeded.get(RECORD_MODELS[link.target_type], link.target_id)
        assert target is not None and target.case_id == evidence.case_id


def test_rows_fit_the_api_schemas(seeded):
    for model, schema in SCHEMAS:
        for row in seeded.scalars(select(model)):
            schema.model_validate(row)


def test_road_accident_case_matches_the_mock(seeded):
    case = CaseOut.model_validate(seeded.get(Case, 1)).model_dump(mode="json")
    assert case["case_number"] == "FX-2026-0001"
    assert case["status"] == "in_progress"
    assert case["investigator_ids"] == [1, 2, 3, 4]
    # 21:40:12 IST on 3 October.
    assert case["incident_at"] == "2026-10-03T16:10:12Z"
    assert case["updated_at"] == "2026-10-05T05:50:00Z"

    assert case_counts(seeded, 1) == {
        "evidence": 8,
        "pending_reviews": 15,
        "open_conflicts": 2,
        "ai_items_total": 22,
        "ai_items_reviewed": 7,
    }

    conflicts = seeded.scalars(select(Conflict).order_by(Conflict.id)).all()
    assert [conflict.observation_ids for conflict in conflicts] == [[1, 2], [3, 4, 5]]

    event = seeded.get(Event, 6)
    assert (event.certainty, event.origin, event.review_status) == ("disputed", "ai", "pending")
    assert seeded.get(CaseObject, 2).details == {"plate": "KA-05-CD-5678", "make": "Tata Ace", "colour": "Blue"}

    for reconstruction_id in (1, 2):
        stages = seeded.scalars(
            select(ReconstructionStage)
            .where(ReconstructionStage.reconstruction_id == reconstruction_id)
            .order_by(ReconstructionStage.stage_order)
        ).all()
        assert [stage.label for stage in stages][:2] == ["Approach", "Impact"]
        for stage in stages:
            placed = seeded.scalars(select(SceneObject).where(SceneObject.stage_id == stage.id)).all()
            assert {(item.object_id, item.model_id) for item in placed} == {(1, "hatchback"), (2, "delivery-van")}

    impact_van = seeded.scalars(select(SceneObject).where(SceneObject.stage_id == 2, SceneObject.object_id == 2)).one()
    assert SceneObjectOut.model_validate(impact_van).model_dump(mode="json")["position"] == [-1.75, 0.0, 0.5]


def test_audit_history_is_in_time_order(seeded):
    entries = seeded.scalars(select(AuditLog).order_by(AuditLog.id)).all()
    assert [entry.at for entry in entries] == sorted(entry.at for entry in entries)
    assert all(entry.case_id is not None for entry in entries)
    uploads = [entry for entry in entries if entry.action == "uploaded"]
    assert sorted(entry.record_id for entry in uploads) == list(range(1, 13))
    assert {entry.actor_name for entry in uploads} == {"Const. Rahul Verma"}
    case_audit = seeded.scalar(select(func.count()).select_from(AuditLog).where(AuditLog.case_id == 1))
    assert case_audit == 12


def test_main_loads_an_empty_database(db, capsys):
    assert seed.main([]) == 0
    assert "4 cases" in capsys.readouterr().out
    assert table_counts(db) == MOCK_COUNTS


def test_main_refuses_a_database_with_cases(db, capsys):
    make_case(db)
    assert seed.main([]) == 1
    assert "--reset" in capsys.readouterr().err
    assert table_counts(db) == {"cases": 1}
    assert stored_files() == []


def test_reset_replaces_the_data_and_the_files(seeded):
    make_case(seeded, title="Not in the seed", case_number="FX-2026-0099")
    old_files = stored_files()
    seeded.close()

    assert seed.main(["--reset"]) == 0

    assert table_counts(seeded) == MOCK_COUNTS
    assert seeded.scalar(select(func.count()).select_from(Case).where(Case.case_number == "FX-2026-0099")) == 0
    new_files = stored_files()
    assert len(new_files) == 12 and not set(old_files) & set(new_files)
    for item in seeded.scalars(select(Evidence)):
        assert sha256_of(item.file_path) == item.sha256


def test_a_failed_load_leaves_no_rows_or_files(db, monkeypatch):
    bad_link = {"evidence_id": 1, "target_type": "event", "target_id": 1, "passage": "Not in the report."}
    monkeypatch.setattr(seed, "LINKS", [*seed.LINKS, bad_link])
    with pytest.raises(ValueError, match="Seed passage not found"):
        seed.load_seed(db)
    assert table_counts(db) == {}
    assert stored_files() == []
