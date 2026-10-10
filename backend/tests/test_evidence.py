"""Tests for the evidence router (task B2): uploads, listings, the detail and its
custody ledger, the file route's audit entries, reprocessing, evidence links and
the integrity check."""

import hashlib
from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from app.core.storage import absolute_path
from app.models import AuditLog, Evidence, EvidenceChunk, EvidenceLink, Reconstruction, ReconstructionStage
from app.routers import evidence as evidence_router
from app.services.evidence.ingest import evidence_type_for, mime_type_for
from tests.factories import make_case, make_event, make_evidence, make_investigator, make_person

REPORT = b"Police report.\n\nThe van entered the junction on amber."


@pytest.fixture
def queued(monkeypatch):
    """Records the evidence IDs handed to the AI pipeline instead of running it."""
    ids: list[int] = []
    monkeypatch.setattr(evidence_router, "process_evidence", ids.append)
    return ids


def upload(client, case_id, *files, headers=None):
    """Posts files as (name, bytes, content type) tuples in the multipart field "files"."""
    return client.post(
        f"/api/cases/{case_id}/evidence",
        files=[("files", file) for file in files],
        headers=headers or {},
    )


def stored_row(db, evidence_id) -> Evidence:
    db.expire_all()
    return db.get(Evidence, evidence_id)


def audit_rows(db, **where) -> list[AuditLog]:
    db.expire_all()
    query = select(AuditLog).filter_by(**where).order_by(AuditLog.id)
    return list(db.scalars(query))


# --- uploads ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("mime", "expected"),
    [
        ("image/jpeg", "photo"),
        ("video/mp4", "video"),
        ("audio/mpeg", "audio"),
        ("application/pdf", "document"),
        ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "document"),
        ("application/msword", "document"),
        ("text/plain", "document"),
        ("application/zip", "other"),
        ("application/octet-stream", "other"),
    ],
)
def test_evidence_type_follows_the_mime_type(mime, expected):
    assert evidence_type_for(mime) == expected


def test_mime_type_drops_parameters_and_falls_back_to_the_file_name():
    assert mime_type_for("Text/Plain; charset=utf-8", "notes.txt") == "text/plain"
    assert mime_type_for("application/octet-stream", "scene.jpg") == "image/jpeg"
    assert mime_type_for(None, "report.pdf") == "application/pdf"
    assert mime_type_for("", "blob") == "application/octet-stream"


def test_upload_stores_each_file_and_queues_processing(client, db, queued):
    investigator = make_investigator(db)
    case = make_case(db)
    photo = b"\xff\xd8\xff fake jpeg"

    response = upload(
        client,
        case.id,
        ("Police report.txt", REPORT, "text/plain"),
        ("Scene photo.jpg", photo, "image/jpeg"),
        ("Call 112.mp3", b"ID3 audio", "audio/mpeg"),
        headers={"X-Investigator-Id": str(investigator.id)},
    )

    assert response.status_code == 201
    rows = response.json()
    assert [(r["title"], r["type"], r["mime_type"]) for r in rows] == [
        ("Police report.txt", "document", "text/plain"),
        ("Scene photo.jpg", "photo", "image/jpeg"),
        ("Call 112.mp3", "audio", "audio/mpeg"),
    ]
    assert all(r["case_id"] == case.id and r["processing_status"] == "queued" for r in rows)
    assert rows[0]["sha256"] == hashlib.sha256(REPORT).hexdigest()
    assert rows[1]["size_bytes"] == len(photo)
    assert "file_path" not in rows[0] and "extracted_text" not in rows[0]
    assert queued == [r["id"] for r in rows]

    stored = stored_row(db, rows[0]["id"])
    assert stored.file_path.startswith(f"{case.id}/")
    assert absolute_path(stored.file_path).read_bytes() == REPORT

    uploaded = audit_rows(db, action="uploaded")
    assert [(e.record_type, e.record_id) for e in uploaded] == [("evidence", r["id"]) for r in rows]
    assert uploaded[0].detail == {"file_name": "Police report.txt", "sha256": rows[0]["sha256"]}
    assert all(e.case_id == case.id and e.actor_id == investigator.id for e in uploaded)


def test_upload_runs_the_pipeline_after_the_response(client, db):
    case = make_case(db)
    response = upload(client, case.id, ("notes.txt", b"Witness notes.", "text/plain"))
    assert response.json()[0]["processing_status"] == "queued"
    # TestClient runs background tasks before returning, so the pipeline has run.
    assert stored_row(db, response.json()[0]["id"]).processing_status != "queued"


def test_upload_guesses_a_missing_mime_type_from_the_file_name(client, db, queued):
    case = make_case(db)
    response = upload(client, case.id, ("dashcam.mp4", b"\x00\x00", "application/octet-stream"))
    assert (response.json()[0]["mime_type"], response.json()[0]["type"]) == ("video/mp4", "video")


def test_upload_needs_files_and_a_case(client, db, queued):
    case = make_case(db)
    assert client.post(f"/api/cases/{case.id}/evidence").status_code == 422
    missing = upload(client, 999, ("a.txt", b"a", "text/plain"))
    assert missing.status_code == 404
    assert missing.json() == {"detail": "Case not found"}
    assert queued == []


# --- listings ---------------------------------------------------------------


def test_case_evidence_is_newest_first(client, db):
    case = make_case(db)
    other = make_case(db)
    older = make_evidence(db, case, title="Older", created_at=datetime(2026, 10, 3, tzinfo=UTC))
    newer = make_evidence(db, case, title="Newer", created_at=datetime(2026, 10, 4, tzinfo=UTC))
    make_evidence(db, other, title="Elsewhere")

    response = client.get(f"/api/cases/{case.id}/evidence")
    assert response.status_code == 200
    assert [e["id"] for e in response.json()] == [newer.id, older.id]
    assert client.get("/api/cases/999/evidence").status_code == 404


def test_global_evidence_search_and_type_filter(client, db):
    ring_road = make_case(db, title="Ring Road junction collision")
    warehouse = make_case(db, title="Warehouse fire", incident_type="fire")
    report = make_evidence(
        db, ring_road, title="Police report.pdf", created_at=datetime(2026, 10, 3, tzinfo=UTC)
    )
    photo = make_evidence(
        db,
        ring_road,
        type="photo",
        title="IMG_0042.jpg",
        mime_type="image/jpeg",
        description="Skid marks before the stop line",
        created_at=datetime(2026, 10, 4, tzinfo=UTC),
    )
    statement = make_evidence(
        db,
        warehouse,
        type="statement",
        title="Statement 1",
        summary="The guard saw smoke at 02:10.",
        created_at=datetime(2026, 10, 5, tzinfo=UTC),
    )

    def ids(**params):
        response = client.get("/api/evidence", params=params)
        assert response.status_code == 200
        return [e["id"] for e in response.json()]

    assert ids() == [statement.id, photo.id, report.id]
    assert ids(type="photo") == [photo.id]
    assert ids(q="police") == [report.id]  # title, any case
    assert ids(q="SKID") == [photo.id]  # description
    assert ids(q="smoke") == [statement.id]  # summary
    assert ids(q="warehouse") == [statement.id]  # case title
    assert ids(q=ring_road.case_number) == [photo.id, report.id]  # case number
    assert ids(q="ring road", type="document") == [report.id]
    assert ids(q="100%") == []  # LIKE wildcards are matched literally
    assert client.get("/api/evidence", params={"type": "hologram"}).status_code == 422

    row = client.get("/api/evidence", params={"type": "statement"}).json()[0]
    assert (row["case_number"], row["case_title"]) == (warehouse.case_number, "Warehouse fire")


# --- detail and edits -------------------------------------------------------


def test_detail_has_text_links_and_custody_oldest_first(client, db):
    investigator = make_investigator(db)
    case = make_case(db)
    evidence = make_evidence(db, case, extracted_text="The van entered on amber.", summary="A police report.")
    person = make_person(db, case)
    db.add(EvidenceLink(evidence_id=evidence.id, target_type="person", target_id=person.id, passage="the van"))
    # Added last but dated first, as the seed's history is: custody follows `at`.
    db.add_all([
        AuditLog(actor_id=investigator.id, action="viewed", record_type="evidence", record_id=evidence.id,
                 case_id=case.id, at=datetime(2026, 10, 4, tzinfo=UTC)),
        AuditLog(actor_id=None, action="uploaded", record_type="evidence", record_id=evidence.id,
                 case_id=case.id, at=datetime(2026, 10, 3, tzinfo=UTC)),
        AuditLog(action="viewed", record_type="person", record_id=evidence.id, case_id=case.id),
    ])
    db.commit()

    response = client.get(f"/api/evidence/{evidence.id}")
    assert response.status_code == 200
    body = response.json()
    assert body["case_number"] == case.case_number
    assert body["case_title"] == case.title
    assert body["extracted_text"] == "The van entered on amber."
    assert [(link["target_type"], link["target_id"], link["passage"]) for link in body["links"]] == [
        ("person", person.id, "the van")
    ]
    assert [(e["action"], e["actor_name"]) for e in body["custody"]] == [
        ("uploaded", None),
        ("viewed", "Insp. Meera Rao"),
    ]
    assert body["custody"][0]["at"] == "2026-10-03T00:00:00Z"
    assert client.get("/api/evidence/999").json() == {"detail": "Evidence not found"}


def test_patch_updates_metadata_with_an_audit_entry(client, db):
    investigator = make_investigator(db)
    case = make_case(db)
    evidence = make_evidence(db, case)

    response = client.patch(
        f"/api/evidence/{evidence.id}",
        json={"type": "statement", "notes": "Signed by the witness.", "title": None},
        headers={"X-Investigator-Id": str(investigator.id)},
    )

    assert response.status_code == 200
    assert response.json()["type"] == "statement"
    assert response.json()["notes"] == "Signed by the witness."
    assert response.json()["title"] == "Police report.pdf"  # null leaves it as it is
    [entry] = audit_rows(db, record_type="evidence")
    assert (entry.action, entry.actor_id, entry.case_id) == ("modified", investigator.id, case.id)
    assert entry.detail == {
        "type": {"before": "document", "after": "statement"},
        "notes": {"before": "", "after": "Signed by the witness."},
    }

    assert client.patch(f"/api/evidence/{evidence.id}", json={"type": "hologram"}).status_code == 422
    assert client.patch(f"/api/evidence/{evidence.id}", json={"sha256": "f" * 64}).status_code == 422
    assert client.patch("/api/evidence/999", json={"notes": "x"}).status_code == 404


# --- the file route ---------------------------------------------------------


def test_file_is_served_inline_and_logged_as_viewed(client, db, queued):
    investigator = make_investigator(db)
    case = make_case(db)
    [row] = upload(client, case.id, ("Police report.txt", REPORT, "text/plain")).json()

    response = client.get(f"/api/evidence/{row['id']}/file", params={"investigator_id": investigator.id})

    assert response.status_code == 200
    assert response.content == REPORT
    assert response.headers["content-type"].startswith("text/plain")
    # Names with spaces use RFC 5987 encoding, which browsers decode.
    assert response.headers["content-disposition"] == "inline; filename*=utf-8''Police%20report.txt"
    [viewed] = audit_rows(db, action="viewed")
    assert (viewed.record_type, viewed.record_id) == ("evidence", row["id"])
    assert (viewed.actor_id, viewed.case_id, viewed.detail) == (investigator.id, case.id, None)


def test_download_is_an_attachment_and_logged_as_exported(client, db, queued):
    investigator = make_investigator(db)
    case = make_case(db)
    [row] = upload(client, case.id, ("scene.jpg", b"\xff\xd8\xff", "image/jpeg")).json()
    client.patch(f"/api/evidence/{row['id']}", json={"title": "Scene photo"})

    response = client.get(
        f"/api/evidence/{row['id']}/file", params={"download": 1, "investigator_id": investigator.id}
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    # Downloaded under its title, keeping the original's extension.
    assert response.headers["content-disposition"] == "attachment; filename*=utf-8''Scene%20photo.jpg"
    [exported] = audit_rows(db, action="exported")
    assert (exported.record_id, exported.actor_id, exported.case_id) == (row["id"], investigator.id, case.id)
    assert audit_rows(db, action="viewed") == []

    custody = client.get(f"/api/evidence/{row['id']}").json()["custody"]
    assert [e["action"] for e in custody] == ["uploaded", "modified", "exported"]


def test_range_requests_log_one_view(client, db, queued):
    case = make_case(db)
    [row] = upload(client, case.id, ("dashcam.mp4", b"0123456789", "video/mp4")).json()
    url = f"/api/evidence/{row['id']}/file"

    first = client.get(url, headers={"Range": "bytes=0-"})
    seek = client.get(url, headers={"Range": "bytes=5-"})

    assert (first.status_code, seek.status_code) == (206, 206)
    assert seek.content == b"56789"
    assert len(audit_rows(db, action="viewed")) == 1


def test_missing_file_is_404_without_an_audit_entry(client, db, queued):
    case = make_case(db)
    no_file = make_evidence(db, case)
    [row] = upload(client, case.id, ("gone.txt", b"gone", "text/plain")).json()
    absolute_path(stored_row(db, row["id"]).file_path).unlink()

    for evidence_id in (no_file.id, row["id"]):
        response = client.get(f"/api/evidence/{evidence_id}/file")
        assert response.status_code == 404
        assert response.json() == {"detail": "File not found"}
    assert client.get("/api/evidence/999/file").json() == {"detail": "Evidence not found"}
    assert audit_rows(db, action="viewed") == []


# --- reprocessing -----------------------------------------------------------


def test_reprocess_queues_the_pipeline_again(client, db, queued):
    investigator = make_investigator(db)
    case = make_case(db)
    evidence = make_evidence(db, case, processing_status="failed", summary="OCR failed")

    response = client.post(
        f"/api/evidence/{evidence.id}/reprocess", headers={"X-Investigator-Id": str(investigator.id)}
    )

    assert response.status_code == 200
    assert response.json()["processing_status"] == "queued"
    assert queued == [evidence.id]
    [entry] = audit_rows(db, record_type="evidence")
    assert (entry.action, entry.actor_id, entry.case_id, entry.detail) == (
        "modified",
        investigator.id,
        case.id,
        {"reprocess": True},
    )
    assert client.post("/api/evidence/999/reprocess").status_code == 404


# --- evidence links ---------------------------------------------------------


def test_link_evidence_to_a_record_and_remove_it(client, db):
    investigator = make_investigator(db)
    case = make_case(db)
    evidence = make_evidence(db, case)
    event = make_event(db, case)
    chunk = EvidenceChunk(evidence_id=evidence.id, ordinal=0, text="The van strikes the hatchback.")
    db.add(chunk)
    db.commit()
    headers = {"X-Investigator-Id": str(investigator.id)}

    response = client.post(
        "/api/evidence-links",
        json={
            "evidence_id": evidence.id,
            "chunk_id": chunk.id,
            "target_type": "event",
            "target_id": event.id,
            "passage": "The van strikes the hatchback.",
        },
        headers=headers,
    )

    assert response.status_code == 201
    link = response.json()
    assert link == {
        "id": link["id"],
        "evidence_id": evidence.id,
        "chunk_id": chunk.id,
        "target_type": "event",
        "target_id": event.id,
        "passage": "The van strikes the hatchback.",
    }
    assert client.get(f"/api/evidence/{evidence.id}").json()["links"] == [link]

    deleted = client.delete(f"/api/evidence-links/{link['id']}", headers=headers)
    assert deleted.status_code == 204
    assert deleted.content == b""
    assert db.scalars(select(EvidenceLink)).all() == []
    assert client.delete(f"/api/evidence-links/{link['id']}").json() == {"detail": "Evidence link not found"}

    # Both changes are audited on the target record, in the case's audit trail.
    entries = audit_rows(db, record_type="event", record_id=event.id)
    assert [(e.action, e.detail) for e in entries] == [
        ("modified", {"linked_evidence": evidence.id}),
        ("modified", {"unlinked_evidence": evidence.id}),
    ]
    assert all(e.case_id == case.id and e.actor_id == investigator.id for e in entries)


def test_link_to_a_reconstruction_stage_finds_its_case(client, db):
    case = make_case(db)
    evidence = make_evidence(db, case)
    reconstruction = Reconstruction(case_id=case.id, scenario="Van ran the red")
    db.add(reconstruction)
    db.commit()
    stage = ReconstructionStage(reconstruction_id=reconstruction.id, label="Impact", stage_order=1)
    db.add(stage)
    db.commit()

    response = client.post(
        "/api/evidence-links", json={"evidence_id": evidence.id, "target_type": "stage", "target_id": stage.id}
    )

    assert response.status_code == 201
    [entry] = audit_rows(db, record_type="stage")
    assert entry.case_id == case.id


def test_link_checks_the_evidence_target_and_chunk(client, db):
    case = make_case(db)
    other = make_case(db)
    evidence = make_evidence(db, case)
    other_evidence = make_evidence(db, other)
    person = make_person(db, case)
    stranger = make_person(db, other)
    foreign_chunk = EvidenceChunk(evidence_id=other_evidence.id, ordinal=0, text="Elsewhere.")
    db.add(foreign_chunk)
    db.commit()

    def link(**body):
        defaults = {"evidence_id": evidence.id, "target_type": "person", "target_id": person.id}
        return client.post("/api/evidence-links", json=defaults | body)

    assert link(evidence_id=999).json() == {"detail": "Evidence not found"}
    assert link(target_id=999).json() == {"detail": "Person not found"}
    assert link(target_type="scene_object", target_id=999).json() == {"detail": "Scene object not found"}
    cross_case = link(target_id=stranger.id)
    assert cross_case.status_code == 422
    assert cross_case.json() == {"detail": "Person is not in the same case as the evidence"}
    assert link(chunk_id=foreign_chunk.id).status_code == 422
    assert link(chunk_id=999).status_code == 422
    assert link(target_type="suspect").status_code == 422
    assert link(target_type="case", target_id=case.id).status_code == 201
    assert db.scalars(select(EvidenceLink)).all()[0].target_type == "case"
    assert len(audit_rows(db, action="modified")) == 1


# --- integrity check --------------------------------------------------------


def test_verify_passes_when_every_file_matches(client, db, queued):
    investigator = make_investigator(db)
    case = make_case(db)
    upload(client, case.id, ("a.txt", b"alpha", "text/plain"), ("b.txt", b"beta", "text/plain"))
    make_evidence(db, case, title="No stored file")  # nothing to check

    response = client.post(f"/api/cases/{case.id}/verify", headers={"X-Investigator-Id": str(investigator.id)})

    assert response.status_code == 200
    body = response.json()
    assert (body["checked"], body["mismatches"]) == (2, [])
    assert body["verified_at"].endswith("Z")
    [entry] = audit_rows(db, action="reviewed")
    assert (entry.record_type, entry.record_id, entry.case_id, entry.actor_id) == (
        "case",
        case.id,
        case.id,
        investigator.id,
    )
    assert entry.detail == {"integrity_check": {"checked": 2, "mismatches": 0}}


def test_verify_reports_tampered_and_missing_files(client, db, queued):
    case = make_case(db)
    other = make_case(db)
    intact, tampered, missing = upload(
        client,
        case.id,
        ("intact.txt", b"intact", "text/plain"),
        ("Statement.txt", b"The light was red.", "text/plain"),
        ("Photo.jpg", b"\xff\xd8\xff", "image/jpeg"),
    ).json()
    upload(client, other.id, ("other.txt", b"other case", "text/plain"))
    absolute_path(stored_row(db, tampered["id"]).file_path).write_bytes(b"The light was green.")
    absolute_path(stored_row(db, missing["id"]).file_path).unlink()

    body = client.post(f"/api/cases/{case.id}/verify").json()

    assert body["checked"] == 3
    assert body["mismatches"] == [
        {
            "evidence_id": tampered["id"],
            "title": "Statement.txt",
            "expected": hashlib.sha256(b"The light was red.").hexdigest(),
            "actual": hashlib.sha256(b"The light was green.").hexdigest(),
        },
        {"evidence_id": missing["id"], "title": "Photo.jpg", "expected": missing["sha256"], "actual": None},
    ]
    assert audit_rows(db, action="reviewed")[0].detail == {"integrity_check": {"checked": 3, "mismatches": 2}}
    assert client.post("/api/cases/999/verify").status_code == 404
