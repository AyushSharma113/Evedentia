"""Tests for the cases routes (task B1): investigators, the case list and its
filters, create and update with their audit entries, and the workspace call."""

from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from app.models import AuditLog, Case, Conflict, EvidenceLink, Injury, Relationship
from tests.factories import (
    make_case,
    make_event,
    make_evidence,
    make_investigator,
    make_location,
    make_object,
    make_person,
)

NEW_CASE = {"title": "Warehouse fire", "incident_type": "fire", "priority": "medium"}


def audit_rows(db):
    db.expire_all()
    return db.scalars(select(AuditLog).order_by(AuditLog.id)).all()


# investigators -------------------------------------------------------------


def test_investigators_are_listed_by_id(client, db):
    first = make_investigator(db)
    second = make_investigator(db, name="SI Arjun Menon", badge_number="TP-2210", role="Field officer")

    response = client.get("/api/investigators")

    assert response.status_code == 200
    assert [i["id"] for i in response.json()] == [first.id, second.id]
    assert response.json()[1] == {
        "id": second.id,
        "name": "SI Arjun Menon",
        "badge_number": "TP-2210",
        "role": "Field officer",
        "clearance_level": "L3",
    }


# list ----------------------------------------------------------------------


def test_cases_are_listed_newest_first_with_evidence_counts(client, db):
    now = datetime.now(UTC)
    older = make_case(db, title="Older", created_at=now - timedelta(days=2))
    newer = make_case(db, title="Newer", created_at=now - timedelta(days=1))
    make_evidence(db, older)
    make_evidence(db, older)
    make_evidence(db, newer)
    empty = make_case(db, title="No evidence", created_at=now - timedelta(days=3))

    response = client.get("/api/cases")

    assert response.status_code == 200
    rows = response.json()
    assert [(r["id"], r["evidence_count"]) for r in rows] == [(newer.id, 1), (older.id, 2), (empty.id, 0)]
    assert rows[0]["case_number"] == newer.case_number
    assert rows[0]["created_at"].endswith("Z")


def test_cases_filter_by_status_and_priority(client, db):
    open_high = make_case(db, status="open", priority="high")
    make_case(db, status="closed", priority="high")
    make_case(db, status="open", priority="low")

    ids = lambda params: [r["id"] for r in client.get("/api/cases", params=params).json()]  # noqa: E731

    assert ids({"status": "open", "priority": "high"}) == [open_high.id]
    assert len(ids({"status": "open"})) == 2
    assert len(ids({"priority": "high"})) == 2
    assert client.get("/api/cases", params={"status": "archived"}).status_code == 422


def test_case_search_matches_every_listed_field(client, db):
    lead = make_investigator(db, name="Insp. Kavya Iyer")
    by_number = make_case(db, case_number="FX-2025-0042", title="A")
    by_title = make_case(db, title="Lakeside drowning")
    by_description = make_case(db, title="B", description="Smoke seen from the godown roof")
    by_lead = make_case(db, title="C", lead_investigator_id=lead.id)
    by_place = make_case(db, title="D")
    make_location(db, by_place, address="14 Marine Drive")
    by_person = make_case(db, title="E")
    make_person(db, by_person, name="Rohan Kapoor")
    make_case(db, title="Unrelated", description="Nothing to see")

    def search(q):
        return {r["id"] for r in client.get("/api/cases", params={"q": q}).json()}

    assert search("fx-2025-0042") == {by_number.id}
    assert search("LAKESIDE") == {by_title.id}
    assert search("godown") == {by_description.id}
    assert search("kavya") == {by_lead.id}
    assert search("marine drive") == {by_place.id}
    assert search("kapoor") == {by_person.id}
    assert search("no match anywhere") == set()


def test_case_search_combines_with_filters_and_takes_wildcards_literally(client, db):
    match = make_case(db, title="Hit and run, 50% damage", status="open")
    make_case(db, title="Hit and run", status="closed")
    make_case(db, title="Fire")

    assert [r["id"] for r in client.get("/api/cases", params={"q": "hit", "status": "open"}).json()] == [match.id]
    assert [r["id"] for r in client.get("/api/cases", params={"q": "%"}).json()] == [match.id]
    assert client.get("/api/cases", params={"q": "_"}).json() == []


def test_case_search_does_not_duplicate_cases_with_several_matches(client, db):
    case = make_case(db, title="Ring Road crash")
    make_person(db, case, name="Ring Road witness")
    make_location(db, case, address="Ring Road")
    make_location(db, case, address="Ring Road service lane")

    assert [r["id"] for r in client.get("/api/cases", params={"q": "ring road"}).json()] == [case.id]


# create --------------------------------------------------------------------


def test_create_case_numbers_it_and_audits(client, db):
    lead = make_investigator(db)
    officer = make_investigator(db, name="SI Arjun Menon")
    make_case(db)

    response = client.post(
        "/api/cases",
        json=NEW_CASE
        | {
            "description": "Fire at a textile warehouse.",
            "incident_at": "2026-10-04T22:15:00+05:30",
            "lead_investigator_id": lead.id,
            "investigator_ids": [lead.id, officer.id],
        },
        headers={"X-Investigator-Id": str(lead.id)},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["case_number"] == f"FX-{body['created_at'][:4]}-{body['id']:04d}"
    assert body["case_number"].endswith("-0002")
    assert body["status"] == "open"
    assert body["incident_at"] == "2026-10-04T16:45:00Z"
    assert body["lead_investigator_id"] == lead.id
    assert body["investigator_ids"] == [lead.id, officer.id]
    assert db.get(Case, body["id"]).case_number == body["case_number"]

    [entry] = audit_rows(db)
    assert (entry.action, entry.record_type, entry.record_id) == ("modified", "case", body["id"])
    assert entry.case_id == body["id"]
    assert entry.actor_id == lead.id
    assert entry.detail == {"created": True}


def test_create_case_defaults(client):
    response = client.post("/api/cases", json=NEW_CASE)

    assert response.status_code == 201
    body = response.json()
    assert body["case_number"] == f"FX-{body['created_at'][:4]}-0001"
    assert body["updated_at"] == body["created_at"]
    assert body["description"] == ""
    assert body["incident_at"] is None
    assert body["lead_investigator_id"] is None
    assert body["investigator_ids"] == []


def test_create_case_rejects_unknown_investigators(client, db):
    lead = make_investigator(db)

    unknown_lead = client.post("/api/cases", json=NEW_CASE | {"lead_investigator_id": 999})
    assert unknown_lead.status_code == 422
    [error] = unknown_lead.json()["detail"]
    assert error["loc"] == ["body", "lead_investigator_id"]
    assert "999" in error["msg"]

    unknown_team = client.post("/api/cases", json=NEW_CASE | {"lead_investigator_id": lead.id, "investigator_ids": [lead.id, 998, 999]})
    assert unknown_team.status_code == 422
    [error] = unknown_team.json()["detail"]
    assert error["loc"] == ["body", "investigator_ids"]
    assert "998, 999" in error["msg"]

    assert db.scalars(select(Case)).all() == []
    assert audit_rows(db) == []


def test_create_case_validates_the_body(client):
    for field in ("title", "incident_type", "priority"):
        body = {k: v for k, v in NEW_CASE.items() if k != field}
        assert client.post("/api/cases", json=body).status_code == 422
    assert client.post("/api/cases", json=NEW_CASE | {"title": ""}).status_code == 422
    assert client.post("/api/cases", json=NEW_CASE | {"incident_type": "theft"}).status_code == 422
    assert client.post("/api/cases", json=NEW_CASE | {"case_number": "FX-1"}).status_code == 422


# read and update -----------------------------------------------------------


def test_get_case(client, db):
    case = make_case(db, investigator_ids=[])

    response = client.get(f"/api/cases/{case.id}")

    assert response.status_code == 200
    assert response.json()["case_number"] == case.case_number
    assert response.json()["incident_at"] == "2026-10-03T16:10:00Z"
    missing = client.get("/api/cases/999")
    assert missing.status_code == 404
    assert missing.json() == {"detail": "Case not found"}


def test_update_case_audits_and_bumps_updated_at(client, db):
    lead = make_investigator(db)
    case = make_case(db, case_number="FX-2026-0001", updated_at=datetime(2026, 10, 3, tzinfo=UTC))

    response = client.patch(
        f"/api/cases/{case.id}",
        json={"status": "in_progress", "priority": "high", "lead_investigator_id": lead.id, "investigator_ids": [lead.id]},
        headers={"X-Investigator-Id": str(lead.id)},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "in_progress"
    assert body["lead_investigator_id"] == lead.id
    assert body["investigator_ids"] == [lead.id]
    assert body["updated_at"] > "2026-10-03T00:00:00Z"
    [entry] = audit_rows(db)
    assert (entry.action, entry.record_type, entry.record_id, entry.case_id) == ("modified", "case", case.id, case.id)
    assert entry.actor_id == lead.id
    # priority was already high, so only the real changes are logged.
    assert entry.detail == {
        "status": {"before": "open", "after": "in_progress"},
        "lead_investigator_id": {"before": None, "after": lead.id},
        "investigator_ids": {"before": [], "after": [lead.id]},
    }


def test_update_case_without_changes_is_not_audited(client, db):
    case = make_case(db, case_number="FX-2026-0001", updated_at=datetime(2026, 10, 3, tzinfo=UTC))

    response = client.patch(f"/api/cases/{case.id}", json={"title": case.title})

    assert response.status_code == 200
    assert response.json()["updated_at"] == "2026-10-03T00:00:00Z"
    assert audit_rows(db) == []


def test_update_case_can_clear_nullable_fields(client, db):
    lead = make_investigator(db)
    case = make_case(db, lead_investigator_id=lead.id)

    response = client.patch(f"/api/cases/{case.id}", json={"lead_investigator_id": None, "incident_at": None})

    assert response.status_code == 200
    assert response.json()["lead_investigator_id"] is None
    assert response.json()["incident_at"] is None


def test_update_case_rejects_bad_values(client, db):
    case = make_case(db)

    unknown_lead = client.patch(f"/api/cases/{case.id}", json={"lead_investigator_id": 999})
    assert unknown_lead.status_code == 422
    assert unknown_lead.json()["detail"][0]["loc"] == ["body", "lead_investigator_id"]
    unknown_team = client.patch(f"/api/cases/{case.id}", json={"investigator_ids": [999]})
    assert unknown_team.status_code == 422
    assert unknown_team.json()["detail"][0]["loc"] == ["body", "investigator_ids"]
    for field in ("title", "status", "priority", "incident_type", "description", "investigator_ids"):
        response = client.patch(f"/api/cases/{case.id}", json={field: None})
        assert response.status_code == 422, field
        assert response.json()["detail"][0]["loc"] == ["body", field]
    assert client.patch(f"/api/cases/{case.id}", json={"case_number": "FX-1"}).status_code == 422
    assert client.patch("/api/cases/999", json={"title": "Missing"}).status_code == 404

    db.expire_all()
    assert db.get(Case, case.id).lead_investigator_id is None
    assert audit_rows(db) == []


# workspace -----------------------------------------------------------------


def test_workspace_returns_the_case_records_links_and_counts(client, db):
    case = make_case(db)
    other = make_case(db)
    evidence = make_evidence(db, case)
    other_evidence = make_evidence(db, other)
    witness = make_person(db, case)
    suspect = make_person(db, case, role="suspect", name="Suresh Nair", origin="ai", review_status="pending")
    van = make_object(db, case)
    site = make_location(db, case)
    event = make_event(db, case, location_id=site.id, origin="ai", review_status="confirmed")
    make_person(db, other)
    make_object(db, other)
    make_location(db, other)
    make_event(db, other)
    drove = Relationship(case_id=case.id, from_type="person", from_id=suspect.id, to_type="object", to_id=van.id, kind="drove")
    injury = Injury(case_id=case.id, person_id=witness.id, body_region="Left forearm", type="Abrasion", severity="minor")
    db.add_all([drove, injury, Conflict(case_id=case.id, description="Signal colour", status="open")])
    db.commit()
    link = EvidenceLink(evidence_id=evidence.id, target_type="person", target_id=suspect.id, passage="The van driver")
    db.add_all([link, EvidenceLink(evidence_id=other_evidence.id, target_type="case", target_id=other.id)])
    db.commit()

    response = client.get(f"/api/cases/{case.id}/workspace")

    assert response.status_code == 200
    body = response.json()
    assert body["case"]["id"] == case.id
    assert [p["id"] for p in body["persons"]] == [witness.id, suspect.id]
    assert [o["id"] for o in body["objects"]] == [van.id]
    assert [loc["id"] for loc in body["locations"]] == [site.id]
    assert [e["id"] for e in body["evidence"]] == [evidence.id]
    assert "extracted_text" not in body["evidence"][0]
    assert "file_path" not in body["evidence"][0]
    assert [e["id"] for e in body["events"]] == [event.id]
    assert [r["id"] for r in body["relationships"]] == [drove.id]
    assert [i["id"] for i in body["injuries"]] == [injury.id]
    assert body["links"] == [
        {
            "id": link.id,
            "evidence_id": evidence.id,
            "chunk_id": None,
            "target_type": "person",
            "target_id": suspect.id,
            "passage": "The van driver",
        }
    ]
    assert body["counts"] == {
        "evidence": 1,
        "pending_reviews": 1,
        "open_conflicts": 1,
        "ai_items_total": 2,
        "ai_items_reviewed": 1,
    }


def test_workspace_of_an_empty_case(client, db):
    case = make_case(db)

    body = client.get(f"/api/cases/{case.id}/workspace").json()

    assert all(body[key] == [] for key in ("persons", "objects", "locations", "evidence", "events", "relationships", "injuries", "links"))
    assert set(body["counts"].values()) == {0}
    assert client.get("/api/cases/999/workspace").status_code == 404
