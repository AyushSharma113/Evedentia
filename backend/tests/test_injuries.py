"""Tests for the injury routes: listing, creating and reviewing injury markers."""

from datetime import UTC, datetime

from sqlalchemy import select

from app.models import AuditLog, Injury
from tests.factories import make_case, make_investigator, make_person


def make_injury(db, case, **fields) -> Injury:
    values = {"body_region": "Head", "type": "Laceration", "severity": "moderate"}
    row = Injury(case_id=case.id, **(values | fields))
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def audit_rows(db):
    return db.scalars(select(AuditLog).where(AuditLog.record_type == "injury").order_by(AuditLog.id)).all()


def test_list_injuries_is_the_case_only_in_id_order(client, db):
    case = make_case(db)
    first = make_injury(db, case)
    make_injury(db, make_case(db, title="Another case"))
    second = make_injury(db, case, body_region="Left femur", type="Fracture", severity="severe")
    response = client.get(f"/api/cases/{case.id}/injuries")
    assert response.status_code == 200
    assert [i["id"] for i in response.json()] == [first.id, second.id]
    assert client.get("/api/cases/999/injuries").status_code == 404


def test_create_injury_is_confirmed_and_audited(client, db):
    investigator = make_investigator(db)
    case = make_case(db)
    victim = make_person(db, case, role="victim", name="Rohan Kapoor")
    response = client.post(
        f"/api/cases/{case.id}/injuries",
        json={
            "person_id": victim.id,
            "body_region": "Head",
            "structure_id": "skull_frontal",
            "type": "Laceration",
            "severity": "moderate",
            "depth": "2 cm",
            "estimated_time": "2026-10-03T21:42:00+05:30",
            "description": "Frontal laceration from the windscreen.",
        },
        headers={"X-Investigator-Id": str(investigator.id)},
    )
    assert response.status_code == 201
    body = response.json()
    assert body == {
        "id": body["id"],
        "case_id": case.id,
        "person_id": victim.id,
        "body_region": "Head",
        "structure_id": "skull_frontal",
        "type": "Laceration",
        "severity": "moderate",
        "depth": "2 cm",
        "estimated_time": "2026-10-03T16:12:00Z",
        "description": "Frontal laceration from the windscreen.",
        "origin": "investigator",
        "review_status": "confirmed",
    }
    [entry] = audit_rows(db)
    assert (entry.action, entry.record_id, entry.case_id, entry.actor_id, entry.detail) == (
        "modified",
        body["id"],
        case.id,
        investigator.id,
        {"created": True},
    )


def test_create_injury_defaults(client, db):
    case = make_case(db)
    response = client.post(
        f"/api/cases/{case.id}/injuries", json={"body_region": "Chest", "type": "Contusion", "severity": "minor"}
    )
    assert response.status_code == 201
    body = response.json()
    assert (body["person_id"], body["structure_id"], body["depth"], body["estimated_time"], body["description"]) == (
        None,
        None,
        None,
        None,
        "",
    )


def test_create_injury_rejects_bad_bodies(client, db):
    case = make_case(db)
    stranger = make_person(db, make_case(db, title="Another case"))
    url = f"/api/cases/{case.id}/injuries"
    valid = {"body_region": "Head", "type": "Laceration", "severity": "moderate"}

    response = client.post(url, json=valid | {"person_id": stranger.id})
    assert response.status_code == 422
    assert response.json()["detail"] == [
        {
            "loc": ["body", "person_id"],
            "msg": f"Person {stranger.id} is not a person of this case",
            "type": "value_error",
        }
    ]
    assert client.post(url, json=valid | {"person_id": 999}).status_code == 422
    assert client.post(url, json=valid | {"severity": "critical"}).status_code == 422
    assert client.post(url, json=valid | {"body_region": ""}).status_code == 422
    assert client.post(url, json=valid | {"origin": "ai"}).status_code == 422
    assert client.post(url, json={"body_region": "Head", "type": "Laceration"}).status_code == 422
    assert client.post("/api/cases/999/injuries", json=valid).status_code == 404
    assert db.scalars(select(Injury)).all() == []


def test_review_an_ai_injury_logs_reviewed(client, db):
    investigator = make_investigator(db)
    case = make_case(db)
    injury = make_injury(db, case, origin="ai", review_status="pending")
    response = client.patch(
        f"/api/injuries/{injury.id}",
        json={"review_status": "confirmed"},
        headers={"X-Investigator-Id": str(investigator.id)},
    )
    assert response.status_code == 200
    assert (response.json()["origin"], response.json()["review_status"]) == ("ai", "confirmed")
    [entry] = audit_rows(db)
    assert (entry.action, entry.record_id, entry.case_id, entry.actor_id) == (
        "reviewed",
        injury.id,
        case.id,
        investigator.id,
    )
    assert entry.detail == {"review_status": {"before": "pending", "after": "confirmed"}}


def test_edit_an_injury_logs_modified_with_before_and_after(client, db):
    case = make_case(db)
    victim = make_person(db, case, role="victim")
    injury = make_injury(db, case)
    response = client.patch(
        f"/api/injuries/{injury.id}",
        json={"severity": "severe", "person_id": victim.id, "body_region": "Head", "depth": None},
    )
    assert response.status_code == 200
    assert (response.json()["severity"], response.json()["person_id"]) == ("severe", victim.id)
    [entry] = audit_rows(db)
    assert (entry.action, entry.case_id) == ("modified", case.id)
    assert entry.detail == {
        "severity": {"before": "moderate", "after": "severe"},
        "person_id": {"before": None, "after": victim.id},
    }

    response = client.patch(f"/api/injuries/{injury.id}", json={"estimated_time": "2026-10-03T21:45:00+05:30"})
    assert response.json()["estimated_time"] == "2026-10-03T16:15:00Z"

    # Unlinking the person is allowed; an unchanged body writes no entry.
    assert client.patch(f"/api/injuries/{injury.id}", json={"person_id": None}).json()["person_id"] is None
    assert client.patch(f"/api/injuries/{injury.id}", json={"severity": "severe"}).status_code == 200
    assert len(audit_rows(db)) == 3


def test_patch_injury_rejects_bad_bodies(client, db):
    case = make_case(db)
    stranger = make_person(db, make_case(db, title="Another case"))
    injury = make_injury(db, case, estimated_time=datetime(2026, 10, 3, 16, 12, tzinfo=UTC))
    url = f"/api/injuries/{injury.id}"

    response = client.patch(url, json={"person_id": stranger.id})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "person_id"]
    for field in ("body_region", "type", "severity", "description", "review_status"):
        response = client.patch(url, json={field: None})
        assert response.status_code == 422, field
        assert response.json()["detail"][0]["loc"] == ["body", field]
    assert client.patch(url, json={"severity": "critical"}).status_code == 422
    assert client.patch(url, json={"case_id": 2}).status_code == 422
    assert client.patch("/api/injuries/999", json={"severity": "minor"}).status_code == 404
    assert audit_rows(db) == []
