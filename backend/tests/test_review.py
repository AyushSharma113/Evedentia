"""Tests for the review routes: observations and conflicts."""

from sqlalchemy import select

from app.core.records import case_counts
from app.models import AuditLog, Conflict
from tests.factories import make_case, make_investigator, make_observation


def make_conflict(db, case, **fields) -> Conflict:
    values = {"observation_ids": [], "description": "Witnesses disagree on the signal colour."}
    row = Conflict(case_id=case.id, **(values | fields))
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def audit_rows(db, record_type: str, record_id: int) -> list[AuditLog]:
    query = select(AuditLog).where(AuditLog.record_type == record_type, AuditLog.record_id == record_id)
    return db.scalars(query.order_by(AuditLog.id)).all()


# Observations ------------------------------------------------------------------


def test_list_observations(client, db):
    case = make_case(db)
    pending = make_observation(db, case, statement="The light was red.", origin="ai", review_status="pending")
    confirmed = make_observation(db, case, statement="The light was amber.")
    make_observation(db, make_case(db, title="Other case"))

    response = client.get(f"/api/cases/{case.id}/observations")
    assert response.status_code == 200
    body = response.json()
    assert [o["id"] for o in body] == [pending.id, confirmed.id]
    assert body[0] == {
        "id": pending.id,
        "case_id": case.id,
        "subject_type": "case",
        "subject_id": case.id,
        "statement": "The light was red.",
        "observed_at": None,
        "author_id": None,
        "origin": "ai",
        "review_status": "pending",
    }

    by_status = client.get(f"/api/cases/{case.id}/observations", params={"status": "pending"}).json()
    assert [o["id"] for o in by_status] == [pending.id]
    assert client.get(f"/api/cases/{case.id}/observations", params={"status": "maybe"}).status_code == 422
    assert client.get("/api/cases/99/observations").status_code == 404


def test_confirming_an_observation_audits_reviewed(client, db):
    investigator = make_investigator(db)
    case = make_case(db)
    observation = make_observation(db, case, origin="ai", review_status="pending")
    assert case_counts(db, case.id)["pending_reviews"] == 1

    response = client.patch(
        f"/api/observations/{observation.id}",
        json={"review_status": "confirmed"},
        headers={"X-Investigator-Id": str(investigator.id)},
    )
    assert response.status_code == 200
    assert response.json()["review_status"] == "confirmed"
    [entry] = audit_rows(db, "observation", observation.id)
    assert (entry.action, entry.actor_id, entry.case_id) == ("reviewed", investigator.id, case.id)
    assert entry.detail == {"review_status": {"before": "pending", "after": "confirmed"}}
    assert case_counts(db, case.id)["pending_reviews"] == 0
    assert case_counts(db, case.id)["ai_items_reviewed"] == 1


def test_editing_an_observation_audits_modified(client, db):
    case = make_case(db)
    observation = make_observation(db, case, statement="The light was amber.", origin="ai", review_status="pending")

    response = client.patch(
        f"/api/observations/{observation.id}",
        json={"statement": "The light had just turned amber.", "review_status": "confirmed"},
    )
    assert response.json()["statement"] == "The light had just turned amber."
    [entry] = audit_rows(db, "observation", observation.id)
    assert entry.action == "modified"
    assert entry.detail == {
        "statement": {"before": "The light was amber.", "after": "The light had just turned amber."},
        "review_status": {"before": "pending", "after": "confirmed"},
    }


def test_update_observation_rejects_bad_bodies(client, db):
    case = make_case(db)
    observation = make_observation(db, case)
    url = f"/api/observations/{observation.id}"

    assert client.patch(url, json={"statement": ""}).status_code == 422
    response = client.patch(url, json={"statement": None})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "statement"]
    assert client.patch(url, json={"review_status": "maybe"}).status_code == 422
    assert client.patch(url, json={"subject_id": 3}).status_code == 422
    assert audit_rows(db, "observation", observation.id) == []
    response = client.patch("/api/observations/99", json={"review_status": "confirmed"})
    assert response.status_code == 404
    assert response.json() == {"detail": "Observation not found"}


# Conflicts ---------------------------------------------------------------------


def test_list_conflicts(client, db):
    case = make_case(db)
    red = make_observation(db, case, statement="The light was red.")
    amber = make_observation(db, case, statement="The light was amber.")
    open_conflict = make_conflict(db, case, observation_ids=[red.id, amber.id])
    resolved = make_conflict(db, case, description="Arrival time unclear.", status="resolved")
    make_conflict(db, make_case(db, title="Other case"))

    response = client.get(f"/api/cases/{case.id}/conflicts")
    assert response.status_code == 200
    assert response.json() == [
        {
            "id": open_conflict.id,
            "case_id": case.id,
            "observation_ids": [red.id, amber.id],
            "description": "Witnesses disagree on the signal colour.",
            "status": "open",
        },
        {
            "id": resolved.id,
            "case_id": case.id,
            "observation_ids": [],
            "description": "Arrival time unclear.",
            "status": "resolved",
        },
    ]

    url = f"/api/cases/{case.id}/conflicts"
    assert [c["id"] for c in client.get(url, params={"status": "open"}).json()] == [open_conflict.id]
    assert [c["id"] for c in client.get(url, params={"status": "resolved"}).json()] == [resolved.id]
    assert client.get(url, params={"status": "closed"}).status_code == 422
    assert client.get("/api/cases/99/conflicts").status_code == 404


def test_resolve_conflict(client, db):
    case = make_case(db)
    conflict = make_conflict(db, case)
    assert case_counts(db, case.id)["open_conflicts"] == 1

    response = client.patch(f"/api/conflicts/{conflict.id}", json={"status": "resolved"})
    assert response.status_code == 200
    assert response.json()["status"] == "resolved"
    [entry] = audit_rows(db, "conflict", conflict.id)
    assert (entry.action, entry.case_id) == ("modified", case.id)
    assert entry.detail == {"status": {"before": "open", "after": "resolved"}}
    assert case_counts(db, case.id)["open_conflicts"] == 0


def test_update_conflict_rejects_bad_bodies(client, db):
    case = make_case(db)
    conflict = make_conflict(db, case)
    url = f"/api/conflicts/{conflict.id}"

    assert client.patch(url, json={}).status_code == 422
    assert client.patch(url, json={"status": None}).status_code == 422
    assert client.patch(url, json={"status": "closed"}).status_code == 422
    assert client.patch(url, json={"status": "resolved", "description": "x"}).status_code == 422
    assert audit_rows(db, "conflict", conflict.id) == []
    response = client.patch("/api/conflicts/99", json={"status": "resolved"})
    assert response.status_code == 404
    assert response.json() == {"detail": "Conflict not found"}
