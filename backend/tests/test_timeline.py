"""Tests for the timeline routes: GET, POST /cases/{id}/events and PATCH, DELETE /events/{id}."""

from datetime import UTC, datetime

from sqlalchemy import select

from app.models import AuditLog, Event, EvidenceLink, Reconstruction, ReconstructionStage
from tests.factories import make_case, make_event, make_evidence, make_investigator, make_location


def at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, 3, hour, minute, tzinfo=UTC)


def audit_rows(db, record_id: int) -> list[AuditLog]:
    return db.scalars(
        select(AuditLog).where(AuditLog.record_type == "event", AuditLog.record_id == record_id).order_by(AuditLog.id)
    ).all()


# GET /cases/{id}/events --------------------------------------------------------


def test_events_are_in_time_order_with_untimed_last(client, db):
    case = make_case(db)
    untimed_a = make_event(db, case, description="Untimed A")
    late = make_event(db, case, time_start=at(16, 30), description="Late")
    untimed_b = make_event(db, case, description="Untimed B")
    early = make_event(db, case, time_start=at(16, 5), description="Early")
    tie = make_event(db, case, time_start=at(16, 5), description="Same time, created later")
    make_event(db, make_case(db, title="Other case"), time_start=at(15), description="Other case")

    response = client.get(f"/api/cases/{case.id}/events")
    assert response.status_code == 200
    assert [e["id"] for e in response.json()] == [early.id, tie.id, late.id, untimed_a.id, untimed_b.id]
    assert response.json()[0]["time_start"] == "2026-10-03T16:05:00Z"


def test_events_filter_by_category(client, db):
    case = make_case(db)
    make_event(db, case, category="collision", time_start=at(16, 10))
    witness = make_event(db, case, category="witness", time_start=at(16, 12))

    response = client.get(f"/api/cases/{case.id}/events", params={"category": "witness"})
    assert [e["id"] for e in response.json()] == [witness.id]
    assert client.get(f"/api/cases/{case.id}/events", params={"category": "medical"}).json() == []
    assert client.get(f"/api/cases/{case.id}/events", params={"category": "party"}).status_code == 422
    assert client.get("/api/cases/99/events").status_code == 404


# POST /cases/{id}/events -------------------------------------------------------


def test_create_event(client, db):
    investigator = make_investigator(db)
    case = make_case(db)
    site = make_location(db, case)

    response = client.post(
        f"/api/cases/{case.id}/events",
        json={
            "category": "emergency",
            "description": "Ambulance arrives.",
            "location_id": site.id,
            "time_start": "2026-10-03T21:58:00+05:30",
        },
        headers={"X-Investigator-Id": str(investigator.id)},
    )
    assert response.status_code == 201
    body = response.json()
    assert body == {
        "id": body["id"],
        "case_id": case.id,
        "location_id": site.id,
        "category": "emergency",
        "time_start": "2026-10-03T16:28:00Z",
        "time_end": None,
        "time_precision": "minute",
        "certainty": "established",
        "description": "Ambulance arrives.",
        "ai_note": None,
        "origin": "investigator",
        "review_status": "confirmed",
    }
    [entry] = audit_rows(db, body["id"])
    assert (entry.action, entry.actor_id, entry.case_id, entry.detail) == (
        "modified",
        investigator.id,
        case.id,
        {"created": True},
    )
    assert [e["id"] for e in client.get(f"/api/cases/{case.id}/events").json()] == [body["id"]]


def test_create_event_location_must_be_in_the_case(client, db):
    case = make_case(db)
    foreign = make_location(db, make_case(db, title="Other case"))
    url = f"/api/cases/{case.id}/events"

    for location_id in (foreign.id, 999):
        response = client.post(url, json={"category": "other", "description": "x", "location_id": location_id})
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "location_id"]
    assert db.scalars(select(Event)).all() == []


def test_create_event_rejects_bad_bodies(client, db):
    case = make_case(db)
    url = f"/api/cases/{case.id}/events"

    assert client.post(url, json={"description": "No category"}).status_code == 422
    assert client.post(url, json={"category": "other", "description": ""}).status_code == 422
    assert client.post(url, json={"category": "other", "description": "x", "certainty": "sure"}).status_code == 422
    assert client.post(url, json={"category": "other", "description": "x", "origin": "ai"}).status_code == 422
    assert client.post("/api/cases/99/events", json={"category": "other", "description": "x"}).status_code == 404


# PATCH /events/{id} ------------------------------------------------------------


def test_update_event(client, db):
    case = make_case(db)
    site = make_location(db, case)
    event = make_event(db, case, time_start=at(16, 10), description="Impact")

    response = client.patch(
        f"/api/events/{event.id}",
        json={"description": "Impact at the stop line", "location_id": site.id, "certainty": "disputed"},
    )
    assert response.status_code == 200
    assert response.json()["description"] == "Impact at the stop line"
    assert response.json()["location_id"] == site.id
    assert response.json()["certainty"] == "disputed"
    [entry] = audit_rows(db, event.id)
    assert entry.action == "modified"
    assert entry.case_id == case.id
    assert entry.detail == {
        "description": {"before": "Impact", "after": "Impact at the stop line"},
        "location_id": {"before": None, "after": site.id},
        "certainty": {"before": "established", "after": "disputed"},
    }

    # The same instant in another offset is not a change; nullable fields can be cleared.
    response = client.patch(
        f"/api/events/{event.id}", json={"time_start": "2026-10-03T21:40:00+05:30", "location_id": None}
    )
    assert response.json()["time_start"] == "2026-10-03T16:10:00Z"
    assert response.json()["location_id"] is None
    assert audit_rows(db, event.id)[-1].detail == {"location_id": {"before": site.id, "after": None}}

    response = client.patch(f"/api/events/{event.id}", json={"time_end": "2026-10-03T21:45:00+05:30"})
    assert response.json()["time_end"] == "2026-10-03T16:15:00Z"
    assert audit_rows(db, event.id)[-1].detail == {
        "time_end": {"before": None, "after": "2026-10-03T16:15:00Z"}
    }


def test_review_only_change_to_event_audits_reviewed(client, db):
    case = make_case(db)
    event = make_event(db, case, origin="ai", review_status="pending")

    response = client.patch(f"/api/events/{event.id}", json={"review_status": "confirmed"})
    assert response.json()["review_status"] == "confirmed"
    [entry] = audit_rows(db, event.id)
    assert entry.action == "reviewed"
    assert entry.detail == {"review_status": {"before": "pending", "after": "confirmed"}}

    # A PATCH that changes nothing writes no audit row.
    client.patch(f"/api/events/{event.id}", json={"review_status": "confirmed"})
    assert len(audit_rows(db, event.id)) == 1


def test_update_event_rejects_bad_changes(client, db):
    case = make_case(db)
    event = make_event(db, case)
    foreign = make_location(db, make_case(db, title="Other case"))

    response = client.patch(f"/api/events/{event.id}", json={"location_id": foreign.id})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "location_id"]
    response = client.patch(f"/api/events/{event.id}", json={"category": None})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "category"]
    assert client.patch(f"/api/events/{event.id}", json={"case_id": 2}).status_code == 422
    assert audit_rows(db, event.id) == []
    response = client.patch("/api/events/99", json={"description": "x"})
    assert response.status_code == 404
    assert response.json() == {"detail": "Event not found"}


# DELETE /events/{id} -----------------------------------------------------------


def test_delete_event_cleans_up_stages_and_links(client, db):
    case = make_case(db)
    evidence = make_evidence(db, case)
    approach_id = make_event(db, case, description="Approach").id
    impact_id = make_event(db, case, description="Impact").id
    reconstruction = Reconstruction(case_id=case.id, scenario="Van ran the red")
    db.add(reconstruction)
    db.flush()
    stage = ReconstructionStage(reconstruction_id=reconstruction.id, label="Impact", event_ids=[approach_id, impact_id])
    other_stage = ReconstructionStage(reconstruction_id=reconstruction.id, label="After", event_ids=[impact_id])
    db.add_all([stage, other_stage])
    db.add(EvidenceLink(evidence_id=evidence.id, target_type="event", target_id=approach_id, passage="..."))
    db.add(EvidenceLink(evidence_id=evidence.id, target_type="event", target_id=impact_id, passage="..."))
    db.commit()

    response = client.delete(f"/api/events/{approach_id}")
    assert response.status_code == 204
    assert response.content == b""
    db.expire_all()
    assert db.get(Event, approach_id) is None
    assert stage.event_ids == [impact_id]
    assert other_stage.event_ids == [impact_id]
    assert [link.target_id for link in db.scalars(select(EvidenceLink))] == [impact_id]
    [entry] = audit_rows(db, approach_id)
    assert (entry.action, entry.case_id, entry.detail) == ("modified", case.id, {"deleted": True})
    assert [e["id"] for e in client.get(f"/api/cases/{case.id}/events").json()] == [impact_id]

    assert client.delete(f"/api/events/{approach_id}").status_code == 404
