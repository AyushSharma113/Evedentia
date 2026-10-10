"""Tests for the dashboard and the alerts feed (task B5)."""

from datetime import UTC, datetime, timedelta

from app.models import Conflict
from tests.factories import (
    make_case,
    make_event,
    make_evidence,
    make_investigator,
    make_location,
    make_observation,
    make_person,
)

T0 = datetime(2026, 10, 3, 16, 0, tzinfo=UTC)


def make_conflict(db, case, **fields):
    values = {"description": "Witnesses disagree on the signal colour.", "status": "open", "observation_ids": []}
    row = Conflict(case_id=case.id, **(values | fields))
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def test_empty_dashboard(client):
    body = client.get("/api/dashboard").json()
    assert body == {
        "total_cases": 0,
        "active_cases": 0,
        "pending_reviews": 0,
        "storage_bytes": 0,
        "status_counts": {"open": 0, "in_progress": 0, "under_review": 0, "closed": 0},
        "incident_points": [],
        "investigators": [],
    }


def test_dashboard_totals_and_status_counts(client, db):
    make_case(db, status="open")
    review = make_case(db, status="under_review")
    closed = make_case(db, status="closed")
    make_evidence(db, review, size_bytes=1000)
    make_evidence(db, closed, size_bytes=24)
    make_person(db, review, origin="ai", review_status="pending")
    make_event(db, review, origin="ai", review_status="pending")
    make_observation(db, closed, origin="ai", review_status="pending")
    make_person(db, review, origin="ai", review_status="confirmed")
    make_person(db, review, review_status="pending")  # not AI, so not a review

    body = client.get("/api/dashboard").json()

    assert body["total_cases"] == 3
    assert body["active_cases"] == 2
    assert body["pending_reviews"] == 3
    assert body["storage_bytes"] == 1024
    # Every status is present, including the ones no case has.
    assert body["status_counts"] == {"open": 1, "in_progress": 0, "under_review": 1, "closed": 1}


def test_incident_points_come_from_incident_sites_with_coordinates(client, db):
    case = make_case(db, title="Ring Road junction collision", priority="critical")
    other = make_case(db, title="Warehouse fire", incident_type="fire", incident_at=None)
    site = make_location(db, case, address="Ring Road and 5th Avenue", lat=12.97, lng=77.59)
    make_location(db, case, kind="witness_position", address="Bus stop", lat=12.98, lng=77.6)
    make_location(db, case, address="No coordinates", lat=None, lng=None)
    make_location(db, case, address="No longitude", lat=12.9, lng=None)
    make_location(db, other, address="Industrial Area", lat=13.01, lng=77.55)

    points = client.get("/api/dashboard").json()["incident_points"]

    assert points == [
        {
            "case_id": case.id,
            "case_number": case.case_number,
            "title": "Ring Road junction collision",
            "incident_type": "road_accident",
            "priority": "critical",
            "address": site.address,
            "lat": 12.97,
            "lng": 77.59,
            "incident_at": "2026-10-03T16:10:00Z",
        },
        {
            "case_id": other.id,
            "case_number": other.case_number,
            "title": "Warehouse fire",
            "incident_type": "fire",
            "priority": "high",
            "address": "Industrial Area",
            "lat": 13.01,
            "lng": 77.55,
            "incident_at": None,
        },
    ]


def test_investigator_solved_and_active_counts(client, db):
    meera = make_investigator(db)
    arjun = make_investigator(db, name="SI Arjun Nair", badge_number="TP-2088")
    idle = make_investigator(db, name="Idle", badge_number="TP-0001")
    make_case(db, status="closed", lead_investigator_id=meera.id, investigator_ids=[meera.id, arjun.id])
    make_case(db, status="closed", lead_investigator_id=meera.id, investigator_ids=[meera.id])
    make_case(db, status="open", lead_investigator_id=arjun.id, investigator_ids=[meera.id, arjun.id])
    make_case(db, status="in_progress", lead_investigator_id=arjun.id, investigator_ids=[arjun.id])

    investigators = client.get("/api/dashboard").json()["investigators"]

    assert investigators[0] == {
        "id": meera.id,
        "name": "Insp. Meera Rao",
        "badge_number": "TP-1042",
        "role": "Lead investigator",
        "clearance_level": "L3",
        "solved_cases": 2,
        "active_cases": 1,
    }
    assert [(i["id"], i["solved_cases"], i["active_cases"]) for i in investigators] == [
        (meera.id, 2, 1),
        (arjun.id, 0, 2),
        (idle.id, 0, 0),
    ]


def test_no_alerts(client, db):
    make_case(db)
    assert client.get("/api/alerts").json() == []


def test_conflict_and_review_alerts(client, db):
    case = make_case(db)
    case.updated_at = T0
    db.commit()
    conflict = make_conflict(db, case)
    make_conflict(db, case, description="Already resolved.", status="resolved")
    make_person(db, case, origin="ai", review_status="pending")
    make_observation(db, case, origin="ai", review_status="pending")
    make_observation(db, case, origin="ai", review_status="rejected")

    alerts = client.get("/api/alerts").json()

    # Conflict and review alerts carry the case's updated_at, as in the mock.
    stamp = "2026-10-03T16:00:00Z"
    assert alerts == [
        {
            "id": f"conflict-{conflict.id}",
            "kind": "conflict",
            "severity": "warning",
            "title": "Conflicting statements",
            "message": "Witnesses disagree on the signal colour.",
            "case_id": case.id,
            "case_number": case.case_number,
            "record_type": "conflict",
            "record_id": conflict.id,
            "created_at": stamp,
        },
        {
            "id": f"review-{case.id}",
            "kind": "review",
            "severity": "info",
            "title": "2 AI proposals to review",
            "message": (
                f"New people, events, links or statements were proposed from the evidence in {case.case_number}."
            ),
            "case_id": case.id,
            "case_number": case.case_number,
            "record_type": None,
            "record_id": None,
            "created_at": stamp,
        },
    ]


def test_processing_alerts_from_processed_at(client, db):
    case = make_case(db)
    done = make_evidence(db, case, title="Witness statement.txt", processed_at=T0)
    failed = make_evidence(
        db, case, title="Corrupt video.mp4", processing_status="failed", processed_at=T0 + timedelta(minutes=5)
    )
    # Never processed, and a reprocess still in flight: neither is an alert.
    make_evidence(db, case, title="Queued.pdf", processing_status="queued")
    make_evidence(db, case, title="Reprocessing.pdf", processing_status="processing", processed_at=T0)

    alerts = client.get("/api/alerts").json()

    assert alerts == [
        {
            "id": f"processing-{failed.id}-2026-10-03T16:05:00Z",
            "kind": "processing",
            "severity": "critical",
            "title": "Processing failed",
            "message": "Corrupt video.mp4",
            "case_id": case.id,
            "case_number": case.case_number,
            "record_type": "evidence",
            "record_id": failed.id,
            "created_at": "2026-10-03T16:05:00Z",
        },
        {
            "id": f"processing-{done.id}-2026-10-03T16:00:00Z",
            "kind": "processing",
            "severity": "info",
            "title": "Processing finished",
            "message": "Witness statement.txt",
            "case_id": case.id,
            "case_number": case.case_number,
            "record_type": "evidence",
            "record_id": done.id,
            "created_at": "2026-10-03T16:00:00Z",
        },
    ]


def test_alerts_are_newest_first_across_kinds_and_cases(client, db):
    old = make_case(db)
    new = make_case(db, title="Another case")
    old.updated_at = T0
    new.updated_at = T0 + timedelta(hours=2)
    db.commit()
    conflict = make_conflict(db, old)
    make_person(db, new, origin="ai", review_status="pending")
    evidence = make_evidence(db, old, processed_at=T0 + timedelta(hours=1))

    alerts = client.get("/api/alerts").json()

    assert [a["id"] for a in alerts] == [
        f"review-{new.id}",
        f"processing-{evidence.id}-2026-10-03T17:00:00Z",
        f"conflict-{conflict.id}",
    ]
    assert [a["created_at"] for a in alerts] == [
        "2026-10-03T18:00:00Z",
        "2026-10-03T17:00:00Z",
        "2026-10-03T16:00:00Z",
    ]
