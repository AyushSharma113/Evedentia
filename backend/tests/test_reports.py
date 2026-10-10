"""Tests for the reports routes and the case audit trail (task B5)."""

import hashlib
import json
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from app.models import AuditLog, Conflict, EvidenceLink, Injury, Reconstruction, Report
from app.services.reports.content import content_sha256
from tests.factories import (
    make_case,
    make_event,
    make_evidence,
    make_investigator,
    make_observation,
    make_person,
)

T0 = datetime(2026, 10, 3, 16, 0, tzinfo=UTC)


def _save(db, row):
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def make_link(db, evidence, target_type, target_id, **fields):
    return _save(db, EvidenceLink(evidence_id=evidence.id, target_type=target_type, target_id=target_id, **fields))


def make_conflict(db, case, observations, **fields):
    values = {"description": "Witnesses disagree on the signal colour.", "status": "open"}
    return _save(db, Conflict(case_id=case.id, observation_ids=[o.id for o in observations], **(values | fields)))


def make_injury(db, case, **fields):
    values = {"body_region": "Head", "type": "Laceration", "severity": "moderate"}
    return _save(db, Injury(case_id=case.id, **(values | fields)))


def make_reconstruction(db, case, **fields):
    values = {"scenario": "Van ran the red", "version": 1, "assumptions": "Signal timing from the controller log."}
    return _save(db, Reconstruction(case_id=case.id, **(values | fields)))


def make_audit(db, case, record_type, record_id, *, at, action="viewed", **fields):
    return _save(
        db, AuditLog(action=action, record_type=record_type, record_id=record_id, case_id=case.id, at=at, **fields)
    )


def build_case(db):
    """A case with a row of every kind each template draws on, plus one rejected
    or out-of-scope row next to each, so the filters show."""
    lead = make_investigator(db)
    case = make_case(db, lead_investigator_id=lead.id, investigator_ids=[lead.id])
    other = make_case(db, title="Another case")
    report = make_evidence(db, case, title="Police report.pdf", summary="Signal was red for the van.")
    medical = make_evidence(db, case, type="medical", title="Discharge summary.pdf", summary="Scalp laceration.")
    photo = make_evidence(db, case, type="photo", title="Scene photo.jpg", mime_type="image/jpeg")
    make_evidence(db, other, title="Other case report.pdf", summary="Not this case.")

    driver = make_person(db, case, role="victim", name="Rahul Verma")
    make_person(db, case, name="Rejected witness", origin="ai", review_status="rejected")
    make_person(db, case, name="Proposed witness", origin="ai", review_status="pending")
    make_person(db, other, name="Other case person")

    late = make_event(db, case, description="Ambulance arrives.", time_start=T0 + timedelta(minutes=20))
    early = make_event(db, case, description="Van enters the junction.", time_start=T0, certainty="hypothesis")
    untimed = make_event(db, case, description="Phone call to the van driver.", certainty="disputed")
    make_event(db, case, description="Rejected event.", time_start=T0, origin="ai", review_status="rejected")

    red = make_observation(db, case, statement="The light was red for the van.")
    amber = make_observation(db, case, statement="The light was amber.", origin="ai", review_status="pending")
    about_driver = make_observation(db, case, subject_type="person", subject_id=driver.id, statement="Driver bled.")
    make_link(db, report, "observation", red.id)
    make_link(db, photo, "observation", red.id)
    make_link(db, report, "observation", red.id, passage="again")
    make_link(db, medical, "observation", about_driver.id)

    make_conflict(db, case, [red, amber])
    make_conflict(db, case, [red], description="Resolved one.", status="resolved")
    make_reconstruction(db, case)
    make_reconstruction(db, other, scenario="Other scenario")
    make_injury(db, case, person_id=driver.id)
    make_injury(db, case, body_region="Arm", origin="ai", review_status="rejected")

    make_audit(db, case, "evidence", report.id, action="uploaded", at=T0 + timedelta(hours=2), actor_id=lead.id)
    make_audit(db, case, "evidence", medical.id, action="uploaded", at=T0 + timedelta(hours=1))
    make_audit(db, case, "person", driver.id, action="modified", at=T0 + timedelta(hours=3))
    return {
        "lead": lead,
        "case": case,
        "evidence": (report, medical, photo),
        "events": (early, late, untimed),
        "driver": driver,
        "observations": (red, amber, about_driver),
    }


def post_report(client, case_id, template, actor=None, **options):
    headers = {"X-Investigator-Id": str(actor.id)} if actor else {}
    return client.post(f"/api/cases/{case_id}/reports", json={"template": template, "options": options}, headers=headers)


def test_court_report_fills_every_section(client, db):
    data = build_case(db)
    case, lead = data["case"], data["lead"]
    report_ev, medical_ev, photo_ev = data["evidence"]
    early, late, untimed = data["events"]

    response = post_report(client, case.id, "court", actor=lead)
    assert response.status_code == 201
    body = response.json()
    assert body["template"] == "court"
    assert body["options"] == {"include_ai_insights": False, "include_custody": False}
    assert body["created_by"] == lead.id
    content = body["content"]

    assert content["case"]["id"] == case.id
    assert content["case"]["case_number"] == case.case_number
    assert content["case"]["lead_investigator_name"] == lead.name
    assert content["summary"] == case.description
    assert [e["id"] for e in content["evidence"]] == [report_ev.id, medical_ev.id, photo_ev.id]
    assert set(content["evidence"][0]) == {"id", "title", "type", "sha256", "size_bytes", "collected_at"}
    # As the mock: untimed events first, then by time; rejected events left out.
    assert [e["id"] for e in content["timeline"]] == [untimed.id, early.id, late.id]
    assert [p["name"] for p in content["persons"]] == ["Rahul Verma", "Proposed witness"]
    assert content["confirmed_findings"] == [
        {"statement": "The light was red for the van.", "evidence_ids": [report_ev.id, photo_ev.id]},
        {"statement": "Driver bled.", "evidence_ids": [medical_ev.id]},
    ]
    assert content["open_conflicts"] == [
        {
            "description": "Witnesses disagree on the signal colour.",
            "statements": ["The light was red for the van.", "The light was amber."],
        }
    ]
    assert content["uncertainties"] == [
        "Phone call to the van driver. (disputed)",
        "Van enters the junction. (hypothesis)",
        "2 AI-proposed items are still waiting for review.",
    ]
    assert content["reconstructions"] == [
        {"scenario": "Van ran the red", "version": 1, "assumptions": "Signal timing from the controller log."}
    ]
    assert [i["body_region"] for i in content["injuries"]] == ["Head"]
    assert content["ai_insights"] is None
    assert content["custody"] is None


def test_report_is_stored_hashed_and_audited(client, db):
    data = build_case(db)
    case, lead = data["case"], data["lead"]

    body = post_report(client, case.id, "summary", actor=lead, include_custody=True).json()

    canonical = json.dumps(body["content"], sort_keys=True, separators=(",", ":"))
    assert body["sha256"] == hashlib.sha256(canonical.encode()).hexdigest()
    stored = db.get(Report, body["id"])
    assert stored.case_id == case.id
    assert stored.content == body["content"]
    assert stored.sha256 == body["sha256"]

    entry = db.scalars(select(AuditLog).where(AuditLog.record_type == "report")).one()
    assert entry.action == "exported"
    assert entry.record_id == body["id"]
    assert entry.case_id == case.id
    assert entry.actor_id == lead.id
    assert entry.detail == {"template": "summary"}

    # Reading it back gives the same content, which still matches its hash.
    fetched = client.get(f"/api/reports/{body['id']}").json()
    assert fetched == body
    assert content_sha256(fetched["content"]) == body["sha256"]


def test_summary_report_leaves_out_reconstructions_and_injuries(client, db):
    data = build_case(db)
    content = post_report(client, data["case"].id, "summary").json()["content"]
    assert len(content["timeline"]) == 3
    assert len(content["open_conflicts"]) == 1
    assert len(content["uncertainties"]) == 3
    assert content["reconstructions"] == []
    assert content["injuries"] == []
    assert content["ai_insights"] is None
    assert content["custody"] is None


def test_custody_report_lists_the_evidence_audit_oldest_first(client, db):
    data = build_case(db)
    report_ev, medical_ev, _ = data["evidence"]

    content = post_report(client, data["case"].id, "custody", include_ai_insights=True).json()["content"]

    assert [(c["record_type"], c["record_id"]) for c in content["custody"]] == [
        ("evidence", medical_ev.id),
        ("evidence", report_ev.id),
    ]
    assert content["custody"][1]["actor_name"] == data["lead"].name
    assert len(content["evidence"]) == 3
    assert content["timeline"] == []
    assert content["open_conflicts"] == []
    assert content["uncertainties"] == []
    assert content["reconstructions"] == []
    assert content["injuries"] == []
    # AI insights never go in a custody report.
    assert content["ai_insights"] is None


def test_medical_report_keeps_medical_evidence_and_findings_about_people(client, db):
    data = build_case(db)
    _, medical_ev, _ = data["evidence"]

    content = post_report(client, data["case"].id, "medical", include_ai_insights=True).json()["content"]

    assert [e["id"] for e in content["evidence"]] == [medical_ev.id]
    assert content["confirmed_findings"] == [{"statement": "Driver bled.", "evidence_ids": [medical_ev.id]}]
    assert [i["body_region"] for i in content["injuries"]] == ["Head"]
    assert content["ai_insights"] == ["Discharge summary.pdf: Scalp laceration."]
    assert content["timeline"] == []
    assert content["open_conflicts"] == []
    assert content["uncertainties"] == []
    assert content["reconstructions"] == []
    assert content["custody"] is None


def test_options_add_ai_insights_and_custody(client, db):
    data = build_case(db)

    body = post_report(client, data["case"].id, "court", include_ai_insights=True, include_custody=True).json()

    assert body["options"] == {"include_ai_insights": True, "include_custody": True}
    assert body["content"]["ai_insights"] == [
        "Police report.pdf: Signal was red for the van.",
        "Discharge summary.pdf: Scalp laceration.",
    ]
    assert len(body["content"]["custody"]) == 2


def test_report_without_lead_or_pending_reviews(client, db):
    case = make_case(db)
    content = post_report(client, case.id, "court").json()["content"]
    assert content["case"]["lead_investigator_name"] is None
    assert content["evidence"] == []
    assert content["uncertainties"] == []


def test_create_report_rejects_bad_input(client, db):
    case = make_case(db)
    assert client.post(f"/api/cases/{case.id}/reports", json={"template": "novel"}).status_code == 422
    assert client.post(f"/api/cases/{case.id}/reports", json={}).status_code == 422
    response = client.post("/api/cases/999/reports", json={"template": "court"})
    assert response.status_code == 404
    assert response.json() == {"detail": "Case not found"}
    assert db.scalars(select(Report)).all() == []


def test_list_reports_newest_first_without_content(client, db):
    case = make_case(db)
    other = make_case(db, title="Another case")
    first = post_report(client, case.id, "court").json()
    second = post_report(client, case.id, "custody").json()
    post_report(client, other.id, "summary")

    rows = client.get(f"/api/cases/{case.id}/reports").json()

    assert [r["id"] for r in rows] == [second["id"], first["id"]]
    assert all("content" not in r for r in rows)
    assert set(rows[0]) == {"id", "case_id", "template", "options", "sha256", "created_by", "created_at"}
    assert rows[0]["template"] == "custody"
    assert client.get("/api/cases/999/reports").status_code == 404


def test_get_report_404(client):
    response = client.get("/api/reports/999")
    assert response.status_code == 404
    assert response.json() == {"detail": "Report not found"}


def test_case_audit_newest_first_by_case_id(client, db):
    lead = make_investigator(db)
    case = make_case(db)
    other = make_case(db, title="Another case")
    oldest = make_audit(db, case, "case", case.id, at=T0, action="modified")
    # A deleted record keeps its case_id, so it still shows.
    deleted = make_audit(db, case, "person", 42, at=T0 + timedelta(hours=2), detail={"deleted": True})
    middle = make_audit(db, case, "evidence", 7, at=T0 + timedelta(hours=1), actor_id=lead.id)
    make_audit(db, other, "case", other.id, at=T0 + timedelta(hours=5))
    _save(db, AuditLog(action="viewed", record_type="case", record_id=case.id, case_id=None, at=T0))

    entries = client.get(f"/api/cases/{case.id}/audit").json()

    assert [e["id"] for e in entries] == [deleted.id, middle.id, oldest.id]
    assert entries[0]["detail"] == {"deleted": True}
    assert entries[1]["actor_id"] == lead.id
    assert entries[1]["actor_name"] == lead.name
    assert entries[2]["actor_name"] is None
    assert entries[0]["at"] == "2026-10-03T18:00:00Z"


def test_case_audit_includes_report_exports(client, db):
    case = make_case(db)
    report = post_report(client, case.id, "court").json()
    entries = client.get(f"/api/cases/{case.id}/audit").json()
    assert [(e["action"], e["record_type"], e["record_id"]) for e in entries] == [("exported", "report", report["id"])]


def test_case_audit_404(client):
    response = client.get("/api/cases/999/audit")
    assert response.status_code == 404
    assert response.json() == {"detail": "Case not found"}
