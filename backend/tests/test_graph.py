"""Tests for the graph routes: GET /cases/{id}/graph and relationship CRUD."""

from sqlalchemy import select

from app.models import AuditLog, EvidenceLink, Relationship
from tests.factories import (
    make_case,
    make_evidence,
    make_investigator,
    make_location,
    make_object,
    make_person,
)


def make_relationship(db, case, from_node, to_node, **fields) -> Relationship:
    values = {
        "case_id": case.id,
        "from_type": from_node[0],
        "from_id": from_node[1],
        "to_type": to_node[0],
        "to_id": to_node[1],
        "kind": "involved_in",
        "label": "Involved in",
    }
    row = Relationship(**(values | fields))
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def audit_rows(db, record_type: str, record_id: int) -> list[AuditLog]:
    query = select(AuditLog).where(AuditLog.record_type == record_type, AuditLog.record_id == record_id)
    return db.scalars(query.order_by(AuditLog.id)).all()


# GET /cases/{id}/graph -------------------------------------------------------


def test_graph_of_unknown_case_is_404(client):
    response = client.get("/api/cases/99/graph")
    assert response.status_code == 404
    assert response.json() == {"detail": "Case not found"}


def test_graph_nodes_and_sublabels(client, db):
    case = make_case(db)
    driver = make_person(db, case, role="suspect", name="Ravi Kumar", condition="injured")
    witness = make_person(db, case, role="witness", name="Anita Desai", origin="ai", review_status="pending")
    van = make_object(db, case, kind="vehicle", label="Delivery van", details={"plate": "KA-05-MN-7788"})
    phone = make_object(db, case, kind="phone", label="Driver's phone", details={})
    site = make_location(db, case, kind="incident_site", address="Ring Road and 5th Avenue")
    make_location(db, case, kind="witness_position", address="Bus stop")
    report = make_evidence(db, case, title="Police report.pdf", type="document")
    make_evidence(db, case, title="Unlinked photo.jpg", type="photo")
    make_relationship(db, case, ("person", driver.id), ("object", van.id), kind="drove", label="Drove")
    make_relationship(db, case, ("object", van.id), ("location", site.id), kind="located_at", label="Located at")
    make_relationship(db, case, ("evidence", report.id), ("person", witness.id), kind="mentions", label="Mentions")

    response = client.get(f"/api/cases/{case.id}/graph")
    assert response.status_code == 200
    nodes = response.json()["nodes"]
    assert nodes == [
        {
            "id": f"person:{driver.id}",
            "type": "person",
            "record_id": driver.id,
            "label": "Ravi Kumar",
            "sublabel": "Suspect, injured",
            "review_status": "confirmed",
        },
        {
            "id": f"person:{witness.id}",
            "type": "person",
            "record_id": witness.id,
            "label": "Anita Desai",
            "sublabel": "Witness",
            "review_status": "pending",
        },
        {
            "id": f"object:{van.id}",
            "type": "object",
            "record_id": van.id,
            "label": "Delivery van",
            "sublabel": "KA-05-MN-7788",
            "review_status": "confirmed",
        },
        {
            "id": f"object:{phone.id}",
            "type": "object",
            "record_id": phone.id,
            "label": "Driver's phone",
            "sublabel": "Phone",
            "review_status": "confirmed",
        },
        {
            "id": f"location:{site.id}",
            "type": "location",
            "record_id": site.id,
            "label": "Ring Road and 5th Avenue",
            "sublabel": "Incident site",
            "review_status": "confirmed",
        },
        {
            "id": f"evidence:{report.id}",
            "type": "evidence",
            "record_id": report.id,
            "label": "Police report.pdf",
            "sublabel": "Document",
            "review_status": "confirmed",
        },
    ]


def test_graph_edges(client, db):
    case = make_case(db)
    driver = make_person(db, case)
    van = make_object(db, case)
    drove = make_relationship(
        db, case, ("person", driver.id), ("object", van.id), kind="drove", label="Drove", confidence=0.8, flagged=True
    )

    edges = client.get(f"/api/cases/{case.id}/graph").json()["edges"]
    assert edges == [
        {
            "id": f"relationship:{drove.id}",
            "relationship_id": drove.id,
            "source": f"person:{driver.id}",
            "target": f"object:{van.id}",
            "kind": "drove",
            "label": "Drove",
            "confidence": 0.8,
            "flagged": True,
            "review_status": "confirmed",
        }
    ]


def test_graph_keeps_to_its_case_and_drops_dangling_edges(client, db):
    case = make_case(db)
    other = make_case(db, title="Other case")
    person = make_person(db, case)
    make_person(db, other, name="Someone else")
    foreign_location = make_location(db, other)
    # Ends that are missing or in another case never become nodes, so no edge either.
    make_relationship(db, case, ("person", person.id), ("person", 999))
    make_relationship(db, case, ("person", person.id), ("location", foreign_location.id))
    make_relationship(db, other, ("person", person.id), ("person", person.id))

    graph = client.get(f"/api/cases/{case.id}/graph").json()
    assert [node["id"] for node in graph["nodes"]] == [f"person:{person.id}"]
    assert graph["edges"] == []


# POST /cases/{id}/relationships ----------------------------------------------


def test_create_relationship(client, db):
    investigator = make_investigator(db)
    case = make_case(db)
    person = make_person(db, case)
    van = make_object(db, case)

    response = client.post(
        f"/api/cases/{case.id}/relationships",
        json={"from_type": "person", "from_id": person.id, "to_type": "object", "to_id": van.id, "kind": "drove"},
        headers={"X-Investigator-Id": str(investigator.id)},
    )
    assert response.status_code == 201
    body = response.json()
    assert body == {
        "id": body["id"],
        "case_id": case.id,
        "from_type": "person",
        "from_id": person.id,
        "to_type": "object",
        "to_id": van.id,
        "kind": "drove",
        "label": "Drove",
        "confidence": None,
        "flagged": False,
        "notes": "",
        "origin": "investigator",
        "review_status": "confirmed",
    }
    [entry] = audit_rows(db, "relationship", body["id"])
    assert (entry.action, entry.actor_id, entry.case_id, entry.detail) == (
        "modified",
        investigator.id,
        case.id,
        {"created": True},
    )

    edges = client.get(f"/api/cases/{case.id}/graph").json()["edges"]
    assert [edge["relationship_id"] for edge in edges] == [body["id"]]


def test_create_relationship_labels(client, db):
    case = make_case(db)
    person = make_person(db, case)
    site = make_location(db, case)
    url = f"/api/cases/{case.id}/relationships"
    ends = {"from_type": "person", "from_id": person.id, "to_type": "location", "to_id": site.id}

    blank = client.post(url, json=ends | {"kind": "involved_in", "label": "  "}).json()
    assert blank["label"] == "Involved in"
    given = client.post(url, json=ends | {"kind": "located_at", "label": "Stood at the corner"}).json()
    assert given["label"] == "Stood at the corner"


def test_create_relationship_ends_must_be_in_the_case(client, db):
    case = make_case(db)
    other = make_case(db, title="Other case")
    person = make_person(db, case)
    stranger = make_person(db, other)
    url = f"/api/cases/{case.id}/relationships"

    response = client.post(
        url,
        json={"from_type": "person", "from_id": stranger.id, "to_type": "person", "to_id": person.id, "kind": "other"},
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "from_id"]

    response = client.post(
        url, json={"from_type": "person", "from_id": person.id, "to_type": "evidence", "to_id": 42, "kind": "other"}
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "to_id"]
    assert db.scalars(select(Relationship)).all() == []


def test_create_relationship_rejects_bad_bodies(client, db):
    case = make_case(db)
    person = make_person(db, case)
    ends = {"from_type": "person", "from_id": person.id, "to_type": "person", "to_id": person.id}

    assert client.post(f"/api/cases/{case.id}/relationships", json=ends | {"kind": "loves"}).status_code == 422
    assert client.post(f"/api/cases/{case.id}/relationships", json=ends).status_code == 422
    bad_type = ends | {"to_type": "event", "kind": "other"}
    assert client.post(f"/api/cases/{case.id}/relationships", json=bad_type).status_code == 422
    assert client.post("/api/cases/99/relationships", json=ends | {"kind": "other"}).status_code == 404


# PATCH, DELETE /relationships/{id} -------------------------------------------


def test_update_relationship(client, db):
    case = make_case(db)
    person = make_person(db, case)
    van = make_object(db, case)
    rel = make_relationship(db, case, ("person", person.id), ("object", van.id))

    response = client.patch(f"/api/relationships/{rel.id}", json={"kind": "drove", "label": "Drove", "confidence": 0.6})
    assert response.status_code == 200
    assert response.json()["kind"] == "drove"
    assert response.json()["label"] == "Drove"
    assert response.json()["confidence"] == 0.6
    [entry] = audit_rows(db, "relationship", rel.id)
    assert entry.action == "modified"
    assert entry.case_id == case.id
    assert entry.detail == {
        "kind": {"before": "involved_in", "after": "drove"},
        "label": {"before": "Involved in", "after": "Drove"},
        "confidence": {"before": None, "after": 0.6},
    }

    # Clearing the label falls back to the humanised kind.
    assert client.patch(f"/api/relationships/{rel.id}", json={"label": ""}).json()["label"] == "Drove"
    # Confidence can be cleared; kind can't be null.
    assert client.patch(f"/api/relationships/{rel.id}", json={"confidence": None}).json()["confidence"] is None
    response = client.patch(f"/api/relationships/{rel.id}", json={"kind": None})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "kind"]


def test_review_only_change_to_relationship_audits_reviewed(client, db):
    case = make_case(db)
    person = make_person(db, case)
    van = make_object(db, case)
    rel = make_relationship(db, case, ("person", person.id), ("object", van.id), origin="ai", review_status="pending")

    response = client.patch(f"/api/relationships/{rel.id}", json={"review_status": "rejected"})
    assert response.json()["review_status"] == "rejected"
    [entry] = audit_rows(db, "relationship", rel.id)
    assert entry.action == "reviewed"
    assert entry.detail == {"review_status": {"before": "pending", "after": "rejected"}}


def test_update_relationship_rejects_unknown_fields_and_ids(client, db):
    case = make_case(db)
    person = make_person(db, case)
    rel = make_relationship(db, case, ("person", person.id), ("person", person.id))

    assert client.patch(f"/api/relationships/{rel.id}", json={"from_id": 5}).status_code == 422
    response = client.patch("/api/relationships/99", json={"label": "x"})
    assert response.status_code == 404
    assert response.json() == {"detail": "Relationship not found"}


def test_delete_relationship(client, db):
    case = make_case(db)
    person = make_person(db, case)
    report = make_evidence(db, case)
    rel_id = make_relationship(db, case, ("evidence", report.id), ("person", person.id)).id
    db.add(EvidenceLink(evidence_id=report.id, target_type="relationship", target_id=rel_id, passage="..."))
    db.commit()

    response = client.delete(f"/api/relationships/{rel_id}")
    assert response.status_code == 204
    assert response.content == b""
    db.expire_all()
    assert db.get(Relationship, rel_id) is None
    assert db.scalars(select(EvidenceLink)).all() == []
    [entry] = audit_rows(db, "relationship", rel_id)
    assert (entry.action, entry.case_id, entry.detail) == ("modified", case.id, {"deleted": True})

    graph = client.get(f"/api/cases/{case.id}/graph").json()
    assert graph["edges"] == []
    assert [node["id"] for node in graph["nodes"]] == [f"person:{person.id}"]
    assert client.delete(f"/api/relationships/{rel_id}").status_code == 404
