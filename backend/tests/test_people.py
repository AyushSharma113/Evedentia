"""Tests for the people routes (task B1): list, create, update and delete of
persons, objects and locations, with their audit entries and cascades."""

import pytest
from sqlalchemy import select

from app.models import (
    AuditLog,
    Building,
    CaseObject,
    EvidenceLink,
    Injury,
    Location,
    Person,
    Reconstruction,
    ReconstructionStage,
    Relationship,
    SceneObject,
)
from tests.factories import (
    make_case,
    make_event,
    make_evidence,
    make_investigator,
    make_location,
    make_object,
    make_person,
)

# collection, record type, model, factory, minimal create body, patch body, null-only fields, not-found message
RESOURCES = [
    pytest.param(
        "persons",
        "person",
        Person,
        make_person,
        {"role": "witness", "name": "Lakshmi Pillai"},
        {"name": "Lakshmi P.", "age": 34},
        ("role", "name", "occupation", "contact", "notes", "details", "review_status"),
        "Person not found",
        id="persons",
    ),
    pytest.param(
        "objects",
        "object",
        CaseObject,
        make_object,
        {"kind": "phone", "label": "Witness phone"},
        {"label": "Grey van", "details": {"plate": "KA-05-MN-7788"}},
        ("kind", "label", "details", "review_status"),
        "Object not found",
        id="objects",
    ),
    pytest.param(
        "locations",
        "location",
        Location,
        make_location,
        {"kind": "witness_position", "address": "Bus stop, 5th Avenue"},
        {"address": "Bus stop, north side", "lat": 12.9721},
        ("kind", "address", "review_status"),
        "Location not found",
        id="locations",
    ),
]

FIELDS = "collection, record_type, model, factory, create_body, patch_body, required, not_found"


def audit_rows(db):
    db.expire_all()
    return db.scalars(select(AuditLog).order_by(AuditLog.id)).all()


# list ----------------------------------------------------------------------


@pytest.mark.parametrize(FIELDS, RESOURCES)
def test_list_returns_the_case_rows_by_id(client, db, collection, record_type, model, factory, create_body, patch_body, required, not_found):
    case = make_case(db)
    other = make_case(db)
    first = factory(db, case)
    factory(db, other)
    second = factory(db, case, origin="ai", review_status="pending")

    response = client.get(f"/api/cases/{case.id}/{collection}")

    assert response.status_code == 200
    rows = response.json()
    assert [r["id"] for r in rows] == [first.id, second.id]
    assert {r["case_id"] for r in rows} == {case.id}
    assert (rows[1]["origin"], rows[1]["review_status"]) == ("ai", "pending")
    missing = client.get(f"/api/cases/999/{collection}")
    assert missing.status_code == 404
    assert missing.json() == {"detail": "Case not found"}


# create --------------------------------------------------------------------


@pytest.mark.parametrize(FIELDS, RESOURCES)
def test_create_adds_a_confirmed_row_and_audits(client, db, collection, record_type, model, factory, create_body, patch_body, required, not_found):
    investigator = make_investigator(db)
    case = make_case(db)

    response = client.post(
        f"/api/cases/{case.id}/{collection}", json=create_body, headers={"X-Investigator-Id": str(investigator.id)}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["case_id"] == case.id
    assert (body["origin"], body["review_status"]) == ("investigator", "confirmed")
    assert body.items() >= create_body.items()
    assert db.get(model, body["id"]) is not None

    [entry] = audit_rows(db)
    assert (entry.action, entry.record_type, entry.record_id) == ("modified", record_type, body["id"])
    assert (entry.case_id, entry.actor_id) == (case.id, investigator.id)
    assert entry.detail == {"created": True}


@pytest.mark.parametrize(FIELDS, RESOURCES)
def test_create_validates_the_body_and_case(client, db, collection, record_type, model, factory, create_body, patch_body, required, not_found):
    case = make_case(db)

    for field in create_body:
        body = {k: v for k, v in create_body.items() if k != field}
        assert client.post(f"/api/cases/{case.id}/{collection}", json=body).status_code == 422, field
    assert client.post(f"/api/cases/{case.id}/{collection}", json=create_body | {"origin": "ai"}).status_code == 422
    assert client.post(f"/api/cases/{case.id}/{collection}", json=create_body | {"case_id": case.id}).status_code == 422
    assert client.post(f"/api/cases/999/{collection}", json=create_body).status_code == 404
    assert db.scalars(select(model)).all() == []
    assert audit_rows(db) == []


def test_create_person_defaults(client, db):
    case = make_case(db)

    body = client.post(f"/api/cases/{case.id}/persons", json={"role": "victim", "name": "Divya Rao"}).json()

    assert body | {"id": 0} == {
        "id": 0,
        "case_id": case.id,
        "role": "victim",
        "name": "Divya Rao",
        "age": None,
        "occupation": "",
        "condition": None,
        "contact": "",
        "notes": "",
        "details": {},
        "origin": "investigator",
        "review_status": "confirmed",
    }


def test_create_object_and_location_defaults(client, db):
    case = make_case(db)

    obj = client.post(f"/api/cases/{case.id}/objects", json={"kind": "vehicle", "label": "Grey van"}).json()
    loc = client.post(f"/api/cases/{case.id}/locations", json={"kind": "zone", "address": "Junction"}).json()

    assert obj["details"] == {}
    assert (loc["lat"], loc["lng"], loc["geometry"], loc["building_id"]) == (None, None, None, None)


def test_location_building_must_exist(client, db):
    case = make_case(db)
    building = Building(name="Textile warehouse")
    db.add(building)
    db.commit()
    body = {"kind": "incident_site", "address": "Warehouse 4"}

    unknown = client.post(f"/api/cases/{case.id}/locations", json=body | {"building_id": 999})
    assert unknown.status_code == 422
    assert unknown.json()["detail"][0]["loc"] == ["body", "building_id"]

    created = client.post(f"/api/cases/{case.id}/locations", json=body | {"building_id": building.id})
    assert created.status_code == 201
    location_id = created.json()["id"]

    assert client.patch(f"/api/locations/{location_id}", json={"building_id": 999}).status_code == 422
    cleared = client.patch(f"/api/locations/{location_id}", json={"building_id": None})
    assert cleared.status_code == 200
    assert cleared.json()["building_id"] is None


# update --------------------------------------------------------------------


@pytest.mark.parametrize(FIELDS, RESOURCES)
def test_update_changes_the_row_and_audits(client, db, collection, record_type, model, factory, create_body, patch_body, required, not_found):
    investigator = make_investigator(db)
    case = make_case(db)
    row = factory(db, case)
    before = {field: getattr(row, field) for field in patch_body}

    response = client.patch(f"/api/{collection}/{row.id}", json=patch_body, headers={"X-Investigator-Id": str(investigator.id)})

    assert response.status_code == 200
    assert response.json().items() >= patch_body.items()
    [entry] = audit_rows(db)
    assert (entry.action, entry.record_type, entry.record_id) == ("modified", record_type, row.id)
    assert (entry.case_id, entry.actor_id) == (case.id, investigator.id)
    assert entry.detail == {field: {"before": before[field], "after": value} for field, value in patch_body.items()}


@pytest.mark.parametrize(FIELDS, RESOURCES)
def test_review_status_change_is_logged_as_reviewed(client, db, collection, record_type, model, factory, create_body, patch_body, required, not_found):
    case = make_case(db)
    row = factory(db, case, origin="ai", review_status="pending")

    response = client.patch(f"/api/{collection}/{row.id}", json={"review_status": "confirmed"})

    assert response.status_code == 200
    assert response.json()["review_status"] == "confirmed"
    [entry] = audit_rows(db)
    assert (entry.action, entry.case_id) == ("reviewed", case.id)
    assert entry.detail == {"review_status": {"before": "pending", "after": "confirmed"}}


@pytest.mark.parametrize(FIELDS, RESOURCES)
def test_update_rejects_bad_values(client, db, collection, record_type, model, factory, create_body, patch_body, required, not_found):
    case = make_case(db)
    row = factory(db, case)

    for field in required:
        response = client.patch(f"/api/{collection}/{row.id}", json={field: None})
        assert response.status_code == 422, field
        assert response.json()["detail"][0]["loc"] == ["body", field]
    assert client.patch(f"/api/{collection}/{row.id}", json={"case_id": 2}).status_code == 422
    assert client.patch(f"/api/{collection}/{row.id}", json={"origin": "ai"}).status_code == 422
    assert client.patch(f"/api/{collection}/{row.id}", json={"review_status": "maybe"}).status_code == 422
    missing = client.patch(f"/api/{collection}/999", json=patch_body)
    assert missing.status_code == 404
    assert missing.json() == {"detail": not_found}
    assert audit_rows(db) == []


def test_update_can_clear_nullable_fields(client, db):
    case = make_case(db)
    person = make_person(db, case, age=40, condition="injured")
    site = make_location(db, case)

    person_body = client.patch(f"/api/persons/{person.id}", json={"age": None, "condition": None}).json()
    site_body = client.patch(f"/api/locations/{site.id}", json={"lat": None, "lng": None}).json()

    assert (person_body["age"], person_body["condition"]) == (None, None)
    assert (site_body["lat"], site_body["lng"]) == (None, None)


def test_update_without_changes_is_not_audited(client, db):
    case = make_case(db)
    person = make_person(db, case)

    assert client.patch(f"/api/persons/{person.id}", json={"name": person.name}).status_code == 200
    assert audit_rows(db) == []


# delete --------------------------------------------------------------------


@pytest.mark.parametrize(FIELDS, RESOURCES)
def test_delete_removes_the_row_its_relationships_and_links(client, db, collection, record_type, model, factory, create_body, patch_body, required, not_found):
    investigator = make_investigator(db)
    case = make_case(db)
    evidence = make_evidence(db, case)
    row = factory(db, case)
    keep = factory(db, case)
    other = make_person(db, case, name="Bystander")
    touching = [
        Relationship(case_id=case.id, from_type=record_type, from_id=row.id, to_type="person", to_id=other.id, kind="mentions"),
        Relationship(case_id=case.id, from_type="evidence", from_id=evidence.id, to_type=record_type, to_id=row.id, kind="mentions"),
    ]
    unrelated = Relationship(case_id=case.id, from_type=record_type, from_id=keep.id, to_type="person", to_id=other.id, kind="mentions")
    db.add_all([*touching, unrelated])
    db.commit()
    db.add_all(
        [
            EvidenceLink(evidence_id=evidence.id, target_type=record_type, target_id=row.id, passage="gone"),
            EvidenceLink(evidence_id=evidence.id, target_type="relationship", target_id=touching[0].id, passage="gone"),
            EvidenceLink(evidence_id=evidence.id, target_type=record_type, target_id=keep.id, passage="kept"),
            EvidenceLink(evidence_id=evidence.id, target_type="relationship", target_id=unrelated.id, passage="kept"),
        ]
    )
    db.commit()

    row_id = row.id

    response = client.delete(f"/api/{collection}/{row_id}", headers={"X-Investigator-Id": str(investigator.id)})

    assert response.status_code == 204
    assert response.content == b""
    db.expire_all()
    assert db.get(model, row_id) is None
    assert db.get(model, keep.id) is not None
    assert [r.id for r in db.scalars(select(Relationship))] == [unrelated.id]
    assert {link.passage for link in db.scalars(select(EvidenceLink))} == {"kept"}
    [entry] = audit_rows(db)
    assert (entry.action, entry.record_type, entry.record_id) == ("modified", record_type, row_id)
    assert (entry.case_id, entry.actor_id) == (case.id, investigator.id)
    assert entry.detail == {"deleted": True}

    missing = client.delete(f"/api/{collection}/{row_id}")
    assert missing.status_code == 404
    assert missing.json() == {"detail": not_found}


def test_delete_clears_references_from_other_tables(client, db):
    case = make_case(db)
    person = make_person(db, case)
    van = make_object(db, case)
    site = make_location(db, case)
    event = make_event(db, case, location_id=site.id)
    injury = Injury(case_id=case.id, person_id=person.id, body_region="Head", type="Laceration", severity="moderate")
    scenario = Reconstruction(case_id=case.id, scenario="Van ran the red")
    db.add_all([injury, scenario])
    db.commit()
    stage = ReconstructionStage(reconstruction_id=scenario.id, label="Impact")
    db.add(stage)
    db.commit()
    placed = SceneObject(stage_id=stage.id, model_id="van", object_id=van.id)
    db.add(placed)
    db.commit()

    for path in (f"/api/persons/{person.id}", f"/api/objects/{van.id}", f"/api/locations/{site.id}"):
        assert client.delete(path).status_code == 204, path

    db.expire_all()
    assert db.get(Injury, injury.id).person_id is None
    assert db.get(SceneObject, placed.id).object_id is None
    assert db.get(type(event), event.id).location_id is None
