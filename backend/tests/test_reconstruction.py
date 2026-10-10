"""Tests for the reconstruction routes: scenarios, stages, scene objects and annotations."""

from sqlalchemy import select

from app.models import (
    AuditLog,
    Building,
    EvidenceLink,
    Reconstruction,
    ReconstructionStage,
    SceneAnnotation,
    SceneObject,
)
from tests.factories import make_case, make_event, make_evidence, make_investigator, make_object


def make_reconstruction(db, case, **fields) -> Reconstruction:
    values = {"scenario": "Van ran the red light", "assumptions": "Signal timing from the controller log."}
    row = Reconstruction(case_id=case.id, **(values | fields))
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def make_stage(db, reconstruction, **fields) -> ReconstructionStage:
    values = {"stage_order": 1, "label": "Approach", "event_ids": []}
    row = ReconstructionStage(reconstruction_id=reconstruction.id, **(values | fields))
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def make_scene_object(db, stage, **fields) -> SceneObject:
    values = {"model_id": "hatchback", "label": "Hatchback", "position": [1.0, 0.0, 2.0]}
    row = SceneObject(stage_id=stage.id, **(values | fields))
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def audit_rows(db, record_type):
    return db.scalars(select(AuditLog).where(AuditLog.record_type == record_type).order_by(AuditLog.id)).all()


# Scenarios ------------------------------------------------------------------


def test_create_reconstruction_starts_at_version_1_and_audits(client, db):
    investigator = make_investigator(db)
    case = make_case(db)
    response = client.post(
        f"/api/cases/{case.id}/reconstructions",
        json={"scenario": "Van ran the red light"},
        headers={"X-Investigator-Id": str(investigator.id)},
    )
    assert response.status_code == 201
    body = response.json()
    assert body == {
        "id": body["id"],
        "case_id": case.id,
        "scenario": "Van ran the red light",
        "version": 1,
        "assumptions": "",
        "scene_file": None,
    }
    [entry] = audit_rows(db, "reconstruction")
    assert (entry.action, entry.record_id, entry.case_id, entry.actor_id) == (
        "modified",
        body["id"],
        case.id,
        investigator.id,
    )
    assert entry.detail == {"created": True}


def test_create_reconstruction_rejects_bad_bodies(client, db):
    case = make_case(db)
    url = f"/api/cases/{case.id}/reconstructions"
    assert client.post(url, json={"scenario": ""}).status_code == 422
    assert client.post(url, json={}).status_code == 422
    assert client.post(url, json={"scenario": "X", "version": 3}).status_code == 422
    assert client.post("/api/cases/999/reconstructions", json={"scenario": "X"}).status_code == 404


def test_list_reconstructions_is_the_case_only_in_id_order(client, db):
    case = make_case(db)
    other = make_case(db, title="Another case")
    first = make_reconstruction(db, case, scenario="Van ran the red")
    make_reconstruction(db, other, scenario="Elsewhere")
    second = make_reconstruction(db, case, scenario="Hatchback ran the amber")
    response = client.get(f"/api/cases/{case.id}/reconstructions")
    assert response.status_code == 200
    assert [r["id"] for r in response.json()] == [first.id, second.id]
    assert client.get("/api/cases/999/reconstructions").status_code == 404


# Detail ---------------------------------------------------------------------


def test_reconstruction_detail_orders_stages_and_nests_objects(client, db):
    case = make_case(db)
    reconstruction = make_reconstruction(db, case)
    impact = make_stage(db, reconstruction, stage_order=2, label="Impact")
    approach = make_stage(db, reconstruction, stage_order=1, label="Approach")
    van = make_scene_object(db, impact, model_id="van", label="Van")
    car = make_scene_object(db, approach, model_id="hatchback")
    note = SceneAnnotation(reconstruction_id=reconstruction.id, stage_id=impact.id, kind="note", data={"text": "Skid"})
    view = SceneAnnotation(reconstruction_id=reconstruction.id, stage_id=None, kind="viewpoint", data={})
    db.add_all([note, view])
    other = make_reconstruction(db, case, scenario="Other")
    make_stage(db, other)
    db.commit()

    response = client.get(f"/api/reconstructions/{reconstruction.id}")
    assert response.status_code == 200
    body = response.json()
    assert body["scenario"] == reconstruction.scenario
    assert [s["label"] for s in body["stages"]] == ["Approach", "Impact"]
    assert [o["id"] for o in body["stages"][0]["scene_objects"]] == [car.id]
    assert body["stages"][1]["scene_objects"] == [
        {
            "id": van.id,
            "stage_id": impact.id,
            "model_id": "van",
            "object_id": None,
            "label": "Van",
            "position": [1.0, 0.0, 2.0],
            "rotation": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        }
    ]
    assert [(a["kind"], a["stage_id"]) for a in body["annotations"]] == [("note", impact.id), ("viewpoint", None)]


def test_reconstruction_detail_without_stages(client, db):
    reconstruction = make_reconstruction(db, make_case(db))
    body = client.get(f"/api/reconstructions/{reconstruction.id}").json()
    assert body["stages"] == []
    assert body["annotations"] == []
    response = client.get("/api/reconstructions/999")
    assert response.status_code == 404
    assert response.json() == {"detail": "Reconstruction not found"}


# Stages ---------------------------------------------------------------------


def test_stage_order_defaults_to_after_the_last_stage(client, db):
    case = make_case(db)
    reconstruction = make_reconstruction(db, case)
    url = f"/api/reconstructions/{reconstruction.id}/stages"

    first = client.post(url, json={"label": "Approach"})
    assert first.status_code == 201
    assert first.json()["stage_order"] == 1
    make_stage(db, reconstruction, stage_order=5, label="Late")
    assert client.post(url, json={"label": "Rest positions"}).json()["stage_order"] == 6
    assert client.post(url, json={"label": "Before", "stage_order": 0}).json()["stage_order"] == 0

    # Another scenario's stages don't count.
    other = make_reconstruction(db, case, scenario="Other")
    assert client.post(f"/api/reconstructions/{other.id}/stages", json={"label": "A"}).json()["stage_order"] == 1


def test_create_stage_keeps_events_of_the_case_and_audits(client, db):
    case = make_case(db)
    reconstruction = make_reconstruction(db, case)
    first = make_event(db, case)
    second = make_event(db, case, category="movement")
    response = client.post(
        f"/api/reconstructions/{reconstruction.id}/stages",
        json={"label": "Impact", "event_ids": [second.id, first.id, second.id]},
    )
    assert response.status_code == 201
    stage = response.json()
    assert stage == {
        "id": stage["id"],
        "reconstruction_id": reconstruction.id,
        "stage_order": 1,
        "label": "Impact",
        "event_ids": [second.id, first.id],
    }
    [entry] = audit_rows(db, "stage")
    assert (entry.action, entry.record_id, entry.case_id, entry.detail) == (
        "modified",
        stage["id"],
        case.id,
        {"created": True},
    )


def test_create_stage_rejects_events_of_another_case(client, db):
    case = make_case(db)
    reconstruction = make_reconstruction(db, case)
    own = make_event(db, case)
    foreign = make_event(db, make_case(db, title="Another case"))
    url = f"/api/reconstructions/{reconstruction.id}/stages"

    response = client.post(url, json={"label": "Impact", "event_ids": [own.id, foreign.id]})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "event_ids", 1]
    assert client.post(url, json={"label": "Impact", "event_ids": [999]}).status_code == 422
    assert client.post(url, json={"label": ""}).status_code == 422
    assert db.scalars(select(ReconstructionStage)).all() == []
    assert client.post("/api/reconstructions/999/stages", json={"label": "Impact"}).status_code == 404


# Scene objects --------------------------------------------------------------


def test_put_scene_objects_replaces_the_stage_list(client, db):
    investigator = make_investigator(db)
    case = make_case(db)
    van = make_object(db, case, label="Delivery van", details={"plate": "KA-05-MN-7788"})
    reconstruction = make_reconstruction(db, case)
    stage = make_stage(db, reconstruction)
    sibling = make_stage(db, reconstruction, stage_order=2, label="Impact")
    old = make_scene_object(db, stage)
    kept = make_scene_object(db, sibling)
    link = EvidenceLink(evidence_id=make_evidence(db, case).id, target_type="scene_object", target_id=old.id)
    db.add(link)
    db.commit()
    old_id, kept_id = old.id, kept.id

    response = client.put(
        f"/api/stages/{stage.id}/scene-objects",
        json=[
            {
                "model_id": "van",
                "object_id": van.id,
                "label": "Van",
                "position": [3.5, 0, -2],
                "rotation": [0, 1.57, 0],
                "scale": [1, 1, 1],
            },
            {"model_id": "cone"},
        ],
        headers={"X-Investigator-Id": str(investigator.id)},
    )
    assert response.status_code == 200
    placed = response.json()
    assert [(o["model_id"], o["object_id"], o["stage_id"]) for o in placed] == [
        ("van", van.id, stage.id),
        ("cone", None, stage.id),
    ]
    assert placed[0]["position"] == [3.5, 0.0, -2.0]
    assert placed[0]["rotation"] == [0.0, 1.57, 0.0]
    assert placed[1] | {"id": 0} == {
        "id": 0,
        "stage_id": stage.id,
        "model_id": "cone",
        "object_id": None,
        "label": "",
        "position": [0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
    }

    db.expire_all()
    assert db.get(SceneObject, old_id) is None
    assert db.get(SceneObject, kept_id) is not None
    assert db.scalars(select(EvidenceLink)).all() == []
    detail = client.get(f"/api/reconstructions/{reconstruction.id}").json()
    assert detail["stages"][0]["scene_objects"] == placed

    [entry] = audit_rows(db, "stage")
    assert (entry.action, entry.record_id, entry.case_id, entry.actor_id) == (
        "modified",
        stage.id,
        case.id,
        investigator.id,
    )
    assert entry.detail == {"scene_objects": {"before": 1, "after": 2}}


def test_put_empty_list_clears_the_stage(client, db):
    case = make_case(db)
    stage = make_stage(db, make_reconstruction(db, case))
    make_scene_object(db, stage)
    response = client.put(f"/api/stages/{stage.id}/scene-objects", json=[])
    assert response.status_code == 200
    assert response.json() == []
    assert db.scalars(select(SceneObject)).all() == []


def test_put_scene_objects_rejects_objects_of_another_case(client, db):
    case = make_case(db)
    stage = make_stage(db, make_reconstruction(db, case))
    existing_id = make_scene_object(db, stage).id
    foreign = make_object(db, make_case(db, title="Another case"))
    url = f"/api/stages/{stage.id}/scene-objects"

    response = client.put(url, json=[{"model_id": "van"}, {"model_id": "van", "object_id": foreign.id}])
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", 1, "object_id"]
    assert client.put(url, json=[{"model_id": "van", "object_id": 999}]).status_code == 422
    assert client.put(url, json=[{"model_id": ""}]).status_code == 422
    assert client.put(url, json=[{"model_id": "van", "position": [1, 2]}]).status_code == 422
    assert client.put(url, json={"model_id": "van"}).status_code == 422
    db.expire_all()
    assert [o.id for o in db.scalars(select(SceneObject))] == [existing_id]
    assert audit_rows(db, "stage") == []

    response = client.put("/api/stages/999/scene-objects", json=[])
    assert response.status_code == 404
    assert response.json() == {"detail": "Stage not found"}


# Annotations ----------------------------------------------------------------


def test_create_and_list_annotations(client, db):
    case = make_case(db)
    reconstruction = make_reconstruction(db, case)
    stage = make_stage(db, reconstruction)
    url = f"/api/reconstructions/{reconstruction.id}/annotations"

    measurement = client.post(
        url, json={"stage_id": stage.id, "kind": "measurement", "data": {"from": [0, 0, 0], "to": [4, 0, 0]}}
    )
    assert measurement.status_code == 201
    assert measurement.json() == {
        "id": measurement.json()["id"],
        "reconstruction_id": reconstruction.id,
        "stage_id": stage.id,
        "kind": "measurement",
        "data": {"from": [0, 0, 0], "to": [4, 0, 0]},
    }
    viewpoint = client.post(url, json={"kind": "viewpoint"})
    assert viewpoint.status_code == 201
    assert (viewpoint.json()["stage_id"], viewpoint.json()["data"]) == (None, {})

    other = make_reconstruction(db, case, scenario="Other")
    db.add(SceneAnnotation(reconstruction_id=other.id, kind="note", data={}))
    db.commit()

    listed = client.get(url)
    assert listed.status_code == 200
    assert [a["id"] for a in listed.json()] == [measurement.json()["id"], viewpoint.json()["id"]]

    entries = audit_rows(db, "annotation")
    assert [(e.record_id, e.case_id, e.detail) for e in entries] == [
        (measurement.json()["id"], case.id, {"created": True}),
        (viewpoint.json()["id"], case.id, {"created": True}),
    ]


def test_annotation_stage_must_belong_to_the_reconstruction(client, db):
    case = make_case(db)
    reconstruction = make_reconstruction(db, case)
    other_stage = make_stage(db, make_reconstruction(db, case, scenario="Other"))
    url = f"/api/reconstructions/{reconstruction.id}/annotations"

    response = client.post(url, json={"stage_id": other_stage.id, "kind": "note"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "stage_id"]
    assert client.post(url, json={"stage_id": 999, "kind": "note"}).status_code == 422
    assert client.post(url, json={"kind": "sketch"}).status_code == 422
    assert db.scalars(select(SceneAnnotation)).all() == []

    assert client.get("/api/reconstructions/999/annotations").status_code == 404
    assert client.post("/api/reconstructions/999/annotations", json={"kind": "note"}).status_code == 404


# Buildings ------------------------------------------------------------------


def test_get_building(client, db):
    building = Building(name="Ring Road Plaza", floors=[{"level": 0, "rooms": []}], description="Corner block")
    db.add(building)
    db.commit()
    response = client.get(f"/api/buildings/{building.id}")
    assert response.status_code == 200
    assert response.json() == {
        "id": building.id,
        "geo_building_id": None,
        "name": "Ring Road Plaza",
        "model_url": None,
        "floors": [{"level": 0, "rooms": []}],
        "description": "Corner block",
    }
    response = client.get("/api/buildings/999")
    assert response.status_code == 404
    assert response.json() == {"detail": "Building not found"}
