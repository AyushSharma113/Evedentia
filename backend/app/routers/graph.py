"""Graph routes (task B3 in docs/BACKEND_TASKS.md):

- GET /api/cases/{id}/graph
- POST /api/cases/{id}/relationships
- PATCH, DELETE /api/relationships/{id}
"""

from fastapi import APIRouter, status
from sqlalchemy import select

from app.core.deps import ActorId, DbSession
from app.core.records import RECORD_MODELS, RECORD_NAMES, apply_update, delete_links_to, get_or_404, record_audit
from app.models import Case, CaseObject, Evidence, Location, Person, Relationship
from app.schemas.graph import CaseGraph, GraphEdge, GraphNode, RelationshipCreate, RelationshipUpdate
from app.schemas.graph import Relationship as RelationshipOut
from app.services.timeline.validation import invalid, reject_nulls

router = APIRouter(tags=["graph"])


def humanize(key: str | None) -> str:
    """Turns a key such as "involved_in" into "Involved in", as the frontend's humanize() does."""
    if not key:
        return ""
    text = key.replace("_", " ")
    return text[:1].upper() + text[1:]


def _node(type_: str, record_id: int, label: str, sublabel: str | None, review_status: str) -> GraphNode:
    return GraphNode(
        id=f"{type_}:{record_id}",
        type=type_,
        record_id=record_id,
        label=label,
        sublabel=sublabel,
        review_status=review_status,
    )


@router.get("/cases/{case_id}/graph", response_model=CaseGraph)
def get_graph(case_id: int, db: DbSession):
    """Every person and object is a node; locations and evidence only when a
    relationship touches them. Edges only where both ends are nodes."""
    case = get_or_404(db, Case, case_id)
    relationships = db.scalars(
        select(Relationship).where(Relationship.case_id == case.id).order_by(Relationship.id)
    ).all()
    referenced = {(r.from_type, r.from_id) for r in relationships} | {(r.to_type, r.to_id) for r in relationships}
    referenced_ids = {
        node_type: [record_id for type_, record_id in referenced if type_ == node_type]
        for node_type in ("location", "evidence")
    }

    nodes: list[GraphNode] = []
    for person in db.scalars(select(Person).where(Person.case_id == case.id).order_by(Person.id)):
        sublabel = ", ".join(part for part in (humanize(person.role), person.condition) if part)
        nodes.append(_node("person", person.id, person.name, sublabel, person.review_status))
    for obj in db.scalars(select(CaseObject).where(CaseObject.case_id == case.id).order_by(CaseObject.id)):
        plate = (obj.details or {}).get("plate")
        nodes.append(_node("object", obj.id, obj.label, str(plate) if plate else humanize(obj.kind), obj.review_status))
    for location in db.scalars(
        select(Location)
        .where(Location.case_id == case.id, Location.id.in_(referenced_ids["location"]))
        .order_by(Location.id)
    ):
        nodes.append(_node("location", location.id, location.address, humanize(location.kind), location.review_status))
    for evidence in db.scalars(
        select(Evidence)
        .where(Evidence.case_id == case.id, Evidence.id.in_(referenced_ids["evidence"]))
        .order_by(Evidence.id)
    ):
        nodes.append(_node("evidence", evidence.id, evidence.title, humanize(evidence.type), "confirmed"))

    node_ids = {node.id for node in nodes}
    edges = [
        GraphEdge(
            id=f"relationship:{r.id}",
            relationship_id=r.id,
            source=f"{r.from_type}:{r.from_id}",
            target=f"{r.to_type}:{r.to_id}",
            kind=r.kind,
            label=r.label,
            confidence=r.confidence,
            flagged=r.flagged,
            review_status=r.review_status,
        )
        for r in relationships
        if f"{r.from_type}:{r.from_id}" in node_ids and f"{r.to_type}:{r.to_id}" in node_ids
    ]
    return CaseGraph(nodes=nodes, edges=edges)


def _require_in_case(db: DbSession, case_id: int, end: str, node_type: str, node_id: int) -> None:
    row = db.get(RECORD_MODELS[node_type], node_id)
    if row is None or row.case_id != case_id:
        raise invalid(f"{end}_id", f"{RECORD_NAMES[node_type]} {node_id} is not in this case")


@router.post("/cases/{case_id}/relationships", response_model=RelationshipOut, status_code=status.HTTP_201_CREATED)
def create_relationship(case_id: int, body: RelationshipCreate, db: DbSession, actor_id: ActorId):
    case = get_or_404(db, Case, case_id)
    _require_in_case(db, case.id, "from", body.from_type, body.from_id)
    _require_in_case(db, case.id, "to", body.to_type, body.to_id)
    values = body.model_dump()
    if not values["label"].strip():
        values["label"] = humanize(body.kind)
    relationship = Relationship(case_id=case.id, **values)
    db.add(relationship)
    db.flush()
    record_audit(db, actor_id, "modified", "relationship", relationship.id, case_id=case.id, detail={"created": True})
    db.commit()
    return relationship


@router.patch("/relationships/{relationship_id}", response_model=RelationshipOut)
def update_relationship(relationship_id: int, body: RelationshipUpdate, db: DbSession, actor_id: ActorId):
    relationship = get_or_404(db, Relationship, relationship_id)
    changes = body.model_dump(exclude_unset=True)
    reject_nulls(relationship, changes)
    if "label" in changes and not changes["label"].strip():
        changes["label"] = humanize(changes.get("kind", relationship.kind))
    apply_update(db, relationship, changes, record_type="relationship", actor_id=actor_id)
    db.commit()
    return relationship


@router.delete("/relationships/{relationship_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_relationship(relationship_id: int, db: DbSession, actor_id: ActorId) -> None:
    relationship = get_or_404(db, Relationship, relationship_id)
    delete_links_to(db, "relationship", relationship.id)
    record_audit(
        db, actor_id, "modified", "relationship", relationship.id, case_id=relationship.case_id, detail={"deleted": True}
    )
    db.delete(relationship)
    db.commit()
