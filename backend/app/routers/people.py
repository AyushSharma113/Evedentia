"""People routes (task B1 in docs/BACKEND_TASKS.md):

- GET, POST /api/cases/{id}/persons | objects | locations
- PATCH, DELETE /api/persons/{id} | objects/{id} | locations/{id}

The three record types share one set of routes, as in the mock's resource() helper.
"""

from typing import Any

from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.deps import ActorId, DbSession
from app.core.records import (
    RECORD_NAMES,
    apply_update,
    delete_links_to,
    delete_relationships_of,
    get_or_404,
    record_audit,
)
from app.models import Base, Building, Case, CaseObject, Location, Person
from app.routers.cases import body_error, reject_nulls, unprocessable
from app.schemas.people import CaseObject as CaseObjectOut
from app.schemas.people import (
    CaseObjectCreate,
    CaseObjectUpdate,
    LocationCreate,
    LocationUpdate,
    PersonCreate,
    PersonUpdate,
)
from app.schemas.people import Location as LocationOut
from app.schemas.people import Person as PersonOut

router = APIRouter(tags=["people"])


def check_building(db: Session, values: dict[str, Any]) -> None:
    building_id = values.get("building_id")
    if building_id is not None and db.get(Building, building_id) is None:
        raise unprocessable([body_error("building_id", f"Building {building_id} not found", building_id)])


def resource(
    collection: str,
    record_type: str,
    model: type[Base],
    out: type[BaseModel],
    create_schema: type[BaseModel],
    update_schema: type[BaseModel],
) -> None:
    """Adds GET and POST /cases/{id}/<collection>, and PATCH and DELETE /<collection>/{id}."""
    name = RECORD_NAMES[record_type]
    validate = check_building if model is Location else lambda db, values: None

    def list_records(case_id: int, db: DbSession):
        case = get_or_404(db, Case, case_id)
        return db.scalars(select(model).where(model.case_id == case.id).order_by(model.id)).all()

    def create_record(case_id: int, body: create_schema, db: DbSession, actor_id: ActorId):
        case = get_or_404(db, Case, case_id)
        values = body.model_dump()
        validate(db, values)
        row = model(case_id=case.id, **values)
        db.add(row)
        db.flush()
        record_audit(db, actor_id, "modified", record_type, row.id, case_id=case.id, detail={"created": True})
        db.commit()
        return row

    def update_record(record_id: int, body: update_schema, db: DbSession, actor_id: ActorId):
        row = get_or_404(db, model, record_id, name)
        changes = body.model_dump(exclude_unset=True)
        reject_nulls(model, changes)
        validate(db, changes)
        apply_update(db, row, changes, record_type=record_type, actor_id=actor_id)
        db.commit()
        return row

    def delete_record(record_id: int, db: DbSession, actor_id: ActorId):
        row = get_or_404(db, model, record_id, name)
        case_id = row.case_id
        delete_relationships_of(db, record_type, record_id)
        delete_links_to(db, record_type, record_id)
        db.delete(row)
        record_audit(db, actor_id, "modified", record_type, record_id, case_id=case_id, detail={"deleted": True})
        db.commit()
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    router.add_api_route(
        f"/cases/{{case_id}}/{collection}",
        list_records,
        methods=["GET"],
        response_model=list[out],
        name=f"list_{collection}",
    )
    router.add_api_route(
        f"/cases/{{case_id}}/{collection}",
        create_record,
        methods=["POST"],
        response_model=out,
        status_code=status.HTTP_201_CREATED,
        name=f"create_{record_type}",
    )
    router.add_api_route(
        f"/{collection}/{{record_id}}",
        update_record,
        methods=["PATCH"],
        response_model=out,
        name=f"update_{record_type}",
    )
    router.add_api_route(
        f"/{collection}/{{record_id}}",
        delete_record,
        methods=["DELETE"],
        status_code=status.HTTP_204_NO_CONTENT,
        response_class=Response,
        name=f"delete_{record_type}",
    )


resource("persons", "person", Person, PersonOut, PersonCreate, PersonUpdate)
resource("objects", "object", CaseObject, CaseObjectOut, CaseObjectCreate, CaseObjectUpdate)
resource("locations", "location", Location, LocationOut, LocationCreate, LocationUpdate)
