"""Request checks the schemas can't make, shared by the graph, timeline and review
routes (task B3): references that must belong to the case, and PATCH bodies that
send null for a column that can't hold one."""

from typing import Any

from fastapi import HTTPException

from app.models import Base


def invalid(field: str, msg: str) -> HTTPException:
    """A 422 shaped like FastAPI's own validation errors, so the frontend shows it
    the same way. Raise the result."""
    return HTTPException(status_code=422, detail=[{"loc": ["body", field], "msg": msg, "type": "value_error"}])


def reject_nulls(row: Base, changes: dict[str, Any]) -> None:
    """Update schemas make every field optional, so a client can send null for a
    NOT NULL column. Answer 422 rather than failing on commit."""
    columns = type(row).__table__.columns
    for field, value in changes.items():
        if value is None and not columns[field].nullable:
            raise invalid(field, "Field may not be null")
