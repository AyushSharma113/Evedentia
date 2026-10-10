"""Checks that a request only points at records of the right case or scenario.

They fail with FastAPI's 422 body (`{"detail": [{"loc", "msg", "type"}]}`), which
the frontend's api client already formats, so a bad reference reads like any other
invalid field.
"""

from collections.abc import Iterable, Sequence

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Base


def invalid(loc: Sequence[str | int], msg: str) -> HTTPException:
    """A 422 for one field. `loc` is the path inside the body, such as ("event_ids", 2)."""
    return HTTPException(
        status_code=422,
        detail=[{"loc": ["body", *loc], "msg": msg, "type": "value_error"}],
    )


def ids_in_case(db: Session, model: type[Base], ids: Iterable[int], case_id: int) -> set[int]:
    """The subset of `ids` that are rows of `model` in the case."""
    wanted = set(ids)
    if not wanted:
        return set()
    return set(db.scalars(select(model.id).where(model.id.in_(wanted), model.case_id == case_id)))
