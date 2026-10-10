"""Dashboard routes (task B5 in docs/BACKEND_TASKS.md):

- GET /api/dashboard
- GET /api/alerts
"""

from typing import get_args

from fastapi import APIRouter
from pydantic_core import to_jsonable_python
from sqlalchemy import func, select

from app.core.deps import DbSession
from app.core.records import case_counts
from app.models import Case, Conflict, Evidence, Investigator, Location
from app.schemas.cases import Investigator as InvestigatorOut
from app.schemas.common import CaseStatus
from app.schemas.dashboard import Alert, Dashboard

router = APIRouter(tags=["dashboard"])


@router.get("/dashboard", response_model=Dashboard)
def get_dashboard(db: DbSession):
    cases = db.scalars(select(Case).order_by(Case.id)).all()
    status_counts = dict.fromkeys(get_args(CaseStatus), 0)
    for case in cases:
        status_counts[case.status] += 1

    sites = db.execute(
        select(Location, Case)
        .join(Case, Location.case_id == Case.id)
        .where(Location.kind == "incident_site", Location.lat.is_not(None), Location.lng.is_not(None))
        .order_by(Location.id)
    ).all()

    investigators = db.scalars(select(Investigator).order_by(Investigator.id)).all()

    return {
        "total_cases": len(cases),
        "active_cases": sum(1 for c in cases if c.status != "closed"),
        "pending_reviews": sum(case_counts(db, c.id)["pending_reviews"] for c in cases),
        "storage_bytes": db.scalar(select(func.coalesce(func.sum(Evidence.size_bytes), 0))),
        "status_counts": status_counts,
        "incident_points": [
            {
                "case_id": case.id,
                "case_number": case.case_number,
                "title": case.title,
                "incident_type": case.incident_type,
                "priority": case.priority,
                "address": location.address,
                "lat": location.lat,
                "lng": location.lng,
                "incident_at": case.incident_at,
            }
            for location, case in sites
        ],
        "investigators": [
            {
                **InvestigatorOut.model_validate(i).model_dump(),
                "solved_cases": sum(1 for c in cases if c.lead_investigator_id == i.id and c.status == "closed"),
                "active_cases": sum(1 for c in cases if i.id in c.investigator_ids and c.status != "closed"),
            }
            for i in investigators
        ],
    }


@router.get("/alerts", response_model=list[Alert])
def list_alerts(db: DbSession):
    alerts: list[dict] = []
    cases = db.scalars(select(Case).order_by(Case.id)).all()
    for case in cases:
        conflicts = db.scalars(
            select(Conflict).where(Conflict.case_id == case.id, Conflict.status == "open").order_by(Conflict.id)
        ).all()
        for conflict in conflicts:
            alerts.append(
                {
                    "id": f"conflict-{conflict.id}",
                    "kind": "conflict",
                    "severity": "warning",
                    "title": "Conflicting statements",
                    "message": conflict.description,
                    "case_id": case.id,
                    "case_number": case.case_number,
                    "record_type": "conflict",
                    "record_id": conflict.id,
                    "created_at": case.updated_at,
                }
            )
        pending = case_counts(db, case.id)["pending_reviews"]
        if pending:
            alerts.append(
                {
                    "id": f"review-{case.id}",
                    "kind": "review",
                    "severity": "info",
                    "title": f"{pending} AI proposals to review",
                    "message": (
                        "New people, events, links or statements were proposed from the evidence "
                        f"in {case.case_number}."
                    ),
                    "case_id": case.id,
                    "case_number": case.case_number,
                    "record_type": None,
                    "record_id": None,
                    "created_at": case.updated_at,
                }
            )

    # Evidence whose last pipeline run finished. A reprocess in flight still has
    # the old processed_at, so only done and failed rows count.
    processed = db.execute(
        select(Evidence, Case)
        .join(Case, Evidence.case_id == Case.id)
        .where(Evidence.processed_at.is_not(None), Evidence.processing_status.in_(("done", "failed")))
        .order_by(Evidence.id)
    ).all()
    for evidence, case in processed:
        failed = evidence.processing_status == "failed"
        alerts.append(
            {
                # The time as the API emits it, so the id changes with every run.
                "id": f"processing-{evidence.id}-{to_jsonable_python(evidence.processed_at)}",
                "kind": "processing",
                "severity": "critical" if failed else "info",
                "title": "Processing failed" if failed else "Processing finished",
                "message": evidence.title,
                "case_id": case.id,
                "case_number": case.case_number,
                "record_type": "evidence",
                "record_id": evidence.id,
                "created_at": evidence.processed_at,
            }
        )

    # Newest first; the sort is stable, so ties keep the order above, as in the mock.
    return sorted(alerts, key=lambda a: a["created_at"], reverse=True)
