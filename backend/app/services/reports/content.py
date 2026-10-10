"""Report assembly, ported from buildReport in frontend/src/mocks/handlers.ts.

Each template picks the sections it fills; the others come back empty, and
ai_insights and custody are null unless their option (or the custody template)
asks for them:

    section            court  summary  custody  medical
    timeline             x       x
    open_conflicts       x       x
    uncertainties        x       x
    reconstructions      x
    injuries             x                         x

The medical template also keeps only medical evidence, and only confirmed
findings about a person.
"""

import hashlib
import json
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.records import case_counts
from app.models import (
    AuditLog,
    Case,
    Conflict,
    Event,
    Evidence,
    EvidenceLink,
    Injury,
    Investigator,
    Observation,
    Person,
    Reconstruction,
)
from app.schemas.cases import Case as CaseOut
from app.schemas.reports import ReportContent, ReportOptions


def content_sha256(content: dict[str, Any]) -> str:
    """SHA-256 of the content as canonical JSON, so the same content always hashes the same."""
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _timeline_key(event: Event) -> tuple[bool, datetime | None, int]:
    # As the mock's sort: untimed events first, then by start time, ties by id.
    return (event.time_start is not None, event.time_start, event.id)


def build_content(db: Session, case: Case, template: str, options: ReportOptions) -> ReportContent:
    lead = db.get(Investigator, case.lead_investigator_id) if case.lead_investigator_id else None
    evidence = db.scalars(select(Evidence).where(Evidence.case_id == case.id).order_by(Evidence.id)).all()
    persons = db.scalars(
        select(Person).where(Person.case_id == case.id, Person.review_status != "rejected").order_by(Person.id)
    ).all()
    events = sorted(
        db.scalars(select(Event).where(Event.case_id == case.id, Event.review_status != "rejected")).all(),
        key=_timeline_key,
    )
    observations = db.scalars(select(Observation).where(Observation.case_id == case.id).order_by(Observation.id)).all()
    statements = {o.id: o.statement for o in observations}

    def linked_evidence(record_type: str, record_id: int) -> list[int]:
        ids = db.scalars(
            select(EvidenceLink.evidence_id)
            .where(EvidenceLink.target_type == record_type, EvidenceLink.target_id == record_id)
            .order_by(EvidenceLink.id)
        ).all()
        return list(dict.fromkeys(ids))

    medical_only = template == "medical"
    narrative = template in ("court", "summary")

    uncertainties: list[str] = []
    open_conflicts: list[dict[str, Any]] = []
    if narrative:
        uncertainties = [f"{e.description} ({e.certainty})" for e in events if e.certainty != "established"]
        pending = case_counts(db, case.id)["pending_reviews"]
        if pending:
            uncertainties.append(f"{pending} AI-proposed items are still waiting for review.")
        conflicts = db.scalars(
            select(Conflict).where(Conflict.case_id == case.id, Conflict.status == "open").order_by(Conflict.id)
        ).all()
        open_conflicts = [
            {"description": c.description, "statements": [statements.get(i, "") for i in c.observation_ids]}
            for c in conflicts
        ]

    reconstructions = []
    if template == "court":
        reconstructions = db.scalars(
            select(Reconstruction).where(Reconstruction.case_id == case.id).order_by(Reconstruction.id)
        ).all()

    injuries = []
    if template == "court" or medical_only:
        injuries = db.scalars(
            select(Injury).where(Injury.case_id == case.id, Injury.review_status != "rejected").order_by(Injury.id)
        ).all()

    ai_insights = None
    if options.include_ai_insights and template != "custody":
        ai_insights = [
            f"{e.title}: {e.summary}" for e in evidence if e.summary and (not medical_only or e.type == "medical")
        ]

    custody = None
    if options.include_custody or template == "custody":
        custody = db.scalars(
            select(AuditLog)
            .where(AuditLog.record_type == "evidence", AuditLog.record_id.in_([e.id for e in evidence]))
            .order_by(AuditLog.at, AuditLog.id)
        ).all()

    return ReportContent.model_validate(
        {
            "case": {**CaseOut.model_validate(case).model_dump(), "lead_investigator_name": lead.name if lead else None},
            "summary": case.description,
            "evidence": [e for e in evidence if not medical_only or e.type == "medical"],
            "timeline": events if narrative else [],
            "persons": persons,
            "confirmed_findings": [
                {"statement": o.statement, "evidence_ids": linked_evidence("observation", o.id)}
                for o in observations
                if o.review_status == "confirmed" and (not medical_only or o.subject_type == "person")
            ],
            "open_conflicts": open_conflicts,
            "uncertainties": uncertainties,
            "reconstructions": reconstructions,
            "injuries": injuries,
            "ai_insights": ai_insights,
            "custody": custody,
        }
    )
