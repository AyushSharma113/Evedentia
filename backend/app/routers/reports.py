"""Reports routes (task B5 in docs/BACKEND_TASKS.md):

- POST /api/cases/{id}/reports
- GET /api/cases/{id}/reports
- GET /api/reports/{id}
- GET /api/cases/{id}/audit
"""

from fastapi import APIRouter, status
from sqlalchemy import select

from app.core.deps import ActorId, DbSession
from app.core.records import get_or_404, record_audit
from app.models import AuditLog, Case, Report
from app.schemas.evidence import AuditEntry
from app.schemas.reports import Report as ReportOut, ReportCreate, ReportListItem
from app.services.reports.content import build_content, content_sha256

router = APIRouter(tags=["reports"])


@router.post("/cases/{case_id}/reports", response_model=ReportOut, status_code=status.HTTP_201_CREATED)
def create_report(case_id: int, body: ReportCreate, db: DbSession, actor_id: ActorId):
    case = get_or_404(db, Case, case_id)
    content = build_content(db, case, body.template, body.options).model_dump(mode="json")
    report = Report(
        case_id=case.id,
        template=body.template,
        options=body.options.model_dump(),
        content=content,
        sha256=content_sha256(content),
        created_by=actor_id,
    )
    db.add(report)
    db.flush()
    record_audit(db, actor_id, "exported", "report", report.id, case_id=case.id, detail={"template": body.template})
    db.commit()
    return report


@router.get("/cases/{case_id}/reports", response_model=list[ReportListItem])
def list_reports(case_id: int, db: DbSession):
    case = get_or_404(db, Case, case_id)
    return db.scalars(
        select(Report).where(Report.case_id == case.id).order_by(Report.created_at.desc(), Report.id.desc())
    ).all()


@router.get("/reports/{report_id}", response_model=ReportOut)
def get_report(report_id: int, db: DbSession):
    return get_or_404(db, Report, report_id)


@router.get("/cases/{case_id}/audit", response_model=list[AuditEntry])
def case_audit(case_id: int, db: DbSession):
    case = get_or_404(db, Case, case_id)
    return db.scalars(
        select(AuditLog).where(AuditLog.case_id == case.id).order_by(AuditLog.at.desc(), AuditLog.id.desc())
    ).all()
