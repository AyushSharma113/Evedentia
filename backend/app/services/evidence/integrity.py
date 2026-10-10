"""File integrity check for POST /cases/{id}/verify."""

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.storage import sha256_of
from app.models import Evidence


@dataclass
class Mismatch:
    evidence_id: int
    title: str
    expected: str
    # None when the stored file is missing.
    actual: str | None


@dataclass
class IntegrityReport:
    checked: int = 0
    mismatches: list[Mismatch] = field(default_factory=list)


def verify_case_files(db: Session, case_id: int) -> IntegrityReport:
    """Recomputes the SHA-256 of every stored file in the case and compares it with
    the hash taken at upload. A file that has gone missing is a mismatch with
    `actual` None. Evidence rows without a stored file are skipped."""
    report = IntegrityReport()
    rows = db.scalars(
        select(Evidence).where(Evidence.case_id == case_id, Evidence.file_path.is_not(None)).order_by(Evidence.id)
    )
    for evidence in rows:
        report.checked += 1
        actual = sha256_of(evidence.file_path)
        if actual != evidence.sha256:
            report.mismatches.append(Mismatch(evidence.id, evidence.title, evidence.sha256, actual))
    return report
