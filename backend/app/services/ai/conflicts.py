"""Conflict detection (step 6 in docs/PLAN.md), run at the end of every
process_evidence:

    detect_conflicts(db, case_id)

The case's observations are grouped by subject, place and time, and one Gemini
call with a JSON response schema says which statements in a group conflict and
what information is missing. Each finding becomes an open `conflicts` row listing
its observation IDs, which shows in the review queue and the alerts feed.

- Rejected observations are left out.
- Only groups holding an observation that no earlier run was asked about are
  sent, so evidence that adds no statements (or a reprocess that finds the same
  ones) makes no call.
- A group, or a finding, whose observations are exactly those of an existing
  conflict (open or resolved) is skipped.
- The call is logged to ai_runs with the case ID, the observations it covered and
  the model's output. Without GEMINI_API_KEY nothing happens, and a failed call is
  logged and skipped, so it never fails the evidence.
"""

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Literal

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    AiRun,
    Case,
    CaseObject,
    Conflict,
    Event,
    Evidence,
    EvidenceLink,
    Location,
    Observation,
    Person,
)
from app.services.ai.calls import decode_json, generate_json, log_run
from app.services.ai.client import get_client

logger = logging.getLogger(__name__)

TASK = "conflicts"
# Timed observations within this span of the first one in a group are about the same moment.
TIME_WINDOW = timedelta(minutes=10)
GAP_PREFIX = "Missing information: "


class Finding(BaseModel):
    kind: Literal["conflict", "gap"] = Field(
        description="conflict when the statements cannot all be true; gap when they leave out something the "
        "investigation needs."
    )
    observation_ids: list[int] = Field(description="IDs of every observation involved, from the list given.")
    description: str = Field(
        description="One sentence for the investigator, such as \"Witnesses disagree on the hatchback's signal "
        'when it entered the junction." or "No account says how fast the van was going."'
    )


class Findings(BaseModel):
    findings: list[Finding]


SYSTEM_INSTRUCTION = """You compare statements in a forensic investigation, for an investigator to review.
Statements are grouped because they are about the same subject, place or time.
Report a conflict when statements in a group cannot all be true, such as two accounts giving different signal colours for the same vehicle at the same moment.
Report a gap when a group leaves out or leaves unclear something the investigation needs to settle.
Don't report statements that agree, that differ only in wording or detail, or that are about different moments.
Report each problem once, listing every observation ID it involves, and use only the IDs given.
Return an empty list when there is nothing to report."""


@dataclass
class Group:
    labels: list[str]
    observation_ids: list[int]


@dataclass
class CaseRecords:
    """Names for the subjects and sources of a case's observations, for the prompt."""

    persons: dict[int, Person] = field(default_factory=dict)
    objects: dict[int, CaseObject] = field(default_factory=dict)
    locations: dict[int, Location] = field(default_factory=dict)
    events: dict[int, Event] = field(default_factory=dict)
    evidence_titles: dict[int, str] = field(default_factory=dict)
    # Observation ID to the titles of the evidence it came from.
    sources: dict[int, list[str]] = field(default_factory=dict)

    @classmethod
    def load(cls, db: Session, case_id: int, observation_ids: list[int]) -> "CaseRecords":
        def rows(model) -> dict:
            return {row.id: row for row in db.scalars(select(model).where(model.case_id == case_id))}

        titles = dict(db.execute(select(Evidence.id, Evidence.title).where(Evidence.case_id == case_id)).all())
        records = cls(rows(Person), rows(CaseObject), rows(Location), rows(Event), titles)
        links = db.execute(
            select(EvidenceLink.target_id, Evidence.title)
            .join(Evidence, Evidence.id == EvidenceLink.evidence_id)
            .where(
                EvidenceLink.target_type == "observation",
                EvidenceLink.target_id.in_(observation_ids),
            )
            .order_by(EvidenceLink.id)
        )
        for observation_id, title in links:
            records.sources.setdefault(observation_id, []).append(title)
        return records

    def subject(self, subject_type: str, subject_id: int) -> str:
        if subject_type == "person" and subject_id in self.persons:
            person = self.persons[subject_id]
            return f"{person.name} ({person.role})"
        if subject_type == "object" and subject_id in self.objects:
            return self.objects[subject_id].label
        if subject_type == "location" and subject_id in self.locations:
            return self.locations[subject_id].address
        if subject_type == "event" and subject_id in self.events:
            return f"event: {self.events[subject_id].description}"
        if subject_type == "evidence" and subject_id in self.evidence_titles:
            return f'evidence "{self.evidence_titles[subject_id]}"'
        if subject_type == "case":
            return "the case"
        return f"{subject_type} {subject_id}"

    def place(self, observation: Observation) -> int | None:
        """The location an observation is about: its subject, or its event's location."""
        if observation.subject_type == "location":
            return observation.subject_id
        if observation.subject_type == "event" and observation.subject_id in self.events:
            return self.events[observation.subject_id].location_id
        return None


def group_observations(observations: list[Observation], records: CaseRecords) -> list[Group]:
    """Groups of two or more observations about the same subject, place or time.
    Groups with the same observations are merged, keeping every label."""
    candidates: list[tuple[str, list[Observation]]] = []

    by_subject: dict[tuple[str, int], list[Observation]] = {}
    by_place: dict[int, list[Observation]] = {}
    for observation in observations:
        by_subject.setdefault((observation.subject_type, observation.subject_id), []).append(observation)
        location_id = records.place(observation)
        if location_id is not None:
            by_place.setdefault(location_id, []).append(observation)
    for (subject_type, subject_id), members in by_subject.items():
        candidates.append((f"Subject: {records.subject(subject_type, subject_id)}", members))
    for location_id, members in by_place.items():
        candidates.append((f"Place: {records.subject('location', location_id)}", members))

    timed = sorted((o for o in observations if o.observed_at), key=lambda o: (o.observed_at, o.id))
    window: list[Observation] = []
    for observation in timed:
        if window and observation.observed_at - window[0].observed_at > TIME_WINDOW:
            candidates.append((_time_label(window), window))
            window = []
        window.append(observation)
    if window:
        candidates.append((_time_label(window), window))

    groups: dict[frozenset[int], Group] = {}
    for label, members in candidates:
        ids = sorted({o.id for o in members})
        if len(ids) < 2:
            continue
        group = groups.setdefault(frozenset(ids), Group([], ids))
        group.labels.append(label)
    return list(groups.values())


def _time_label(window: list[Observation]) -> str:
    start, end = (moment.strftime("%Y-%m-%d %H:%M UTC") for moment in (window[0].observed_at, window[-1].observed_at))
    return f"Time: {start}" if start == end else f"Time: {start} to {end}"


def build_prompt(
    case: Case,
    observations: list[Observation],
    groups: list[Group],
    records: CaseRecords,
) -> str:
    lines = [f"Case: {case.title} ({case.incident_type.replace('_', ' ')})."]
    if case.description:
        lines.append(f"Case description: {case.description}")
    lines += ["", "Observations, by ID:"]
    wanted = {i for group in groups for i in group.observation_ids}
    for observation in observations:
        if observation.id not in wanted:
            continue
        about = [f"about {records.subject(observation.subject_type, observation.subject_id)}"]
        if observation.observed_at:
            about.append(f"at {observation.observed_at.isoformat()}")
        sources = records.sources.get(observation.id)
        if sources:
            about.append("from " + ", ".join(f'"{title}"' for title in sources))
        lines.append(f"[{observation.id}] ({'; '.join(about)}) {observation.statement}")
    lines += ["", "Groups to compare:"]
    for group in groups:
        lines.append(f"- {'; '.join(group.labels)}: {', '.join(map(str, group.observation_ids))}")
    return "\n".join(lines)


def detect_conflicts(db: Session, case_id: int) -> list[Conflict]:
    """Saves and commits one open conflict per new finding, with the ai_runs row,
    and returns the new conflicts."""
    client = get_client()
    case = db.get(Case, case_id)
    if client is None or case is None:
        return []

    observations = list(
        db.scalars(
            select(Observation)
            .where(Observation.case_id == case_id, Observation.review_status != "rejected")
            .order_by(Observation.id)
        )
    )
    known = {
        frozenset(ids or []) for ids in db.scalars(select(Conflict.observation_ids).where(Conflict.case_id == case_id))
    }
    seen = _examined_ids(db, case_id)
    records = CaseRecords.load(db, case_id, [o.id for o in observations])
    groups = [
        group
        for group in group_observations(observations, records)
        if frozenset(group.observation_ids) not in known and not set(group.observation_ids) <= seen
    ]
    if not groups:
        return []

    prompt = build_prompt(case, observations, groups, records)
    try:
        raw = generate_json(client, [prompt], Findings, system_instruction=SYSTEM_INSTRUCTION)
    except Exception:  # Conflict detection never fails the evidence that triggered it.
        logger.warning("Conflict detection failed for case %s", case_id, exc_info=True)
        return []
    decoded = decode_json(raw)
    log_run(
        db,
        TASK,
        {
            "observation_ids": [o.id for o in observations],
            "groups": [{"labels": g.labels, "observation_ids": g.observation_ids} for g in groups],
            "response": raw if decoded is None else decoded,
        },
        case_id=case_id,
    )

    sent = {i for group in groups for i in group.observation_ids}
    created: list[Conflict] = []
    for finding in _findings(decoded):
        ids = sorted({i for i in finding.observation_ids if i in sent})
        description = finding.description.strip()
        if len(ids) < (2 if finding.kind == "conflict" else 1) or not description or frozenset(ids) in known:
            continue
        if finding.kind == "gap" and not description.lower().startswith("missing"):
            description = GAP_PREFIX + description
        conflict = Conflict(case_id=case_id, observation_ids=ids, description=description, status="open")
        db.add(conflict)
        known.add(frozenset(ids))
        created.append(conflict)
    db.commit()
    return created


def _examined_ids(db: Session, case_id: int) -> set[int]:
    """Observations an earlier run sent to Gemini and got a readable answer about."""
    seen: set[int] = set()
    for output in db.scalars(select(AiRun.output).where(AiRun.case_id == case_id, AiRun.task == TASK)):
        if isinstance(output, dict) and isinstance(output.get("response"), dict):
            seen.update(i for i in output.get("observation_ids") or [] if isinstance(i, int))
    return seen


def _findings(decoded) -> list[Finding]:
    """The findings that fit the schema; the rest are dropped."""
    items = decoded.get("findings") if isinstance(decoded, dict) else None
    findings = []
    for item in items if isinstance(items, list) else []:
        try:
            findings.append(Finding.model_validate(item))
        except ValidationError:
            continue
    return findings
