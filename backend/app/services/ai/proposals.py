"""Saving the structured output (structure.py) as reviewable case records.

Each item becomes an origin="ai", review_status="pending" row with an
evidence_links row holding its passage and the chunk that contains it. Persons,
vehicles and phones, and locations already in the case are matched by name,
plate, number or address and linked instead of duplicated, and so are identical
events, relationships and observations.

Reprocessing first clears this evidence's earlier proposals that are still
pending and that nothing else uses, so a re-run replaces them.
"""

import re
from datetime import UTC, datetime

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from app.core.records import REVIEWABLE_MODELS, delete_links_to
from app.models import (
    CaseObject,
    Conflict,
    Event,
    Evidence,
    EvidenceChunk,
    EvidenceLink,
    Injury,
    Location,
    Observation,
    Person,
    Reconstruction,
    ReconstructionStage,
    Relationship,
    SceneObject,
)
from app.services.ai.structure import EVIDENCE_REF, StructuredEvidence

AI_ROW = {"origin": "ai", "review_status": "pending"}
GRAPH_NODE_TYPES = {"person", "object", "location", "evidence"}
# Words dropped when comparing names, so "Mr. Rohan Kapoor" matches "Rohan Kapoor".
TITLES = {"mr", "mrs", "ms", "miss", "dr", "shri", "sri", "smt", "kumari", "late"}
# A passage that isn't found verbatim links to the chunk sharing most of its words, from this share up.
MIN_WORD_SHARE = 0.6


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def name_key(name: str) -> str:
    return " ".join(word for word in _words(name) if word not in TITLES)


def plate_key(plate: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(plate or "").upper())


def number_key(number: object) -> str:
    digits = re.sub(r"\D", "", str(number or ""))
    return digits[-10:] if len(digits) >= 6 else ""


def text_key(text: str) -> str:
    return " ".join(_words(text))


def parse_time(value: str | None) -> datetime | None:
    """ISO 8601 from the model; a time without an offset is taken as UTC."""
    if not value or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def humanize(key: str) -> str:
    text = key.replace("_", " ")
    return text[:1].upper() + text[1:]


class PassageIndex:
    """Finds the chunk that holds a passage: the one containing it (ignoring case,
    punctuation and spacing), else the one sharing most of its words."""

    def __init__(self, chunks: list[EvidenceChunk]):
        self.chunks = [(chunk.id, f" {text_key(chunk.text)} ", set(_words(chunk.text))) for chunk in chunks]

    def chunk_for(self, passage: str | None) -> int | None:
        words = _words(passage or "")
        if not words:
            return None
        needle = f" {' '.join(words)} "
        wanted = set(words)
        best_id, best_share = None, MIN_WORD_SHARE
        for chunk_id, text, chunk_words in self.chunks:
            if needle in text:
                return chunk_id
            share = len(wanted & chunk_words) / len(wanted)
            if share >= best_share and (best_id is None or share > best_share):
                best_id, best_share = chunk_id, share
        return best_id


def relink_passages(db: Session, evidence_id: int, passages: PassageIndex) -> None:
    """Points this evidence's existing links at the new chunks after re-indexing."""
    for link in db.scalars(select(EvidenceLink).where(EvidenceLink.evidence_id == evidence_id)):
        link.chunk_id = passages.chunk_for(link.passage)


def clear_pending_proposals(db: Session, evidence: Evidence) -> None:
    """Removes the still-pending AI rows that only this evidence proposed and that no
    other record uses (an observation, relationship, event, injury, stage, scene
    object or conflict). Reviewed rows, and rows other evidence also supports, stay."""
    links = db.scalars(select(EvidenceLink).where(EvidenceLink.evidence_id == evidence.id)).all()
    # Dependents first, so a person whose relationships are being removed can go too.
    for record_type in ("observation", "relationship", "event", "location", "object", "person"):
        ids = {link.target_id for link in links if link.target_type == record_type}
        if not ids:
            continue
        model = REVIEWABLE_MODELS[record_type]
        rows = db.scalars(
            select(model).where(model.id.in_(ids), model.origin == "ai", model.review_status == "pending")
        ).all()
        for row in rows:
            if _linked_elsewhere(db, evidence.id, record_type, row.id) or _in_use(db, record_type, row):
                continue
            delete_links_to(db, record_type, row.id)
            db.delete(row)
        db.flush()


def _linked_elsewhere(db: Session, evidence_id: int, record_type: str, record_id: int) -> bool:
    query = select(EvidenceLink.id).where(
        EvidenceLink.target_type == record_type,
        EvidenceLink.target_id == record_id,
        EvidenceLink.evidence_id != evidence_id,
    )
    return db.scalar(query.limit(1)) is not None


def _in_use(db: Session, record_type: str, row) -> bool:
    checks = []
    if record_type != "observation":
        checks.append(select(Observation.id).where(Observation.subject_type == record_type, Observation.subject_id == row.id))
    if record_type in GRAPH_NODE_TYPES:
        checks.append(
            select(Relationship.id).where(
                or_(
                    and_(Relationship.from_type == record_type, Relationship.from_id == row.id),
                    and_(Relationship.to_type == record_type, Relationship.to_id == row.id),
                )
            )
        )
    if record_type == "person":
        checks.append(select(Injury.id).where(Injury.person_id == row.id))
    if record_type == "location":
        checks.append(select(Event.id).where(Event.location_id == row.id))
    if record_type == "object":
        checks.append(select(SceneObject.id).where(SceneObject.object_id == row.id))
    if any(db.scalar(check.limit(1)) is not None for check in checks):
        return True
    if record_type == "event":
        stages = db.scalars(
            select(ReconstructionStage.event_ids)
            .join(Reconstruction, Reconstruction.id == ReconstructionStage.reconstruction_id)
            .where(Reconstruction.case_id == row.case_id)
        )
        return any(row.id in (event_ids or []) for event_ids in stages)
    if record_type == "observation":
        conflicts = db.scalars(select(Conflict.observation_ids).where(Conflict.case_id == row.case_id))
        return any(row.id in (observation_ids or []) for observation_ids in conflicts)
    return False


class ProposalWriter:
    """Saves one evidence's structured output. Refs from the output map to the
    (record type, id) they were saved or matched as."""

    def __init__(self, db: Session, evidence: Evidence, passages: PassageIndex):
        self.db = db
        self.evidence = evidence
        self.case_id = evidence.case_id
        self.passages = passages
        self.refs: dict[str, tuple[str, int]] = {EVIDENCE_REF: ("evidence", evidence.id)}
        self.linked = {
            (link.target_type, link.target_id)
            for link in db.scalars(select(EvidenceLink).where(EvidenceLink.evidence_id == evidence.id))
        }
        self.persons: dict[str, Person] = {}
        for person in self._case_rows(Person):
            self.persons.setdefault(name_key(person.name), person)
        self.objects: dict[tuple[str, str], CaseObject] = {}
        for obj in self._case_rows(CaseObject):
            self._remember_object(obj)
        self.locations: dict[str, Location] = {}
        for location in self._case_rows(Location):
            self.locations.setdefault(text_key(location.address), location)

    def save(self, output: StructuredEvidence) -> None:
        for person in output.persons:
            self._save_person(person)
        for obj in output.objects:
            self._save_object(obj)
        for location in output.locations:
            self._save_location(location)
        for event in output.events:
            self._save_event(event)
        for relationship in output.relationships:
            self._save_relationship(relationship)
        for observation in output.observations:
            self._save_observation(observation)
        self.db.flush()

    def _case_rows(self, model) -> list:
        return list(self.db.scalars(select(model).where(model.case_id == self.case_id).order_by(model.id)))

    def _add(self, row):
        self.db.add(row)
        self.db.flush()
        return row

    def _ref(self, ref: str | None) -> tuple[str, int] | None:
        return self.refs.get((ref or "").strip().lower())

    def _keep(self, ref: str, record_type: str, row, passage: str) -> None:
        self.refs[ref.strip().lower()] = (record_type, row.id)
        self.link(record_type, row.id, passage)

    def link(self, record_type: str, record_id: int, passage: str) -> None:
        """One link per record from this evidence: the first passage that supports it."""
        if (record_type, record_id) in self.linked:
            return
        self.linked.add((record_type, record_id))
        passage = passage.strip()
        self.db.add(
            EvidenceLink(
                evidence_id=self.evidence.id,
                chunk_id=self.passages.chunk_for(passage),
                target_type=record_type,
                target_id=record_id,
                passage=passage or None,
            )
        )

    def _save_person(self, item) -> None:
        key = name_key(item.name)
        if not key:
            return
        person = self.persons.get(key)
        if person is None:
            age = item.age if item.age is not None and 0 <= item.age <= 130 else None
            person = self._add(
                Person(
                    case_id=self.case_id,
                    role=item.role,
                    name=item.name.strip(),
                    age=age,
                    occupation=item.occupation.strip(),
                    condition=item.condition,
                    contact=item.contact.strip(),
                    notes=item.notes.strip(),
                    details={},
                    **AI_ROW,
                )
            )
            self.persons[key] = person
        self._keep(item.ref, "person", person, item.passage)

    @staticmethod
    def _object_keys(kind: str, label: str, details: dict) -> list[tuple[str, str]]:
        """Plate first, then phone or serial number, then the same label for the same kind."""
        keys = [
            ("plate", plate_key(details.get("plate"))),
            ("number", number_key(details.get("number"))),
            (f"label:{kind}", text_key(label)),
        ]
        return [key for key in keys if key[1]]

    def _remember_object(self, obj: CaseObject) -> None:
        for key in self._object_keys(obj.kind, obj.label, obj.details or {}):
            self.objects.setdefault(key, obj)

    def _save_object(self, item) -> None:
        details = {key: getattr(item, key).strip() for key in ("plate", "number", "make", "colour")}
        details = {key: value for key, value in details.items() if value}
        label = item.label.strip() or details.get("make") or details.get("plate") or humanize(item.kind)
        keys = self._object_keys(item.kind, label, details)
        obj = next((self.objects[key] for key in keys if key in self.objects), None)
        if obj is None:
            obj = self._add(CaseObject(case_id=self.case_id, kind=item.kind, label=label, details=details, **AI_ROW))
            self._remember_object(obj)
        self._keep(item.ref, "object", obj, item.passage)

    def _save_location(self, item) -> None:
        key = text_key(item.address)
        if not key:
            return
        location = self.locations.get(key)
        if location is None:
            location = self._add(Location(case_id=self.case_id, kind=item.kind, address=item.address.strip(), **AI_ROW))
            self.locations[key] = location
        self._keep(item.ref, "location", location, item.passage)

    def _save_event(self, item) -> None:
        description = item.description.strip()
        if not description:
            return
        event = self.db.scalar(
            select(Event).where(Event.case_id == self.case_id, Event.description == description).limit(1)
        )
        if event is None:
            location = self._ref(item.location_ref)
            event = self._add(
                Event(
                    case_id=self.case_id,
                    location_id=location[1] if location and location[0] == "location" else None,
                    category=item.category,
                    time_start=parse_time(item.time_start),
                    time_end=parse_time(item.time_end),
                    time_precision=item.time_precision,
                    certainty=item.certainty,
                    description=description,
                    ai_note=item.note.strip() or None,
                    **AI_ROW,
                )
            )
        self._keep(item.ref, "event", event, item.passage)

    def _save_relationship(self, item) -> None:
        source, target = self._ref(item.from_ref), self._ref(item.to_ref)
        if not source or not target or source == target:
            return
        if source[0] not in GRAPH_NODE_TYPES or target[0] not in GRAPH_NODE_TYPES:
            return
        ends = {"from_type": source[0], "from_id": source[1], "to_type": target[0], "to_id": target[1], "kind": item.kind}
        relationship = self.db.scalar(select(Relationship).filter_by(case_id=self.case_id, **ends).limit(1))
        if relationship is None:
            label = item.label.strip() or humanize(item.kind)
            relationship = self._add(Relationship(case_id=self.case_id, label=label, **ends, **AI_ROW))
        self.link("relationship", relationship.id, item.passage)

    def _save_observation(self, item) -> None:
        statement = item.statement.strip()
        if not statement:
            return
        subject_type, subject_id = self._ref(item.subject_ref) or ("evidence", self.evidence.id)
        observation = self.db.scalar(
            select(Observation)
            .where(
                Observation.case_id == self.case_id,
                Observation.subject_type == subject_type,
                Observation.subject_id == subject_id,
                Observation.statement == statement,
            )
            .limit(1)
        )
        if observation is None:
            observation = self._add(
                Observation(
                    case_id=self.case_id,
                    subject_type=subject_type,
                    subject_id=subject_id,
                    statement=statement,
                    observed_at=parse_time(item.observed_at),
                    **AI_ROW,
                )
            )
        self.link("observation", observation.id, item.passage)


def save_proposals(db: Session, evidence: Evidence, output: StructuredEvidence, passages: PassageIndex) -> None:
    """Adds the rows and links to the session and flushes; the caller commits."""
    ProposalWriter(db, evidence, passages).save(output)
