"""Structuring (step 3 in docs/PLAN.md): one Gemini call with a JSON response
schema turns an evidence's text into a summary plus persons, objects, locations,
events, relationships and observations, each with the passage it came from.

Items refer to each other by refs the model assigns (P1, O1, L1, E1), and to the
evidence itself as "evidence". proposals.py saves them.
"""

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.models import Case, CaseObject, Evidence, Person
from app.schemas.common import (
    Certainty,
    EventCategory,
    LocationKind,
    ObjectKind,
    PersonCondition,
    PersonRole,
    RelationshipKind,
    TimePrecision,
)
from app.services.ai.calls import generate_json

EVIDENCE_REF = "evidence"

PASSAGE = "Exact quote from the evidence text that supports this item, copied character for character: one or two sentences."
TIME = (
    "ISO 8601 date and time, with the UTC offset when the text states or clearly implies it, "
    "such as 2026-10-03T21:40:10+05:30, or a date alone when only the day is known; null when the text gives no time."
)


def _ref(prefix: str) -> Any:
    return Field(description=f"Your own unique id for this item ({prefix}1, {prefix}2...), used by other items.")


class PersonProposal(BaseModel):
    ref: str = _ref("P")
    name: str
    role: PersonRole
    age: int | None
    occupation: str = Field(description="As stated, or empty.")
    condition: PersonCondition | None = Field(description="Physical condition after the incident, if stated.")
    contact: str = Field(description="Phone number, email or address, if stated, or empty.")
    notes: str = Field(description="Anything else the text says about this person, briefly, or empty.")
    passage: str = Field(description=PASSAGE)


class ObjectProposal(BaseModel):
    ref: str = _ref("O")
    kind: ObjectKind
    label: str = Field(description="Short description, such as 'White Maruti Swift hatchback'.")
    plate: str = Field(description="Registration plate as written, or empty.")
    number: str = Field(description="Phone or serial number as written, or empty.")
    make: str = Field(description="Make and model, or empty.")
    colour: str = Field(description="Colour, or empty.")
    passage: str = Field(description=PASSAGE)


class LocationProposal(BaseModel):
    ref: str = _ref("L")
    kind: LocationKind
    address: str = Field(description="The place as the text names it, such as 'Ring Road and 5th Avenue junction'.")
    passage: str = Field(description=PASSAGE)


class EventProposal(BaseModel):
    ref: str = _ref("E")
    category: EventCategory
    description: str = Field(description="What happened, in one sentence.")
    time_start: str | None = Field(description=TIME)
    time_end: str | None = Field(description="When it ended, in the same format, or null.")
    time_precision: TimePrecision = Field(description="exact (to the second), minute, approximate, or day (date only).")
    certainty: Certainty = Field(
        description="established for instrument or official records, hypothesis for one person's account or an "
        "estimate, disputed when the text itself reports disagreement."
    )
    location_ref: str | None = Field(description="ref of the location where it happened, or null.")
    note: str = Field(description="Why the time or certainty is uncertain, or empty.")
    passage: str = Field(description=PASSAGE)


class RelationshipProposal(BaseModel):
    from_ref: str = Field(description=f"ref of a person, object or location, or '{EVIDENCE_REF}' for this evidence.")
    to_ref: str = Field(description=f"ref of a person, object or location, or '{EVIDENCE_REF}' for this evidence.")
    kind: RelationshipKind
    label: str = Field(description="Short label for the link, such as 'Drove'.")
    passage: str = Field(description=PASSAGE)


class ObservationProposal(BaseModel):
    subject_ref: str = Field(
        description=f"ref of the person, object, location or event the statement is about, or '{EVIDENCE_REF}'."
    )
    statement: str = Field(description="One factual claim as a full sentence, naming whose account it is when it is one.")
    observed_at: str | None = Field(description="When the thing it describes happened. " + TIME)
    passage: str = Field(description=PASSAGE)


class StructuredEvidence(BaseModel):
    summary: str = Field(description="Two to four sentences on what this evidence establishes, for an investigator.")
    persons: list[PersonProposal]
    objects: list[ObjectProposal]
    locations: list[LocationProposal]
    events: list[EventProposal]
    relationships: list[RelationshipProposal]
    observations: list[ObservationProposal]


ITEM_MODELS: dict[str, type[BaseModel]] = {
    "persons": PersonProposal,
    "objects": ObjectProposal,
    "locations": LocationProposal,
    "events": EventProposal,
    "relationships": RelationshipProposal,
    "observations": ObservationProposal,
}

SYSTEM_INSTRUCTION = """You extract facts from one piece of evidence in a forensic investigation, for an investigator to review.
Use only what the evidence text says. Don't guess names, ages, times or places it doesn't give, and don't merge different people or vehicles.
Record each distinct factual claim as an observation, so that accounts which disagree can be compared later.
Every item has a passage: an exact quote from the evidence text that supports it, copied character for character.
Use empty lists when there is nothing of a kind."""


def build_prompt(
    case: Case, evidence: Evidence, text: str, persons: list[Person], objects: list[CaseObject]
) -> str:
    lines = [f"Case: {case.title} ({case.incident_type.replace('_', ' ')})."]
    if case.incident_at:
        lines.append(f"Incident time: {case.incident_at.isoformat()} (UTC).")
    if case.description:
        lines.append(f"Case description: {case.description}")
    about = ", ".join(part for part in (evidence.type, evidence.source) if part)
    lines.append(f'Evidence: "{evidence.title}" ({about}).' + (f" {evidence.description}" if evidence.description else ""))
    if persons:
        known = "; ".join(f"{person.name} ({person.role})" for person in persons)
        lines.append(f"People already recorded in the case, to be spelled the same way when the text means them: {known}.")
    if objects:
        known = "; ".join(_describe_object(obj) for obj in objects)
        lines.append(f"Vehicles and objects already recorded in the case: {known}.")
    lines += ["", "Evidence text:", "-----", text, "-----"]
    return "\n".join(lines)


def _describe_object(obj: CaseObject) -> str:
    extras = [f"{key} {obj.details[key]}" for key in ("plate", "number") if obj.details.get(key)]
    return obj.label + (f" ({', '.join(extras)})" if extras else "")


def request_structure(
    client, case: Case, evidence: Evidence, text: str, persons: list[Person], objects: list[CaseObject]
) -> str:
    """The model's raw output, for ai_runs; decode it, then parse it with parse_output."""
    prompt = build_prompt(case, evidence, text, persons, objects)
    return generate_json(client, [prompt], StructuredEvidence, system_instruction=SYSTEM_INSTRUCTION)


def parse_output(raw: Any) -> StructuredEvidence:
    """Validates the model's JSON item by item, dropping items that don't fit the
    schema instead of failing the whole evidence over one of them."""
    if not isinstance(raw, dict):
        raise ValueError("Gemini returned JSON that is not an object")
    summary = raw.get("summary")
    lists: dict[str, list[BaseModel]] = {}
    for key, model in ITEM_MODELS.items():
        items = raw.get(key)
        lists[key] = []
        for item in items if isinstance(items, list) else []:
            try:
                lists[key].append(model.model_validate(item))
            except ValidationError:
                continue
    return StructuredEvidence(summary=summary.strip() if isinstance(summary, str) else "", **lists)
