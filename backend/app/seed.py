"""Loads the seed data: the road-accident case from frontend/src/mocks/seed.ts and
three smaller cases, with the same IDs, names, times and texts, so the screens look
the same with the mock switched off.

Evidence files are real. The PDFs, statements, scene photos (SVG) and the call
recording (WAV) are generated here, as frontend/src/mocks/files.ts does; the CCTV
clip is committed in seed_data/ (see seed_data/make_cctv_clip.py). Each file is
stored with save_bytes, so its hash is real and POST /cases/{id}/verify passes,
and its text is indexed, so search works without Gemini.

    python -m app.seed           # loads into an empty database; refuses if it has data
    python -m app.seed --reset   # drops and recreates every table first
"""

import argparse
import io
import math
import shutil
import sys
import wave
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from xml.sax.saxutils import escape

import pymupdf
from alembic import command
from alembic.config import Config
from sqlalchemy import func, inspect, select, text
from sqlalchemy.orm import Session

from app.core.config import BACKEND_DIR
from app.core.db import SessionLocal, engine
from app.core.storage import absolute_path, save_bytes, storage_root
from app.models import (
    AuditLog,
    Base,
    Case,
    CaseObject,
    Conflict,
    Event,
    Evidence,
    EvidenceChunk,
    EvidenceLink,
    Injury,
    Investigator,
    Location,
    Observation,
    Person,
    Reconstruction,
    ReconstructionStage,
    Relationship,
    SceneAnnotation,
    SceneObject,
)
from app.services.search.indexing import index_evidence

SEED_DATA_DIR = BACKEND_DIR / "seed_data"

IST = timezone(timedelta(hours=5, minutes=30))


def at(day: str, time: str) -> datetime:
    """Local time in IST on a day of 2026, such as at("10-03", "21:40:12")."""
    month, day_of_month = (int(part) for part in day.split("-"))
    hour, minute, second = (int(part) for part in time.split(":"))
    return datetime(2026, month, day_of_month, hour, minute, second, tzinfo=IST)


def oct3(time: str) -> datetime:
    return at("10-03", time)


# ---------------------------------------------------------------------------
# Evidence texts for case 1
# ---------------------------------------------------------------------------

POLICE_REPORT = [
    """TRAFFIC POLICE - SECTOR 12 STATION
Accident report no. TPS12/2026/0412

Date and time of incident: 3 October 2026, about 21:40. Location: junction of Ring Road and 5th Avenue, Sector 12.

Vehicle 1: white Maruti Swift hatchback, KA-01-AB-1234, driven by Rohan Kapoor (34), travelling east on Ring Road.

Vehicle 2: blue Tata Ace delivery van, KA-05-CD-5678, driven by Suresh Nair (46), travelling north on 5th Avenue, registered to QuickCart Logistics.

The van struck the driver's side of the hatchback in the middle of the junction. The hatchback came to rest against the north-east kerb. The van stopped near the centre line of 5th Avenue.

Rohan Kapoor was taken to City General Hospital by ambulance at 21:52 with a shoulder injury and a cut to the forehead. Suresh Nair was not injured.""",
    """Statements at the scene:

Suresh Nair stated that he waited at the stop line and moved off when his light went green. His phone was in a dashboard mount and he says he was not using it.

Rohan Kapoor stated that his light was amber when he entered the junction.

Scene examination at 23:10: skid marks from the hatchback measured 8.4 m before the point of impact. The van left no skid marks. Signal heads on all four arms were working. The signal controller log has been requested from the traffic management centre.

Witnesses: Anita Desai (pedestrian, north-east corner) and Vikram Singh (shopkeeper, south-west corner). CCTV from junction camera 2 seized.""",
]

DESAI_STATEMENT = """Witness statement - Anita Desai
Recorded by SI Arjun Mehta, 3 October 2026, 22:30, at the scene.

I was waiting to cross 5th Avenue at the north-east corner of the junction. The pedestrian signal for my crossing was red.

The white car was coming along Ring Road from the west. Its light was still yellow when it went into the junction. I am sure of that because I was watching the lights to see when I could cross.

The blue delivery van on 5th Avenue started moving before its light turned green. It had been creeping forward over the line, and it went as soon as the car's light changed. It hit the side of the white car in the middle of the junction.

I called 112 straight away. A man on a bicycle was also waiting at the corner, but he rode off before the police came."""

SINGH_STATEMENT = """Witness statement - Vikram Singh
Recorded by SI Arjun Mehta, 3 October 2026, 22:45, at his shop.

I run the corner store on the south-west corner of the junction. I was standing at the shop door, facing the junction.

The white car came fast along Ring Road. The light for Ring Road was red before the car reached the junction, and it did not slow down.

The delivery van had a green light. It pulled away normally from the stop line and the car went straight across in front of it. The van could not stop in time.

I have seen many cars jump that light at night."""

CCTV_DESCRIPTION = """Junction camera 2, facing north along 5th Avenue. Recording from 21:38:00 to 21:42:00. The signal heads are outside the camera's view. The camera clock has not yet been checked against the signal controller clock.

21:39:48 The blue delivery van arrives at the 5th Avenue stop line and stops.

21:40:06 The van's front wheels move about one metre past the stop line.

21:40:10 The white hatchback enters the frame from the left on Ring Road, travelling east.

21:40:11 The van moves off into the junction.

21:40:12 The van strikes the driver's side of the hatchback near the centre of the junction.

21:40:20 A cyclist at the north-east corner rides away north along 5th Avenue.

21:40:35 A pedestrian at the north-east corner makes a phone call."""

CALL_TRANSCRIPT = """Emergency call to 112, 3 October 2026, 21:41:05. Duration 1 minute 12 seconds.

Caller (Anita Desai): There's been an accident at the Ring Road and 5th Avenue junction. A van has hit a car.

Operator: Is anyone hurt?

Caller: The driver of the car is hurt, he's holding his shoulder. There's blood on his face.

Operator: An ambulance is on its way. Is the road blocked?

Caller: Yes, both of them are in the middle of the junction."""

MEDICAL_REPORT = [
    """CITY GENERAL HOSPITAL - EMERGENCY DEPARTMENT

Patient: Rohan Kapoor, male, 34. Arrived 3 October 2026, 22:05, by ambulance.

Finding 1: displaced fracture of the right clavicle, consistent with a side impact on the driver's side.

Finding 2: laceration of the forehead, 3 cm, sutured.

Finding 3: bruising to the right upper arm.

Blood alcohol: not detected. Condition: stable, admitted for observation.

Examining doctor: Dr. Kavya Iyer"""
]

SIGNAL_LOG = """Traffic management centre - signal controller log
Junction: Ring Road / 5th Avenue (controller J-12-04). Date: 3 October 2026.

21:39:20 Ring Road green, 5th Avenue red.

21:40:08 Ring Road amber, 5th Avenue red.

21:40:11 Ring Road red, 5th Avenue red (all-red phase).

21:40:13 Ring Road red, 5th Avenue green.

21:41:03 Ring Road red, 5th Avenue amber.

No faults were recorded between 21:00 and 22:00."""

BUS_DRIVER_STATEMENT = (
    "Witness statement - route 220 bus driver\n\n"
    "A dark hatchback hit a man crossing behind my bus at about 23:15. It did not stop and drove towards Lakeview Circle."
)

FIRE_REPORT = [
    "FIRE SERVICE INCIDENT REPORT\n\n"
    "Fire reported at 03:20 in a textile warehouse. Fire started near the electrical panel on the ground floor. No injuries."
]

COMPLAINANT_STATEMENT = (
    "Complainant statement\n\nWe returned at 17:30 to find the balcony door forced and jewellery missing."
)

# ---------------------------------------------------------------------------
# Evidence files
# ---------------------------------------------------------------------------


def make_pdf(pages: list[str], title: str) -> bytes:
    """An A4 PDF with one Helvetica text block per page and a real text layer."""
    document = pymupdf.open()
    for page_text in pages:
        page = document.new_page(width=595, height=842)
        overflow = page.insert_textbox(pymupdf.Rect(56, 56, 539, 786), page_text, fontname="helv", fontsize=11)
        if overflow < 0:
            raise ValueError(f"Seed PDF page does not fit: {title}")
    document.set_metadata({"title": title, "producer": "Forensix seed", "creator": "Forensix seed"})
    data = document.tobytes(garbage=4, deflate=True)
    document.close()
    return data


def make_wav(seconds: int = 4, sample_rate: int = 8000) -> bytes:
    """A short mono WAV: a two-tone ring followed by a quiet hum, standing in for a call recording."""
    frames = bytearray()
    for i in range(seconds * sample_rate):
        t = i / sample_rate
        ringing = t < 1.6 and t % 0.8 < 0.4
        if ringing:
            value = 0.25 * (math.sin(2 * math.pi * 440 * t) + math.sin(2 * math.pi * 480 * t))
        else:
            value = 0.04 * math.sin(2 * math.pi * 120 * t)
        frames += int(max(-1.0, min(1.0, value)) * 0x7FFF).to_bytes(2, "little", signed=True)
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(frames))
    return out.getvalue()


def make_scene_photo(caption: str) -> bytes:
    """A schematic stand-in for a scene photograph."""
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500" width="800" height="500">
  <rect width="800" height="500" fill="#1f2933"/>
  <rect x="0" y="190" width="800" height="120" fill="#3e4c59"/>
  <rect x="330" y="0" width="140" height="500" fill="#3e4c59"/>
  <g stroke="#e4e7eb" stroke-width="4" stroke-dasharray="24 18">
    <line x1="0" y1="250" x2="330" y2="250"/><line x1="470" y1="250" x2="800" y2="250"/>
    <line x1="400" y1="0" x2="400" y2="190"/><line x1="400" y1="310" x2="400" y2="500"/>
  </g>
  <g transform="translate(520 150) rotate(22)">
    <rect x="-60" y="-28" width="120" height="56" rx="12" fill="#f5f7fa"/>
    <path d="M -10 28 l 18 -16 l 16 18" fill="none" stroke="#cf1124" stroke-width="5"/>
  </g>
  <g transform="translate(385 285)">
    <rect x="-30" y="-55" width="60" height="110" rx="8" fill="#2186eb"/>
  </g>
  <path d="M 110 232 L 470 232" stroke="#9aa5b1" stroke-width="3" stroke-dasharray="4 6"/>
  <text x="24" y="40" fill="#f5f7fa" font-family="monospace" font-size="20">{escape(caption)}</text>
  <text x="24" y="470" fill="#9aa5b1" font-family="monospace" font-size="14">Mock evidence photo, generated for the demo case</text>
</svg>"""
    return svg.encode()


@dataclass(frozen=True)
class EvidenceSeed:
    id: int
    case_id: int
    type: str
    title: str
    description: str
    source: str
    collected_at: datetime
    file_name: str
    mime_type: str
    data: bytes
    # Extracted text; None for photos.
    text: str | None
    summary: str
    # One string per page, for PDFs.
    pages: list[str] | None = None


def evidence_seeds() -> list[EvidenceSeed]:
    pdf, txt = "application/pdf", "text/plain"
    return [
        EvidenceSeed(
            id=1,
            case_id=1,
            type="document",
            title="Police accident report TPS12/2026/0412",
            description="Report by the first officers at the scene, with statements and scene examination.",
            source="Traffic Police, Sector 12 station",
            collected_at=oct3("23:40:00"),
            file_name="police-report-TPS12-2026-0412.pdf",
            mime_type=pdf,
            data=make_pdf(POLICE_REPORT, "Police accident report TPS12/2026/0412"),
            text="\n\n".join(POLICE_REPORT),
            pages=POLICE_REPORT,
            summary=(
                "Van struck the driver side of the hatchback in the junction at about 21:40. Hatchback driver "
                "injured. Hatchback skid marks 8.4 m. Van driver says he moved off on green; hatchback driver says "
                "his light was amber."
            ),
        ),
        EvidenceSeed(
            id=2,
            case_id=1,
            type="statement",
            title="Witness statement: Anita Desai",
            description="Pedestrian waiting at the north-east corner.",
            source="Recorded by SI Arjun Mehta",
            collected_at=oct3("22:30:00"),
            file_name="witness-statement-anita-desai.txt",
            mime_type=txt,
            data=DESAI_STATEMENT.encode(),
            text=DESAI_STATEMENT,
            summary=(
                "Says the hatchback entered on yellow and the van moved off before its light turned green. "
                "Called 112. Mentions a cyclist who left before police arrived."
            ),
        ),
        EvidenceSeed(
            id=3,
            case_id=1,
            type="statement",
            title="Witness statement: Vikram Singh",
            description="Shopkeeper at the south-west corner.",
            source="Recorded by SI Arjun Mehta",
            collected_at=oct3("22:45:00"),
            file_name="witness-statement-vikram-singh.txt",
            mime_type=txt,
            data=SINGH_STATEMENT.encode(),
            text=SINGH_STATEMENT,
            summary="Says the hatchback ran a red light at speed and the van pulled away on green.",
        ),
        EvidenceSeed(
            id=4,
            case_id=1,
            type="video",
            title="Junction camera 2, 21:38 to 21:42",
            description="CCTV facing north along 5th Avenue. Signal heads not in view.",
            source="Traffic management centre",
            collected_at=oct3("23:55:00"),
            file_name="junction-camera-2.mp4",
            mime_type="video/mp4",
            data=(SEED_DATA_DIR / "junction-camera-2.mp4").read_bytes(),
            text=CCTV_DESCRIPTION,
            summary=(
                "Van creeps past the stop line at 21:40:06, moves off at 21:40:11 and strikes the hatchback at "
                "21:40:12. A cyclist leaves the scene at 21:40:20. Camera clock not yet checked."
            ),
        ),
        EvidenceSeed(
            id=5,
            case_id=1,
            type="photo",
            title="Scene photo: hatchback at rest",
            description="Hatchback against the north-east kerb, driver-side damage.",
            source="Const. Rahul Verma",
            collected_at=oct3("23:10:00"),
            file_name="scene-photo-hatchback-at-rest.svg",
            mime_type="image/svg+xml",
            data=make_scene_photo("Scene photo 1: hatchback at rest, NE kerb"),
            text=None,
            summary="Hatchback at rest against the north-east kerb with damage to the driver door and B-pillar.",
        ),
        EvidenceSeed(
            id=6,
            case_id=1,
            type="audio",
            title="Emergency call recording (112)",
            description="Call made by Anita Desai at 21:41.",
            source="Emergency response centre",
            collected_at=at("10-04", "10:00:00"),
            file_name="emergency-call-112.wav",
            mime_type="audio/wav",
            data=make_wav(),
            text=CALL_TRANSCRIPT,
            summary="Anita Desai reports a van hitting a car at 21:41; the car driver is hurt and the junction is blocked.",
        ),
        EvidenceSeed(
            id=7,
            case_id=1,
            type="medical",
            title="Medical report: Rohan Kapoor",
            description="Emergency department findings, City General Hospital.",
            source="City General Hospital",
            collected_at=at("10-04", "09:00:00"),
            file_name="medical-report-rohan-kapoor.pdf",
            mime_type=pdf,
            data=make_pdf(MEDICAL_REPORT, "Medical report: Rohan Kapoor"),
            text="\n\n".join(MEDICAL_REPORT),
            pages=MEDICAL_REPORT,
            summary="Fractured right clavicle, 3 cm forehead laceration, bruised right upper arm. No alcohol detected.",
        ),
        EvidenceSeed(
            id=8,
            case_id=1,
            type="document",
            title="Signal controller log, junction J-12-04",
            description="Signal phase changes between 21:39 and 21:41.",
            source="Traffic management centre",
            collected_at=at("10-04", "16:30:00"),
            file_name="signal-controller-log-J-12-04.txt",
            mime_type=txt,
            data=SIGNAL_LOG.encode(),
            text=SIGNAL_LOG,
            summary="Ring Road amber 21:40:08, red 21:40:11; all-red until 5th Avenue turns green at 21:40:13. No faults.",
        ),
        EvidenceSeed(
            id=9,
            case_id=2,
            type="photo",
            title="Bus stop 14 camera still",
            description="Still frame from the bus shelter camera.",
            source="City transport authority",
            collected_at=at("10-08", "08:00:00"),
            file_name="bus-stop-14-camera-still.svg",
            mime_type="image/svg+xml",
            data=make_scene_photo("Bus stop 14 camera, 23:15"),
            text=None,
            summary="Dark hatchback with a damaged left headlight passing the bus stop at 23:15.",
        ),
        EvidenceSeed(
            id=10,
            case_id=2,
            type="statement",
            title="Witness statement: bus driver",
            description="Driver of route 220, parked at bus stop 14.",
            source="Recorded by SI Arjun Mehta",
            collected_at=at("10-08", "01:10:00"),
            file_name="witness-statement-bus-driver.txt",
            mime_type=txt,
            data=BUS_DRIVER_STATEMENT.encode(),
            text=BUS_DRIVER_STATEMENT,
            summary="Dark hatchback hit a pedestrian at about 23:15 and drove towards Lakeview Circle.",
        ),
        EvidenceSeed(
            id=11,
            case_id=3,
            type="document",
            title="Fire service incident report",
            description="Report from the attending fire station.",
            source="Fire station 7",
            collected_at=at("09-22", "12:00:00"),
            file_name="fire-service-incident-report.pdf",
            mime_type=pdf,
            data=make_pdf(FIRE_REPORT, "Fire service incident report"),
            text="\n\n".join(FIRE_REPORT),
            pages=FIRE_REPORT,
            summary="Fire started near the ground-floor electrical panel at about 03:20. No injuries.",
        ),
        EvidenceSeed(
            id=12,
            case_id=4,
            type="statement",
            title="Complainant statement",
            description="Resident of flat 4B.",
            source="Recorded by Insp. Meera Rao",
            collected_at=at("08-30", "18:30:00"),
            file_name="complainant-statement.txt",
            mime_type=txt,
            data=COMPLAINANT_STATEMENT.encode(),
            text=COMPLAINANT_STATEMENT,
            summary="Balcony door forced; jewellery missing. Discovered at 17:30.",
        ),
    ]


# ---------------------------------------------------------------------------
# Records, written as tuples under a header of column names
# ---------------------------------------------------------------------------


def table(columns: str, rows: list[tuple], **common) -> list[dict]:
    """Turns tuples into keyword arguments for the model. A `review` column holds
    one of the origin and review_status pairs below."""
    names = columns.split()
    out = []
    for values in rows:
        row = deepcopy(common) | dict(zip(names, values, strict=True))
        out.append(row | row.pop("review", {}))
    return out


INVESTIGATOR = dict(origin="investigator", review_status="confirmed")
AI_PENDING = dict(origin="ai", review_status="pending")
AI_CONFIRMED = dict(origin="ai", review_status="confirmed")

INVESTIGATORS = table("id name badge_number role clearance_level", [
    (1, "Insp. Meera Rao", "TPD-2187", "Lead Investigator", "Level 4"),
    (2, "SI Arjun Mehta", "TPD-3044", "Accident Reconstruction Specialist", "Level 3"),
    (3, "Dr. Kavya Iyer", "FMO-0912", "Forensic Medical Officer", "Level 3"),
    (4, "Const. Rahul Verma", "TPD-4410", "Evidence Officer", "Level 2"),
])  # fmt: skip

CASES = table(
    "id case_number title incident_type status priority description incident_at lead_investigator_id "
    "investigator_ids created_at updated_at",
    [
        (1, "FX-2026-0001", "Collision at Ring Road and 5th Avenue junction", "road_accident", "in_progress", "high",
         "A delivery van and a hatchback collided in a signal-controlled junction at about 21:40. The hatchback "
         "driver was injured. Witnesses disagree on which vehicle had the right of way.",
         oct3("21:40:12"), 1, [1, 2, 3, 4], oct3("22:10:00"), at("10-05", "11:20:00")),
        (2, "FX-2026-0002", "Hit-and-run on Lakeview Road", "road_accident", "open", "critical",
         "A pedestrian was struck near bus stop 14 by a vehicle that did not stop.",
         at("10-07", "23:15:00"), 2, [2, 4], at("10-08", "00:05:00"), at("10-08", "09:40:00")),
        (3, "FX-2026-0003", "Warehouse fire, Industrial Area Phase 2", "fire", "under_review", "medium",
         "Fire in a textile warehouse during the night shift. Cause under review.",
         at("09-21", "03:20:00"), 3, [3], at("09-21", "06:00:00"), at("09-30", "15:00:00")),
        (4, "FX-2026-0004", "Burglary at Greenpark Apartments, flat 4B", "burglary", "closed", "low",
         "Daytime burglary while the residents were away. Property recovered.",
         at("08-30", "14:00:00"), 1, [1, 4], at("08-30", "18:00:00"), at("09-12", "10:00:00")),
    ],
)  # fmt: skip

PERSONS = table("id case_id role name age occupation condition contact notes review", [
    (1, 1, "victim", "Rohan Kapoor", 34, "Software engineer", "injured", "+91 90000 00101",
     "Driver of the hatchback. Says his light was amber when he entered the junction.", INVESTIGATOR),
    (2, 1, "other", "Suresh Nair", 46, "Delivery driver, QuickCart Logistics", "uninjured", "+91 90000 00102",
     "Driver of the van. Says he moved off when his light went green.", INVESTIGATOR),
    (3, 1, "witness", "Anita Desai", 29, "Pedestrian at the north-east corner", "uninjured", "+91 90000 00103",
     "Watched the signals while waiting to cross. Called 112.", INVESTIGATOR),
    (4, 1, "witness", "Vikram Singh", 52, "Shopkeeper, south-west corner", "uninjured", "+91 90000 00104",
     "Watched from his shop door.", INVESTIGATOR),
    (5, 1, "witness", "Unidentified cyclist", None, "", None, "",
     "Waiting at the north-east corner; rode away north before the police arrived.", AI_PENDING),
    (6, 2, "victim", "Unidentified pedestrian", 60, "", "injured", "",
     "Struck while crossing behind the route 220 bus.", INVESTIGATOR),
], details={})  # fmt: skip

OBJECTS = table("id case_id kind label details review", [
    (1, 1, "vehicle", "White hatchback", {"plate": "KA-01-AB-1234", "make": "Maruti Swift", "colour": "White"},
     INVESTIGATOR),
    (2, 1, "vehicle", "Delivery van", {"plate": "KA-05-CD-5678", "make": "Tata Ace", "colour": "Blue"}, INVESTIGATOR),
    (3, 1, "phone", "Van driver's phone", {"number": "+91 90000 00102"}, AI_PENDING),
    (4, 2, "vehicle", "Dark hatchback (unidentified)", {"colour": "Dark grey"}, INVESTIGATOR),
])  # fmt: skip

LOCATIONS = table("id case_id kind address lat lng", [
    (1, 1, "incident_site", "Ring Road and 5th Avenue junction, Sector 12", 12.9716, 77.5946),
    (2, 1, "witness_position", "North-east corner, pedestrian crossing", 12.97175, 77.59475),
    (3, 1, "witness_position", "Corner store, south-west corner", 12.97145, 77.59445),
    (4, 1, "vehicle_position", "Hatchback rest position, north-east kerb", 12.97168, 77.59472),
    (5, 1, "vehicle_position", "Van rest position, 5th Avenue centre line", 12.97158, 77.59458),
    (6, 2, "incident_site", "Lakeview Road near bus stop 14", 12.9352, 77.6245),
    (7, 3, "incident_site", "Plot 22, Industrial Area Phase 2", 13.0158, 77.5367),
    (8, 4, "incident_site", "Greenpark Apartments, flat 4B", 12.9279, 77.6271),
], geometry=None, building_id=None, review=INVESTIGATOR)  # fmt: skip

EVENTS = table(
    "id case_id location_id category time_start time_end time_precision certainty description ai_note review",
    [
        (1, 1, None, "movement", oct3("21:39:00"), oct3("21:40:10"), "approximate", "established",
         "Hatchback travels east on Ring Road towards the junction", None, INVESTIGATOR),
        (2, 1, 1, "movement", oct3("21:39:48"), None, "exact", "established",
         "Van stops at the 5th Avenue stop line", None, AI_CONFIRMED),
        (3, 1, 1, "movement", oct3("21:40:06"), None, "exact", "established",
         "Van creeps about one metre past the stop line", None, AI_CONFIRMED),
        (4, 1, 1, "other", oct3("21:40:08"), None, "exact", "established",
         "Ring Road signal turns amber", None, INVESTIGATOR),
        (5, 1, 1, "other", oct3("21:40:11"), None, "exact", "established",
         "Ring Road signal turns red; all-red phase begins", None, INVESTIGATOR),
        (6, 1, 1, "movement", oct3("21:40:11"), None, "exact", "disputed",
         "Van moves off into the junction",
         "Camera time is before 5th Avenue's green at 21:40:13, matching Anita Desai. Vikram Singh says the van "
         "had green. The camera clock has not been checked.", AI_PENDING),
        (7, 1, 1, "collision", oct3("21:40:12"), None, "exact", "established",
         "Van strikes the driver's side of the hatchback", None, INVESTIGATOR),
        (8, 1, 1, "other", oct3("21:40:13"), None, "exact", "established",
         "5th Avenue signal turns green", None, INVESTIGATOR),
        (9, 1, 2, "witness", oct3("21:40:20"), None, "exact", "hypothesis",
         "Unidentified cyclist leaves the scene northbound",
         "Seen on camera 2 and mentioned by Anita Desai. A possible third witness.", AI_PENDING),
        (10, 1, 2, "emergency", oct3("21:41:05"), None, "exact", "established",
         "Anita Desai calls 112", None, INVESTIGATOR),
        (11, 1, 1, "emergency", oct3("21:52:00"), None, "minute", "established",
         "Ambulance takes Rohan Kapoor to City General Hospital", None, INVESTIGATOR),
        (12, 1, None, "medical", oct3("22:05:00"), None, "minute", "established",
         "Rohan Kapoor examined at City General Hospital", None, INVESTIGATOR),
        (13, 1, 1, "investigation", oct3("23:10:00"), None, "minute", "established",
         "Scene examined; hatchback skid marks of 8.4 m measured", None, INVESTIGATOR),
        (14, 2, 6, "collision", at("10-07", "23:15:00"), None, "approximate", "established",
         "Pedestrian struck behind the route 220 bus", None, INVESTIGATOR),
    ],
)  # fmt: skip

RELATIONSHIPS = table("id case_id from_type from_id to_type to_id kind label confidence flagged notes review", [
    (1, 1, "person", 1, "object", 1, "drove", "Driver", None, False, "", INVESTIGATOR),
    (2, 1, "person", 2, "object", 2, "drove", "Driver", None, False, "", INVESTIGATOR),
    (3, 1, "object", 1, "location", 1, "involved_in", "Struck in junction", None, False, "", INVESTIGATOR),
    (4, 1, "object", 2, "location", 1, "involved_in", "Struck the hatchback", None, False, "", INVESTIGATOR),
    (5, 1, "person", 3, "location", 1, "witnessed", "Watched from NE crossing", None, False, "", INVESTIGATOR),
    (6, 1, "person", 4, "location", 1, "witnessed", "Watched from shop door", None, False, "", INVESTIGATOR),
    (7, 1, "evidence", 2, "evidence", 3, "contradicts", "Disagree on signal colour", 0.92, True,
     "Desai: hatchback on yellow, van early. Singh: hatchback on red, van on green.", AI_PENDING),
    (8, 1, "evidence", 4, "object", 2, "mentions", "Van moves off at 21:40:11", 0.88, False, "", AI_CONFIRMED),
    (9, 1, "person", 2, "object", 3, "owns", "Phone in dashboard mount", 0.75, False, "", AI_PENDING),
    (10, 1, "evidence", 8, "evidence", 2, "supports", "Signal log fits amber entry", 0.7, False, "", AI_PENDING),
    (11, 1, "person", 5, "location", 1, "witnessed", "Left before police arrived", 0.6, True, "", AI_PENDING),
    (12, 2, "object", 4, "person", 6, "involved_in", "Struck pedestrian", None, False, "", INVESTIGATOR),
])  # fmt: skip

OBSERVATIONS = table("id subject_type subject_id statement observed_at review", [
    (1, "object", 1, "The hatchback's signal was amber when it entered the junction.", oct3("21:40:10"), AI_PENDING),
    (2, "object", 1, "The hatchback's signal was red before it reached the junction.", oct3("21:40:10"), AI_PENDING),
    (3, "object", 2, "The van moved off before its signal turned green.", oct3("21:40:11"), AI_PENDING),
    (4, "object", 2, "The van had a green signal when it pulled away.", oct3("21:40:11"), AI_PENDING),
    (5, "person", 2, "The van driver says he moved off when his light went green.", oct3("21:40:11"), AI_PENDING),
    (6, "location", 1,
     "Ring Road was amber from 21:40:08 and red from 21:40:11; 5th Avenue turned green at 21:40:13.",
     oct3("21:40:08"), AI_CONFIRMED),
    (7, "person", 1,
     "Rohan Kapoor has a fractured right clavicle and a 3 cm forehead laceration; no alcohol detected.",
     oct3("22:05:00"), AI_CONFIRMED),
    (8, "object", 2,
     "By the camera clock, the van moved off at 21:40:11. The camera clock has not been checked against the "
     "signal controller.", oct3("21:40:11"), AI_PENDING),
], case_id=1, author_id=None)  # fmt: skip

# The signal-colour conflict, in two parts: the hatchback's light and the van's.
CONFLICTS = table("id observation_ids description", [
    (1, [1, 2], "Witnesses disagree on the hatchback's signal when it entered the junction."),
    (2, [3, 4, 5], "Accounts disagree on whether the van moved off before its signal turned green."),
], case_id=1, status="open")  # fmt: skip

INJURIES = table("id body_region structure_id type severity depth description review", [
    (1, "chest", "clavicle_right", "Fracture", "moderate", None,
     "Displaced fracture of the right clavicle, consistent with a driver-side impact.", AI_CONFIRMED),
    (2, "head", None, "Laceration", "minor", "3 cm long", "Forehead laceration, sutured.", AI_CONFIRMED),
    (3, "right_arm", None, "Bruising", "minor", None, "Bruising to the right upper arm.", AI_PENDING),
], case_id=1, person_id=1, estimated_time=oct3("21:40:12"))  # fmt: skip

# The source passage behind each record, in the mock's order so the IDs match.
LINKS = table("evidence_id target_type target_id passage", [
    (4, "event", 1, "The white hatchback enters the frame from the left on Ring Road, travelling east."),
    (4, "event", 2, "The blue delivery van arrives at the 5th Avenue stop line and stops."),
    (4, "event", 3, "The van's front wheels move about one metre past the stop line."),
    (8, "event", 4, "Ring Road amber, 5th Avenue red."),
    (8, "event", 5, "Ring Road red, 5th Avenue red (all-red phase)."),
    (4, "event", 6, "The van moves off into the junction."),
    (2, "event", 6, "The blue delivery van on 5th Avenue started moving before its light turned green."),
    (4, "event", 7, "The van strikes the driver's side of the hatchback near the centre of the junction."),
    (1, "event", 7, "The van struck the driver's side of the hatchback in the middle of the junction."),
    (8, "event", 8, "Ring Road red, 5th Avenue green."),
    (4, "event", 9, "A cyclist at the north-east corner rides away north along 5th Avenue."),
    (6, "event", 10, "There's been an accident at the Ring Road and 5th Avenue junction. A van has hit a car."),
    (1, "event", 11, "Rohan Kapoor was taken to City General Hospital by ambulance at 21:52"),
    (7, "event", 12, "Arrived 3 October 2026, 22:05, by ambulance."),
    (1, "event", 13, "skid marks from the hatchback measured 8.4 m before the point of impact."),
    (2, "observation", 1, "Its light was still yellow when it went into the junction."),
    (3, "observation", 2, "The light for Ring Road was red before the car reached the junction, and it did not slow down."),
    (2, "observation", 3, "The blue delivery van on 5th Avenue started moving before its light turned green."),
    (3, "observation", 4, "The delivery van had a green light."),
    (1, "observation", 5, "Suresh Nair stated that he waited at the stop line and moved off when his light went green."),
    (8, "observation", 6, "Ring Road amber, 5th Avenue red."),
    (7, "observation", 7, "displaced fracture of the right clavicle, consistent with a side impact on the driver's side."),
    (4, "observation", 8, "The van moves off into the junction."),
    (2, "person", 5, "A man on a bicycle was also waiting at the corner, but he rode off before the police came."),
    (1, "object", 3, "His phone was in a dashboard mount and he says he was not using it."),
    (2, "relationship", 7, "Its light was still yellow when it went into the junction."),
    (3, "relationship", 7, "The delivery van had a green light."),
    (1, "person", 1, "Rohan Kapoor stated that his light was amber when he entered the junction."),
    (10, "event", 14, "A dark hatchback hit a man crossing behind my bus at about 23:15."),
    (7, "injury", 1, "displaced fracture of the right clavicle, consistent with a side impact on the driver's side."),
    (7, "injury", 2, "laceration of the forehead, 3 cm, sutured."),
    (7, "injury", 3, "bruising to the right upper arm."),
])  # fmt: skip

# ---------------------------------------------------------------------------
# Reconstructions: two competing scenarios
# ---------------------------------------------------------------------------

RECONSTRUCTIONS = table("id scenario assumptions", [
    (1, "Van moved off before its green",
     "The hatchback entered on amber at about 21:40:09. The van moved off during the all-red phase at 21:40:11. "
     "Assumes the camera clock matches the signal controller. Supported by Anita Desai and the signal log."),
    (2, "Hatchback entered on red",
     "The hatchback entered after 21:40:11 on red. The van moved off on green at 21:40:13. Assumes the camera "
     "clock runs about 2 seconds behind the signal controller. Supported by Vikram Singh."),
], case_id=1, version=1, scene_file="road-accident/junction.json")  # fmt: skip

STAGES = table("id reconstruction_id stage_order label event_ids", [
    (1, 1, 1, "Approach", [1, 2, 3, 4]),
    (2, 1, 2, "Impact", [5, 6, 7]),
    (3, 1, 3, "Rest positions", [8, 10, 11]),
    (4, 2, 1, "Approach", [1, 2, 5]),
    (5, 2, 2, "Impact", [7, 8]),
])  # fmt: skip

HALF_PI = math.pi / 2
MODELS = {1: ("hatchback", "White hatchback"), 2: ("delivery-van", "Delivery van")}
# Positions are metres from the junction centre (x east, z south); headings are radians.
PLACEMENTS = [
    # (stage, object, position, heading)
    (1, 1, [-25, 0, -1.75], 0), (1, 2, [-1.75, 0, 9], HALF_PI),
    (2, 1, [-1, 0, -1.75], 0), (2, 2, [-1.75, 0, 0.5], HALF_PI),
    (3, 1, [6, 0, -5], 0.4), (3, 2, [-1.5, 0, -1], HALF_PI + 0.2),
    (4, 1, [-30, 0, -1.75], 0), (4, 2, [-1.75, 0, 8], HALF_PI),
    (5, 1, [-1, 0, -1.75], 0), (5, 2, [-1.75, 0, 0.5], HALF_PI),
]  # fmt: skip
SCENE_OBJECTS = [
    dict(
        id=index,
        stage_id=stage_id,
        model_id=MODELS[object_id][0],
        object_id=object_id,
        label=MODELS[object_id][1],
        position=[float(value) for value in position],
        rotation=[0.0, float(heading), 0.0],
        scale=[1.0, 1.0, 1.0],
    )
    for index, (stage_id, object_id, position, heading) in enumerate(PLACEMENTS, start=1)
]

ANNOTATIONS = table("id reconstruction_id stage_id kind data", [
    (1, 1, 2, "measurement",
     {"label": "Hatchback skid marks", "from": [-9.4, 0, -1.75], "to": [-1, 0, -1.75], "length_m": 8.4}),
    (2, 1, None, "viewpoint", {"label": "Anita Desai's view", "position": [8, 1.6, -8], "target": [0, 0, 0]}),
    (3, 1, None, "note", {"text": "Camera clock offset not yet checked against the signal controller."}),
])  # fmt: skip

# The audit trail besides the "uploaded" entry of each evidence file.
AUDIT = table("actor_id action record_type record_id case_id detail at", [
    (2, "viewed", "evidence", 4, 1, None, at("10-04", "11:00:00")),
    (1, "reviewed", "event", 2, 1, {"review_status": {"before": "pending", "after": "confirmed"}}, at("10-05", "10:15:00")),
    (1, "reviewed", "event", 3, 1, {"review_status": {"before": "pending", "after": "confirmed"}}, at("10-05", "10:16:00")),
    (1, "modified", "case", 1, 1, {"status": {"before": "open", "after": "in_progress"}}, at("10-05", "11:20:00")),
])  # fmt: skip

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def chunk_for(db: Session, evidence_id: int, passage: str) -> EvidenceChunk:
    """The first chunk of the evidence that contains the passage."""
    chunks = db.scalars(
        select(EvidenceChunk).where(EvidenceChunk.evidence_id == evidence_id).order_by(EvidenceChunk.ordinal)
    )
    for chunk in chunks:
        if passage in chunk.text:
            return chunk
    raise ValueError(f"Seed passage not found in evidence {evidence_id}: {passage}")


def load_seed(db: Session) -> None:
    """Adds the seed rows and stores the evidence files, then commits. Expects
    empty tables; IDs are set explicitly so they match the mock."""
    stored_files: list[str] = []
    try:
        _load(db, stored_files)
        db.commit()
    except BaseException:
        db.rollback()
        for file_path in stored_files:
            absolute_path(file_path).unlink(missing_ok=True)
        raise


def _load(db: Session, stored_files: list[str]) -> None:
    def add(model: type[Base], rows: list[dict]) -> None:
        # Table by table, so each insert finds the rows its foreign keys point at.
        db.add_all(model(**row) for row in rows)
        db.flush()

    add(Investigator, INVESTIGATORS)
    add(Case, CASES)

    seeds = evidence_seeds()
    for seed in seeds:
        stored = save_bytes(seed.case_id, seed.file_name, seed.data)
        stored_files.append(stored.file_path)
        add(Evidence, [dict(
            id=seed.id,
            case_id=seed.case_id,
            type=seed.type,
            title=seed.title,
            description=seed.description,
            notes="",
            file_path=stored.file_path,
            mime_type=seed.mime_type,
            size_bytes=stored.size_bytes,
            sha256=stored.sha256,
            collected_at=seed.collected_at,
            source=seed.source,
            processing_status="done",
            summary=seed.summary,
            extracted_text=seed.text,
            created_at=seed.collected_at,
            # Null, as in the mock: the alerts feed only lists evidence the pipeline processed.
            processed_at=None,
        )])  # fmt: skip
        if seed.text is not None:
            index_evidence(db, seed.id, seed.pages or [seed.text], paged=seed.pages is not None)
            db.flush()

    add(Person, PERSONS)
    add(CaseObject, OBJECTS)
    add(Location, LOCATIONS)
    add(Event, EVENTS)
    add(Relationship, RELATIONSHIPS)
    add(Observation, OBSERVATIONS)
    add(Conflict, CONFLICTS)
    add(Injury, INJURIES)
    add(Reconstruction, RECONSTRUCTIONS)
    add(ReconstructionStage, STAGES)
    add(SceneObject, SCENE_OBJECTS)
    add(SceneAnnotation, ANNOTATIONS)
    add(EvidenceLink, [
        dict(link, id=index, chunk_id=chunk_for(db, link["evidence_id"], link["passage"]).id)
        for index, link in enumerate(LINKS, start=1)
    ])  # fmt: skip

    # Const. Rahul Verma, the evidence officer, logged each file in when it was collected.
    uploads = table("record_id case_id detail at", [
        (seed.id, seed.case_id, {"title": seed.title}, seed.collected_at) for seed in seeds
    ], actor_id=4, action="uploaded", record_type="evidence")  # fmt: skip
    entries = sorted(uploads + AUDIT, key=lambda entry: entry["at"])
    add(AuditLog, [dict(entry, id=index) for index, entry in enumerate(entries, start=1)])

    _sync_id_sequences(db)


def _sync_id_sequences(db: Session) -> None:
    """The seed sets IDs explicitly. SQLite carries on from the highest ID by itself;
    PostgreSQL needs its sequences moved past them."""
    if db.get_bind().dialect.name != "postgresql":
        return
    for table_ in Base.metadata.sorted_tables:
        if "id" in table_.c:
            db.execute(
                text(
                    f"SELECT setval(pg_get_serial_sequence('{table_.name}', 'id'), "
                    f"coalesce(max(id), 0) + 1, false) FROM {table_.name}"
                )
            )


def row_counts(db: Session) -> dict[str, int]:
    """Rows per table, leaving out empty tables."""
    counts = {t.name: db.scalar(select(func.count()).select_from(t)) or 0 for t in Base.metadata.sorted_tables}
    return {name: count for name, count in counts.items() if count}


def upgrade_schema() -> None:
    """Runs `alembic upgrade head`."""
    command.upgrade(Config(str(BACKEND_DIR / "alembic.ini")), "head")


def reset_database() -> None:
    """Drops every table and rebuilds the schema with the migrations, as on a new
    database, then removes the stored evidence files (storage/<case_id>/), which
    no row points at any more."""
    Base.metadata.drop_all(engine)
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS alembic_version"))
    upgrade_schema()
    for folder in storage_root().iterdir():
        if folder.is_dir() and folder.name.isdigit():
            shutil.rmtree(folder)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.seed", description="Load the Forensix seed cases.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="drop and recreate every table, and remove the stored evidence files, before loading",
    )
    args = parser.parse_args(argv)

    if args.reset:
        reset_database()
    elif not inspect(engine).has_table(Case.__tablename__):
        upgrade_schema()

    with SessionLocal() as db:
        existing = row_counts(db)
        if existing:
            print(
                f"The database already has data ({existing.get(Case.__tablename__, 0)} cases). "
                "Run `python -m app.seed --reset` to replace it with the seed.",
                file=sys.stderr,
            )
            return 1
        load_seed(db)
        loaded = row_counts(db)

    print(
        f"Loaded the seed: {loaded['investigators']} investigators, {loaded['cases']} cases and "
        f"{loaded['evidence']} evidence files, stored under {storage_root()}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
