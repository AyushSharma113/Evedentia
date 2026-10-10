"""Helpers around the shared Gemini client (client.py): plain-text and JSON calls,
media parts for files, and the ai_runs log that records every AI output.

Every function takes the client from get_client() as an argument, so callers
decide what happens without a key and tests pass a fake.
"""

import json
import time
from pathlib import Path
from typing import Any

from google.genai import types
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import AiRun

# Gemini takes at most 20 MB per request inline; larger files go through the Files API.
INLINE_LIMIT = 18 * 1024 * 1024
UPLOAD_TIMEOUT_SECONDS = 300
UPLOAD_POLL_SECONDS = 2


def model_name() -> str:
    return get_settings().gemini_model


def generate_text(client, contents: list[Any], *, system_instruction: str | None = None) -> str:
    config = types.GenerateContentConfig(system_instruction=system_instruction) if system_instruction else None
    response = client.models.generate_content(model=model_name(), contents=contents, config=config)
    return (response.text or "").strip()


def generate_json(client, contents: list[Any], schema: Any, *, system_instruction: str | None = None) -> str:
    """One call with a JSON response schema. Returns the raw text, so it can be
    logged even when it doesn't decode; decode it with decode_json."""
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=schema,
    )
    response = client.models.generate_content(model=model_name(), contents=contents, config=config)
    return response.text or ""


def decode_json(raw: str) -> Any:
    """The decoded JSON, or None when the output is empty or not JSON."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def generate_from_file(client, path: Path, mime_type: str, prompt: str) -> str:
    """A plain-text call about one file (OCR, transcription). Small files go inline;
    larger ones are uploaded, used once and deleted."""
    if path.stat().st_size <= INLINE_LIMIT:
        part = types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)
        return generate_text(client, [part, prompt])
    uploaded = client.files.upload(file=path, config=types.UploadFileConfig(mime_type=mime_type))
    try:
        uploaded = _wait_until_active(client, uploaded)
        return generate_text(client, [uploaded, prompt])
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:  # noqa: BLE001 - Gemini deletes uploads after 48 hours anyway
            pass


def _wait_until_active(client, file):
    deadline = time.monotonic() + UPLOAD_TIMEOUT_SECONDS
    while file.state == types.FileState.PROCESSING:
        if time.monotonic() > deadline:
            raise TimeoutError("Gemini took too long to process the uploaded file")
        time.sleep(UPLOAD_POLL_SECONDS)
        file = client.files.get(name=file.name)
    if file.state == types.FileState.FAILED:
        raise RuntimeError("Gemini could not process the uploaded file")
    return file


def log_run(
    db: Session,
    task: str,
    output: dict | list | str | None,
    *,
    evidence_id: int | None = None,
    case_id: int | None = None,
) -> AiRun:
    """Adds an ai_runs row with the raw output of one call; the caller commits."""
    run = AiRun(evidence_id=evidence_id, case_id=case_id, task=task, model=model_name(), output=output)
    db.add(run)
    return run
