"""The one Gemini client, shared by the pipeline (A1), search and answers (A2) and
conflict detection (A3). Model names come from settings (GEMINI_MODEL,
GEMINI_EMBED_MODEL).

get_client() returns None when GEMINI_API_KEY is empty: callers skip the AI step
or fall back to local logic. Tests patch it to return a fake client.
"""

from functools import lru_cache

from google import genai

from app.core.config import get_settings


@lru_cache
def get_client() -> genai.Client | None:
    settings = get_settings()
    if not settings.ai_enabled:
        return None
    return genai.Client(api_key=settings.gemini_api_key)
