"""Gemini embeddings for evidence chunks and questions.

Vectors are stored on evidence_chunks.embedding as float32 bytes. Every vector
has EMBED_DIMENSIONS values, so a stored vector of any other length (say, from a
different embedding model) counts as missing and is embedded again.
"""

from collections.abc import Sequence

import numpy as np
from google.genai import types

from app.core.config import get_settings
from app.models import EvidenceChunk

EMBED_DIMENSIONS = 768
# The most texts the Gemini API embeds in one request.
EMBED_BATCH_SIZE = 100


def to_bytes(vector: np.ndarray | Sequence[float]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def from_bytes(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32)


def has_embedding(chunk: EvidenceChunk) -> bool:
    return chunk.embedding is not None and len(chunk.embedding) == EMBED_DIMENSIONS * 4


def embed_texts(client, texts: Sequence[str], *, task_type: str) -> list[np.ndarray]:
    """Embeds texts in batches. task_type is RETRIEVAL_DOCUMENT for chunks and
    RETRIEVAL_QUERY for questions. Raises if the API fails."""
    model = get_settings().gemini_embed_model
    config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBED_DIMENSIONS)
    vectors: list[np.ndarray] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = list(texts[start : start + EMBED_BATCH_SIZE])
        response = client.models.embed_content(model=model, contents=batch, config=config)
        embeddings = response.embeddings or []
        if len(embeddings) != len(batch):
            raise ValueError(f"Expected {len(batch)} embeddings, got {len(embeddings)}")
        vectors.extend(np.asarray(e.values, dtype=np.float32) for e in embeddings)
    return vectors


def embed_chunks(client, chunks: Sequence[EvidenceChunk]) -> None:
    """Sets the embedding of each chunk, a batch at a time, so a failure part way
    keeps the batches already done. The caller flushes and commits."""
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start : start + EMBED_BATCH_SIZE]
        vectors = embed_texts(client, [chunk.text for chunk in batch], task_type="RETRIEVAL_DOCUMENT")
        for chunk, vector in zip(batch, vectors, strict=True):
            chunk.embedding = to_bytes(vector)


def embed_query(client, text: str) -> np.ndarray:
    return embed_texts(client, [text], task_type="RETRIEVAL_QUERY")[0]
