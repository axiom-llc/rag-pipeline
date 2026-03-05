"""Text chunking strategies.

Both functions return list[str].  Strategy is selected at ingest time via
pipeline.ingest(strategy=...).

chunk_size is measured in **words**, not tokens.  For English prose,
word count ≈ 0.75 × token count, so the default of 512 words corresponds
to roughly 680 tokens — comfortably within Gemini's embedding context window.
Adjust RAG_CHUNK_SIZE empirically for your domain and script.

Fixed-size (default)
    Splits on word boundaries to approximately chunk_size words, with a
    configurable word-level overlap between consecutive chunks.  Overlap
    preserves context that would otherwise be severed at a boundary.

Sentence
    Groups sentences (split on sentence-ending punctuation) until chunk_size
    words is reached.  Better recall for prose; less predictable for structured
    or technical documents.
"""
from __future__ import annotations
import re


def chunk_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping fixed-size chunks (word-boundary aligned).

    Args:
        text:       Input text.
        chunk_size: Maximum words per chunk.
        overlap:    Words of overlap between consecutive chunks. Must be < chunk_size.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")
    words = text.split()
    chunks: list[str] = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + chunk_size]))
        i += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def chunk_sentences(text: str, chunk_size: int) -> list[str]:
    """Group sentences into chunks of up to chunk_size words.

    Args:
        text:       Input text.
        chunk_size: Maximum words per chunk.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        word_count = len(sentence.split())
        if count + word_count > chunk_size and current:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += word_count
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]
