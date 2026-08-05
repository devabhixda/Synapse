"""Split an episode body into the spans that get embedded.

An episode is a whole conversation, and that is the right unit to store: it is
what a citation points at, what consolidation reasons over, and what a re-ingest
must recognise. It is the wrong unit to embed. One vector for a ten-day thread
describes its opening and averages away everything after it, so a question whose
answer is in the last reply never retrieves the thread that contains it.

So the two are separated here. The episode stays whole; the index gets one
vector per chunk, each pointing back at the episode. Retrieval matches a span
and returns the conversation it came from.

Chunks are cut on line boundaries because a body is rendered one message per
line — cutting mid-line would split a speaker from what they said. A line longer
than the budget is emitted whole rather than divided: an over-long chunk still
embeds, while a half-sentence embeds as something nobody wrote.
"""

from __future__ import annotations

DEFAULT_CHUNK_CHARS = 700


def chunk_text(text: str, max_chars: int = DEFAULT_CHUNK_CHARS) -> list[str]:
    """Split ``text`` into consecutive chunks of at most ``max_chars``.

    The fold is greedy and left-to-right, so it is prefix-stable: appending to
    the text cannot change the chunks already produced from it. Re-indexing a
    conversation that has grown therefore rewrites nothing that was already
    stored.

    Returns ``[]`` for empty input, and never returns an empty chunk.
    """
    if not text or not text.strip():
        return []

    chunks: list[str] = []
    current: list[str] = []
    size = 0

    for line in text.splitlines():
        if not line.strip():
            continue
        cost = len(line) + (1 if current else 0)
        if current and size + cost > max_chars:
            chunks.append("\n".join(current))
            current, size = [], 0
            cost = len(line)
        current.append(line)
        size += cost

    if current:
        chunks.append("\n".join(current))
    return chunks
