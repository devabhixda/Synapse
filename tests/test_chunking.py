"""How an episode body is cut into the spans that get embedded."""

from synapse.memory.chunking import chunk_text


def test_short_text_is_one_chunk() -> None:
    assert chunk_text("a single line", 700) == ["a single line"]


def test_empty_text_yields_nothing() -> None:
    assert chunk_text("", 700) == []
    assert chunk_text("   \n\n  ", 700) == []


def test_chunks_respect_the_budget_and_keep_every_line() -> None:
    lines = [f"user{i}: {'x' * 90}" for i in range(10)]
    chunks = chunk_text("\n".join(lines), 300)

    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    # Nothing is dropped and nothing is duplicated.
    assert "\n".join(chunks).splitlines() == lines


def test_a_line_longer_than_the_budget_is_kept_whole() -> None:
    """Cutting mid-line would embed half a sentence as if someone wrote it."""
    long_line = "rishi: " + "y" * 2000
    chunks = chunk_text(long_line, 300)

    assert chunks == [long_line]


def test_chunking_is_prefix_stable() -> None:
    """Appending must not re-cut what came before, or a re-index rewrites it all."""
    before = "\n".join(f"user{i}: {'x' * 90}" for i in range(6))
    after = before + "\n" + "user6: something new"

    early = chunk_text(before, 300)
    later = chunk_text(after, 300)

    assert later[: len(early) - 1] == early[:-1]


def test_a_late_passage_reaches_the_front_of_its_own_chunk() -> None:
    """The point of chunking: the answer stops being buried mid-document.

    Embedded whole, this body is described by its first line. Chunked, the
    passage that answers a question sits at the top of a chunk of its own.
    """
    filler = "\n".join(f"user{i}: {'x' * 90}" for i in range(20))
    body = f"{filler}\nrishi: GB300 c=4 0.51 / 0.72"

    chunks = chunk_text(body, 300)
    hit = next(c for c in chunks if "GB300" in c)

    assert hit.index("GB300") < 300
