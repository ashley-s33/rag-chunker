"""End-to-end check against a realistic document, the shape used in the README.

The unit tests in test_chunker.py exercise the packer against short synthetic
docs, one behaviour at a time. This one runs the whole pipeline against a
document with a real intro, a code block and a table, and pins the exact
token counts a maintainer would otherwise have to eyeball from `--stats`.
"""

from pathlib import Path

from rag_chunker.chunker import chunk_markdown

DOC = (Path(__file__).parent / "fixtures" / "doc.md").read_text()


def test_generous_budget_yields_one_chunk_per_section():
    chunks = chunk_markdown(DOC, max_tokens=512, overlap=64)

    assert [c.heading_path for c in chunks] == [
        ["Vector index runbook"],
        ["Vector index runbook", "Reindex"],
        ["Vector index runbook", "Checks"],
    ]
    assert [c.token_estimate for c in chunks] == [23, 58, 101]
    assert [c.oversized for c in chunks] == [False, False, False]

    reindex, checks = chunks[1], chunks[2]
    assert reindex.start_line == 7 and reindex.end_line == 14
    assert "```bash" in reindex.body and "```" in reindex.body
    assert checks.start_line == 18 and checks.end_line == 24
    assert checks.body.count("|") > 0


def test_tight_budget_forces_code_and_table_into_their_own_oversized_chunks():
    chunks = chunk_markdown(DOC, max_tokens=40, overlap=10)

    assert [c.heading_path for c in chunks] == [
        ["Vector index runbook"],
        ["Vector index runbook", "Reindex"],
        ["Vector index runbook", "Reindex"],
        ["Vector index runbook", "Checks"],
        ["Vector index runbook", "Checks"],
    ]
    assert [c.token_estimate for c in chunks] == [23, 22, 44, 22, 86]
    assert [c.oversized for c in chunks] == [False, False, True, False, True]

    code_chunk, table_chunk = chunks[2], chunks[4]
    assert code_chunk.body.startswith("```bash")
    assert table_chunk.body.startswith("| Check |")
    # a carried sentence never gets glued to the front of an atomic piece
    assert not code_chunk.body.startswith("Run the job")
    assert not table_chunk.body.startswith("Confirm")
