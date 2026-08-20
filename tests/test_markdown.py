from rag_chunker.markdown import parse_blocks


def test_empty_input():
    assert parse_blocks("") == []
    assert parse_blocks("   \n  ") == []


def test_heading_and_paragraph():
    blocks = parse_blocks("# Title\n\nSome text.\n")
    assert [b.kind for b in blocks] == ["heading", "paragraph"]
    assert blocks[0].level == 1
    assert blocks[0].title == "Title"
    assert blocks[0].start_line == 1
    assert blocks[0].end_line == 1
    assert blocks[1].text == "Some text."
    assert blocks[1].start_line == 3
    assert blocks[1].end_line == 3


def test_heading_levels():
    blocks = parse_blocks("### Deep heading\n")
    assert blocks[0].level == 3
    assert blocks[0].title == "Deep heading"


def test_list_block_keeps_items_together():
    blocks = parse_blocks("- one\n- two\n- three\n")
    assert len(blocks) == 1
    assert blocks[0].kind == "list"
    assert blocks[0].text == "- one\n- two\n- three"
    assert blocks[0].start_line == 1
    assert blocks[0].end_line == 3


def test_fenced_code_block():
    blocks = parse_blocks("```python\nprint('hi')\n```\n")
    assert len(blocks) == 1
    assert blocks[0].kind == "code"
    assert blocks[0].text == "```python\nprint('hi')\n```"
    assert blocks[0].start_line == 1
    assert blocks[0].end_line == 3


def test_unterminated_fence_still_closes_the_block():
    blocks = parse_blocks("```\nprint('hi')")
    assert len(blocks) == 1
    assert blocks[0].kind == "code"
    assert blocks[0].end_line == 2


def test_pipe_table():
    blocks = parse_blocks("| a | b |\n| - | - |\n| 1 | 2 |\n")
    assert len(blocks) == 1
    assert blocks[0].kind == "table"
    assert blocks[0].start_line == 1
    assert blocks[0].end_line == 3


def test_paragraph_stops_before_a_following_heading():
    blocks = parse_blocks("Some text\nMore text\n# Heading\n")
    assert len(blocks) == 2
    assert blocks[0].kind == "paragraph"
    assert blocks[0].text == "Some text\nMore text"
    assert blocks[1].kind == "heading"


def test_paragraph_is_not_mistaken_for_a_table_without_a_separator_row():
    blocks = parse_blocks("a | b\nnot a separator\n")
    assert len(blocks) == 1
    assert blocks[0].kind == "paragraph"
