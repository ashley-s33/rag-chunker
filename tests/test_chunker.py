import pytest

from rag_chunker.chunker import _split_list_items, chunk_markdown


def test_single_short_section_is_one_chunk_with_heading_prefix():
    doc = "# Title\n\nHello world.\n"
    chunks = chunk_markdown(doc, max_tokens=512, overlap=64)
    assert len(chunks) == 1
    assert chunks[0].heading_path == ["Title"]
    assert chunks[0].body == "Hello world."
    assert chunks[0].text == "Title\n\nHello world."
    assert chunks[0].oversized is False


def test_no_heading_prefix_option_drops_the_prefix_but_keeps_the_path():
    doc = "# Title\n\nHello world.\n"
    chunks = chunk_markdown(doc, heading_prefix=False)
    assert chunks[0].text == "Hello world."
    assert chunks[0].heading_path == ["Title"]


def test_oversized_code_block_is_emitted_whole_rather_than_split():
    doc = "# T\n\n```\n" + "x" * 300 + "\n```\n"
    chunks = chunk_markdown(doc, max_tokens=10, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].oversized is True
    assert chunks[0].heading_path == ["T"]
    assert "x" * 300 in chunks[0].text


def test_overlap_repeats_the_trailing_sentence_in_the_next_chunk():
    doc = "# H\n\nOne. Two. Three. Four. Five.\n"
    chunks = chunk_markdown(doc, max_tokens=8, overlap=2)
    assert len(chunks) == 2
    assert chunks[0].heading_path == ["H"]
    assert chunks[1].heading_path == ["H"]
    assert chunks[0].body == "One. Two. Three."
    assert chunks[1].body == "Three. Four. Five."


def test_sections_carry_their_full_heading_path_and_empty_ones_are_dropped():
    doc = "# A\n## B\n\nContent B.\n# C\n\nContent C.\n"
    chunks = chunk_markdown(doc, max_tokens=512, overlap=0)
    assert len(chunks) == 2
    assert chunks[0].heading_path == ["A", "B"]
    assert chunks[0].body == "Content B."
    assert chunks[1].heading_path == ["C"]
    assert chunks[1].body == "Content C."


def test_chunk_indices_are_assigned_in_order():
    doc = "# A\n\nFirst.\n# B\n\nSecond.\n"
    chunks = chunk_markdown(doc)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_max_tokens_must_be_positive():
    with pytest.raises(ValueError):
        chunk_markdown("text", max_tokens=0)


def test_overlap_must_not_be_negative():
    with pytest.raises(ValueError):
        chunk_markdown("text", overlap=-1)


def test_overlap_must_be_smaller_than_max_tokens():
    with pytest.raises(ValueError):
        chunk_markdown("text", max_tokens=10, overlap=10)


def test_split_list_items_splits_unordered_bullets():
    assert _split_list_items("- one\n- two\n- three") == ["- one", "- two", "- three"]


def test_split_list_items_handles_star_and_plus_markers():
    assert _split_list_items("* one\n+ two") == ["* one", "+ two"]


def test_split_list_items_splits_ordered_dot_and_paren_markers():
    assert _split_list_items("1. one\n2. two") == ["1. one", "2. two"]
    assert _split_list_items("1) one\n2) two") == ["1) one", "2) two"]


def test_split_list_items_keeps_wrapped_continuation_lines_attached():
    assert _split_list_items("- one\n  continued\n- two") == ["- one\n  continued", "- two"]


def test_list_block_packs_each_item_as_its_own_piece():
    doc = "# H\n\n- one\n- two\n- three\n"
    chunks = chunk_markdown(doc, max_tokens=512, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].body == "- one\n- two\n- three"


def test_ordered_list_block_packs_each_item_as_its_own_piece():
    doc = "# H\n\n1. one\n2. two\n3. three\n"
    chunks = chunk_markdown(doc, max_tokens=512, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].body == "1. one\n2. two\n3. three"


def test_list_items_split_across_chunks_when_they_do_not_fit_together():
    doc = "# H\n\n- one\n- two\n- three\n"
    chunks = chunk_markdown(doc, max_tokens=6, overlap=0)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.body.count("\n") < 2
