from rag_chunker.tokens import estimate_tokens, fits_budget


def test_empty_string_is_zero_tokens():
    assert estimate_tokens("") == 0


def test_short_word_rounds_up_to_one_token():
    # "cat": 3 chars / 4 chars-per-token = 0.75, floored to the 1 token minimum.
    assert estimate_tokens("cat") == 1


def test_longer_word_crosses_a_token_boundary():
    # "hello": 5 / 4 = 1.25, ceil'd to 2.
    assert estimate_tokens("hello") == 2


def test_digit_run_uses_the_number_rate():
    # "12345": 5 / 3 = 1.667, ceil'd to 2.
    assert estimate_tokens("12345") == 2


def test_cjk_is_one_token_per_character():
    assert estimate_tokens("中文") == 2


def test_newline_runs_merge_before_scoring():
    # A run of newlines is one match worth len(run) * 0.5 tokens, not one per line.
    assert estimate_tokens("\n\n\n\n") == 2


def test_spaces_are_free():
    assert estimate_tokens("cat") == estimate_tokens("  cat  ")


def test_sentence_combines_word_and_punctuation_costs():
    # "cat" (1.0) + "!" (0.6) = 1.6, ceil'd to 2.
    assert estimate_tokens("cat!") == 2


def test_fits_budget_true_at_exact_boundary():
    assert fits_budget("cat", 1) is True


def test_fits_budget_false_when_over():
    assert fits_budget("cat", 0) is False
