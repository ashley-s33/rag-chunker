from rag_chunker.sentences import split_sentences


def test_empty_input():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_two_plain_sentences():
    result = split_sentences("Hello world. Goodbye now.")
    assert result == ["Hello world.", "Goodbye now."]


def test_abbreviation_does_not_split():
    result = split_sentences("Dr. Chen arrived early.")
    assert result == ["Dr. Chen arrived early."]


def test_decimal_number_does_not_split():
    result = split_sentences("Version 1.4 shipped today.")
    assert result == ["Version 1.4 shipped today."]


def test_middle_initial_does_not_split():
    result = split_sentences("J. Robert Oppenheimer led the project.")
    assert result == ["J. Robert Oppenheimer led the project."]


def test_eg_abbreviation_does_not_split():
    result = split_sentences("Bring snacks, e.g. chips, before the meeting.")
    assert result == ["Bring snacks, e.g. chips, before the meeting."]


def test_question_and_exclamation_marks_are_boundaries():
    result = split_sentences("Is this on? It is now!")
    assert result == ["Is this on?", "It is now!"]


def test_closing_quote_after_terminator_stays_attached():
    result = split_sentences('She said "stop." Then she left.')
    assert result == ['She said "stop."', "Then she left."]
