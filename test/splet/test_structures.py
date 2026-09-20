"""Tests for the structured-text types."""

from splet.structures import Session, Turn


def test_a_plain_utterance_is_a_one_turn_session():
    session = Session.from_text("utt1", "hello world")
    assert session.text() == "hello world"
    assert session.speakers == []
    assert not session.is_timed()


def test_speakers_are_listed_in_order_of_first_appearance():
    session = Session(
        "meeting",
        [Turn("a", "B"), Turn("b", "A"), Turn("c", "B")],
    )
    assert session.speakers == ["B", "A"]


def test_text_can_be_taken_per_speaker():
    """Per-speaker concatenation lives here because every long-form metric
    needs it and they must all do it identically."""
    session = Session(
        "meeting",
        [Turn("hello", "A"), Turn("hi there", "B"), Turn("how are you", "A")],
    )
    assert session.text("A") == "hello how are you"
    assert session.text("B") == "hi there"
    assert session.text() == "hello hi there how are you"


def test_empty_turns_do_not_introduce_separators():
    session = Session("s", [Turn("hello", "A"), Turn("", "A"), Turn("world", "A")])
    assert session.text() == "hello world"


def test_timing():
    session = Session("s", [Turn("hello", "A", 0.0, 1.5)])
    assert session.is_timed()
    assert session.turns[0].duration == 1.5
    assert Session("s", [Turn("hello", "A")]).turns[0].duration is None
