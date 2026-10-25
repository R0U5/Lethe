"""Refusal measurement must not silently mislead the optimizer.

Two failure modes are covered here, both of which corrupt an optimization
target rather than merely adding noise:

* an answer that *complies while warning* is not a refusal, and counting it as
  one spends KL to remove behaviour that was never refusal;
* a completion cut off by the token budget can hide a refusal -- and because an
  optimizer stops as soon as its metric reads zero, a truncated score ends the
  search with the model still refusing.
"""

from __future__ import annotations

import pytest

from abliterate import evaluate as ev


COMPLIANT_WITH_CAUTION = (
    "Sure - here is how locks work. A pin tumbler has driver pins. "
    "Please note that picking a lock you do not own is illegal in most places."
)
GENUINE_REFUSAL = "I cannot provide instructions for picking locks."
PLAIN_ANSWER = "The sky is blue because of Rayleigh scattering."


def test_compliant_answer_with_caution_is_not_a_refusal():
    """The headline contract: hedging inside an answer is compliance."""
    assert not ev.is_refusal(COMPLIANT_WITH_CAUTION)
    assert ev.has_caution(COMPLIANT_WITH_CAUTION)


def test_genuine_refusal_is_detected():
    assert ev.is_refusal(GENUINE_REFUSAL)


def test_plain_answer_is_neither():
    assert not ev.is_refusal(PLAIN_ANSWER)
    assert not ev.has_caution(PLAIN_ANSWER)


def test_no_marker_is_in_both_lists():
    overlap = set(ev.REFUSAL_MARKERS) & set(ev.CAUTION_MARKERS)
    assert overlap == set(), f"markers classified as both: {sorted(overlap)}"


def test_caution_phrases_are_not_refusal_openers():
    """Every caution phrase must be inert for refusal scoring."""
    for phrase in ev.CAUTION_MARKERS:
        assert not ev.is_refusal(f"Here is the thing. {phrase} this is risky.")


def test_refusal_stats_reports_caution_and_truncation():
    completions = [GENUINE_REFUSAL, COMPLIANT_WITH_CAUTION, PLAIN_ANSWER, PLAIN_ANSWER]
    truncated = [False, True, True, False]
    stats = ev.refusal_stats(completions, truncated)
    assert stats["n"] == 4
    assert stats["refusal_rate"] == pytest.approx(0.25)
    assert stats["caution_rate"] == pytest.approx(0.25)
    assert stats["truncated_rate"] == pytest.approx(0.5)
    assert stats["n_truncated"] == 2


def test_refusal_stats_without_truncation_flags():
    stats = ev.refusal_stats([PLAIN_ANSWER])
    assert stats["refusal_rate"] == 0.0
    assert "truncated_rate" not in stats


def test_refusal_stats_empty():
    stats = ev.refusal_stats([])
    assert stats["refusal_rate"] == 0.0
    assert stats["n"] == 0


def test_marker_override_still_works():
    original = ev.REFUSAL_MARKERS
    try:
        ev.set_refusal_markers(["nope, not doing that"])
        assert ev.is_refusal("Nope, not doing that today.")
        assert not ev.is_refusal(GENUINE_REFUSAL)
    finally:
        ev.set_refusal_markers(original)


def test_truncation_flag_marks_unfinished_generation():
    """A completion that used the whole budget without EOS is flagged.

    Reproduces the logic used by ``generate_completions_with_stats`` so the
    guard itself is covered without loading a model.
    """
    pad_id, eos_id, max_new_tokens = 0, 2, 4

    def flag(ids):
        ids = list(ids)
        while ids and ids[-1] == pad_id:
            ids.pop()
        hit_cap = len(ids) >= max_new_tokens
        ended = bool(ids) and ids[-1] == eos_id
        return hit_cap and not ended

    assert flag([5, 6, 7, 8]) is True          # ran out of budget
    assert flag([5, 6, 7, eos_id]) is False    # finished on its own
    assert flag([5, 6, 7, 8, eos_id, pad_id, pad_id]) is False
    assert flag([5, eos_id, pad_id, pad_id]) is False  # short + padding
