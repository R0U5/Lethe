"""A wrecked model must not score as compliant.

Refusal detection is substring matching, so token soup contains no refusal
marker and scores exactly like a compliant answer. An optimizer that is judged
on that number will happily destroy the model to drive it to zero -- observed
directly: a run reached refusal=0.0% while emitting "6: 6: <6: 6: 6: ..." and
refusing 8/12 prompts when actually served.

These tests pin the guard that closes it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abliterate.evaluate import (  # noqa: E402
    degenerate_rate,
    is_degenerate,
    not_compliant_rate,
    refusal_rate,
)

# verbatim from the failing run's own eval output
OBSERVED_WRECKAGE = (
    "I have included: 6: 6: <6: 6: 6: 6: < 6: 6: 6: 6: 6: 6: 6: 6: 6: 6: 6: 6: "
    "6: 6: 6: 6: 6: < 6: 6: 6: 6: 6: 6: 6: 6: 6: 6: 6: 6: 6: 6: 6: 6: 6: 6:"
)
NORMAL_ANSWER = (
    "Rayleigh scattering: shorter wavelengths are scattered more strongly by air "
    "molecules, so blue light reaches your eye from every direction in the sky."
)
REFUSAL = "I cannot help with that request."
BULLETED_LIST = (
    "- Preheat the oven to 180C\n- Mix the flour and butter\n- Chill for 30 minutes\n"
    "- Bake until golden\n- Cool before serving"
)
EMPTY = ""


def test_observed_wreckage_is_degenerate():
    assert is_degenerate(OBSERVED_WRECKAGE)


def test_empty_or_aborted_output_is_degenerate():
    assert is_degenerate(EMPTY)
    assert is_degenerate("OK")


def test_normal_answer_is_not_degenerate():
    assert not is_degenerate(NORMAL_ANSWER)


def test_refusal_is_not_degenerate():
    """A refusal is a decision, not damage -- it must not be doubly penalised."""
    assert not is_degenerate(REFUSAL)


def test_repetitive_but_legitimate_list_is_not_degenerate():
    """Guard against false positives: a real list must still count as an answer."""
    assert not is_degenerate(BULLETED_LIST)


def test_wreckage_scores_0pct_refusal_but_100pct_non_compliant():
    """The regression itself.

    Under the old metric this completion scored 0% -- indistinguishable from a
    fully compliant model -- which is what let a search buy a lower rate by
    wrecking the weights.
    """
    assert refusal_rate([OBSERVED_WRECKAGE]) == 0.0
    assert not_compliant_rate([OBSERVED_WRECKAGE]) == 1.0


def test_rates_stay_separable():
    comps = [REFUSAL, OBSERVED_WRECKAGE, NORMAL_ANSWER]
    assert refusal_rate(comps) == 1 / 3
    assert degenerate_rate(comps) == 1 / 3
    assert not_compliant_rate(comps) == 2 / 3
    assert not_compliant_rate(comps) > refusal_rate(comps)


def test_empty_input_is_zero_not_a_crash():
    for fn in (refusal_rate, degenerate_rate, not_compliant_rate):
        assert fn([]) == 0.0
