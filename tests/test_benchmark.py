"""Scoring the pipeline against the annotations.

This is the instrument every decision about the floorplan pipeline is now made
with, so its own arithmetic is worth holding still. The subtle part is that
"nothing to say" is three different outcomes - correctly silent, abstained, and
wrong - and collapsing them would flatter or damn the pipeline by a third of the
corpus, since 7 of the 24 plans carry no compass at all.
"""

from __future__ import annotations

import pytest

from flat_scout.annotate import Annotation
from flat_scout.benchmark import (
    found_of_true,
    score_aspect,
    score_north,
    score_windows,
    true_aspects,
)


def plan(**fields) -> Annotation:
    base = {"portal_id": "1", "file": "1.png", "done": True}
    return Annotation(**{**base, **fields})


@pytest.mark.parametrize(
    "truth,got,verdict",
    [
        (270, 270, "right"),
        (270, 300, "right"),  # within a compass point, so the same answer
        (270, 320, "wrong"),  # 50 degrees out, a different point
        (358, 2, "right"),  # across the wrap
        (270, None, "abstained"),
        (None, None, "correctly-silent"),  # no indicator drawn, and none claimed
        (None, 90, "wrong"),  # invented a bearing where nothing is drawn
    ],
)
def test_north_is_scored_against_what_was_drawn(truth, got, verdict):
    assert score_north(plan(north_clock=truth), got).verdict == verdict


def test_a_plan_with_no_compass_is_not_counted_as_a_failure():
    """Seven of twenty-four have none, so this is a third of the corpus.

    Counting silence as a miss would say the pipeline fails a third of the time
    at a job that cannot be done from those drawings at all.
    """
    outcome = score_north(plan(north_clock=None), None)
    assert outcome.verdict == "correctly-silent"
    assert outcome.truth == "none"


@pytest.mark.parametrize(
    "truth,got,verdict",
    [
        ((180,), [180], "right"),
        ((0, 270), [270], "right"),  # one of two real walls, which is a partial answer
        ((0, 270), [0, 270], "right"),  # both, which is the whole answer
        ((0, 270), [90], "wrong"),  # a wall the flat does not have
        ((0, 270), [0, 90], "wrong"),  # one real and one invented is still wrong
        ((0, 270), [], "abstained"),
        ((), [180], "wrong"),  # glazing claimed where the annotator saw none
    ],
)
def test_a_wall_is_right_only_if_the_flat_really_has_it(truth, got, verdict):
    """Asymmetric on purpose.

    Missing one wall of a dual aspect costs the couple a fact they never had.
    Naming a wall the flat does not have surfaces a wrong one, and they cannot
    check it without opening the floorplan.
    """
    assert score_windows(plan(windows_clock=truth), got).verdict == verdict


def test_completeness_is_counted_separately_from_safety():
    """One wall of a dual aspect is a safe answer and half a description."""
    entry = plan(windows_clock=(0, 270))
    assert score_windows(entry, [270]).verdict == "right"
    assert found_of_true(entry, [270]) == (1, 2)
    assert found_of_true(entry, [0, 270]) == (2, 2)


def test_every_aspect_a_dual_aspect_flat_has_counts_as_the_truth():
    entry = plan(north_clock=0, windows_clock=(0, 90))
    assert true_aspects(entry) == {"N", "E"}
    assert score_aspect(entry, ["E"]).verdict == "right"
    assert score_aspect(entry, ["N", "E"]).verdict == "right"
    assert score_aspect(entry, ["S"]).verdict == "wrong"
    # One real and one invented is not half right; the invented one is surfaced too.
    assert score_aspect(entry, ["N", "S"]).verdict == "wrong"


def test_no_compass_means_no_aspect_is_knowable_from_the_drawing():
    entry = plan(north_clock=None, windows_clock=(0, 90))
    assert true_aspects(entry) == set()
    assert score_aspect(entry, []).verdict == "correctly-silent"
    # ...so any answer at all is a fabrication, however plausible.
    assert score_aspect(entry, ["N"]).verdict == "wrong"


def test_the_aspect_turns_with_the_compass():
    """Same windows, different north: the whole point of measuring the compass."""
    windows = (0,)
    assert true_aspects(plan(north_clock=0, windows_clock=windows)) == {"N"}
    assert true_aspects(plan(north_clock=270, windows_clock=windows)) == {"E"}
    assert true_aspects(plan(north_clock=90, windows_clock=windows)) == {"W"}


def test_one_prediction_can_only_account_for_one_real_wall():
    """The figure quoted as "N of 41 walls found" was overstating the pipeline.

    A single wall at 45 degrees sits within tolerance of annotated walls at both
    0 and 90, and used to score two found from one answer.
    """
    entry = plan(windows_clock=(0, 90))
    assert found_of_true(entry, [45]) == (1, 2)
    assert found_of_true(entry, [0, 90]) == (2, 2)
    assert found_of_true(entry, []) == (0, 2)
