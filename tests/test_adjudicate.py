from flat_scout.adjudicate import Grade, adjudicate
from flat_scout.criteria import Criterion


def a_criterion(slug: str, weight: float, **kwargs) -> Criterion:
    return Criterion(
        slug=slug,
        name=slug.replace("-", " ").capitalize(),
        description="d",
        weight=weight,
        grade={"model": True},
        body="rubric",
        **kwargs,
    )


def determined(slug: str, value: float, evidence: str = "because") -> Grade:
    return Grade(criterion=slug, value=value, determined=True, evidence=evidence)


def skipped(slug: str) -> Grade:
    return Grade(criterion=slug, value=None, determined=False, evidence="unknown")


def defaulted(slug: str, value: float) -> Grade:
    return Grade(criterion=slug, value=value, determined=False, evidence="default")


def judge(pairs, threshold=6.0, floor=0.5):
    return adjudicate(pairs, hopeful_threshold=threshold, coverage_floor=floor)


def test_score_is_the_weighted_mean_of_the_determined_grades():
    result = judge(
        [
            (a_criterion("a", 3), determined("a", 10)),
            (a_criterion("b", 1), determined("b", 2)),
        ]
    )
    assert result.score == 8.0
    assert result.coverage == 1.0


def test_a_skipped_criterion_drags_the_score_toward_the_prior():
    """Changed on 2026-08-20, and it is the point of that change.

    A skip used to leave the score untouched, so an advert that answered one
    Criterion at 10 and stayed silent on the rest scored 10. Of the
    Listings the couple had decided on, the highest-scoring was the one that
    answered least - 24% of the brief, an 8.33, and a rejection.

    Three parts weight at 10 and one part unread at the 4.0 prior is 8.5.
    """
    result = judge(
        [
            (a_criterion("a", 3), determined("a", 10)),
            (a_criterion("b", 1), skipped("b")),
        ]
    )
    assert result.score == 8.5
    assert result.coverage == 0.75


def test_silence_cannot_beat_an_answered_weakness():
    """The comparison the old rule got backwards, as one assertion.

    The thin flat answers a quarter of the brief perfectly. The documented one
    answers all of it, mostly well, with a single 5. The documented flat must
    win, and under the old mean-of-what-was-read it lost 10.0 to 8.75.
    """
    thin = judge(
        [
            (a_criterion("a", 1), determined("a", 10)),
            (a_criterion("b", 1), skipped("b")),
            (a_criterion("c", 1), skipped("c")),
            (a_criterion("d", 1), skipped("d")),
        ]
    )
    documented = judge(
        [
            (a_criterion("a", 1), determined("a", 10)),
            (a_criterion("b", 1), determined("b", 10)),
            (a_criterion("c", 1), determined("c", 10)),
            (a_criterion("d", 1), determined("d", 5)),
        ]
    )
    assert documented.score > thin.score


def test_a_defaulted_grade_counts_towards_the_score_and_not_the_coverage():
    """A default came from the config, not from the flat."""
    result = judge(
        [
            (a_criterion("a", 1), determined("a", 10)),
            (a_criterion("b", 1), defaulted("b", 4)),
        ]
    )
    assert result.score == 7.0
    assert result.coverage == 0.5


def test_no_determined_criterion_gives_no_score_and_never_hopeful():
    result = judge([(a_criterion("a", 1), skipped("a"))], threshold=0.0, floor=0.0)
    assert result.score is None
    assert result.coverage == 0.0
    assert result.verdict == "borderline"


def test_an_empty_criteria_set_does_not_divide_by_zero():
    result = judge([])
    assert result.score is None
    assert result.coverage == 0.0
    assert result.verdict == "borderline"


def test_hopeful_needs_both_the_threshold_and_the_floor():
    high = [(a_criterion("a", 1), determined("a", 9))]
    assert judge(high, threshold=6.0, floor=0.5).verdict == "hopeful"
    assert judge(high, threshold=9.5, floor=0.5).verdict == "borderline"
    thin = [
        (a_criterion("a", 1), determined("a", 9)),
        (a_criterion("b", 3), skipped("b")),
    ]
    assert judge(thin, threshold=6.0, floor=0.5).verdict == "borderline"


def test_a_veto_rejects_and_names_itself():
    result = judge(
        [
            (a_criterion("a", 1), determined("a", 10)),
            (
                a_criterion("bedroom-window", 4, veto_at_or_below=0),
                determined("bedroom-window", 0, "the plan shows an internal bedroom"),
            ),
        ]
    )
    assert result.verdict == "reject"
    assert result.red_flags[0] == "the plan shows an internal bedroom"


def test_a_veto_never_fires_on_an_unknown_grade():
    """The bedroom-window asymmetry: False rejects, None must not."""
    result = judge(
        [
            (a_criterion("a", 1), determined("a", 9)),
            (a_criterion("bedroom-window", 4, veto_at_or_below=0), skipped("bedroom-window")),
        ],
        # The unread veto Criterion carries four fifths of the weight, so the
        # prior drags this to 5.0. The threshold is lowered to keep the test
        # about the veto - the assertion is that nothing REJECTED the flat.
        threshold=5.0,
        floor=0.0,
    )
    assert result.verdict == "hopeful"


def test_a_veto_never_fires_on_a_defaulted_grade():
    result = judge(
        [
            (a_criterion("a", 1), determined("a", 9)),
            (
                a_criterion("bedroom-window", 4, veto_at_or_below=0),
                defaulted("bedroom-window", 0),
            ),
        ],
        floor=0.0,
    )
    assert result.verdict != "reject"


def test_a_veto_rejects_even_when_hopeful():
    """A veto beats hopeful: excellent elsewhere does not save a flat."""
    result = judge(
        [
            (a_criterion("bedroom-window", 1, veto_at_or_below=0), determined("bedroom-window", 0)),
            (a_criterion("layout", 4), determined("layout", 10)),
            (a_criterion("transport", 1), determined("transport", 10)),
        ]
    )
    # Score: (1*0 + 4*10 + 1*10) / 6 = 50/6 ≈ 8.33, Coverage: 6/6 = 1.0
    # Would be hopeful (score >= 6.0 and coverage >= 0.5) but veto rejects it.
    assert result.score >= 6.0
    assert result.coverage >= 0.5
    assert result.verdict == "reject"


def test_reasons_come_from_above_the_pivot_and_flags_from_below():
    result = judge(
        [
            (a_criterion("good", 3), determined("good", 9, "modern build")),
            (a_criterion("bad", 3), determined("bad", 2, "west facing")),
            (a_criterion("middling", 3), determined("middling", 5, "exactly five")),
        ]
    )
    assert result.reasons == ["modern build"]
    assert "west facing" in result.red_flags
    assert "exactly five" not in result.reasons
    assert "exactly five" not in result.red_flags


def test_a_criterion_is_never_both_a_reason_and_a_flag():
    result = judge([(a_criterion("only", 1), determined("only", 8, "the one thing"))])
    assert result.reasons == ["the one thing"]
    assert "the one thing" not in result.red_flags


def test_reasons_are_ranked_by_weight_times_distance_and_capped_at_three():
    result = judge(
        [
            (a_criterion("small", 1), determined("small", 10, "small but perfect")),
            (a_criterion("big", 5), determined("big", 9, "big and nearly perfect")),
            (a_criterion("c", 1), determined("c", 7, "third")),
            (a_criterion("d", 1), determined("d", 6, "fourth")),
        ]
    )
    assert result.reasons[0] == "big and nearly perfect"
    assert len(result.reasons) == 3
    assert "fourth" not in result.reasons


def test_a_concern_is_a_flag_whatever_the_grade():
    grade = Grade(
        criterion="layout",
        value=9,
        determined=True,
        evidence="good layout",
        concern="the advert says one bed and the plan shows a studio",
    )
    result = judge([(a_criterion("layout", 1), grade)])
    assert "the advert says one bed and the plan shows a studio" in result.red_flags


def test_low_coverage_says_so_and_names_the_heaviest_unknowns():
    result = judge(
        [
            (a_criterion("a", 1), determined("a", 9)),
            (a_criterion("layout", 5), skipped("layout")),
            (a_criterion("aspect", 2), skipped("aspect")),
        ]
    )
    flags = " ".join(result.red_flags)
    assert "%" in flags
    assert "Layout" in flags
    assert "Aspect" in flags


def test_full_coverage_says_nothing_about_coverage():
    result = judge([(a_criterion("a", 1), determined("a", 9))])
    assert not any("%" in flag for flag in result.red_flags)


def test_an_ask_template_becomes_a_question_only_when_unknown():
    asked = a_criterion("bike", 1, ask="What is the bike storage, and does it lock?")
    assert judge([(asked, skipped("bike"))]).agent_questions == [
        "What is the bike storage, and does it lock?"
    ]
    assert judge([(asked, determined("bike", 8))]).agent_questions == []


def test_a_grade_may_carry_a_question_even_when_it_is_determined():
    """price-value grades a cheap flat as neutral and still asks what explains it."""
    grade = Grade(
        criterion="price-value",
        value=7,
        determined=True,
        evidence="below the range for SE17",
        question="What explains the rent being below the local range?",
    )
    result = judge([(a_criterion("price-value", 1, ask="unused"), grade)])
    assert result.agent_questions == [
        "What explains the rent being below the local range?"
    ]
