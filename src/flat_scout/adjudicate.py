"""Grade vector in, Verdict out. No model, no database, no file.

The rules that used to be instructions in `criteria.md` are arithmetic here.
"Absent means unknown, never bad" is the `determined` flag. The pets asymmetry
is one condition in `vetoes_fired`. "Ask only what the listing has not
answered" is the fact that a determined Criterion proposes nothing.

Purity is the point: every rule argued for here is testable in a millisecond,
which is why this module holds them and the pipeline holds none.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from flat_scout.models import Evaluation

if TYPE_CHECKING:  # pragma: no cover - needed only for the annotation
    from flat_scout.criteria import Criterion

# Above this is something good to say, below it something wrong. A grade
# exactly here is neither, which is what keeps one Criterion off both lists.
PIVOT = 5.0
SHOWN = 3


@dataclass(frozen=True)
class Grade:
    """One Criterion's answer about one Listing."""

    criterion: str
    value: float | None
    # False when nothing was read off the flat: the Criterion was skipped, or a
    # configured default supplied the value. A default is not evidence, so it
    # moves the score and never the coverage, and it can never fire a veto.
    determined: bool
    # One sentence naming what the grade was read off. This is what makes a
    # score auditable, and it is what a result and a replay print.
    evidence: str
    graded_by: str = ""
    concern: str | None = None
    question: str | None = None


Pair = tuple["Criterion", Grade]


def vetoes_fired(pairs: list[Pair]) -> list[Pair]:
    """The Criteria whose veto this grade vector trips.

    Public because a Verdict of `reject` is not self-explaining: `report
    --shift` has to name which Criterion disqualified the flat, and a second
    copy of this condition living there would be free to drift from this one.
    """
    return [
        (criterion, grade)
        for criterion, grade in pairs
        if criterion.veto_at_or_below is not None
        and grade.determined
        and grade.value is not None
        and grade.value <= criterion.veto_at_or_below
    ]


def _ranked(pairs: list[Pair], above: bool) -> list[str]:
    chosen = [
        (criterion, grade)
        for criterion, grade in pairs
        if grade.determined
        and grade.value is not None
        and (grade.value > PIVOT if above else grade.value < PIVOT)
    ]
    chosen.sort(key=lambda pair: pair[0].weight * (pair[1].value - PIVOT), reverse=above)
    return [grade.evidence for _, grade in chosen[:SHOWN]]


def _coverage_flag(pairs: list[Pair], coverage: float) -> str:
    unread = sorted(
        (criterion for criterion, grade in pairs if not grade.determined),
        key=lambda criterion: criterion.weight,
        reverse=True,
    )
    named = ", ".join(criterion.name for criterion in unread[:SHOWN])
    return f"Only {coverage:.0%} of the brief could be read - {named} unread."


# What an unread Criterion is worth. Silence is priced rather than ignored:
# see `adjudicate`. Four rather than five because an advert omits what does not
# help it - agents print the square footage when it flatters the flat - so an
# unanswered Criterion is a little worse than an average one, and that is the
# whole claim being made.
#
# The default here is deliberately the NEW rule rather than the old one. Several
# call sites pass this from settings, and a site that is ever missed must fall
# on the rule everything else uses rather than quietly keeping the old one.
UNKNOWN_PRIOR = 4.0


def adjudicate(
    pairs: list[Pair],
    *,
    hopeful_threshold: float,
    coverage_floor: float,
    unknown_prior: float = UNKNOWN_PRIOR,
) -> Evaluation:
    """The score, the coverage and the Verdict, computed in that order.

    The score is a weighted mean over the WHOLE brief, with unread Criteria
    entering at `unknown_prior`. It used to be a mean over the answered
    Criteria only, which quietly rewarded an advert for saying less: the flat
    that scored highest of the ones the couple had decided on was the one that
    answered least - 24% of the brief, an 8.33, and a rejection. An advert
    answering three things well beat a documented flat with one real weakness,
    because the three were all it was asked about.

    So Coverage is no longer only a gate at `coverage_floor`. It moves the
    score continuously, which is what the couple asked for: a flat that
    answers half the brief is judged on half the brief and a middling guess at
    the rest, and it cannot outrank a flat that answered everything well.

    `coverage_floor` stays as a backstop for the extreme end, where the score
    is mostly prior and means very little either way.

    A Grade with a value but `determined=False` - a configured default - counts
    at its value here and not at the prior. It is not evidence, which is why it
    never lifts Coverage, but it is a considered answer rather than a silence.
    """
    scored = [(c, g) for c, g in pairs if g.value is not None]
    total = sum(c.weight for c, _ in pairs)
    answered = sum(c.weight * g.value for c, g in scored)
    unanswered = total - sum(c.weight for c, _ in scored)
    # Still None when NOTHING was read. A Listing nobody could grade at all has
    # no score rather than a score made entirely of prior, which would be a
    # number about the brief and not about the flat.
    score = (answered + unanswered * unknown_prior) / total if scored and total else None

    covered = sum(c.weight for c, g in pairs if g.determined)
    coverage = covered / total if total else 0.0

    vetoes = vetoes_fired(pairs)
    if vetoes:
        verdict = "reject"
    elif score is not None and score >= hopeful_threshold and coverage >= coverage_floor:
        verdict = "hopeful"
    else:
        verdict = "borderline"

    # Ordered by how much a reader needs it: what disqualified the flat, then
    # how much of the brief went unread, then what a Criterion noticed but does
    # not measure, then the grades that came out low.
    red_flags = [grade.evidence for _, grade in vetoes]
    if coverage < coverage_floor:
        red_flags.append(_coverage_flag(pairs, coverage))
    red_flags += [g.concern for _, g in pairs if g.concern]
    red_flags += [
        evidence for evidence in _ranked(pairs, above=False) if evidence not in red_flags
    ]

    questions = []
    for criterion, grade in pairs:
        if grade.question:
            questions.append(grade.question)
        elif not grade.determined and criterion.ask:
            questions.append(criterion.ask)

    return Evaluation(
        verdict=verdict,
        score=score,
        coverage=coverage,
        reasons=_ranked(pairs, above=True),
        red_flags=red_flags,
        agent_questions=questions,
    )
