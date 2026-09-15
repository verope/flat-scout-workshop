"""The criteria/ directory as shipped. These test the content, not the code."""

from pathlib import Path

from flat_scout.criteria import load_criteria, load_preamble
from flat_scout.graders import GRADERS

CRITERIA = Path("criteria")


def shipped():
    return load_criteria(CRITERIA, known_graders=set(GRADERS))


def test_the_shipped_criteria_load():
    assert len(shipped()) == 5


def test_every_criterion_has_a_rubric_with_anchors():
    for criterion in shipped():
        assert "## Rubric" in criterion.body, criterion.slug
        assert "**10**" in criterion.body, criterion.slug
        assert "**0**" in criterion.body, criterion.slug


def test_the_shared_files_are_not_criteria():
    """`_brief.md`, `_fields.md` and `criteria.md` sit in the same directory
    and are prose, not Criteria. The loader must not turn them into slugs."""
    slugs = {criterion.slug for criterion in shipped()}
    assert "_brief" not in slugs
    assert "_fields" not in slugs
    assert "criteria" not in slugs


def test_the_non_tradeables_carry_a_veto():
    vetoes = {c.slug for c in shipped() if c.veto_at_or_below is not None}
    assert vetoes == {"pets"}


def test_every_veto_skips_rather_than_defaulting():
    """A default could otherwise supply the very value that rejects a flat."""
    for criterion in shipped():
        if criterion.veto_at_or_below is not None:
            assert criterion.unknown == "skip", criterion.slug


def test_advert_silence_has_explicit_priors_only_where_it_speaks():
    priors = {
        criterion.slug: criterion.silence_prior
        for criterion in shipped()
        if criterion.silence_prior is not None
    }
    assert priors == {"pets": {0.0: 0.7, 10.0: 0.3}}


def test_the_total_weight_is_ten():
    """10 over the five active Criteria, since floor area came down from 4 to
    3 for the workshop. Every Score moves when this moves, so the number is
    asserted rather than left to drift unnoticed."""
    assert sum(criterion.weight for criterion in shipped()) == 10


def test_only_the_aspect_criterion_falls_back_to_the_model():
    """`aspect` grades off the plan's compass where there is one and asks the
    model where there is not. Nothing else in the set has a second route."""
    fallbacks = {c.slug for c in shipped() if c.fallback}
    assert fallbacks == {"aspect"}


def test_the_model_reachable_set_is_the_three_model_criteria_plus_the_fallback():
    """What `measure` prices and what a rubric edit can make unsteady.
    `floor-area` is arithmetic and can never reach a model, so it is not here."""
    reachable = {c.slug for c in shipped() if c.is_model or c.fallback == "model"}
    assert reachable == {"pets", "hard-floors", "lift", "aspect"}


def test_every_criterion_that_can_be_unknown_can_ask_or_needs_no_question():
    """A Criterion with no `ask` and no model to propose one asks nothing at
    all, which is fine only where a question would be unanswerable."""
    silent = {c.slug for c in shipped() if not c.ask and not c.is_model and not c.fallback}
    assert silent == set()


def test_the_preamble_carries_the_brief_and_the_field_semantics():
    preamble = load_preamble(CRITERIA)
    assert "Iris" in preamble
    assert "epc_caption" in preamble
    assert "window_aspects" in preamble
