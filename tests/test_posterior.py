import numpy as np

from flat_scout.adjudicate import Grade
from flat_scout.criteria import Criterion
from flat_scout.fit import GRADE_BINS, CriterionFit, Fit
from flat_scout.posterior import Posterior, _unknown_flag, adjudicate_bayesian, score_posterior


def a_criterion(slug, weight, **kwargs):
    return Criterion(
        slug=slug, name=slug.capitalize(), description="d", weight=weight,
        grade={"model": True}, body="rubric", **kwargs,
    )


def determined(slug, value, graded_by="map"):
    return Grade(criterion=slug, value=value, determined=True, evidence="e", graded_by=graded_by)


def silent(slug, graded_by="model:sonnet"):
    return Grade(criterion=slug, value=None, determined=False, evidence="unknown", graded_by=graded_by)


def a_fit(sigma=1.0, weights=None):
    flat = tuple(weights or [1 / 11] * 11)
    return Fit(
        criteria={
            slug: CriterionFit(sigma=sigma, prior_weights=flat, n_grades=0, n_runs=0)
            for slug in ("a", "b", "c")
        },
        pooled_sigma=sigma, input_hash="t", fitted_at="t",
    )


def test_all_deterministic_is_a_point_mass():
    pairs = [(a_criterion("a", 3), determined("a", 8.0)), (a_criterion("b", 1), determined("b", 4.0))]
    got = score_posterior(pairs, a_fit(), threshold=6.0, seed=1)
    assert got.mean == got.low == got.high == 7.0
    assert got.p_hopeful == 1.0


def test_a_model_grade_carries_judge_noise():
    pairs = [(a_criterion("a", 1), determined("a", 7.0, graded_by="model:m"))]
    got = score_posterior(pairs, a_fit(sigma=1.5), threshold=6.0, seed=1)
    assert got.low < 7.0 < got.high
    assert 0.5 < got.p_hopeful < 1.0


def test_an_unknown_widens_by_its_weight():
    heavy = [(a_criterion("a", 5), silent("a")), (a_criterion("b", 1), determined("b", 8.0))]
    light = [(a_criterion("a", 1), silent("a")), (a_criterion("b", 5), determined("b", 8.0))]
    wide = score_posterior(heavy, a_fit(), threshold=6.0, seed=1)
    narrow = score_posterior(light, a_fit(), threshold=6.0, seed=1)
    assert (wide.high - wide.low) > (narrow.high - narrow.low)
    assert wide.width_shares["a"] > 0.9


def test_silence_prior_shifts_advert_silence_only():
    prior = {0.0: 0.9, 10.0: 0.1}
    advert = [(a_criterion("a", 1, silence_prior=prior), silent("a", graded_by="model:m"))]
    extraction = [(a_criterion("a", 1, silence_prior=prior), silent("a", graded_by="map"))]
    assert (
        score_posterior(advert, a_fit(), threshold=6.0, seed=1).mean
        < score_posterior(extraction, a_fit(), threshold=6.0, seed=1).mean
    )


def test_a_silence_prior_within_loader_tolerance_does_not_crash_rng_choice():
    # The loader accepts a silence_prior summing within 1e-6 of 1.0; rng.choice
    # wants ~1.5e-8. This must be normalized here, not tightened in the loader.
    prior = {0.0: 0.3333333, 10.0: 0.6666666}
    pairs = [(a_criterion("a", 1, silence_prior=prior), silent("a", graded_by="model:m"))]
    got = score_posterior(pairs, a_fit(), threshold=6.0, seed=1)
    assert 0.0 < got.mean < 10.0


def test_a_failed_model_call_is_not_advert_silence():
    prior = {0.0: 1.0}
    pairs = [(a_criterion("a", 1, silence_prior=prior), silent("a", graded_by="model:m (failed)"))]
    got = score_posterior(pairs, a_fit(), threshold=6.0, seed=1)
    assert got.mean > 1.0  # the empirical prior, not the damning silence prior


def test_a_default_is_a_point_not_a_distribution():
    pairs = [(a_criterion("a", 1), Grade("a", 4.0, False, "default", graded_by="default"))]
    got = score_posterior(pairs, a_fit(), threshold=6.0, seed=1)
    assert got.mean == got.low == got.high == 4.0


def test_a_criterion_outside_the_fit_falls_back_to_pooled_sigma_and_uniform_prior():
    pairs = [(a_criterion("outside", 1), silent("outside"))]
    got = score_posterior(pairs, a_fit(), threshold=6.0, seed=1)
    assert 4.0 < got.mean < 6.0


def test_every_criterion_unknown_gives_the_prior_predictive():
    pairs = [(a_criterion("a", 1), silent("a"))]
    got = score_posterior(pairs, a_fit(), threshold=6.0, seed=1)
    assert 4.0 < got.mean < 6.0  # uniform prior mean is 5


def test_scoring_is_deterministic_by_seed():
    pairs = [(a_criterion("a", 1), silent("a"))]
    one = score_posterior(pairs, a_fit(), threshold=6.0, seed=42)
    two = score_posterior(pairs, a_fit(), threshold=6.0, seed=42)
    assert (one.mean, one.low, one.high, one.p_hopeful) == (two.mean, two.low, two.high, two.p_hopeful)


def test_verdict_gates_on_probability_not_mean():
    sure = [(a_criterion("a", 1), determined("a", 6.5))]
    unsure = [(a_criterion("a", 5), silent("a")), (a_criterion("b", 1), determined("b", 10.0))]
    hopeful = adjudicate_bayesian(sure, a_fit(), hopeful_threshold=6.0, hopeful_confidence=0.6, seed=1)
    held = adjudicate_bayesian(unsure, a_fit(), hopeful_threshold=6.0, hopeful_confidence=0.6, seed=1)
    assert hopeful.verdict == "hopeful"
    assert held.verdict == "borderline"
    assert any("unknown" in flag.lower() for flag in held.red_flags)


def test_a_veto_still_fires_and_an_unknown_veto_still_does_not():
    veto = a_criterion("a", 1, veto_at_or_below=0.0)
    fired = adjudicate_bayesian(
        [(veto, determined("a", 0.0))], a_fit(),
        hopeful_threshold=6.0, hopeful_confidence=0.6, seed=1,
    )
    spared = adjudicate_bayesian(
        [(veto, silent("a"))], a_fit(),
        hopeful_threshold=6.0, hopeful_confidence=0.6, seed=1,
    )
    assert fired.verdict == "reject"
    assert spared.verdict != "reject"


def test_questions_go_out_in_voi_order():
    # The heavy unknown decides the Verdict; its question must lead.
    pairs = [
        (a_criterion("a", 1, ask="Small thing?"), silent("a")),
        (a_criterion("b", 5, ask="Big thing?"), silent("b")),
        (a_criterion("c", 3), determined("c", 8.0)),
    ]
    got = adjudicate_bayesian(pairs, a_fit(), hopeful_threshold=6.0, hopeful_confidence=0.6, seed=1)
    assert got.agent_questions[0] == "Big thing?"


def test_unknown_flag_sums_share_over_the_named_criteria_only():
    # Four unknowns, only the top SHOWN=3 are named; the sentence must not
    # claim they carry the fourth's share too.
    pairs = [(a_criterion(s, 1), silent(s)) for s in "abcd"]
    posterior = Posterior(
        mean=5.0, low=0.0, high=10.0, p_hopeful=0.5,
        width_shares={"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1},
        voi={},
    )
    flag = _unknown_flag(pairs, posterior)
    named = flag.split(" - ")[1].split(" carry")[0]
    assert named == "A, B, C"
    assert "90%" in flag


def test_unknown_flag_never_names_a_configured_default():
    # A default has a value and zero variance: it is undetermined but it is
    # not a source of uncertainty, so it must not be named here.
    pairs = [
        (a_criterion("a", 1), silent("a")),
        (a_criterion("b", 1), Grade("b", 4.0, False, "default", graded_by="default")),
    ]
    posterior = Posterior(
        mean=5.0, low=0.0, high=10.0, p_hopeful=0.5,
        width_shares={"a": 1.0, "b": 0.0},
        voi={},
    )
    flag = _unknown_flag(pairs, posterior)
    named = flag.split(" - ")[1].split(" carry")[0]
    assert named == "A"


def test_unknown_flag_names_noisy_grades_when_they_dominate_a_partial_read():
    """One small unknown must not hide the model Grade making the interval wide."""
    pairs = [
        (a_criterion("a", 1), silent("a")),
        (a_criterion("b", 1), determined("b", 5.0, graded_by="model:test")),
    ]
    posterior = Posterior(
        mean=5.0,
        low=2.0,
        high=8.0,
        p_hopeful=0.4,
        width_shares={"a": 0.1, "b": 0.9},
        voi={},
    )

    flag = _unknown_flag(pairs, posterior)

    assert flag.startswith("Too uncertain to call")
    assert "B, A carry 100%" in flag


def test_the_posterior_fields_reach_the_evaluation():
    pairs = [(a_criterion("a", 1), determined("a", 7.0, graded_by="model:m"))]
    got = adjudicate_bayesian(pairs, a_fit(), hopeful_threshold=6.0, hopeful_confidence=0.6, seed=1)
    assert got.score_low is not None and got.score_high is not None
    assert got.p_hopeful is not None
    assert got.posterior_fit_hash == "t"
    assert got.score is not None and got.score_low <= got.score <= got.score_high


def test_a_narrow_borderline_result_says_it_is_confidently_mediocre():
    got = adjudicate_bayesian(
        [(a_criterion("a", 1), determined("a", 5.0))],
        a_fit(),
        hopeful_threshold=7.0,
        hopeful_confidence=0.6,
        seed=1,
    )
    assert got.verdict == "borderline"
    assert any("confidently mediocre" in flag.lower() for flag in got.red_flags)


def test_a_wide_model_only_result_names_uncertainty_without_an_empty_source_list():
    got = adjudicate_bayesian(
        [(a_criterion("a", 1), determined("a", 5.0, graded_by="model:test"))],
        a_fit(sigma=5.0),
        hopeful_threshold=8.0,
        hopeful_confidence=0.9,
        seed=1,
    )
    assert got.verdict == "borderline"
    assert any("too uncertain to call" in flag.lower() for flag in got.red_flags)
    assert all("-  carry" not in flag for flag in got.red_flags)
