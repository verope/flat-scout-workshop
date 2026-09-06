import pytest

from flat_scout.adjudicate import Grade
from flat_scout.criteria import Criterion
from flat_scout.db import Database
from flat_scout.fit import (
    DEFAULT_POOLED_SIGMA,
    GRADE_BINS,
    CriterionFit,
    Fit,
    determined_grades,
    empirical_priors,
    fit_input_hash,
    fit_sigmas,
    load_fit,
    run_fit,
)
from flat_scout.models import ListingData


def a_criterion() -> Criterion:
    return Criterion(
        slug="quiet",
        name="Quiet",
        description="d",
        weight=1,
        grade={"model": True},
        body="rubric",
        rubric_hash="rubric",
    )


def add_graded_listing(
    db: Database, criterion: Criterion, portal_id: str, grade: float, *, evaluated: bool
) -> int:
    listing_id = db.upsert(
        ListingData(portal="rightmove", portal_id=portal_id, url=f"u{portal_id}"),
        source="manual",
    )
    if evaluated:
        db.conn.execute(
            "UPDATE listings SET evaluated_at = ? WHERE id = ?",
            ("2026-08-21T12:00:00+00:00", listing_id),
        )
        db.conn.commit()
    db.set_criterion_grades(
        listing_id,
        [(criterion, Grade(criterion.slug, grade, True, "read", graded_by="model:test"))],
    )
    return listing_id


def test_priors_keep_a_bimodal_criterion_bimodal():
    # 7 stated no-pets, 3 pet-friendly operators.
    weights = empirical_priors({"pets": [0.0] * 7 + [10.0] * 3})["pets"]
    assert len(weights) == len(GRADE_BINS) == 11
    assert abs(sum(weights) - 1.0) < 1e-9
    assert weights[0] > weights[5] and weights[10] > weights[5]


def test_priors_pool_toward_uniform_when_thin():
    weights = empirical_priors({"quiet": [8.0]})["quiet"]
    # One observation shifts the prior; it must not dominate it.
    assert max(weights) < 0.5
    assert min(weights) > 0.0


def test_priors_for_an_ungraded_criterion_are_near_uniform():
    weights = empirical_priors({"new-thing": []})["new-thing"]
    assert all(abs(w - 1 / 11) < 1e-9 for w in weights)


def test_off_scale_grades_land_in_the_nearest_bin():
    weights = empirical_priors({"aspect": [7.4]})["aspect"]
    assert weights[7] == max(weights)


def test_fit_cache_round_trip(tmp_path):
    db = Database(tmp_path / "flats.db")
    assert db.fit_cache("deadbeef") is None
    db.set_fit_cache("deadbeef", '{"pooled_sigma": 1.5}')
    assert db.fit_cache("deadbeef") == '{"pooled_sigma": 1.5}'


def test_fit_hash_changes_when_the_model_version_moves(tmp_path, monkeypatch):
    from flat_scout import fit as fit_module

    db = Database(tmp_path / "flats.db")
    criterion = a_criterion()
    before = fit_input_hash([criterion], db)

    monkeypatch.setattr(
        fit_module, "FIT_MODEL_VERSION", fit_module.FIT_MODEL_VERSION + 1
    )

    assert fit_input_hash([criterion], db) != before


def test_empirical_priors_only_read_the_corpus_that_passed_the_filters(tmp_path):
    db = Database(tmp_path / "flats.db")
    criterion = a_criterion()
    add_graded_listing(db, criterion, "seen", 8.0, evaluated=True)
    add_graded_listing(db, criterion, "stopped", 1.0, evaluated=False)

    assert determined_grades(db, [criterion]) == {"quiet": [8.0]}


def test_empirical_priors_exclude_grades_from_a_stale_rubric(tmp_path):
    from dataclasses import replace

    db = Database(tmp_path / "flats.db")
    old = a_criterion()
    add_graded_listing(db, old, "old", 9.0, evaluated=True)
    live = replace(old, rubric_hash="rewritten")

    assert determined_grades(db, [live]) == {"quiet": []}


def test_no_runs_means_pooled_default():
    sigmas, pooled = fit_sigmas({})
    assert sigmas == {} and pooled == DEFAULT_POOLED_SIGMA


@pytest.mark.slow
def test_fit_recovers_a_noisy_and_a_steady_criterion():
    import numpy as np
    rng = np.random.default_rng(0)
    runs = {
        "steady": [list(5 + rng.normal(0, 0.3, 3)) for _ in range(8)],
        "shaky": [list(5 + rng.normal(0, 2.5, 3)) for _ in range(8)],
    }
    sigmas, pooled = fit_sigmas(runs)
    assert sigmas["shaky"] > 2 * sigmas["steady"]
    assert 0 < pooled < 5


@pytest.mark.slow
def test_load_fit_reuses_the_cache(tmp_path, monkeypatch):
    from flat_scout import fit as fit_module
    db = Database(tmp_path / "flats.db")
    criteria = []  # no Criteria: empty fit, but the cache contract is the same
    first = load_fit(db, criteria)
    calls = []
    monkeypatch.setattr(fit_module, "fit_sigmas", lambda runs: calls.append(1) or ({}, 1.5))
    second = load_fit(db, criteria)
    assert second.input_hash == first.input_hash
    assert calls == []  # cache hit: no sampling


def test_scoring_keeps_the_last_good_fit_until_the_scheduled_refit(tmp_path, monkeypatch):
    from flat_scout import fit as fit_module

    db = Database(tmp_path / "flats.db")
    criterion = a_criterion()
    add_graded_listing(db, criterion, "1", 7.0, evaluated=True)
    first = load_fit(db, [criterion])

    add_graded_listing(db, criterion, "2", 8.0, evaluated=True)

    def sampling_here_would_block_the_listing(*_args):
        raise AssertionError("live scoring must not refit")

    monkeypatch.setattr(fit_module, "fit_sigmas", sampling_here_would_block_the_listing)
    still_current = load_fit(db, [criterion])
    assert still_current.input_hash == first.input_hash

    monkeypatch.setattr(fit_module, "fit_sigmas", lambda _runs: ({}, 1.5))
    refreshed = load_fit(db, [criterion], refit=True)
    assert refreshed.input_hash != first.input_hash
    assert refreshed.criteria[criterion.slug].n_grades == 2


def test_live_scoring_never_samples_when_measure_runs_exist_without_a_cache(
    tmp_path, monkeypatch
):
    from flat_scout import fit as fit_module

    db = Database(tmp_path / "flats.db")
    criterion = a_criterion()
    db.add_measure_runs(1, criterion.slug, criterion.rubric_hash, [4.0, 6.0])

    def sampling_here_would_block_the_listing(*_args):
        raise AssertionError("live scoring must not start PyMC")

    monkeypatch.setattr(fit_module, "fit_sigmas", sampling_here_would_block_the_listing)
    with pytest.raises(RuntimeError, match="flat-scout fit"):
        load_fit(db, [criterion])


def test_run_fit_hashes_the_same_snapshot_that_it_samples(tmp_path, monkeypatch):
    from flat_scout import fit as fit_module

    db = Database(tmp_path / "flats.db")
    criterion = a_criterion()
    add_graded_listing(db, criterion, "1", 7.0, evaluated=True)

    def mutate_after_the_snapshot(_runs):
        add_graded_listing(db, criterion, "2", 8.0, evaluated=True)
        return {}, 1.5

    monkeypatch.setattr(fit_module, "fit_sigmas", mutate_after_the_snapshot)
    fitted = run_fit(db, [criterion])

    assert fitted.criteria[criterion.slug].n_grades == 1
    assert fitted.input_hash != fit_input_hash([criterion], db)


def test_the_fit_table_prints_the_categorical_prior_not_only_its_mean():
    from flat_scout.report import fit_report

    weights = (0.7, *([0.0] * 9), 0.3)
    fitted = Fit(
        criteria={
            "quiet": CriterionFit(
                sigma=1.2, prior_weights=weights, n_grades=10, n_runs=30
            )
        },
        pooled_sigma=1.2,
        input_hash="hash",
        fitted_at="now",
    )

    text = fit_report(fitted, [a_criterion()])
    assert "0:70%" in text
    assert "10:30%" in text
