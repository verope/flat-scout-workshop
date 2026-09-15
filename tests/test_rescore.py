"""Replaying an edited brief over the Listings already in hand.

The point of the command is that a brief edit is otherwise untestable: an
evaluated Listing is never evaluated again, so criteria.md can only be judged
against Listings that have not arrived yet. The properties worth holding are
that it writes nothing, that it reuses the stored image reading rather than
paying to look again, and that noise is not reported as movement.
"""

from __future__ import annotations

import pytest
from pydantic_ai.models.test import TestModel

from flat_scout.config import Settings
from flat_scout.db import Database
from flat_scout.models import Evaluation, ImageReading, ListingData
from flat_scout.pipeline import rescore
from flat_scout.report import MOVED, Rescored, crossings, rescore_report


def settings_for(tmp_path, brief: str = "Bike storage matters.") -> Settings:
    settings = Settings()
    # The holistic rescore, which is no longer the default path. Pinned rather
    # than inherited: the weighted rescue is `_rescore_weighted` and has its
    # own tests, and a test that reaches a path by omission starts testing a
    # different path the day the default moves.
    settings.features.weighted_criteria = False
    settings.evaluation.criteria_path = str(tmp_path / "criteria.md")
    (tmp_path / "criteria.md").write_text(brief)
    return settings


def evaluated(
    db: Database,
    portal_id: str,
    score: float,
    *,
    verdict: str = "borderline",
    **fields,
) -> int:
    listing_id = db.upsert(
        ListingData(portal="rightmove", portal_id=portal_id, url=f"u{portal_id}", **fields),
        source="alert",
    )
    db.set_evaluation(
        listing_id,
        Evaluation(verdict=verdict, score=score, reasons=["as it stood"]),
        "openrouter:z-ai/glm-5.3-flash",
    )
    db.transition(listing_id, "evaluated")
    return listing_id


def model_scoring(score: float, verdict: str = "borderline") -> TestModel:
    return TestModel(
        custom_output_args={
            "verdict": verdict,
            "score": score,
            "reasons": ["under the new brief"],
        }
    )


@pytest.mark.asyncio
async def test_rescore_judges_every_scored_listing_again(tmp_path):
    db = Database(tmp_path / "flats.db")
    evaluated(db, "a", 2.5)
    evaluated(db, "b", 7.5)
    changes = await rescore(db, settings_for(tmp_path), model=model_scoring(6.0))
    assert sorted(change.was for change in changes) == [2.5, 7.5]
    assert {change.now for change in changes} == {6.0}


@pytest.mark.asyncio
async def test_rescore_writes_nothing(tmp_path):
    """The stored score is evidence a Decision was taken against.

    Overwriting it would leave `report --decisions` holding a new score against
    an old Decision, which is a comparison with nothing behind it.
    """
    db = Database(tmp_path / "flats.db")
    listing_id = evaluated(db, "a", 2.5)
    await rescore(db, settings_for(tmp_path), model=model_scoring(9.0))
    row = db.get(listing_id)
    assert row["score"] == 2.5
    assert row["status"] == "evaluated"
    assert "under the new brief" not in (row["reasons"] or "")


@pytest.mark.asyncio
async def test_rescore_leaves_out_listings_that_were_never_scored(tmp_path):
    db = Database(tmp_path / "flats.db")
    evaluated(db, "a", 2.5)
    db.upsert(ListingData(portal="rightmove", portal_id="b", url="ub"), source="alert")
    changes = await rescore(db, settings_for(tmp_path), model=model_scoring(6.0))
    assert len(changes) == 1


@pytest.mark.asyncio
async def test_rescore_reuses_the_stored_image_reading(tmp_path):
    """No vision spend, and the same evidence the first evaluation had.

    A rescore that dropped the reading would score every Listing against less
    than the original saw, and report the loss as the work of the brief.
    """
    db = Database(tmp_path / "flats.db")
    listing_id = evaluated(db, "a", 2.5)
    db.set_image_reading(
        listing_id,
        ImageReading(epc_band="C", epc_band_source="graph", layout_verdict="good"),
        "openrouter:z-ai/glm-5.3-flash",
    )
    import flat_scout.pipeline as pipeline

    seen: list[ImageReading | None] = []

    async def spy(listing, criteria, settings, model=None, reading=None):
        seen.append(reading)
        return Evaluation(verdict="borderline", score=6.0, reasons=["fine"])

    original = pipeline.evaluate_listing
    pipeline.evaluate_listing = spy
    try:
        await rescore(db, settings_for(tmp_path))
    finally:
        pipeline.evaluate_listing = original
    assert len(seen) == 1
    assert seen[0] is not None and seen[0].epc_band == "C"


@pytest.mark.asyncio
async def test_one_failed_listing_does_not_lose_the_run(tmp_path):
    db = Database(tmp_path / "flats.db")
    evaluated(db, "a", 2.5)
    evaluated(db, "b", 3.5)
    calls = {"n": 0}
    import flat_scout.pipeline as pipeline

    async def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("the provider fell over")
        return Evaluation(verdict="borderline", score=6.0, reasons=["fine"])

    original = pipeline.evaluate_listing
    pipeline.evaluate_listing = flaky
    try:
        changes = await rescore(db, settings_for(tmp_path))
    finally:
        pipeline.evaluate_listing = original
    assert len(changes) == 1


@pytest.mark.asyncio
async def test_rescore_can_be_limited_to_a_handful(tmp_path):
    db = Database(tmp_path / "flats.db")
    for index in range(5):
        evaluated(db, str(index), 2.0 + index)
    changes = await rescore(db, settings_for(tmp_path), model=model_scoring(6.0), limit=2)
    # Highest first, so a limited run samples the top of the ranking rather
    # than whatever happens to have the lowest id.
    assert sorted(change.was for change in changes) == [5.0, 6.0]


def change(listing_id: int, was: float, now: float | None, **fields) -> Rescored:
    return Rescored(
        listing_id=listing_id,
        was=was,
        now=now,
        was_verdict=fields.get("was_verdict", "borderline"),
        now_verdict=fields.get("now_verdict", "borderline"),
        address=fields.get("address", "1 Test Street"),
        postcode="SW11",
        price_pcm=2000,
        reasons=["a reason"],
    )


def test_the_report_leaves_out_movement_too_small_to_mean_anything(tmp_path):
    text = rescore_report([change(1, 2.5, 2.5 + MOVED / 2)], threshold=3.0)
    assert "0 moved" in text
    assert "#1" not in text


def test_the_report_leads_with_the_listing_that_moved_furthest(tmp_path):
    changes = [change(1, 2.5, 3.5), change(2, 2.5, 6.5), change(3, 5.0, 5.0)]
    text = rescore_report(changes, threshold=3.0)
    assert "2 moved" in text
    assert text.index("#2") < text.index("#1")


def test_the_report_says_which_way_a_listing_moved(tmp_path):
    text = rescore_report([change(1, 7.0, 2.0)], threshold=3.0)
    assert "down 5.0" in text


def test_crossings_count_the_listings_a_brief_edit_would_show(tmp_path):
    gained, lost = crossings(
        [change(1, 2.5, 6.0), change(2, 7.0, 1.0), change(3, 4.0, 5.0)], threshold=3.0
    )
    assert (gained, lost) == (1, 1)


def test_a_verdict_carries_a_listing_across_without_the_score(tmp_path):
    # `evaluate_weighted` surfaces a hopeful Verdict whatever the score, so a
    # brief that changes only the Verdict still changes what arrives.
    gained, lost = crossings(
        [change(1, 1.0, 1.0, now_verdict="hopeful")], threshold=3.0
    )
    assert (gained, lost) == (1, 0)


def test_a_regrade_with_no_new_score_crosses_neither_direction(tmp_path):
    """The weighted path can regrade a Listing down to no score at all - a
    Criterion vector with nothing determined adjudicates to `score=None` - and
    a Listing that crossed nothing cannot be counted on either side.

    Without the `change.now is not None` guard in `crossings`, this raises:
    `was=1.0` is below the threshold, so the "gained" arm evaluates
    `would_be_shown(Decided(None, ...))`, which compares `None >= 3.0`.
    `was=7.0` is above it, so the "lost" arm hits the same comparison from the
    other side. Both rows are needed to exercise both arms of the tuple.
    """
    gained, lost = crossings(
        [change(1, 1.0, None), change(2, 7.0, None)], threshold=3.0
    )
    assert (gained, lost) == (0, 0)


def test_the_report_says_when_an_edit_changed_nothing_that_matters(tmp_path):
    text = rescore_report([change(1, 5.0, 6.0)], threshold=3.0)
    assert "No Listing crosses" in text


def test_the_report_does_not_crash_on_a_listing_regraded_to_no_score(tmp_path):
    """`Rescored.moved` subtracts `was` from `now`. Without the guard in
    `rescore_report` that excludes a `None` `now` from the "moved" selection,
    `abs(change.moved)` raises the same TypeError `crossings` used to raise -
    None has no `-`. The Listing is still counted in the total; it is just not
    listed as having moved, because there is nothing to compare it against.
    """
    text = rescore_report([change(1, 7.0, None)], threshold=3.0)
    assert "Rescored 1 Listings" in text
    assert "#1" not in text


def test_the_report_says_so_when_there_is_nothing_to_rescore(tmp_path):
    assert "Nothing to rescore" in rescore_report([], threshold=3.0)


def weighted_settings() -> Settings:
    settings = Settings()
    settings.features.weighted_criteria = True
    return settings


def a_corpus_led_by_unreportable_rows(tmp_path, unreportable: int = 3) -> Database:
    """Rows `rescore` can never report on, sitting on the low ids.

    `_rescore_weighted` drops every change whose `was` is None, so a Listing
    that was never scored produces nothing however it regrades. Two of these
    are terminal `prefiltered_out` rows and one is simply never evaluated;
    all three sort ahead of the real corpus by id.
    """
    db = Database(tmp_path / "flats.db")
    for index in range(unreportable):
        listing_id = db.upsert(
            ListingData(
                portal="rightmove", portal_id=f"skipped{index}", url=f"d{index}",
                postcode="SW11", price_pcm=3000, sqft=600,
                description="A modern flat.",
            ),
            source="alert",
        )
        if index < 2:
            db.transition(listing_id, "prefiltered_out")
    for portal_id, score in (("live0", 8.0), ("live1", 6.0), ("live2", 4.0)):
        evaluated(
            db, portal_id, score,
            postcode="SW11", price_pcm=3000, sqft=600, description="A modern flat.",
        )
    return db


@pytest.mark.asyncio
async def test_the_weighted_rescore_limit_counts_listings_it_can_report_on(tmp_path):
    """`--limit 2` must buy two reported Listings, not two rows off the top.

    `backfill_grades` walks every row in id order and applies `limit` before
    anything filters, so `rescore --limit` used to spend its whole budget on
    `prefiltered_out` and never-scored rows - paying for model calls whose
    results `_rescore_weighted` then throws away for having no `was` to compare
    against, and reporting nothing at all.
    """
    db = a_corpus_led_by_unreportable_rows(tmp_path)
    model = TestModel(custom_output_args={"grade": 7.0, "evidence": "fine"})
    changes = await rescore(db, weighted_settings(), model=model, limit=2)
    assert len(changes) == 2


@pytest.mark.asyncio
async def test_the_weighted_rescore_does_not_grade_what_it_cannot_report(tmp_path):
    """The waste is the point, not only the empty report: every model call on
    an unreportable row is spend against a result that is thrown away."""
    from flat_scout import pipeline as pipeline_module

    graded: list[str] = []
    real = pipeline_module.grade_listing

    async def capture(criteria, listing, reading, settings, preamble, model=None, gate=None):
        graded.append(listing.url)
        return await real(criteria, listing, reading, settings, preamble, model, gate=gate)

    db = a_corpus_led_by_unreportable_rows(tmp_path)
    model = TestModel(custom_output_args={"grade": 7.0, "evidence": "fine"})
    import unittest.mock

    with unittest.mock.patch.object(pipeline_module, "grade_listing", capture):
        await rescore(db, weighted_settings(), model=model, limit=2)

    assert graded, "grade_listing was never called - the fixture reaches nothing"
    assert not [url for url in graded if url.startswith("d")]


@pytest.mark.asyncio
async def test_the_weighted_rescore_without_a_limit_still_reports_the_whole_corpus(
    tmp_path,
):
    """Scoping the population must not narrow an unlimited run."""
    db = a_corpus_led_by_unreportable_rows(tmp_path)
    model = TestModel(custom_output_args={"grade": 7.0, "evidence": "fine"})
    changes = await rescore(db, weighted_settings(), model=model)
    assert sorted(change.was for change in changes) == [4.0, 6.0, 8.0]
