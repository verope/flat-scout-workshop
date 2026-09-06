from pathlib import Path

import httpx
import pytest
import respx
from pydantic_ai.models.test import TestModel

from flat_scout.adjudicate import Grade, adjudicate
from flat_scout.config import Settings
from flat_scout.criteria import Criterion
from flat_scout.db import Database
from flat_scout.models import ListingData
from flat_scout.pipeline import (
    adjudicate_current,
    evaluate_weighted,
    process_url,
)

# Same fixture and URL as tests/test_pipeline.py's process_url tests - a
# genuine downloadable Rightmove listing under the filters' price and beds
# caps, so it reaches "evaluated" with nothing but the postcode filter opened up.
FIXTURES = Path(__file__).parent / "fixtures"
URL = "https://www.rightmove.co.uk/properties/92053296"
GOOD_HTML = (FIXTURES / "rightmove_listing.html").read_text()


def a_listing(**kwargs) -> ListingData:
    return ListingData(
        portal="rightmove", portal_id="1", url="u", postcode="SW11",
        price_pcm=3000, sqft=600, nearest_station_miles=0.4,
        description="A modern flat with a secure bike store.", **kwargs,
    )


def a_db(tmp_path) -> Database:
    return Database(tmp_path / "flats.db")


@pytest.mark.asyncio
async def test_the_weighted_path_scores_stores_and_returns(tmp_path):
    db = a_db(tmp_path)
    listing = a_listing()
    listing_id = db.upsert(listing, source="manual")
    model = TestModel(custom_output_args={"grade": 7.0, "evidence": "seems fine"})

    result = await evaluate_weighted(db, Settings(), listing_id, listing, None, model=model)

    assert result.score is not None
    assert 0 < result.coverage <= 1
    assert result.verdict in {"hopeful", "borderline", "reject"}
    stored = db.criterion_grades(listing_id)
    assert "pets" in stored and "floor-area" in stored
    assert len(stored) == 5


@pytest.mark.asyncio
async def test_the_stored_grades_carry_the_rubric_hash_of_the_files_on_disk(tmp_path):
    from pathlib import Path

    from flat_scout.criteria import load_criteria
    from flat_scout.graders import GRADERS

    db = a_db(tmp_path)
    listing = a_listing()
    listing_id = db.upsert(listing, source="manual")
    model = TestModel(custom_output_args={"grade": 7.0, "evidence": "fine"})
    await evaluate_weighted(db, Settings(), listing_id, listing, None, model=model)

    criteria = load_criteria(Path("criteria"), known_graders=set(GRADERS))
    assert db.stale_criteria(listing_id, criteria) == []


@pytest.mark.asyncio
@respx.mock
async def test_process_url_dispatches_to_the_weighted_path_when_the_flag_is_on(tmp_path):
    """The flag's only production effect is the branch inside `process_url` -
    every other test in this file drives `evaluate_weighted` or
    `adjudicate_current` directly, which proves nothing about that branch
    actually being taken on a real ingest. `Evaluation.coverage` defaults to
    0.0 and the holistic path never sets it (see evaluate.py), so a stored row
    with non-zero coverage can only have come through `evaluate_weighted`; the
    stored grade vector confirms the storage side of it ran too.
    """
    respx.get(URL).mock(return_value=httpx.Response(200, html=GOOD_HTML))
    db = a_db(tmp_path)
    settings = Settings()
    settings.fetch.delay_seconds = 0
    settings.filters.postcodes = []  # the fixture is in E9, outside the shortlist
    settings.features.weighted_criteria = True
    model = TestModel(custom_output_args={"grade": 7.0, "evidence": "seems fine"})

    async with httpx.AsyncClient() as client:
        listing_id = await process_url(URL, db, settings, "manual", client, model=model)

    row = db.get(listing_id)
    assert row["status"] == "evaluated"
    assert row["evaluation_mode"] == "weighted"
    assert row["coverage"] > 0.0
    assert db.criterion_grades(listing_id) != {}


def test_flag_off_keeps_the_point_path(tmp_path):
    """`bayesian_verdict = False` must be byte-identical to `adjudicate` today."""
    db = a_db(tmp_path)
    listing_id = db.upsert(a_listing(), source="manual")
    settings = Settings()
    settings.features.bayesian_verdict = False
    criterion = Criterion(
        slug="quality", name="Quality", description="d", weight=1,
        grade={"model": True}, body="rubric",
    )
    pairs = [(criterion, Grade("quality", 8.0, True, "Good."))]

    result = adjudicate_current(db, pairs, settings, [criterion], listing_id)

    expected = adjudicate(
        pairs,
        hopeful_threshold=settings.evaluation.hopeful_threshold,
        coverage_floor=settings.evaluation.coverage_floor,
        unknown_prior=settings.evaluation.unknown_prior,
    )
    assert result == expected
    assert result.score_low is None


def test_flag_on_gates_on_probability(tmp_path):
    """On, the Verdict carries the posterior fields and replays by seed."""
    from dataclasses import replace

    from flat_scout.fit import load_fit
    from flat_scout.posterior import adjudicate_bayesian

    db = a_db(tmp_path)
    listing_id = db.upsert(a_listing(), source="manual")
    other_listing_id = db.upsert(replace(a_listing(), portal_id="2"), source="manual")
    settings = Settings()
    settings.features.bayesian_verdict = True
    criterion = Criterion(
        slug="quality", name="Quality", description="d", weight=1,
        grade={"model": True}, body="rubric",
    )
    # graded_by="model:..." carries judge noise (unlike a deterministic
    # grader's point value), so the posterior actually spreads and the seed
    # has something to draw differently - see test_posterior.py's
    # test_a_model_grade_carries_judge_noise.
    pairs = [(criterion, Grade("quality", 8.0, True, "Good.", graded_by="model:test"))]

    first = adjudicate_current(db, pairs, settings, [criterion], listing_id)
    second = adjudicate_current(db, pairs, settings, [criterion], listing_id)
    different_seed = adjudicate_current(db, pairs, settings, [criterion], other_listing_id)

    assert first.p_hopeful is not None
    assert first.score_low is not None
    assert first.score_high is not None
    assert first == second

    # The seed IS the Listing id, not an incidental constant: a direct
    # `adjudicate_bayesian` call seeded on `listing_id` reproduces
    # `adjudicate_current`'s numbers exactly, and a different Listing id
    # draws a different interval.
    direct = adjudicate_bayesian(
        pairs,
        load_fit(db, [criterion]),
        hopeful_threshold=settings.evaluation.hopeful_threshold,
        hopeful_confidence=settings.evaluation.hopeful_confidence,
        seed=listing_id,
    )
    assert first == direct
    assert (first.score_low, first.score_high) != (
        different_seed.score_low,
        different_seed.score_high,
    )
