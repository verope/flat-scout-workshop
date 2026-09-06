import asyncio
import logging
import sqlite3
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from flat_scout.config import Settings
from flat_scout.criteria import PREAMBLE_FILES
from flat_scout.db import Database
from flat_scout.models import Evaluation, ListingData
from flat_scout.pipeline import backfill_grades


def a_db_with_two_listings(tmp_path) -> Database:
    db = Database(tmp_path / "flats.db")
    for portal_id in ("1", "2"):
        db.upsert(
            ListingData(
                portal="rightmove", portal_id=portal_id, url=f"u{portal_id}",
                postcode="SW11", price_pcm=3000, sqft=600,
                description="A modern flat.",
            ),
            source="manual",
        )
    return db


@pytest.mark.asyncio
async def test_the_backfill_grades_every_listing(tmp_path):
    db = a_db_with_two_listings(tmp_path)
    model = TestModel(custom_output_args={"grade": 7.0, "evidence": "fine"})
    done = await backfill_grades(db, Settings(), model=model)
    assert len(done) == 2
    assert all(row.coverage > 0 for row in done)


@pytest.mark.asyncio
async def test_calls_counts_what_actually_happened_not_what_might_have(tmp_path):
    """`aspect` is `fallback: model`, but a plan with a compass on it
    lets `best_aspect` resolve on its own - it never reaches the model.
    Counting every Criterion merely *eligible* to reach the model reports 4
    (3 `model: true` + 1 `fallback: model`); only 3 of those are real spend,
    since aspect's fallback is never taken.
    """
    from flat_scout.models import ImageReading

    db = a_db_with_two_listings(tmp_path)
    for row in db.conn.execute("SELECT id FROM listings").fetchall():
        db.set_image_reading(row["id"], ImageReading(window_aspects=["N"]), model="test")
    model = TestModel(custom_output_args={"grade": 7.0, "evidence": "fine"})
    done = await backfill_grades(db, Settings(), model=model)
    assert all(row.calls == 3 for row in done)


@pytest.mark.asyncio
async def test_the_backfill_writes_no_score_and_no_verdict(tmp_path):
    """listings.score is what the stored result said. It is never rewritten.

    A fixture that starts both columns NULL only proves the backfill does not
    *newly populate* them. The far more important direction is the opposite:
    that it *preserves* a score the couple already decided against - the
    scenario that matters, where a Listing approved at 8.0 would end up
    compared against a number that no longer exists anywhere. So one Listing
    is seeded with a real score and Verdict first, the way a Listing that has
    already been through a result and a Decision actually looks.
    """
    db = a_db_with_two_listings(tmp_path)
    seeded_id = db.conn.execute("SELECT id FROM listings ORDER BY id LIMIT 1").fetchone()["id"]
    db.set_evaluation(
        seeded_id,
        Evaluation(verdict="hopeful", score=8.0, coverage=0.9, reasons=["good bones"]),
        model="test",
    )

    model = TestModel(custom_output_args={"grade": 7.0, "evidence": "fine"})
    await backfill_grades(db, Settings(), model=model)

    seeded = db.get(seeded_id)
    assert seeded["score"] == 8.0
    assert seeded["verdict"] == "hopeful"
    for row in db.conn.execute("SELECT score, verdict FROM listings WHERE id != ?", (seeded_id,)):
        assert row["score"] is None
        assert row["verdict"] is None


class Counting(TestModel):
    """A TestModel that counts requests instead of raising on one.

    A raising stand-in is unreliable here: `grading.by_model` wraps the model
    call in a broad `except Exception`, and an `AssertionError` is an
    `Exception` - the raise would be swallowed and the code would fall
    through to its normal "model unreachable" path, which is indistinguishable
    from the path this test means to rule out. Counting calls and asserting
    on the count survives that swallow; a raise asserted on the *returned*
    value does not.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls = 0

    async def request(self, *args, **kwargs):
        self.calls += 1
        return await super().request(*args, **kwargs)


@pytest.mark.asyncio
async def test_a_second_run_regrades_nothing_and_calls_no_model(tmp_path):
    db = a_db_with_two_listings(tmp_path)
    model = TestModel(custom_output_args={"grade": 7.0, "evidence": "fine"})
    await backfill_grades(db, Settings(), model=model)

    counting = Counting(custom_output_args={"grade": 7.0, "evidence": "fine"})
    again = await backfill_grades(db, Settings(), model=counting)
    assert all(row.regraded == [] for row in again)
    assert all(row.calls == 0 for row in again)
    assert counting.calls == 0


@pytest.mark.asyncio
async def test_force_regrades_against_an_unchanged_rubric(tmp_path):
    """`rubric_hash` covers a Criterion file's `grade` block, its `fallback`
    and its body - and those inputs are fixed so that a weight edit
    replays for free. Everything else a grade depends on is therefore outside
    it: `graders.py` and the shared preamble. Edit one
    and `stale_criteria` reports the whole corpus current while newly graded
    Listings use the new values, with nothing on screen to say the corpus has
    split. --force is the escape hatch.

    The model is counted rather than made to raise: `by_model` wraps its call
    in a broad `except Exception`, which would swallow an AssertionError and
    land on a default that looks like a grade.
    """
    db = a_db_with_two_listings(tmp_path)
    model = TestModel(custom_output_args={"grade": 7.0, "evidence": "fine"})
    await backfill_grades(db, Settings(), model=model)

    counting = Counting(custom_output_args={"grade": 7.0, "evidence": "fine"})
    again = await backfill_grades(db, Settings(), model=counting, force=True)
    assert counting.calls > 0
    assert all(row.calls == 4 for row in again)
    assert all(len(row.regraded) == 5 for row in again)


@pytest.mark.asyncio
async def test_criterion_scopes_the_forced_regrade_to_the_slugs_named(tmp_path):
    """An edit that could touch one Criterion should not pay for the rest."""
    db = a_db_with_two_listings(tmp_path)
    model = TestModel(custom_output_args={"grade": 7.0, "evidence": "fine"})
    await backfill_grades(db, Settings(), model=model)
    assert db.criterion_grades(1)["hard-floors"]["grade"] == 7.0

    # A different answer, so a Criterion that was re-graded says so in the
    # stored row rather than only in the report.
    counting = Counting(custom_output_args={"grade": 2.0, "evidence": "carpeted"})
    again = await backfill_grades(
        db, Settings(), model=counting, force=True, only=["hard-floors"]
    )
    assert all(row.regraded == ["hard-floors"] for row in again)
    assert counting.calls == 2  # one Criterion, two Listings

    stored = db.criterion_grades(1)
    assert stored["hard-floors"]["grade"] == 2.0
    # Every other model-graded Criterion kept the first run's answer.
    assert stored["pets"]["grade"] == 7.0
    assert stored["lift"]["grade"] == 7.0
    assert stored["aspect"]["grade"] == 7.0
    # And the coverage still spans the whole brief, not the one slug asked for.
    assert all(row.coverage > 0.5 for row in again)


@pytest.mark.asyncio
async def test_a_criterion_slug_that_does_not_exist_is_named_not_ignored(tmp_path):
    """Grading nothing looks exactly like a corpus with nothing stale, which
    is the false all-clear --force exists to fix."""
    from flat_scout.criteria import CriteriaError

    db = a_db_with_two_listings(tmp_path)
    with pytest.raises(CriteriaError, match="no_such_criterion"):
        await backfill_grades(db, Settings(), only=["no_such_criterion"])


@pytest.mark.asyncio
async def test_the_limit_stops_early(tmp_path):
    db = a_db_with_two_listings(tmp_path)
    model = TestModel(custom_output_args={"grade": 7.0, "evidence": "fine"})
    assert len(await backfill_grades(db, Settings(), model=model, limit=1)) == 1


class Tracking(TestModel):
    """Counts requests in flight, keeping the high-water mark.

    `grade_listing` fans out one concurrent call per stale Criterion, so a cap
    that only bounded the Listings in flight would let each Listing's own
    calls multiply the real concurrency past the configured number - the
    fixture below has 4 model-reachable Criteria per Listing across 2
    Listings, 8 possible calls at once against a cap of 2. The `asyncio.sleep` widens
    the window in which two calls can overlap, so an unbounded run reliably
    shows it rather than merely getting lucky on scheduling.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.current = 0
        self.high_water = 0

    async def request(self, *args, **kwargs):
        self.current += 1
        self.high_water = max(self.high_water, self.current)
        await asyncio.sleep(0.01)
        try:
            return await super().request(*args, **kwargs)
        finally:
            self.current -= 1


@pytest.mark.asyncio
async def test_the_concurrency_cap_bounds_model_calls_not_listings(tmp_path):
    db = a_db_with_two_listings(tmp_path)
    model = Tracking(custom_output_args={"grade": 7.0, "evidence": "fine"})
    await backfill_grades(db, Settings(), model=model, concurrency=2)
    assert model.high_water > 0  # the model was genuinely exercised
    assert model.high_water <= 2


# A Criterion pairing `fallback: model` with a numeric `unknown`. None of the
# five shipped Criteria does this yet - all five use `unknown: skip` - so the
# defect below is dormant on disk and would arrive with the first one that
# does not.
_DEFAULTING = """---
name: Quiet and well built
description: d
weight: 4
grade:
  field: epc_band
  map: {A: 10, B: 10, C: 8, D: 5, E: 2, F: 0, G: 0}
unknown: 4
fallback: model
---

## Rubric
**5** - a rubric body, which the loader requires.
"""


def a_criteria_dir(tmp_path):
    directory = tmp_path / "criteria"
    directory.mkdir()
    (directory / "quiet.md").write_text(_DEFAULTING)
    for name in PREAMBLE_FILES:
        (directory / name).write_text("The couple are looking for a flat.")
    return directory


@pytest.mark.asyncio
async def test_a_model_call_behind_a_numeric_default_is_still_counted_as_spend(tmp_path):
    """`apply_unknown` used to hardcode `graded_by="default"`.

    The model was called, the call cost money, it returned null, and the
    default supplied the value - and the spend count, which reads
    `graded_by.startswith("model:")`, saw "default" and reported zero. That
    is the number somebody reads before running the backfill over the whole
    corpus.

    The model is counted here as well as asserted on, because `by_model`
    wraps its call in a broad `except Exception`: a stand-in that raised to
    prove it ran would be swallowed and land on the same default.
    """
    db = a_db_with_two_listings(tmp_path)
    settings = Settings()
    settings.evaluation.criteria_dir = str(a_criteria_dir(tmp_path))

    class Counting(TestModel):
        calls: int = 0

        async def request(self, *args, **kwargs):
            self.calls += 1
            return await super().request(*args, **kwargs)

    model = Counting(custom_output_args={"grade": None, "evidence": "nothing stated"})
    done = await backfill_grades(db, settings, model=model)

    assert model.calls == 2
    assert [row.calls for row in done] == [1, 1]
    # And the default did arrive: the value is on the score, off the coverage.
    assert all(row.coverage == 0 for row in done)


@pytest.mark.asyncio
async def test_one_broken_listing_does_not_cancel_the_backfill(tmp_path, caplog):
    """Rollout step 2 is this command over the whole corpus.

    `one(row)` had no `try`, and a non-model error - a db write, a malformed
    row - propagated out of `asyncio.gather` and cancelled every other
    Listing in flight. A deterministic row failure then blocked the backfill
    on every retry, and the weighted `rescore` delegates here, so it silently
    lost the isolation its holistic twin documents as "one Listing, not the
    run".

    The failure is raised from a db write rather than the model: `by_model`
    already swallows model errors per Criterion, so a model-shaped failure
    would prove nothing about this path.
    """
    db = a_db_with_two_listings(tmp_path)
    settings = Settings()
    settings.evaluation.criteria_dir = str(a_criteria_dir(tmp_path))
    doomed = db.conn.execute("SELECT id FROM listings ORDER BY id LIMIT 1").fetchone()["id"]

    real = db.set_criterion_grades

    def flaky(listing_id, pairs):
        if listing_id == doomed:
            raise sqlite3.OperationalError("database is locked")
        return real(listing_id, pairs)

    db.set_criterion_grades = flaky
    model = TestModel(custom_output_args={"grade": 7.0, "evidence": "fine"})
    with caplog.at_level(logging.ERROR):
        done = await backfill_grades(db, settings, model=model)

    assert [row.listing_id for row in done] == [
        row["id"] for row in db.conn.execute("SELECT id FROM listings WHERE id != ?", (doomed,))
    ]
    assert str(doomed) in caplog.text
    assert "database is locked" in caplog.text


async def aspect_grade_after_backfill(tmp_path, image_signals: bool):
    """The stored `aspect` grade for a Listing whose plan was read.

    `process_url` gates the stored reading on the flag rather than only the
    new vision call, precisely so that turning the feature off withholds old
    readings from the evaluator. The backfill has to gate it the same way or
    the corpus rollout step 3 reads describes a different evaluator from the
    one that will grade tomorrow's Listings.
    """
    from flat_scout.criteria import load_criteria
    from flat_scout.graders import GRADERS
    from flat_scout.models import ImageReading

    db = a_db_with_two_listings(tmp_path)
    listing_id = db.conn.execute("SELECT id FROM listings ORDER BY id LIMIT 1").fetchone()["id"]
    db.set_image_reading(
        listing_id,
        ImageReading(north_clock=0, windows_clock=[180], window_aspects=["S"]),
        model="test",
    )

    settings = Settings()
    settings.features.image_signals = image_signals
    model = TestModel(custom_output_args={"grade": 7.0, "evidence": "off the advert"})
    await backfill_grades(db, settings, model=model, only=["aspect"])

    criteria = load_criteria(Path("criteria"), known_graders=set(GRADERS))
    pairs = dict(
        (criterion.slug, grade) for criterion, grade in db.graded_pairs(listing_id, criteria)
    )
    return pairs["aspect"]


@pytest.mark.asyncio
async def test_the_backfill_grades_off_a_stored_reading_when_image_signals_is_on(tmp_path):
    """The control on the test below, and the half that proves the fixture
    reaches `best_aspect` at all. South off the plan's compass grades 10, which
    is a determined answer and so never reaches the model fallback."""
    grade = await aspect_grade_after_backfill(tmp_path, image_signals=True)
    assert grade.graded_by == "best_aspect"
    assert grade.value == 10


@pytest.mark.asyncio
async def test_the_backfill_withholds_a_stored_reading_when_image_signals_is_off(tmp_path):
    """`backfill_grades` read `image_reading_from_row(row)` unconditionally.

    With the flag off, it therefore graded `aspect` - and `floor-area`
    with it - off vision data the live evaluator would never see, so the
    coverage figures and the score distribution used to set
    `hopeful_threshold` described an evaluator that was not going to run.
    """
    grade = await aspect_grade_after_backfill(tmp_path, image_signals=False)
    assert grade.graded_by != "best_aspect"
