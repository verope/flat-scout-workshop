import asyncio

import pytest
from pydantic_ai.models.test import TestModel

from flat_scout.config import Settings
from flat_scout.criteria import CriteriaError
from flat_scout.db import Database
from flat_scout.models import Evaluation, ListingData
from flat_scout.pipeline import measure_criteria
from flat_scout.report import CriterionSpread, measure_report


def a_db(tmp_path, count: int = 2) -> Database:
    """Listings the evaluator has actually read: `evaluated_at` is set on each,
    which is what `measure_criteria` now samples on."""
    db = Database(tmp_path / "flats.db")
    for portal_id in range(count):
        listing_id = db.upsert(
            ListingData(
                portal="rightmove", portal_id=str(portal_id), url=f"u{portal_id}",
                postcode="SW11", price_pcm=3000, sqft=600, description="A modern flat.",
            ),
            source="manual",
        )
        db.set_evaluation(
            listing_id,
            Evaluation(verdict="borderline", score=5.0, reasons=["r"]),
            "test",
        )
        db.transition(listing_id, "evaluated")
    return db


class Recording(TestModel):
    """A TestModel that records when each call starts and ends."""

    def __init__(self, events: list[str], **kwargs):
        super().__init__(**kwargs)
        self._events = events

    async def request(self, *args, **kwargs):
        self._events.append("start")
        await asyncio.sleep(0.01)
        result = await super().request(*args, **kwargs)
        self._events.append("end")
        return result


@pytest.mark.asyncio
async def test_the_first_replicate_warms_the_cache_for_the_rest(tmp_path):
    """`measure` sends the SAME prompt `runs` times, which is the best cache
    prefix in the codebase and was the worst thing to fire all at once.

    Every replicate carries the breakpoints `build_criterion_prompt` marks, so
    gathering them all made every one a cache write at 1.25x and a cache read
    never - measure got 23% dearer the moment caching was switched on. One
    replicate has to land before the others for the rest to read it back.
    """
    events: list[str] = []
    await measure_criteria(
        a_db(tmp_path),
        Settings(),
        model=Recording(events, custom_output_args={"grade": 7.0, "evidence": "fine"}),
        runs=3,
    )
    assert events[:2] == ["start", "end"], events[:6]


@pytest.mark.asyncio
async def test_asking_for_no_runs_spends_no_money(tmp_path):
    """`--runs 0` asked for nothing and used to cost nothing.

    The warm call above must not turn "measure zero replicates" into one paid
    grading per Criterion per Listing, persisted through `add_measure_runs` and
    reported under a heading that says zero runs. The CLI takes `runs` as a
    plain int with no floor, so this is reachable from the command line.
    """
    events: list[str] = []
    await measure_criteria(
        a_db(tmp_path),
        Settings(),
        model=Recording(events, custom_output_args={"grade": 7.0, "evidence": "fine"}),
        runs=0,
    )
    assert events == []


@pytest.mark.asyncio
async def test_a_steady_model_reports_no_spread(tmp_path):
    model = TestModel(custom_output_args={"grade": 7.0, "evidence": "fine"})
    spreads = await measure_criteria(a_db(tmp_path), Settings(), model=model, runs=3)
    graded = [row for row in spreads if row.unknown_rate < 1.0]
    assert graded, "no model Criterion was exercised"
    assert all(row.worst_spread == 0 for row in graded)


@pytest.mark.asyncio
async def test_measure_persists_raw_runs(tmp_path):
    """The replicates `measure_criteria` gathers must survive the run - the
    noise fit (Task 5) reads them back through `db.measure_runs()`, and a
    spread reported once and discarded proves nothing on the next call."""
    model = TestModel(custom_output_args={"grade": 7.0, "evidence": "fine"})
    db = a_db(tmp_path)
    spreads = await measure_criteria(db, Settings(), model=model, runs=2)
    rows = db.measure_runs()
    assert rows, "measure runs must be persisted for the noise fit"
    assert {row["criterion"] for row in rows} <= {s.slug for s in spreads}


@pytest.mark.asyncio
async def test_measure_keeps_deterministic_fallbacks_out_of_judge_noise(tmp_path):
    """A fallback that resolves without a model says nothing about model noise."""
    from flat_scout.models import ImageReading

    model = TestModel(custom_output_args={"grade": 7.0, "evidence": "fine"})
    db = a_db(tmp_path, count=1)
    listing_id = db.conn.execute("SELECT id FROM listings").fetchone()["id"]
    # A plan with a compass on it, so `best_aspect` resolves and `aspect`
    # never takes its `fallback: model` branch.
    db.set_image_reading(listing_id, ImageReading(window_aspects=["N"]), model="test")

    await measure_criteria(db, Settings(), model=model, runs=2)

    measured = {row["criterion"] for row in db.measure_runs()}
    assert "pets" in measured  # a primary model Criterion proves the fixture ran
    assert "aspect" not in measured  # the plan resolves this fallback exactly


@pytest.mark.asyncio
async def test_a_model_that_never_answers_reports_a_full_unknown_rate(tmp_path):
    model = TestModel(custom_output_args={"grade": None, "evidence": "says nothing"})
    spreads = await measure_criteria(a_db(tmp_path), Settings(), model=model, runs=2)
    by_slug = {row.slug: row for row in spreads}
    assert by_slug["pets"].unknown_rate == 1.0


@pytest.mark.asyncio
async def test_measure_can_be_scoped_to_one_criterion(tmp_path):
    """`grade --measure --criterion pets` pays for pets and nothing else."""
    model = TestModel(custom_output_args={"grade": 7.0, "evidence": "fine"})
    spreads = await measure_criteria(
        a_db(tmp_path), Settings(), model=model, runs=1, only=["pets"]
    )
    assert [row.slug for row in spreads] == ["pets"]


@pytest.mark.asyncio
async def test_measure_refuses_a_slug_it_cannot_measure(tmp_path):
    """A typo, or a Criterion graded by code, must fail before any call."""
    model = TestModel(custom_output_args={"grade": 7.0, "evidence": "fine"})
    with pytest.raises(CriteriaError):
        await measure_criteria(
            a_db(tmp_path), Settings(), model=model, runs=1, only=["floor-area"]
        )


@pytest.mark.asyncio
async def test_only_the_model_criteria_are_measured(tmp_path):
    """Grading `floor-area` three times would spend nothing and prove nothing."""
    model = TestModel(custom_output_args={"grade": 7.0, "evidence": "fine"})
    spreads = await measure_criteria(a_db(tmp_path), Settings(), model=model, runs=2)
    slugs = {row.slug for row in spreads}
    assert "floor-area" not in slugs
    assert {"pets", "hard-floors", "lift"} <= slugs


@pytest.mark.asyncio
async def test_measure_criteria_withholds_a_stored_reading_when_image_signals_is_off(
    tmp_path, monkeypatch
):
    """`measure_criteria` read `image_reading_from_row(row)` unconditionally -
    the same defect just fixed in `backfill_grades`, and the one `process_url`
    has always got right.

    `build_criterion_prompt` folds a reading into the prompt of whichever
    model Criterion is being graded, not only a vision-specific one, so a
    stored reading reaching here with the flag off would still put the plan
    in front of the model on every run - a steadiness measurement taken
    against a reading the live evaluator would never see.
    """
    from flat_scout import pipeline as pipeline_module
    from flat_scout.models import ImageReading

    db = a_db(tmp_path, count=1)
    listing_id = db.conn.execute("SELECT id FROM listings").fetchone()["id"]
    db.set_image_reading(
        listing_id,
        ImageReading(north_clock=0, windows_clock=[180], window_aspects=["S"]),
        model="test",
    )

    seen: list = []

    async def capture(criterion, listing, reading, settings, preamble, model=None, gate=None):
        seen.append(reading)
        from flat_scout.adjudicate import Grade

        return Grade(criterion.slug, 7.0, True, "fine")

    monkeypatch.setattr(pipeline_module, "grade_one", capture)

    settings = Settings()
    settings.features.image_signals = False
    await measure_criteria(db, settings, model=TestModel(), sample=1, runs=1)

    assert seen, "grade_one was never called - the fixture reaches nothing"
    assert all(reading is None for reading in seen)


def test_the_report_names_a_criterion_that_disagrees_with_itself():
    noisy = CriterionSpread(
        "quiet", "Quiet", runs=3, sampled=2, unknown_rate=0.0, worst_spread=7.0
    )
    text = measure_report([noisy], aspect_agreement=None)
    assert "Quiet" in text
    assert "7" in text


def test_the_report_says_when_the_corpus_has_no_aspect_overlap():
    text = measure_report([], aspect_agreement=None)
    assert "no annotated plan" in text.lower()


def test_measure_report_says_how_many_listings_it_sampled():
    row = CriterionSpread(
        "quiet", "Quiet", runs=3, sampled=5, unknown_rate=0.0, worst_spread=0.0
    )
    text = measure_report([row], aspect_agreement=None)
    assert "5 Listings sampled" in text
    assert "3 runs" in text


@pytest.mark.asyncio
async def test_measure_report_states_plainly_when_no_listings_were_sampled(tmp_path):
    """Narrowing the corpus to `evaluated_at IS NOT NULL` makes an empty
    corpus reachable - nothing evaluated yet. That must not render as every
    Criterion at 100% unknown with the alarming rubric prose and no
    explanation; it must say plainly that nothing was sampled.
    """
    db = Database(tmp_path / "flats.db")  # no rows at all
    spreads = await measure_criteria(db, Settings(), model=TestModel(), sample=10, runs=1)
    text = measure_report(spreads, aspect_agreement=None)
    assert "0 Listings sampled" in text
    assert "nothing to measure" in text.lower()
    assert "100%" not in text
    assert "unknown " not in text.lower()


def an_annotated_listing(tmp_path, monkeypatch, windows_clock: tuple[int, ...]):
    """One stored Listing with a reading, and a human annotation of its plan.

    The pair `aspect_against_annotations` needs and nothing else built: a
    `done` annotation carrying a compass whose `portal_id` matches a row.
    North is up the page and the windows point down it on the stored reading,
    so the machine reads south; `windows_clock` is what the human read.
    """
    from flat_scout import annotate
    from flat_scout.annotate import Annotation
    from flat_scout.models import ImageReading

    db = a_db(tmp_path, count=1)
    listing_id = db.conn.execute("SELECT id FROM listings").fetchone()["id"]
    db.set_image_reading(
        listing_id,
        ImageReading(north_clock=0, windows_clock=[180], window_aspects=["S"]),
        model="test",
    )
    monkeypatch.setattr(
        annotate,
        "load_annotations",
        lambda *args, **kwargs: {
            "0": Annotation(
                portal_id="0", file="0.png", north_clock=0,
                windows_clock=windows_clock, done=True,
            )
        },
    )
    return db


def test_the_aspect_comparison_grades_both_readings(tmp_path, monkeypatch):
    """The loop that holds the human's plan against the machine's.

    Nothing reached it before: it runs only where a `done` annotation carrying
    a compass matches a stored Listing, and no fixture built that pair. So a
    grader signature change went in with both call sites left on the old
    arity, and `grade --measure` raised `TypeError` in the field - after
    `measure_criteria` had already paid for its model calls.
    """
    from flat_scout.pipeline import aspect_against_annotations

    db = an_annotated_listing(tmp_path, monkeypatch, (180,))
    assert aspect_against_annotations(db, Settings()) == (1, 1)


def test_the_aspect_comparison_counts_a_disagreement(tmp_path, monkeypatch):
    """The same loop and the same two grader calls, where the readings differ.

    Without this, a comparison hard-wired to agree would still pass above.
    Grades are what is compared, so the disagreement has to be one that moves
    the grade: the human's windows point left across the page against a north
    that points up it, so they read west, which grades 4 against south's 10.
    North would not do here - it grades 10, the same as south, and the two
    readings would agree on the grade while differing on the compass.
    """
    from flat_scout.pipeline import aspect_against_annotations

    db = an_annotated_listing(tmp_path, monkeypatch, (270,))
    assert aspect_against_annotations(db, Settings()) == (0, 1)


@pytest.mark.asyncio
async def test_a_criterion_that_reaches_the_model_by_fallback_is_measured(tmp_path):
    """`aspect` grades deterministically when its input is there and
    goes to the model when it is not - which, on a Listing with no vision
    reading, is most of the time.

    Filtering on `is_model` matched only a primary `grade: {model: true}`
    block, so it was invisible to the very tool built to detect an unsteady
    rubric. The couple set `coverage_floor` off this measurement.
    """
    model = TestModel(custom_output_args={"grade": 7.0, "evidence": "fine"})
    spreads = await measure_criteria(a_db(tmp_path), Settings(), model=model, runs=2)
    slugs = {row.slug for row in spreads}
    assert {"aspect"} <= slugs
    # Still not the purely deterministic ones: `floor-area` is arithmetic over
    # two measurements and can never reach a model, so measuring it proves
    # nothing.
    assert "floor-area" not in slugs


@pytest.mark.asyncio
async def test_a_fallback_criterion_is_actually_graded_and_not_merely_listed(tmp_path):
    """It has to reach `grade_one`, not just appear in the output with a stub.

    `aspect` reads a compass off the floorplan, and the fixture Listing
    has no reading at all, so its deterministic pass always returns
    `value=None` - the same `unknown_rate == 1.0` a fallback that was never
    actually taken would also produce, so that outcome can't tell the two
    apart. An answering stand-in can: `aspect` comes back known only if
    `grade_one` actually took the `fallback: model` branch and reached it, so
    `unknown_rate == 0.0` is reachable exclusively through `by_model`.
    """
    model = TestModel(custom_output_args={"grade": 7.0, "evidence": "fine"})
    spreads = await measure_criteria(a_db(tmp_path), Settings(), model=model, runs=2)
    by_slug = {row.slug: row for row in spreads}
    assert by_slug["aspect"].unknown_rate == 0.0


def a_db_led_by_dead_ends(tmp_path, dead: int = 2, live: int = 1) -> Database:
    """Dead-end Listings on the low ids, one real one behind them.

    `prefiltered_out` and `unavailable` have an empty transition set: neither
    can ever reach the evaluator, so neither ever calls `set_evaluation` and
    `evaluated_at` stays unset. They sort first by id, so `ORDER BY id LIMIT`
    handed them the whole sample.
    """
    db = Database(tmp_path / "flats.db")
    for index in range(dead):
        listing_id = db.upsert(
            ListingData(portal="rightmove", portal_id=f"dead{index}", url=f"d{index}"),
            source="alert",
        )
        db.transition(listing_id, "prefiltered_out" if index % 2 == 0 else "unavailable")
    for index in range(live):
        listing_id = db.upsert(
            ListingData(
                portal="rightmove", portal_id=f"live{index}", url=f"l{index}",
                postcode="SW11", price_pcm=3000, sqft=600, description="A modern flat.",
            ),
            source="alert",
        )
        db.set_evaluation(
            listing_id,
            Evaluation(verdict="borderline", score=5.0, reasons=["r"]),
            "test",
        )
        db.transition(listing_id, "evaluated")
    return db


@pytest.mark.asyncio
async def test_measure_skips_listings_that_never_reached_the_evaluator(
    tmp_path, monkeypatch
):
    """A `prefiltered_out` or `unavailable` row carries only the fields it was
    discovered with, so grading it repeatedly manufactures unknowns off a
    Listing page nobody ever read - and the command then recommends rewriting a
    rubric that works.

    Worse, they sort first by id, so `ORDER BY id LIMIT ?` let terminal rows
    eat the entire `--sample` and the real corpus was never measured at all.
    """
    from flat_scout import pipeline as pipeline_module
    from flat_scout.adjudicate import Grade

    graded: list[str] = []

    async def capture(criterion, listing, reading, settings, preamble, model=None, gate=None):
        graded.append(listing.url)
        return Grade(criterion.slug, 7.0, True, "fine")

    monkeypatch.setattr(pipeline_module, "grade_one", capture)

    db = a_db_led_by_dead_ends(tmp_path)
    await measure_criteria(db, Settings(), model=TestModel(), sample=2, runs=1)

    assert graded, "grade_one was never called - the fixture reaches nothing"
    assert set(graded) == {"l0"}


@pytest.mark.asyncio
async def test_measure_samples_only_listings_the_evaluator_has_read(tmp_path, monkeypatch):
    """The corpus is `evaluated_at IS NOT NULL`, checked directly rather than
    through a status list. Two Listings share the same `status` here - only
    `set_evaluation` tells them apart - so this fails if the `WHERE` clause is
    ever dropped or swapped for a status check that happens to agree with it
    on the fixtures elsewhere in this file.
    """
    from flat_scout import pipeline as pipeline_module
    from flat_scout.adjudicate import Grade

    graded: list[str] = []

    async def capture(criterion, listing, reading, settings, preamble, model=None, gate=None):
        graded.append(listing.url)
        return Grade(criterion.slug, 7.0, True, "fine")

    monkeypatch.setattr(pipeline_module, "grade_one", capture)

    db = Database(tmp_path / "flats.db")
    db.upsert(
        ListingData(
            portal="rightmove", portal_id="unread", url="unread",
            postcode="SW11", price_pcm=3000, sqft=600, description="A modern flat.",
        ),
        source="alert",
    )
    evaluated_id = db.upsert(
        ListingData(
            portal="rightmove", portal_id="read", url="read",
            postcode="SW11", price_pcm=3000, sqft=600, description="A modern flat.",
        ),
        source="alert",
    )
    db.set_evaluation(
        evaluated_id, Evaluation(verdict="borderline", score=5.0, reasons=["r"]), "test"
    )

    await measure_criteria(db, Settings(), model=TestModel(), sample=10, runs=1)

    assert graded, "grade_one was never called - the fixture reaches nothing"
    assert set(graded) == {"read"}
