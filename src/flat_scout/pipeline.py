"""Orchestration behind the CLI: one Listing to a Verdict, and the corpus-wide replays."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from flat_scout.adjudicate import Pair, adjudicate
from flat_scout.config import Settings
from flat_scout.criteria import CriteriaError, load_criteria, load_preamble
from flat_scout.db import Database
from flat_scout.evaluate import evaluate_listing, prefilter
from flat_scout.fetch import (
    BotChallenge,
    FetchError,
    ListingUnavailable,
    fetch_listing,
    is_fetchable,
)
from flat_scout.graders import GRADERS
from flat_scout.grading import grade_listing, grade_one
from flat_scout.images import cache_images
from flat_scout.models import Evaluation, ImageReading, ListingData
from flat_scout.observe import Progress, event, span
from flat_scout.report import Backfilled, CriterionSpread, Rescored
from flat_scout.urls import canonicalise
from flat_scout.vision import read_images

if TYPE_CHECKING:
    from flat_scout.criteria import Criterion
    from flat_scout.fit import Fit

log = logging.getLogger(__name__)

SEED_FIELDS = ("price_pcm", "address", "postcode", "beds")

def listing_from_row(row: sqlite3.Row) -> ListingData:
    """Rebuild a ListingData from the database, which is where layers merge."""
    return ListingData(
        portal=row["portal"],
        portal_id=row["portal_id"],
        url=row["url"],
        address=row["address"],
        postcode=row["postcode"],
        price_pcm=row["price_pcm"],
        beds=row["beds"],
        sqft=row["sqft"],
        floor=row["floor"],
        furnished=row["furnished"],
        pet_notes=row["pet_notes"],
        description=row["description"],
        listed_on=row["listed_on"],
        image_url=row["image_url"],
        epc_caption=row["epc_caption"],
        epc_image_url=row["epc_image_url"],
        floorplan_url=row["floorplan_url"],
        image_path=row["image_path"],
        epc_image_path=row["epc_image_path"],
        floorplan_path=row["floorplan_path"],
        council_tax_band=row["council_tax_band"],
        nearest_station=row["nearest_station"],
        nearest_station_miles=row["nearest_station_miles"],
        latitude=row["latitude"],
        longitude=row["longitude"],
    )


def image_reading_from_row(row: sqlite3.Row) -> ImageReading | None:
    """What the vision step read, or None if it never ran on this Listing.

    `images_read_at` is the marker, not the fields: a reading in which every
    field is unknown is a real answer — the graph was illegible — and reusing it
    is what stops a retry paying to look at the same images again.
    """
    if not row["images_read_at"]:
        return None
    return ImageReading(
        epc_band=row["epc_band"],
        epc_band_source=row["epc_band_source"],
        epc_floor_area_sqm=row["epc_floor_area_sqm"],
        epc_property_type=row["epc_property_type"],
        layout_verdict=row["layout_verdict"],
        layout_notes=json.loads(row["layout_notes"] or "[]"),
        reception_is_separate=(
            None if row["reception_is_separate"] is None else bool(row["reception_is_separate"])
        ),
        desk_space=None if row["desk_space"] is None else bool(row["desk_space"]),
        bedroom_has_window=(
            None if row["bedroom_has_window"] is None else bool(row["bedroom_has_window"])
        ),
        floor_text=row["floor_text"],
        north_clock=row["north_clock"],
        windows_clock=json.loads(row["windows_clock"] or "[]"),
        window_aspects=json.loads(row["window_aspects"] or "[]"),
        north_source=row["north_source"],
    )


async def evaluate_weighted(
    db: Database,
    settings: Settings,
    listing_id: int,
    listing: ListingData,
    reading: ImageReading | None,
    model=None,
) -> Evaluation:
    """Grade every Criterion, store the vector, and adjudicate it.

    The grade vector is stored before the Evaluation is returned, and stored
    whatever the Verdict. It is the evidence a weight change is replayed over,
    so a rejected Listing's grades matter as much as an approved one's.
    """
    directory = Path(settings.evaluation.criteria_dir)
    criteria = load_criteria(directory, known_graders=set(GRADERS))
    preamble = load_preamble(directory)
    pairs = await grade_listing(criteria, listing, reading, settings, preamble, model)
    db.set_criterion_grades(listing_id, pairs)

    return adjudicate_current(db, pairs, settings, criteria, listing_id)


async def process_url(
    url: str,
    db: Database,
    settings: Settings,
    source: str,
    client: httpx.AsyncClient,
    model=None,
    vision_model=None,
    fields: dict | None = None,
) -> int | None:
    """Take one Listing URL as far as a Verdict, under a span. See `_take`.

    The span is the whole point of splitting this in two: the download, the
    vision calls and the evaluation all nest under it, which is what makes
    "why did that one Listing take ninety seconds" a question with an answer.
    """
    with span("process listing", url=url, source=source):
        return await _take(
            url, db, settings, source, client, model, vision_model, fields
        )


async def _take(
    url: str,
    db: Database,
    settings: Settings,
    source: str,
    client: httpx.AsyncClient,
    model=None,
    vision_model=None,
    fields: dict | None = None,
) -> int | None:
    """Take one Listing URL as far as a Verdict. Returns the listing id.

    `fields` carries a best-effort extraction from whatever discovered the
    Listing. It lets a Listing be pre-filtered before any download, and is the
    only data source for Portals we never download at all.

    `vision_model` falls back to `model`, so a caller overriding the model for a
    test overrides both agents and neither reaches a real provider.
    """
    parsed = canonicalise(url)
    if parsed is None:
        return None
    portal, portal_id, canonical = parsed

    # The listing page is the primary source and wins where the layers disagree.
    # Once the page has been read, a re-ingest must not write over it — a
    # Listing turns up again weeks later carrying the price it was advertised at
    # then, and a row whose price contradicts the Verdict computed from it is
    # worse than a row with no price at all.
    existing = db.conn.execute(
        "SELECT fetch_status FROM listings WHERE portal = ? AND portal_id = ?",
        (portal, portal_id),
    ).fetchone()
    seed: dict = {}
    if existing is None or existing["fetch_status"] != "ok":
        seed = {key: value for key, value in (fields or {}).items() if key in SEED_FIELDS}
    listing_id = db.upsert(
        ListingData(portal=portal, portal_id=portal_id, url=canonical, **seed),
        source=source,
    )
    if db.get(listing_id)["status"] not in ("new", "fetch_pending"):
        return listing_id  # already decided or already evaluated

    # Pre-filter on the fields we already have, before spending a download.
    reason = prefilter(listing_from_row(db.get(listing_id)), settings.filters)
    if reason:
        db.transition(listing_id, "prefiltered_out")
        db.record_event(
            listing_id, "prefiltered_out", f"on the fields already on the row: {reason}"
        )
        return listing_id

    # A retry after an evaluation failure must not download the page again:
    # the data is already in the row.
    already_downloaded = db.get(listing_id)["fetch_status"] == "ok"

    if is_fetchable(portal) and not already_downloaded:
        db.transition(listing_id, "fetch_pending")
        try:
            fetched = await fetch_listing(canonical, settings, client)
        except ListingUnavailable as exc:
            db.set_fetch_result(listing_id, None, ok=False)
            db.record_event(listing_id, "unavailable", str(exc))
            db.transition(listing_id, "unavailable")
            return listing_id
        except BotChallenge as exc:
            # Caught before the generic branch, and deliberately not retried.
            # The block is aimed at this host's address rather than at this
            # Listing, so the remaining attempts would each fetch the same
            # challenge page and write the same log line - which is what
            # listing #107 did, four times, before the couple got anything at
            # all. Going straight to `fetch_failed` gets them the link on the
            # next run instead of the fourth.
            #
            # `fetch_failed` and not `unavailable`, because the two say
            # different things: `unavailable` is a dead end for a flat that is
            # gone, whereas this flat is alive and merely out of our sight, and
            # `fetch_failed` still earns a link-only result.
            db.set_fetch_result(listing_id, None, ok=False)
            db.record_event(listing_id, "bot_challenge", str(exc))
            db.transition(listing_id, "fetch_failed")
            return listing_id
        except (FetchError, httpx.HTTPError) as exc:
            db.set_fetch_result(listing_id, None, ok=False)
            db.record_event(listing_id, "fetch_failed", str(exc))
            if db.get(listing_id)["fetch_attempts"] >= settings.fetch.max_attempts:
                db.transition(listing_id, "fetch_failed")
            return listing_id
        try:
            db.set_fetch_result(listing_id, fetched, ok=True)
        except Exception as exc:  # noqa: BLE001 - one odd Listing must not stop a batch
            # A Portal field of an unexpected shape used to escape here, past
            # every guard, and abort the caller. In a batch run that would take
            # every Listing after it down too.
            log.error("could not store listing %s: %s", listing_id, exc)
            db.record_event(listing_id, "store_failed", str(exc))
            db.transition(listing_id, "fetch_failed")
            return listing_id

        reason = prefilter(listing_from_row(db.get(listing_id)), settings.filters)
        if reason:
            db.transition(listing_id, "prefiltered_out")
            db.record_event(listing_id, "prefiltered_out", reason)
            return listing_id

    listing = listing_from_row(db.get(listing_id))

    # Every picture the Listing links is fetched once into data/images, so the
    # vision reading below, `annotate --db` and any later re-run read from disk
    # and nothing depends on the Portal's media host still answering.
    try:
        cached = await cache_images(listing, settings, client)
    except Exception as exc:  # noqa: BLE001 - a cache miss must not lose the Listing
        log.warning("image cache failed for listing %s: %s", listing_id, exc)
        db.record_event(listing_id, "image_cache_failed", str(exc))
    else:
        if cached:
            db.set_image_paths(listing_id, cached)
            listing = listing_from_row(db.get(listing_id))

    # The images are read after the pre-filter and before the evaluation, so a
    # Listing that is already out never costs a vision call, and a Listing that
    # is in reaches the evaluator with the band and the layout in hand.
    # The flag governs what the evaluator sees, not merely whether a new call
    # is made. A reading stored before the flag was turned off must not keep
    # reaching the evaluator on every retry.
    reading = None
    if settings.features.image_signals:
        reading = image_reading_from_row(db.get(listing_id))
    if settings.features.image_signals and reading is None:
        try:
            reading = await read_images(
                listing, settings, client, model=vision_model or model
            )
        except Exception as exc:  # noqa: BLE001 - an outage must not lose the Listing
            # Deliberately not fatal, and deliberately not retried here: the
            # text alone still produces a Verdict, and the result still reaches
            # the couple. The event is the record that it happened.
            log.warning("image reading failed for listing %s: %s", listing_id, exc)
            db.record_event(listing_id, "image_reading_failed", str(exc))
            reading = None
        else:
            if reading is not None:
                db.set_image_reading(
                    listing_id, reading, settings.evaluation.vision_model
                )

    # A floorplan states the floor far more often than a Portal does - Rightmove
    # has sent `entranceFloor` as null on every Listing seen, against 14 of the
    # 24 sampled plans that print it - so this is the only structured source
    # there is, and `exclude_ground_floor` has never once fired in production.
    #
    # It is deliberately NOT wired to that filter. The first version of this
    # re-ran the pre-filter on the new floor, which would have silently removed
    # the Listing behind one plan in the corpus: the model answered "Ground
    # Floor" for a drawing that does not contain the word anywhere, confirmed
    # by OCR over the whole image. A terminal status reached on an invented fact
    # is the worst outcome available here. The floor informs the evaluator
    # instead - the brief already treats ground-floor hints as a red flag - and
    # a human sees the result.
    if reading is not None and reading.floor_text:
        db.set_floor_from_plan(listing_id, reading.floor_text)

    try:
        if settings.features.weighted_criteria:
            evaluation = await evaluate_weighted(
                db, settings, listing_id, listing, reading, model=model
            )
        else:
            criteria = Path(settings.evaluation.criteria_path).read_text()
            evaluation = await evaluate_listing(
                listing, criteria, settings, model=model, reading=reading
            )
    except Exception as exc:  # noqa: BLE001 - an outage must not lose the Listing
        # Leave the status alone. The Listing stays in a state process_url will
        # pick up again, and the download above is not repeated.
        log.error("evaluation failed for listing %s: %s", listing_id, exc)
        db.record_event(listing_id, "evaluation_failed", str(exc))
        return listing_id
    db.set_evaluation(
        listing_id,
        evaluation,
        settings.evaluation.model,
        mode="weighted" if settings.features.weighted_criteria else "holistic",
    )
    db.transition(listing_id, "evaluated")
    return listing_id


async def rescore(
    db: Database,
    settings: Settings,
    model=None,
    limit: int = 0,
    concurrency: int = 4,
) -> list[Rescored]:
    """Replay the brief over Listings already evaluated. Writes nothing.

    An evaluated Listing is never evaluated again - `process_url` returns early
    for any status past `fetch_pending` - so an edit to criteria.md acts on
    tomorrow's Listings and on nothing that has already arrived. That makes a
    brief impossible to change with any confidence: the evidence that would
    show whether the edit worked is exactly the evidence the edit cannot touch.

    This runs the edited brief over the stored rows and reports what would have
    changed. Nothing is written, deliberately. Overwriting a stored score would
    silently rewrite the calibration data that `report --decisions` reads: a
    Listing they approved at 8.0 would be compared against a Decision taken on
    a score that no longer exists anywhere.

    The images are not read again. The stored reading is reused exactly as a
    retry reuses it, so a rescore costs one text call per Listing and no
    vision spend at all.
    """
    # On the weighted path a rescore is not a re-run. Only the Criteria whose
    # `rubric_hash` moved are graded again, so an edit to a weight - or to
    # `unknown`, `veto_at_or_below` or `ask`, none of which are hashed - costs
    # nothing at all and still says which Listings crossed the threshold.
    if settings.features.weighted_criteria:
        return await _rescore_weighted(
            db, settings, model=model, limit=limit, concurrency=concurrency
        )
    rows = db.conn.execute(
        "SELECT * FROM listings WHERE score IS NOT NULL ORDER BY score DESC, id"
    ).fetchall()
    if limit:
        rows = rows[:limit]
    criteria = Path(settings.evaluation.criteria_path).read_text()
    gate = asyncio.Semaphore(concurrency)
    watch = Progress(len(rows), "rescoring Listings")

    async def one(row: sqlite3.Row) -> Rescored | None:
        async with gate:
            try:
                with span("rescore listing", listing_id=row["id"]):
                    listing = listing_from_row(row)
                    evaluation = await evaluate_listing(
                        listing,
                        criteria,
                        settings,
                        model=model,
                        reading=image_reading_from_row(row),
                    )
            except Exception as exc:  # noqa: BLE001 - one Listing, not the run
                log.error("rescore failed for listing %s: %s", row["id"], exc)
                watch.tick(failed=True)
                return None
        watch.tick()
        return Rescored(
            listing_id=row["id"],
            was=row["score"],
            now=evaluation.score,
            was_verdict=row["verdict"],
            now_verdict=evaluation.verdict,
            address=row["address"] or row["postcode"] or "?",
            postcode=row["postcode"],
            price_pcm=row["price_pcm"],
            reasons=evaluation.reasons,
        )

    with span("rescore", listings=len(rows)), watch:
        done = await asyncio.gather(*(one(row) for row in rows))
    return [change for change in done if change is not None]


async def _rescore_weighted(
    db: Database, settings: Settings, model=None, limit: int = 0, concurrency: int = 4
) -> list[Rescored]:
    directory = Path(settings.evaluation.criteria_dir)
    criteria = load_criteria(directory, known_graders=set(GRADERS))
    # The population this command reports on, chosen before anything is graded.
    #
    # A change with no `was` is dropped at the bottom of this function - there
    # is no stored score for the regrade to be held against - so a Listing with
    # `score IS NULL` can never appear in the report however it grades.
    # `backfill_grades` walks the whole table in id order and applies `limit`
    # before any of that, so `rescore --limit 5` used to spend all five on
    # `prefiltered_out` and never-scored rows and report nothing, having paid
    # for every model call it then threw away.
    #
    # Ordered as the holistic path orders it, highest score first: under a
    # limit the Listings worth replaying an edited brief over are the ones
    # nearest the threshold, not the ones that happen to have the lowest ids.
    ids = [
        row["id"]
        for row in db.conn.execute(
            "SELECT id FROM listings WHERE score IS NOT NULL ORDER BY score DESC, id"
        )
    ]
    if limit:
        ids = ids[:limit]
    graded = await backfill_grades(
        db, settings, model=model, listing_ids=ids, concurrency=concurrency
    )
    log.info(
        "rescore regraded %s Criteria across %s Listings with %s model calls",
        sum(len(row.regraded) for row in graded),
        len(graded),
        sum(row.calls for row in graded),
    )

    # One last-good fit for the whole report, not one per Listing.
    # `flat-scout fit` owns PyMC; replaying stored Listings stays numpy-only
    # and every row in this report must use the same fitted prior.
    fit = None
    if settings.features.bayesian_verdict:
        from flat_scout.fit import load_fit

        fit = load_fit(db, criteria)

    changes = []
    for done in graded:
        row = db.get(done.listing_id)
        result = adjudicate_current(
            db,
            db.graded_pairs(done.listing_id, criteria),
            settings,
            criteria,
            done.listing_id,
            fit=fit,
        )
        changes.append(
            Rescored(
                listing_id=done.listing_id,
                # What the stored result said, against what the brief says now.
                # The stored score is never overwritten - that is the point.
                was=row["score"],
                now=result.score,
                was_verdict=row["verdict"],
                now_verdict=result.verdict,
                address=row["address"] or row["postcode"] or "?",
                postcode=row["postcode"],
                price_pcm=row["price_pcm"],
                reasons=result.reasons,
            )
        )
    # A Listing that was never evaluated has `row["score"] is None`, so
    # `rescore_report` must skip it rather than subtract from None.
    return [c for c in changes if c.was is not None]


async def backfill_grades(
    db: Database,
    settings: Settings,
    model=None,
    limit: int = 0,
    concurrency: int = 4,
    force: bool = False,
    only: list[str] | None = None,
    listing_ids: list[int] | None = None,
) -> list[Backfilled]:
    """Grade every stored Listing into `criterion_grades`.

    Neither `listings.score` nor `listings.verdict` is touched. Those are what
    the stored result said, and the couple's Decision answered them; rewriting
    one would silently rewrite the calibration data `report --decisions` reads.

    Idempotent by rubric hash: a Criterion already graded against the rubric now
    on disk is skipped, and a Listing with nothing stale costs no model call.

    That hash covers a Criterion file's `grade` block, its `fallback` and its
    body, and those inputs are fixed deliberately - widening it would cost the
    property the whole design rests on, that a weight edit replays over the
    corpus for free. The price is that everything else a grade depends on is
    outside it: `graders.py`, and the shared preamble in
    `_brief.md` and `_fields.md`. Edit one of those and nothing goes stale, so
    `stale_criteria` reports the corpus current while newly graded Listings
    use the new values and the old ones keep the old - two versions of the
    data with nothing on screen to say so.

    `force` is the escape hatch for exactly that: it re-grades regardless of
    the stored hashes. `only` scopes the run to the named slugs, so an edit
    that could touch one Criterion need not re-run the model over the rest.

    `listing_ids` scopes the run to a population the caller has already chosen.
    `limit` cannot do that job for a caller who filters afterwards: it is
    applied here, over every row in id order, so the budget goes on whatever
    sorts first rather than on whatever the caller can use. `rescore` picks its
    Listings first and passes them in.
    """
    directory = Path(settings.evaluation.criteria_dir)
    criteria = load_criteria(directory, known_graders=set(GRADERS))
    preamble = load_preamble(directory)
    selected = criteria
    if only:
        wanted = set(only)
        # Named before any work rather than quietly grading nothing: a typo'd
        # slug would otherwise report a clean run over an untouched corpus,
        # which is the same false all-clear this flag exists to fix.
        missing = wanted - {criterion.slug for criterion in criteria}
        if missing:
            raise CriteriaError(
                f"no Criterion in {directory} is named {', '.join(sorted(missing))}"
            )
        selected = [criterion for criterion in criteria if criterion.slug in wanted]
    rows = db.conn.execute("SELECT * FROM listings ORDER BY id").fetchall()
    if listing_ids is not None:
        # The caller's order is not preserved; `limit` is theirs to have
        # applied already. Everything downstream reads the whole vector back
        # per Listing, so the rows are independent and the order is only ever
        # about which rows a budget buys.
        wanted = set(listing_ids)
        rows = [row for row in rows if row["id"] in wanted]
    if limit:
        rows = rows[:limit]
    # Bounds the model calls in flight, not the Listings: `grade_listing` fans
    # out one concurrent call per stale Criterion, so gating only the
    # per-Listing task would let the real concurrency multiply by however many
    # Criteria happen to be stale at once - `concurrency=4` would mean 4
    # Listings, not 4 calls, and a corpus-wide backfill is exactly where that
    # difference meets a provider's rate limit. `grade_listing` threads this
    # gate down to each model call itself, so Listings are left to run
    # concurrently and only the calls are capped.
    gate = asyncio.Semaphore(concurrency)
    # AND a second bound, on the Listings actually open at once. Not a
    # replacement for the one above - read it before changing either.
    #
    # `grade_listing` grades one Criterion first and fans the rest out behind
    # it, so that the fan-out reads a warm cache instead of writing its own.
    # That only holds if the fan-out runs while the entry is alive, and an
    # `asyncio.Semaphore` hands its slots out first-come-first-served. Opening
    # every row at once put all N warm calls in the queue ahead of any
    # fan-out: measured on 8 Listings at concurrency 2, the last Listing's six
    # siblings were spread over 40 of the run's 48 calls, and the spread grows
    # with the corpus. On the several-hundred-Listing backfills this path is
    # for, the entry is long dead by then and the siblings miss and write
    # concurrently - the exact arrangement the warm call exists to avoid.
    #
    # Bounding the rows costs no throughput, because `gate` is what decides
    # how many calls are in flight and it is unchanged: `concurrency` open
    # Listings can still only have `concurrency` calls out between them. What
    # it buys is that a Listing's fan-out queues behind a handful of calls
    # rather than behind the whole corpus.
    rows_gate = asyncio.Semaphore(concurrency)
    # One last-good fit for the whole run. `flat-scout fit` owns refitting;
    # this command threads one stable fit through every Listing.
    fit = None
    if settings.features.bayesian_verdict:
        from flat_scout.fit import load_fit

        fit = load_fit(db, criteria)
    # The command this whole module was written for. Several hundred model
    # calls and ten minutes, and until now not one character of output until
    # the last Listing landed.
    watch = Progress(len(rows), "grading Listings")

    async def graded(row: sqlite3.Row) -> Backfilled | None:
        # One Listing, not the run. This is rollout step 2 over the whole
        # corpus, and a deterministic failure on one row - a db write, a
        # malformed row - used to propagate out of the gather and cancel
        # every other Listing in flight, blocking the backfill on every
        # retry. A model failure is already isolated per Criterion inside
        # `by_model`; everything else is isolated here.
        try:
            if force:
                todo = list(selected)
            else:
                stale = set(db.stale_criteria(row["id"], criteria))
                todo = [c for c in selected if c.slug in stale]
            listing = listing_from_row(row)
            # Gated exactly as `process_url` gates it. The flag governs what
            # the evaluator sees and not merely whether a new vision call is
            # made, so a reading stored before it was turned off must not keep
            # reaching the graders here either. Ungated, the backfill graded
            # `aspect` and `floor-area` off data the live evaluator
            # would not see - so the coverage and the score
            # distribution rollout step 3 reads would describe an evaluator
            # that is not the one grading tomorrow's Listings.
            reading = image_reading_from_row(row) if settings.features.image_signals else None
            calls = 0
            if todo:
                pairs = await grade_listing(
                    todo, listing, reading, settings, preamble, model, gate=gate
                )
                db.set_criterion_grades(row["id"], pairs)
                # Count what actually happened, not what might have: a
                # `fallback: model` Criterion whose deterministic grader resolves
                # never reaches `by_model` at all, and `graded_by` is the record of
                # whether it did - set on both a successful call and a failed one,
                # since a failed call still costs money.
                calls = sum(
                    1 for _, grade in pairs if grade.graded_by.startswith("model:")
                )
            # A Criterion this run did not touch keeps its stored value, which
            # is why the whole vector is read back.
            stored = db.graded_pairs(row["id"], criteria)
            result = adjudicate_current(db, stored, settings, criteria, row["id"], fit=fit)
        except Exception as exc:  # noqa: BLE001 - one Listing, not the run
            # Named and counted rather than swallowed: a Listing missing from
            # the coverage distribution with nothing on screen to say why
            # would read as a corpus that is simply thinner than it is.
            log.error("backfill failed for listing %s: %s", row["id"], exc)
            return None
        return Backfilled(
            listing_id=row["id"],
            coverage=result.coverage,
            calls=calls,
            regraded=[criterion.slug for criterion in todo],
        )

    async def one(row: sqlite3.Row) -> Backfilled | None:
        """The same work, seen from outside: a span, and one step of the bar.

        Wrapped rather than folded in, so the isolation above stays exactly as
        it is. A Listing that fails still ticks - the count on screen is of
        Listings finished with, not of Listings that went well - and the two
        are told apart by the failure count beside it.
        """
        async with rows_gate:
            with span("grade listing", listing_id=row["id"]):
                outcome = await graded(row)
        watch.tick(failed=outcome is None)
        return outcome

    # `watch` closes on the way out however that happens: its beat is a thread,
    # and a run that raises must not leave one talking.
    with span("backfill", listings=len(rows), criteria=len(selected)), watch:
        done = await asyncio.gather(*(one(row) for row in rows))
    graded_rows = [row for row in done if row is not None]
    event(
        "backfill",
        listings=len(graded_rows),
        failed=len(done) - len(graded_rows),
        calls=sum(row.calls for row in graded_rows),
    )
    return graded_rows


async def measure_criteria(
    db: Database,
    settings: Settings,
    model=None,
    sample: int = 10,
    runs: int = 3,
    only: list[str] | None = None,
) -> list[CriterionSpread]:
    """Grade the model Criteria repeatedly, and report how steady they are.

    `only` scopes the run to the named slugs, so one rubric can be measured
    on its own rather than paying for every model-reachable Criterion. A slug
    that names nothing measurable - a typo, or a Criterion graded by code -
    raises before any call is made.

    Only the Criteria that can reach a model. Grading `floor-area` three
    times would spend nothing and prove nothing: it is arithmetic, and it is
    already covered by a test.

    That set is not `is_model`, which matches only a primary
    `grade: {model: true}` block. `aspect` declares `fallback: model` and
    goes to the model whenever its deterministic input is missing - on a
    Listing with no vision reading, most of the time. Leaving it out made a
    Criterion that can produce an unsteady answer invisible to the one tool
    built to find one, and `coverage_floor` is set off exactly this
    measurement. The condition is the same one `grade_one` branches on,
    deliberately: this must measure what the grader actually does.

    A Criterion whose deterministic pass resolves is still measured, and comes
    back perfectly steady - which is the true answer for that Listing, and the
    contrast that shows which Listings sent it to the model at all.

    This measures steadiness and silence, never correctness. A rubric cannot be
    graded right by a machine - but one that answers differently every time, or
    never answers at all, is not doing the job whatever it says.
    """
    directory = Path(settings.evaluation.criteria_dir)
    criteria = [
        criterion
        for criterion in load_criteria(directory, known_graders=set(GRADERS))
        if criterion.is_model or criterion.fallback == "model"
    ]
    if only:
        wanted = set(only)
        missing = wanted - {criterion.slug for criterion in criteria}
        if missing:
            raise CriteriaError(
                f"no model-reachable Criterion in {directory} is named "
                f"{', '.join(sorted(missing))}"
            )
        criteria = [criterion for criterion in criteria if criterion.slug in wanted]
    preamble = load_preamble(directory)
    # Listings the evaluator actually read. `evaluated_at` is set only by
    # `set_evaluation`, so "IS NOT NULL" means precisely that - one column,
    # rather than a second status list that has to be kept in step with
    # `NEVER_CARDED`. It subsumes both of that list's dead ends
    # (`prefiltered_out` and `unavailable` never reach `set_evaluation`), and
    # it also excludes `new`, `fetch_pending` and `fetch_failed`: those carry
    # no fetched page either, so every prose Criterion could only return
    # unknown, which inflated `unknown_rate` and made the command recommend
    # rewriting a rubric that works perfectly well. A `fetched_at` test would
    # get this wrong the other way, wrongly excluding the Portals that are
    # never downloaded by design but still reach `evaluated`.
    #
    # This answers a different question from `NEVER_CARDED` and
    # `weighted_distribution` ("could this ever be surfaced"), which are left
    # exactly as they are.
    #
    # Sorted by id, so `ORDER BY id LIMIT ?` samples deterministically rather
    # than however SQLite happens to return the rows.
    rows = db.conn.execute(
        "SELECT * FROM listings WHERE evaluated_at IS NOT NULL ORDER BY id LIMIT ?",
        (sample,),
    ).fetchall()

    spreads = []
    # Criteria by Listings by runs, one call each and none of them skippable:
    # the slowest thing this project does, and the one most worth a bar.
    # Closed however this exits: the beat is a thread, and a provider outage
    # part-way through must not leave one counting.
    with Progress(len(criteria) * len(rows), "measuring Criteria") as watch:
        for criterion in criteria:
            seen: list[list[float | None]] = []
            for row in rows:
                listing = listing_from_row(row)
                reading = (
                    image_reading_from_row(row) if settings.features.image_signals else None
                )
                with span(
                    "measure", criterion=criterion.slug, listing_id=row["id"], runs=runs
                ):
                    # The first replicate alone, then the rest together, for
                    # the reason `grade_listing` does the same: all `runs`
                    # prompts here are byte-identical, which makes this the
                    # strongest cache prefix in the codebase and the worst
                    # thing to fire all at once. Anthropic's cache entry does
                    # not exist until the first response begins, so a plain
                    # gather would make every replicate a write at 1.25x and
                    # none of them a read.
                    #
                    # This is a billing change and not a measurement one: what
                    # `runs` is sampling is the model's judge noise, and a
                    # cached prefix is the same tokens read from somewhere
                    # cheaper.
                    #
                    # Guarded on `runs`, because the warm call is unconditional
                    # where the gather it replaced was not: `--runs 0` asked for
                    # no replicates and issued no calls, and an unguarded warm-up
                    # turned that into one paid grading per Criterion per Listing,
                    # persisted, under a report headed "0 runs". `runs` reaches
                    # here from the CLI as a plain int with no floor.
                    warm = (
                        [
                            await grade_one(
                                criterion, listing, reading, settings, preamble, model
                            )
                        ]
                        if runs > 0
                        else []
                    )
                    rest = await asyncio.gather(
                        *(
                            grade_one(criterion, listing, reading, settings, preamble, model)
                            for _ in range(runs - 1)
                        )
                    )
                    grades = [*warm, *rest]
                watch.tick()
                # A fallback can resolve deterministically. Those identical
                # answers belong in the steadiness report above, but they say
                # nothing about the model's judge noise and must not shrink σ.
                model_grades = [
                    grade for grade in grades if grade.graded_by.startswith("model:")
                ]
                if model_grades:
                    db.add_measure_runs(
                        row["id"],
                        criterion.slug,
                        criterion.rubric_hash,
                        # A numeric `unknown` default has a value but is not a
                        # model answer. Keep it NULL in the noise evidence.
                        [
                            grade.value if grade.determined else None
                            for grade in model_grades
                        ],
                    )
                seen.append([grade.value for grade in grades])
            answered = [[v for v in values if v is not None] for values in seen]
            total = sum(len(values) for values in seen)
            unknown = total - sum(len(values) for values in answered)
            spreads.append(
                CriterionSpread(
                    slug=criterion.slug,
                    name=criterion.name,
                    runs=runs,
                    sampled=len(rows),
                    unknown_rate=(unknown / total) if total else 1.0,
                    worst_spread=max(
                        (max(values) - min(values) for values in answered if values),
                        default=0.0,
                    ),
                )
            )
    return spreads


def aspect_against_annotations(db: Database, settings: Settings) -> tuple[int, int] | None:
    """How often the aspect Criterion agrees with itself across two readings.

    The human annotated the plan in image terms - which way north points on the
    page, which way the windows face on the page - so the same arithmetic that
    produces `window_aspects` produces theirs. Grading both and comparing tests
    the Criterion end to end without asking a person to agree with the machine.
    """
    from flat_scout.annotate import load_annotations
    from flat_scout.graders import GRADERS as G
    from flat_scout.vision import aspect_from_bearings

    annotations = load_annotations()
    if not annotations:
        return None
    directory = Path(settings.evaluation.criteria_dir)
    # Found by the grader it declares rather than by slug: the Criterion the
    # comparison is about is whichever one `best_aspect` grades, and renaming
    # the file must not silently turn this into a KeyError on the way past.
    criterion = next(
        (
            c
            for c in load_criteria(directory, known_graders=set(G))
            if c.grade.get("grader") == "best_aspect"
        ),
        None,
    )
    if criterion is None:
        return None
    agreed = total = 0
    for portal_id, annotation in annotations.items():
        row = db.conn.execute(
            "SELECT * FROM listings WHERE portal_id = ?", (portal_id,)
        ).fetchone()
        if row is None or not annotation.done or annotation.north_clock is None:
            continue
        theirs = [
            aspect for clock in annotation.windows_clock
            if (aspect := aspect_from_bearings(annotation.north_clock, clock))
        ]
        if not theirs:
            continue
        total += 1
        listing = listing_from_row(row)
        human = G["best_aspect"](
            criterion, listing, ImageReading(window_aspects=theirs), directory
        )
        machine = G["best_aspect"](
            criterion, listing, image_reading_from_row(row), directory
        )
        if human.value == machine.value:
            agreed += 1
    return (agreed, total) if total else None


def adjudicate_current(
    db: Database,
    pairs: list[Pair],
    settings: Settings,
    criteria: list[Criterion] | None,
    listing_id: int,
    *,
    fit: Fit | None = None,
) -> Evaluation:
    """The Verdict under the rules in force now, whichever path is live.

    The seed is the Listing id, so a replay - a look at the evidence, a report,
    the queue re-adjudication - reproduces the stored numbers exactly.

    `fit` lets a multi-Listing caller (backfill, rescore, a batch run) load
    it once and pass it through, rather than have every Listing reload it -
    see the comment at each of those call sites for why that reload is not
    merely wasteful but actively wrong mid-run.
    """
    if settings.features.bayesian_verdict:
        from flat_scout.fit import load_fit
        from flat_scout.posterior import adjudicate_bayesian

        return adjudicate_bayesian(
            pairs,
            fit if fit is not None else load_fit(db, criteria),
            hopeful_threshold=settings.evaluation.hopeful_threshold,
            hopeful_confidence=settings.evaluation.hopeful_confidence,
            seed=listing_id,
        )
    return adjudicate(
        pairs,
        hopeful_threshold=settings.evaluation.hopeful_threshold,
        coverage_floor=settings.evaluation.coverage_floor,
        unknown_prior=settings.evaluation.unknown_prior,
    )
