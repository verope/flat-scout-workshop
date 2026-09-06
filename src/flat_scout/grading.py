"""Determining one Criterion about one Listing.

Arithmetic wherever a field answers the question, because a model call over
`epc_band` spends money to add variance to a lookup. A model is called only
where the question is genuinely about prose, and then for one Criterion at a
time - isolation is the point of the design, so the calls are not combined.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, CachePoint

from flat_scout.adjudicate import Grade
from flat_scout.config import Settings
from flat_scout.criteria import Criterion
from flat_scout.evaluate import IMAGE_HEADING, LISTING_HEADING, facts_block
from flat_scout.models import ImageReading, ListingData
from flat_scout.observe import span
from flat_scout.vision import render_reading

log = logging.getLogger(__name__)


def unknown_grade(criterion: Criterion, why: str) -> Grade:
    return Grade(criterion=criterion.slug, value=None, determined=False, evidence=why)


def field_value(listing: ListingData, reading: ImageReading | None, field: str) -> Any:
    """The value of `field`, from the Image Reading first and the Listing next.

    The Image Reading wins because it is a reading of the document itself,
    which `criteria.md` says outranks the advert's prose wherever the two
    disagree. An empty list counts as absent: `window_aspects` is `[]` on a
    plan with no compass, and that is unknown rather than "faces nowhere".
    """
    if reading is not None:
        value = getattr(reading, field, None)
        if value is not None and value != []:
            return value
    value = getattr(listing, field, None)
    return None if value == [] else value


def _evidence(criterion: Criterion, value: Any) -> str:
    template = criterion.grade.get("evidence")
    if template:
        return str(template).format(value=value)
    return f"{criterion.grade['field']} is {value}."


def _capped(criterion: Criterion, grade: float) -> float:
    ceiling = criterion.grade.get("max")
    return min(float(grade), float(ceiling)) if ceiling is not None else float(grade)


def by_map(
    criterion: Criterion, listing: ListingData, reading: ImageReading | None
) -> Grade:
    field = criterion.grade["field"]
    value = field_value(listing, reading, field)
    if value is None:
        return unknown_grade(criterion, f"{field} was not read.")
    mapping = criterion.grade["map"]
    if value not in mapping:
        # The loader checks closed sets, so this is an open-set field that grew
        # a value nobody anticipated. Unknown is the safe answer: a value we
        # cannot interpret is not evidence of anything.
        return unknown_grade(criterion, f"{field} is {value}, which the map does not cover.")
    return Grade(
        criterion=criterion.slug,
        value=_capped(criterion, mapping[value]),
        determined=True,
        evidence=_evidence(criterion, value),
        graded_by="map",
    )


def by_bands(
    criterion: Criterion, listing: ListingData, reading: ImageReading | None
) -> Grade:
    field = criterion.grade["field"]
    value = field_value(listing, reading, field)
    if value is None:
        return unknown_grade(criterion, f"{field} was not read.")
    for band in criterion.grade["bands"]:
        if "at_most" not in band or value <= band["at_most"]:
            return Grade(
                criterion=criterion.slug,
                value=_capped(criterion, band["grade"]),
                determined=True,
                evidence=_evidence(criterion, value),
                graded_by="bands",
            )
    # No catch-all entry. Unknown rather than the worst band, because a gap in
    # the bands is an error in the file and not a fact about the flat, and
    # inventing a low grade would hide it.
    return unknown_grade(criterion, f"{field} is {value}, which no band covers.")


def deterministic(
    criterion: Criterion,
    listing: ListingData,
    reading: ImageReading | None,
    directory: Path,
) -> Grade:
    """The grade a field can settle, or an unknown Grade if none can.

    A Criterion graded by a model returns unknown from here. `grade_one` sends
    it on to the model; nothing else needs to know the difference.

    `directory` is the configured criteria directory, passed on to every
    grader so a grader that reads a file beside the Criterion files reads the
    one belonging to the Criteria it is grading against rather than whatever
    sits below the process working directory.
    """
    if criterion.grade.get("map"):
        return by_map(criterion, listing, reading)
    if criterion.grade.get("bands"):
        return by_bands(criterion, listing, reading)
    name = criterion.grade.get("grader")
    if name:
        # Import lazily: graders.py imports unknown_grade back out of this module,
        # creating a cycle. A short lazy import keeps both modules importable.
        from flat_scout.graders import GRADERS

        return GRADERS[name](criterion, listing, reading, directory)
    return unknown_grade(criterion, "This Criterion is judged by a model.")


def apply_unknown(criterion: Criterion, grade: Grade) -> Grade:
    """The `unknown:` rule, applied last of all.

    A default is not evidence. It supplies a value so the score has something
    to weigh, and leaves `determined` false so the coverage does not count it
    and no veto can fire on it.

    `graded_by` carries through from whatever was attempted. A model call that
    returned null still cost money, and `backfill_grades` counts spend with
    `graded_by.startswith("model:")` - overwriting it with "default" here made
    a `fallback: model` Criterion with a numeric `unknown` undercount its own
    bill, in the number somebody reads before running the corpus backfill.
    """
    if grade.value is not None or criterion.unknown == "skip":
        return grade
    return Grade(
        criterion=criterion.slug,
        value=float(criterion.unknown),
        determined=False,
        evidence=f"Not determined, so the brief's default of {criterion.unknown} applies.",
        graded_by=grade.graded_by or "default",
    )


CRITERION_SYSTEM_PROMPT = """You are judging one London rental listing against
ONE criterion from a couple's brief. Judge that criterion and nothing else:
another criterion covers each of the other things they care about, and a grade
that smuggles in your overall impression makes the whole score unreadable.

Return a grade from 0 to 10 against the rubric you are given, and one short
sentence of evidence naming what you read it off.

Return grade = null if the listing does not answer this criterion. Null is a
real answer and the right one wherever you would otherwise be guessing. Absent
is never the same as bad: a listing that says nothing about flooring has
not told you there is carpet.

Use `concern` for something you noticed that this criterion does not measure -
for instance the advert calling a flat a one-bed where the floorplan shows a
studio. Leave it null otherwise.

Use `question` for what a letting agent would have to answer, and only where
the advert has genuinely left it open. It is sent under the couple's names.
Never ask about this system, its fields or its scores: the agent has never
heard of any of them."""


class ModelGrade(BaseModel):
    """What a model returns about one Criterion."""

    # Nullable on purpose. Without it the model must invent a number, and an
    # invented number is indistinguishable from a read one.
    grade: float | None = Field(default=None, ge=0, le=10)
    evidence: str
    concern: str | None = None
    question: str | None = None


def build_criterion_prompt(
    criterion: Criterion,
    listing: ListingData,
    reading: ImageReading | None,
    preamble: str,
) -> list[str | CachePoint]:
    """The shared preamble, then this Listing, then this Criterion's rubric.

    Content blocks rather than one string, because the boundaries between them
    are where the cache breakpoints go. The text the model reads is unchanged:
    a provider concatenates the blocks.

    The order was always deliberate - shared material first, the rubric last -
    but ordering alone cached nothing. This function's docstring claimed for
    months that the prefix was one "the provider can cache", and it was: the
    provider simply never had been asked to. Over three days in August 2026
    this app spent 1.67M input tokens across 417 model calls - the graders
    being 60% of the bill - with a cache read count of exactly zero and a cache
    write count of exactly zero. A breakpoint is opt-in, and nothing opted in.

    TWO breakpoints, because two spans repeat at two different rates:

      preamble        identical on every grader call this app makes, for as
                      long as `criteria/` is unchanged. ~2,900 tokens with the
                      system prompt and the tool definitions ahead of it, which
                      a breakpoint in the first user block also covers.
      + the Listing   identical across the calls about one Listing. ~1,100
                      tokens more.
      + the criterion  ~370 tokens, different every call, deliberately left
                      outside: caching a block that never repeats writes the
                      cache and never reads it.

    Both prefixes clear Claude Sonnet 5's 1,024-token minimum, which is the
    reason the system prompt does not get a breakpoint of its own. On its own
    the system prompt and the tool definitions come to about 390 tokens, under
    the minimum, so `anthropic_cache_instructions` would mark a boundary
    Anthropic declines to cache - a wasted breakpoint of the four available.
    """
    listing_block = f"{LISTING_HEADING}\n\n{facts_block(listing)}\n"
    if reading is not None:
        listing_block += f"\n{IMAGE_HEADING}\n\n{render_reading(reading)}\n"
    return [
        f"{preamble}\n\n",
        CachePoint(),
        listing_block,
        CachePoint(),
        f"\n# The criterion: {criterion.name}\n\n{criterion.body}\n",
    ]


async def by_model(
    criterion: Criterion,
    listing: ListingData,
    reading: ImageReading | None,
    settings: Settings,
    preamble: str,
    model=None,
    gate: asyncio.Semaphore | None = None,
) -> Grade:
    """Ask the model, holding `gate` for the call and nothing else.

    `gate` is optional and unbounded by default, so every existing caller -
    the live pipeline, `rescore`, every test in this module - is unaffected.
    Only a caller that fans out many of these at once, namely the backfill,
    passes one, and it bounds the calls in flight rather than the Listings.
    """
    # `model_name`, not `name`: pydantic-ai's `name` is the span's label, and
    # a local of the same name holding the model read as a bug beside it.
    model_name = model or settings.evaluation.model
    agent = Agent(
        model_name,
        output_type=ModelGrade,
        system_prompt=CRITERION_SYSTEM_PROMPT,
        name="criterion-grader",
    )
    prompt = build_criterion_prompt(criterion, listing, reading, preamble)
    try:
        # pydantic-ai's own instrumentation reports the call - tokens, latency,
        # retries - but not which Criterion asked for it, and
        # that is the whole question when one of them is slow or expensive.
        # The wait for `gate` is inside the span deliberately: time queued
        # behind the concurrency cap is time the run is spending.
        with span("grade criterion", criterion=criterion.slug, model=str(model_name)):
            if gate is not None:
                async with gate:
                    result = await agent.run(prompt)
            else:
                result = await agent.run(prompt)
    except Exception as exc:  # noqa: BLE001 - one Criterion, not the Listing
        # An outage costs this Criterion and nothing else. The Listing still
        # gets a score off whatever else was determined, and the coverage
        # figure is what says the score is thin.
        #
        # `graded_by` still records the attempt rather than staying blank: a
        # call that failed part-way still cost money, and a spend count that
        # only sees successes would undercount exactly the runs worth knowing
        # about - the ones where the provider was struggling.
        log.warning("grading %s failed: %s", criterion.slug, exc)
        return Grade(
            criterion=criterion.slug,
            value=None,
            determined=False,
            evidence="The model could not be reached.",
            graded_by=f"model:{model_name} (failed)",
        )
    answer = result.output
    return Grade(
        criterion=criterion.slug,
        value=answer.grade,
        determined=answer.grade is not None,
        evidence=answer.evidence,
        graded_by=f"model:{model_name}",
        concern=answer.concern,
        question=answer.question,
    )


async def grade_one(
    criterion: Criterion,
    listing: ListingData,
    reading: ImageReading | None,
    settings: Settings,
    preamble: str,
    model=None,
    gate: asyncio.Semaphore | None = None,
) -> Grade:
    """Deterministic first, then the model fallback, then the unknown rule."""
    grade = deterministic(
        criterion, listing, reading, Path(settings.evaluation.criteria_dir)
    )
    if grade.value is None and (criterion.is_model or criterion.fallback == "model"):
        asked = await by_model(criterion, listing, reading, settings, preamble, model, gate=gate)
        # The model answers the Criterion; it is not asked about anything the
        # deterministic pass noticed on the way past. `floor_area` hands over
        # a floor area the advert and the certificate disagree on, and that is
        # a question for an agent whatever grade comes back.
        #
        # The deterministic finding wins where there are two, and deliberately:
        # it is grounded in a contradiction between two documents that the model
        # was never asked to resolve, and the model's own question about the
        # same Criterion is the vaguer version of it. The cost is a model
        # concern dropped on the one Criterion that sets one of its own, which
        # is a great deal cheaper than losing the contradiction.
        grade = replace(
            asked,
            concern=grade.concern or asked.concern,
            question=grade.question or asked.question,
        )
    return apply_unknown(criterion, grade)


async def grade_listing(
    criteria: list[Criterion],
    listing: ListingData,
    reading: ImageReading | None,
    settings: Settings,
    preamble: str,
    model=None,
    gate: asyncio.Semaphore | None = None,
) -> list[tuple[Criterion, Grade]]:
    """Every Criterion about one Listing: one call first, then the rest at once.

    Concurrent because the calls are independent by construction, which is what
    isolating each Criterion buys. The wall clock is two calls, not N.

    THE FIRST CALL GOES ALONE, AND THAT IS THE WHOLE POINT. Every model call
    about one Listing sends the same ~4,000-token prefix - the brief, then the
    Listing - and `build_criterion_prompt` marks it cacheable. But Anthropic's
    cache entry does not exist until the first response begins: "If you need
    cache hits for parallel requests, wait for the first response before
    sending subsequent requests." Firing all seven together means seven misses,
    seven writes, and seven times the 1.25x write premium - measured at 23%
    MORE than sending no breakpoints at all. Warming the cache with one call
    turns the other six into reads at 0.1x, and 30,450 billed input tokens per
    Listing into about 6,600.

    So the cost of this arrangement is one model call's latency per Listing,
    paid once, and the thing it buys is the entire saving. Anyone tempted to
    put the plain `gather` back should first check what the graders are
    actually spending: `cache_read_input_tokens` at zero is what this looked
    like before.

    The warm call is the first Criterion the model always grades - `is_model`,
    not `fallback: model`, because a fallback resolves deterministically often
    enough that it cannot be relied on to make a call at all. With no such
    Criterion in the list nothing is warmed and nothing is serialised, which is
    exactly the old behaviour.

    `gate`, passed through to every model call, is what lets a caller fanning
    this out over many Listings at once - the backfill - cap the calls in
    flight across the whole run rather than only within one Listing.
    """
    warm = next((criterion for criterion in criteria if criterion.is_model), None)
    graded: dict[str, Grade] = {}
    if warm is not None:
        graded[warm.slug] = await grade_one(
            warm, listing, reading, settings, preamble, model, gate=gate
        )
    rest = [criterion for criterion in criteria if criterion.slug not in graded]
    fanned = await asyncio.gather(
        *(
            grade_one(criterion, listing, reading, settings, preamble, model, gate=gate)
            for criterion in rest
        )
    )
    graded.update(zip((criterion.slug for criterion in rest), fanned, strict=True))
    # Rebuilt in the caller's order. The warm Criterion is graded out of turn
    # and every caller downstream reads these pairs positionally.
    return [(criterion, graded[criterion.slug]) for criterion in criteria]
