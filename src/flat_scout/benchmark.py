"""Scoring the floorplan pipeline against what a human said was there.

Every number here is against `tests/fixtures/floorplans/annotations.toml`, and
never against a second run of the pipeline. Twice now, agreement between two
readings has certified an answer that was simply wrong - a model said "north is
up" three times running on a plan where north points left - so self-consistency
is not evidence and is not reported.

The stages are scored apart because they fail apart and are fixed by different
work: north is measured off the pixels by `compass.py`, while which wall the
windows are on is still read by a model. A single end-to-end number would hide
which of those is costing the answers.

Abstention is scored as its own outcome rather than folded in with being wrong.
On this corpus 7 plans of 24 carry no compass at all, so "there is nothing here
to read" is the correct answer roughly a third of the time, and a pipeline that
says so is doing well rather than failing to answer.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai import BinaryContent

import logging
from contextlib import contextmanager

from flat_scout.annotate import SUFFIXES, Annotation, load_annotations
from flat_scout.compass import _angle_between
from flat_scout.config import Settings
from flat_scout.models import ImageReading
from flat_scout.observe import Progress, span
from flat_scout.vision import _agreed_aspect, _Source, aspect_from_bearings

# One compass point. The aspect is bucketed into eight of them, so a bearing
# this close produces the same answer, and anything tighter would score a
# difference the couple never see.
WITHIN = 45

log = logging.getLogger(__name__)


@contextmanager
def _watch_for_failures(troubled: list[str], portal_id: str):
    """Record a plan whose reading logged a failure rather than answering."""

    class Catcher(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING and "could not read the compass" in record.getMessage():
                troubled.append(portal_id)

    catcher = Catcher()
    vision_log = logging.getLogger("flat_scout.vision")
    vision_log.addHandler(catcher)
    try:
        yield
    finally:
        vision_log.removeHandler(catcher)


@dataclass
class Outcome:
    """What happened on one plan, at one stage."""

    portal_id: str
    truth: str
    got: str
    verdict: str  # right | wrong | abstained | correctly-silent


@dataclass
class Stage:
    name: str
    outcomes: list[Outcome] = field(default_factory=list)
    # How many real walls were found across the corpus, out of how many exist.
    # The verdicts say whether an answer was safe; this says how complete it was,
    # and a dual-aspect flat is only half described by one of its walls.
    found: int = 0
    wanted: int = 0

    def count(self, verdict: str) -> int:
        return sum(1 for outcome in self.outcomes if outcome.verdict == verdict)

    @property
    def answerable(self) -> int:
        """Plans where there was something to get right."""
        return sum(1 for o in self.outcomes if o.verdict != "correctly-silent")


def _bearings(entry: Annotation) -> list[int]:
    return list(entry.windows_clock)


def score_north(entry: Annotation, north: int | None) -> Outcome:
    """Did we find the compass, and did we read it the way a human did?"""
    truth = "none" if entry.north_clock is None else str(entry.north_clock)
    got = "none" if north is None else str(north)
    if entry.north_clock is None:
        # Nothing is drawn. Saying so is the right answer; inventing a bearing
        # is the failure this corpus exists to catch.
        verdict = "correctly-silent" if north is None else "wrong"
    elif north is None:
        verdict = "abstained"
    else:
        verdict = "right" if _angle_between(north, entry.north_clock) <= WITHIN else "wrong"
    return Outcome(entry.portal_id, truth, got, verdict)


def score_windows(entry: Annotation, walls: list[int]) -> Outcome:
    """Right if every wall named is one the annotator listed, and at least one is.

    Not a match on the whole set. A flat with three glazed walls where two were
    found is a good answer, not a failure - what must not happen is naming a
    wall the flat does not have, because that becomes an aspect on the Listing.
    """
    truth = "/".join(str(bearing) for bearing in _bearings(entry)) or "none"
    got = "/".join(str(bearing) for bearing in walls) or "none"
    if not _bearings(entry):
        return Outcome(entry.portal_id, truth, got, "correctly-silent" if not walls else "wrong")
    if not walls:
        return Outcome(entry.portal_id, truth, got, "abstained")
    real = [
        wall
        for wall in walls
        if any(_angle_between(wall, bearing) <= WITHIN for bearing in _bearings(entry))
    ]
    return Outcome(entry.portal_id, truth, got, "right" if len(real) == len(walls) else "wrong")


def found_of_true(entry: Annotation, walls: list[int]) -> tuple[int, int]:
    """How many of the flat's real walls were named, out of how many there are.

    One prediction may account for one real wall, and no more. Counting matches
    without consuming them inflates the figure: a single wall at 45 degrees sits
    within tolerance of annotated walls at both 0 and 90, and would score two
    walls found from one answer. This is the number quoted as "N of 41 real
    glazed walls were found", so it had been overstating the pipeline.
    """
    unclaimed = list(walls)
    hit = 0
    for bearing in _bearings(entry):
        nearest = min(
            unclaimed, key=lambda wall: _angle_between(wall, bearing), default=None
        )
        if nearest is None or _angle_between(nearest, bearing) > WITHIN:
            continue
        unclaimed.remove(nearest)
        hit += 1
    return hit, len(_bearings(entry))


def true_aspects(entry: Annotation) -> set[str]:
    """Every compass direction this flat genuinely faces, per the annotation."""
    if entry.north_clock is None:
        return set()
    found = {aspect_from_bearings(entry.north_clock, bearing) for bearing in _bearings(entry)}
    return {aspect for aspect in found if aspect is not None}


def score_aspect(entry: Annotation, aspects: list[str]) -> Outcome:
    """Right if every aspect claimed is one the flat really has.

    Same asymmetry as the walls. Missing one of a dual aspect costs the couple a
    fact they never had; inventing one surfaces a wrong fact, and they cannot
    check it without opening the floorplan.
    """
    truth = "/".join(sorted(true_aspects(entry))) or "none"
    got = "/".join(aspects) or "none"
    if not true_aspects(entry):
        return Outcome(entry.portal_id, truth, got, "correctly-silent" if not aspects else "wrong")
    if not aspects:
        return Outcome(entry.portal_id, truth, got, "abstained")
    ok = all(aspect in true_aspects(entry) for aspect in aspects)
    return Outcome(entry.portal_id, truth, got, "right" if ok else "wrong")


async def run(
    settings: Settings,
    corpus: Path,
    annotations: dict[str, Annotation] | None = None,
    model=None,
    limit: int = 0,
) -> tuple[list[Stage], dict[str, ImageReading]]:
    """Read every annotated plan and score the result. Writes nothing."""
    entries = [
        entry
        for entry in (annotations or load_annotations()).values()
        if entry.done and (corpus / entry.file).exists()
    ]
    entries.sort(key=lambda entry: entry.portal_id)
    if limit:
        entries = entries[:limit]

    # A provider failure inside `_read_aspect` is caught and returned as an empty
    # reading, which the scorer would count as the model abstaining - or worse,
    # as correctly silent on a plan with no compass. Every entry runs at once, so
    # one rate-limit could move the numbers a whole experiment is read from. The
    # warnings are counted here and reported, so a run that hit trouble says so.
    troubled: list[str] = []

    # Two vision calls per plan, twenty-four plans, all in flight at once.
    watch = Progress(len(entries), "reading floorplans")

    async def one(entry: Annotation) -> tuple[str, ImageReading]:
        data = (corpus / entry.file).read_bytes()
        media = {suffix: kind for kind, suffix in SUFFIXES.items()}
        source = _Source(BinaryContent(data=data, media_type=media[(corpus / entry.file).suffix]))
        with _watch_for_failures(troubled, entry.portal_id):
            with span("benchmark plan", portal_id=entry.portal_id, file=entry.file):
                reading = await _agreed_aspect(ImageReading(), source, settings, model)
        watch.tick(failed=entry.portal_id in troubled)
        return entry.portal_id, reading

    # Closed on the way out however that happens; the beat is a thread.
    with watch:
        readings = dict(await asyncio.gather(*(one(entry) for entry in entries)))
    if troubled:
        stages_note = ", ".join(sorted(set(troubled)))
        log.warning("provider trouble on %d plan(s): %s", len(set(troubled)), stages_note)
    stages = [Stage("north"), Stage("window walls"), Stage("aspects")]
    for entry in entries:
        reading = readings[entry.portal_id]
        stages[0].outcomes.append(score_north(entry, reading.north_clock))
        stages[1].outcomes.append(score_windows(entry, reading.windows_clock))
        stages[2].outcomes.append(score_aspect(entry, reading.window_aspects))
        found, total = found_of_true(entry, reading.windows_clock)
        stages[1].found += found
        stages[1].wanted += total
    return stages, readings, troubled


def render(
    stages: list[Stage], readings: dict[str, ImageReading], troubled: list[str] | None = None
) -> str:
    lines: list[str] = ["Scored against the annotations, not against a second run.", ""]
    if troubled:
        lines.append(
            f"  WARNING: {len(set(troubled))} plan(s) hit a provider failure and are "
            "scored as if the model had nothing to say. Re-run before believing this."
        )
        lines.append("")
    for stage in stages:
        right, wrong = stage.count("right"), stage.count("wrong")
        quiet, silent = stage.count("abstained"), stage.count("correctly-silent")
        of = stage.answerable
        share = f"{right}/{of}" if of else "0/0"
        lines.append(
            f"  {stage.name:<12} {share:>7} right   {wrong} wrong   "
            f"{quiet} abstained   {silent} correctly silent"
        )
    aspects = stages[2]
    partly = sum(
        1
        for outcome in aspects.outcomes
        if outcome.verdict in ("right", "wrong")
        and set(outcome.got.split("/")) & set(outcome.truth.split("/"))
    )
    lines.append("")
    lines.append(
        f"  {partly} of {aspects.answerable} plans got at least one aspect right, "
        "against the stricter count above, which requires every aspect named to be real."
    )
    walls = stages[1]
    if walls.wanted:
        lines.append("")
        lines.append(
            f"  {walls.found} of {walls.wanted} real glazed walls were found "
            f"across the corpus."
        )
    lines.append("")
    lines.append("Where it went wrong")
    seen = False
    for stage in stages:
        for outcome in stage.outcomes:
            if outcome.verdict != "wrong":
                continue
            seen = True
            source = readings[outcome.portal_id].north_source or "-"
            lines.append(
                f"    {outcome.portal_id}  {stage.name:<12} said {outcome.got:<10} "
                f"truth {outcome.truth:<14} via {source}"
            )
    if not seen:
        lines.append("    nothing")
    return "\n".join(lines)
