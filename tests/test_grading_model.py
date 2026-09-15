import asyncio

import pytest
from pydantic_ai import CachePoint
from pydantic_ai.messages import UserPromptPart
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.test import TestModel

from flat_scout.config import Settings
from flat_scout.criteria import Criterion
from flat_scout.grading import build_criterion_prompt, grade_listing, grade_one
from flat_scout.models import ImageReading, ListingData

PREAMBLE = "# Their brief\n\nThey want a warm flat."
BAND_MAP = {"A": 10, "B": 10, "C": 8, "D": 5, "E": 2, "F": 0, "G": 0}


def prompt_text(parts) -> str:
    """The prompt as the model reads it, with the cache markers dropped.

    `build_criterion_prompt` returns content blocks rather than one string so
    it can mark cache boundaries between them. The model sees the blocks
    concatenated, and that is what the tests about wording assert against.
    """
    return "".join(part for part in parts if isinstance(part, str))


def boundaries(parts) -> list[int]:
    return [i for i, part in enumerate(parts) if isinstance(part, CachePoint)]


def a_listing(**kwargs) -> ListingData:
    return ListingData(
        portal="rightmove", portal_id="1", url="u", address="Riverlight Quay",
        description="A modern flat with a secure bike store.", **kwargs,
    )


def a_criterion(slug: str, grade: dict, **kwargs) -> Criterion:
    return Criterion(
        slug=slug, name=slug, description="d", weight=1, grade=grade,
        body="## Rubric\n**10** — good.", **kwargs,
    )


def test_the_prompt_puts_the_shared_material_before_the_rubric():
    """Order is what makes the prefix cacheable across a Listing's calls."""
    prompt = prompt_text(
        build_criterion_prompt(a_criterion("quiet", {"model": True}), a_listing(), None, PREAMBLE)
    )
    assert prompt.index("They want a warm flat") < prompt.index("Riverlight Quay")
    assert prompt.index("Riverlight Quay") < prompt.index("**10** — good.")


def test_the_prompt_carries_one_rubric_and_names_the_criterion():
    prompt = prompt_text(
        build_criterion_prompt(a_criterion("quiet", {"model": True}), a_listing(), None, PREAMBLE)
    )
    assert prompt.count("## Rubric") == 1
    assert "# The criterion: quiet" in prompt


def test_a_cache_boundary_follows_the_brief_and_another_follows_the_listing():
    """Two boundaries, because two different things repeat at two rates.

    The brief is identical on every grader call this app ever makes; the
    Listing block is identical across the calls about one Listing. Marking
    both means the second Listing of a batch reads the brief back rather than
    paying to write it again.
    """
    parts = build_criterion_prompt(
        a_criterion("quiet", {"model": True}), a_listing(), None, PREAMBLE
    )
    first, second = boundaries(parts)
    brief = prompt_text(parts[:first])
    assert "They want a warm flat" in brief
    assert "Riverlight Quay" not in brief, "the Listing is inside the brief's prefix"
    assert "Riverlight Quay" in prompt_text(parts[first:second])


def test_the_criterion_stays_outside_the_cached_prefix():
    """Caching a block that changes on every call would write the cache and
    never read it - the 1.25x premium with none of the discount."""
    parts = build_criterion_prompt(
        a_criterion("quiet", {"model": True}), a_listing(), None, PREAMBLE
    )
    last = boundaries(parts)[-1]
    assert "**10** — good." in prompt_text(parts[last:])
    assert "**10** — good." not in prompt_text(parts[:last])


class Recording(TestModel):
    """A TestModel that records when each call starts and ends.

    Defined at module level with a caller-supplied list because TestModel is a
    dataclass: a mutable class attribute on a subclass is shared between
    instances, and a per-test list is what makes the assertions readable.
    """

    def __init__(self, events: list[str], **kwargs):
        super().__init__(**kwargs)
        self._events = events

    async def request(self, *args, **kwargs):
        self._events.append("start")
        # A real await, so a sibling running concurrently genuinely gets to
        # start here. `sleep(0)` yields only once and is too weak to tell the
        # two arrangements apart.
        await asyncio.sleep(0.01)
        result = await super().request(*args, **kwargs)
        self._events.append("end")
        return result


def four_model_criteria() -> list[Criterion]:
    return [a_criterion(f"m{i}", {"model": True}) for i in range(4)]


@pytest.mark.asyncio
async def test_the_first_model_call_finishes_before_the_others_begin():
    """The whole saving depends on this and nothing else.

    A prefix cache entry does not exist until the first response begins, so
    seven calls fired together all miss and all pay the 1.25x write premium -
    the change costs 23% more than no caching at all. One call has to land
    first for the other six to have something to read.
    """
    events: list[str] = []
    await grade_listing(
        four_model_criteria(),
        a_listing(),
        None,
        Settings(),
        PREAMBLE,
        model=Recording(events, custom_output_args={"grade": 7.0, "evidence": "fine"}),
    )
    assert events[:2] == ["start", "end"], events


@pytest.mark.asyncio
async def test_the_calls_after_the_first_still_run_concurrently():
    """Warming the cache costs one call's latency, not the Listing's.

    Serialising all of them would be the easy way to get the assertion above
    and would turn one call's wall clock into seven.
    """
    events: list[str] = []
    await grade_listing(
        four_model_criteria(),
        a_listing(),
        None,
        Settings(),
        PREAMBLE,
        model=Recording(events, custom_output_args={"grade": 7.0, "evidence": "fine"}),
    )
    assert events[2:5] == ["start", "start", "start"], events


@pytest.mark.asyncio
async def test_a_model_criterion_is_graded_by_the_model():
    model = TestModel(custom_output_args={"grade": 8.0, "evidence": "modern build"})
    grade = await grade_one(
        a_criterion("quiet", {"model": True}), a_listing(), None, Settings(), PREAMBLE, model=model
    )
    assert grade.value == 8.0
    assert grade.evidence == "modern build"
    assert grade.determined is True
    assert grade.graded_by.startswith("model")


@pytest.mark.asyncio
async def test_a_model_that_cannot_tell_returns_unknown_and_not_a_low_grade():
    model = TestModel(custom_output_args={"grade": None, "evidence": "the advert says nothing"})
    grade = await grade_one(
        a_criterion("quiet", {"model": True}), a_listing(), None, Settings(), PREAMBLE, model=model
    )
    assert grade.value is None
    assert grade.determined is False
    # The evidence can only have come from the model, so this is what tells the
    # null apart from an unreached deterministic() fallthrough - both leave
    # value and determined identical.
    assert grade.evidence == "the advert says nothing"


@pytest.mark.asyncio
async def test_a_determined_field_costs_no_model_call_even_with_a_fallback():
    criterion = a_criterion("warmth", {"field": "epc_band", "map": BAND_MAP}, fallback="model")

    class Explode(TestModel):
        async def request(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("the model was called for a determined Criterion")

    grade = await grade_one(
        criterion, a_listing(), ImageReading(epc_band="C"), Settings(), PREAMBLE, model=Explode()
    )
    assert grade.value == 8
    assert grade.graded_by == "map"


@pytest.mark.asyncio
async def test_the_fallback_runs_where_the_field_is_silent():
    criterion = a_criterion("warmth", {"field": "epc_band", "map": BAND_MAP}, fallback="model")
    model = TestModel(custom_output_args={"grade": 6.0, "evidence": "double glazing mentioned"})
    grade = await grade_one(criterion, a_listing(), ImageReading(), Settings(), PREAMBLE, model=model)
    assert grade.value == 6.0
    assert grade.determined is True


@pytest.mark.asyncio
async def test_the_unknown_default_applies_after_a_fallback_also_fails():
    criterion = a_criterion(
        "warmth", {"field": "epc_band", "map": BAND_MAP}, fallback="model", unknown=4
    )

    class Counting(TestModel):
        # apply_unknown overwrites the evidence with the default's own
        # sentence, so evidence cannot tell us the fallback ran. Counting
        # calls can: the default of 4 is what both a real fallback attempt
        # and a skipped one land on, so only the call count tells them apart.
        calls: int = 0

        async def request(self, *args, **kwargs):
            self.calls += 1
            return await super().request(*args, **kwargs)

    model = Counting(custom_output_args={"grade": None, "evidence": "nothing stated"})
    grade = await grade_one(criterion, a_listing(), ImageReading(), Settings(), PREAMBLE, model=model)
    assert grade.value == 4
    assert grade.determined is False
    assert model.calls == 1


@pytest.mark.asyncio
async def test_a_failed_model_call_costs_one_criterion_and_not_the_listing():
    class Broken(TestModel):
        async def request(self, *args, **kwargs):
            raise RuntimeError("the provider is down")

    grade = await grade_one(
        a_criterion("quiet", {"model": True}), a_listing(), None, Settings(), PREAMBLE, model=Broken()
    )
    assert grade.determined is False
    assert grade.value is None


@pytest.mark.asyncio
async def test_the_model_fallback_keeps_what_the_deterministic_pass_noticed():
    """A floor area two documents disagree on is not a question for the model.

    `floor_area` hands the Criterion over unresolved, and the model may well
    grade it off the prose. The contradiction between the advert and the
    certificate is still worth an agent's time, and replacing the Grade
    wholesale threw it away.
    """
    criterion = a_criterion("floor-area", {"grader": "floor_area"}, fallback="model")
    # The model asks its own vaguer version of the same thing, which is what a
    # model with no floor area in front of it does. The specific question, the
    # one naming the contradiction, is the one worth an agent's single answer.
    model = TestModel(
        custom_output_args={
            "grade": 6.0,
            "evidence": "the advert calls it spacious",
            "question": "How big is the flat?",
        }
    )
    grade = await grade_one(
        criterion,
        a_listing(sqft=529),
        ImageReading(epc_floor_area_sqm=74.0),
        Settings(),
        PREAMBLE,
        model=model,
    )
    assert grade.value == 6.0
    assert grade.determined is True
    assert grade.concern is not None and "529" in grade.concern
    assert grade.question is not None and "EPC" in grade.question


@pytest.mark.asyncio
async def test_the_cache_boundaries_survive_the_trip_to_the_model():
    """The one regression no other test here would notice.

    `by_model` could join the blocks back into a single string and every other
    assertion in this file would still pass: the model reads the same words,
    grades the same way, and caches nothing at all. What makes the prefix
    cacheable is that the markers are still there when the request is built.
    """
    seen: list = []

    class Capturing(TestModel):
        async def request(self, messages, *args, **kwargs):
            seen.append(messages)
            return await super().request(messages, *args, **kwargs)

    await grade_one(
        a_criterion("quiet", {"model": True}),
        a_listing(),
        None,
        Settings(),
        PREAMBLE,
        model=Capturing(custom_output_args={"grade": 7.0, "evidence": "fine"}),
    )
    [messages] = seen
    prompts = [
        part
        for message in messages
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]
    [user] = prompts
    assert isinstance(user.content, list), "the blocks were flattened into one string"
    assert sum(isinstance(block, CachePoint) for block in user.content) == 2


@pytest.mark.asyncio
async def test_grade_listing_returns_one_pair_for_every_criterion_in_order():
    criteria = [
        a_criterion("a-first", {"field": "sqft", "bands": [{"grade": 5}]}),
        a_criterion("b-second", {"model": True}),
    ]
    model = TestModel(custom_output_args={"grade": 7.0, "evidence": "fine"})
    pairs = await grade_listing(criteria, a_listing(sqft=600), None, Settings(), PREAMBLE, model=model)
    assert [criterion.slug for criterion, _ in pairs] == ["a-first", "b-second"]
    assert [grade.criterion for _, grade in pairs] == ["a-first", "b-second"]


class FlakyOnce(TestModel):
    """A 429 on the first request, then the ordinary answer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._failed = False

    async def request(self, *args, **kwargs):
        if not self._failed:
            self._failed = True
            raise ModelHTTPError(status_code=429, model_name="flaky", body=None)
        return await super().request(*args, **kwargs)


@pytest.mark.asyncio
async def test_a_rate_limited_grade_is_asked_again_rather_than_lost(monkeypatch):
    """One 429 used to cost the Criterion: `by_model` reports "could not be
    reached" on any exception, and the coverage drops on a fine Listing."""
    from flat_scout import backoff

    async def no_wait(seconds: float) -> None:
        pass

    monkeypatch.setattr(backoff, "_sleep", no_wait)
    grade = await grade_one(
        a_criterion("m", {"model": True}), a_listing(), None, Settings(), PREAMBLE,
        model=FlakyOnce(custom_output_args={"grade": 7.0, "evidence": "fine"}),
    )
    assert grade.value == 7.0
    assert grade.determined is True
