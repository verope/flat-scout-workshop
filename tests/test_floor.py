"""The floor, which the Portal does not send and the drawing usually states.

Rightmove's `entranceFloor` resolved to null on all 24 Listings sampled and on
the page fixture, so `exclude_ground_floor` - a hard filter the couple set on
purpose - had never once fired in production. The floorplan names the floor in
its title block far more often, and that is the only structured source there is.

The asymmetry to hold on to: a floor read off a drawing REMOVES a flat from the
search, so it may only come from words actually printed on that drawing. Never
from the stairs, never from the entrance door, and never from sales prose, which
says "moments from the ground floor cafe" as readily as it says anything.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart

from flat_scout.config import Settings
from flat_scout.db import Database
from flat_scout.models import ImageReading, ListingData
from flat_scout.pipeline import image_reading_from_row
from flat_scout.vision import read_images, render_reading

PLAN_URL = "https://media.example.com/plan.png"
PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
    b"\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05"
    b"\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def a_listing(**overrides) -> ListingData:
    fields = {"portal": "rightmove", "portal_id": "1", "url": "u", "floorplan_url": PLAN_URL}
    fields.update(overrides)
    return ListingData(**fields)


def test_a_floor_from_the_plan_fills_an_empty_one(tmp_path):
    db = Database(tmp_path / "flats.db")
    listing_id = db.upsert(a_listing(), source="alert")
    assert db.set_floor_from_plan(listing_id, "14th Floor") is True
    assert db.get(listing_id)["floor"] == "14th Floor"


def test_a_floor_the_portal_gave_is_never_argued_with(tmp_path):
    """A value on the row was read from the page. A drawing does not outrank it."""
    db = Database(tmp_path / "flats.db")
    listing_id = db.upsert(a_listing(floor="Second"), source="alert")
    assert db.set_floor_from_plan(listing_id, "Ground Floor") is False
    assert db.get(listing_id)["floor"] == "Second"


def test_where_the_floor_came_from_is_recorded(tmp_path):
    db = Database(tmp_path / "flats.db")
    listing_id = db.upsert(a_listing(), source="alert")
    db.set_floor_from_plan(listing_id, "Ground Floor")
    kinds = [event["kind"] for event in db.events_for(listing_id)]
    assert "floor_from_plan" in kinds


def test_the_reading_round_trips_the_plan_s_own_words(tmp_path):
    db = Database(tmp_path / "flats.db")
    listing_id = db.upsert(a_listing(), source="alert")
    db.set_image_reading(listing_id, ImageReading(floor_text="Lower Ground"), "m")
    assert image_reading_from_row(db.get(listing_id)).floor_text == "Lower Ground"


def test_the_floor_reaches_the_evaluator_named_for_what_it_is():
    """`floor_from_the_plan`, not `floor`: the evaluator should know the source."""
    rendered = render_reading(ImageReading(floor_text="14th Floor"))
    assert "floor_from_the_plan: 14th Floor" in rendered
    assert "floor_from_the_plan" not in render_reading(ImageReading(epc_band="C"))


def reader(answer: dict) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, answer)])

    return FunctionModel(respond)


@pytest.mark.asyncio
@respx.mock
async def test_a_floor_claimed_with_no_plan_to_read_it_from_is_discarded():
    """The guard that does not depend on the model behaving.

    No floorplan was sent, so there was no title block, so the words were
    invented - and an invented "Ground Floor" removes the flat outright.
    """
    settings = Settings()
    settings.evaluation.vision_model = "openrouter:z-ai/glm-5.3-flash"
    respx.get("https://media.example.com/epc.png").mock(
        return_value=httpx.Response(200, content=PNG, headers={"content-type": "image/png"})
    )
    async with httpx.AsyncClient() as client:
        reading = await read_images(
            a_listing(floorplan_url=None, epc_image_url="https://media.example.com/epc.png"),
            settings,
            client,
            model=reader({"floor_text": "Ground Floor", "epc_band": "C"}),
        )
    assert reading.floor_text is None
    assert reading.epc_band == "C"


@pytest.mark.asyncio
@respx.mock
async def test_two_reads_must_print_the_same_words():
    """The guard one plan in the corpus bought.

    That drawing contains neither "floor" nor "ground" anywhere - confirmed by
    OCR over the whole image - and came back as "Ground Floor". A ground-floor
    answer is the one that removes a Listing outright, so it may not rest on a
    single reading.
    """
    from flat_scout.vision import _agreed_floor

    assert _agreed_floor("Ground Floor", "Ground Floor") == "Ground Floor"
    assert _agreed_floor("GROUND FLOOR", "Ground  Floor") == "GROUND FLOOR"  # same words
    assert _agreed_floor("Ground Floor", "First Floor") is None
    assert _agreed_floor("Ground Floor", None) is None
    assert _agreed_floor(None, None) is None


def test_a_floor_from_a_plan_never_removes_a_listing_by_itself(tmp_path):
    """It informs the evaluator; it does not reach the hard filter.

    The filter is terminal and invisible, and the floor here comes from a model
    reading a drawing. A human sees the Listing and decides instead.
    """
    import inspect

    from flat_scout import pipeline

    # `_take` is the body of `process_url`; the public name is now a span
    # around it. The invariant is about the code that does the work, so it
    # follows the work rather than the name.
    source = inspect.getsource(pipeline._take)
    after = source[source.index("set_floor_from_plan") :]
    assert "prefiltered_out" not in after
