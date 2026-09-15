"""Which way the windows face, read off the floorplan's compass.

Measured before it was built, on eight live floorplans. Two findings shaped
every decision here, and each has a test named after it:

- Asked outright which way the windows face, the model reads the compass
  correctly and then rotates the plan wrongly. One plan whose arrow it read at
  222 degrees came back NW, NE and SE on three consecutive runs.
- Where the indicator is subtle it does not read it at all, it assumes north is
  up. On the Koa House plan - a 40-pixel circle with a rotated N pointing left -
  it answered "north is up" three times out of three, which made a west-facing
  flat read as south-facing. Consistency across runs is not correctness, and
  that is the whole reason for the agreement check being a SECOND read rather
  than a repeat of the first question.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from PIL import Image
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart

from flat_scout.config import Settings
from flat_scout.models import ImageReading, ListingData
from flat_scout.vision import aspect_from_bearings, read_images, render_reading

EPC_URL = "https://media.example.com/epc.png"
PLAN_URL = "https://media.example.com/plan.png"
PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
    b"\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05"
    b"\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def settings_for() -> Settings:
    settings = Settings()
    settings.evaluation.vision_model = "openrouter:z-ai/glm-5.3-flash"
    return settings


def a_listing(**overrides) -> ListingData:
    fields = {
        "portal": "rightmove",
        "portal_id": "1",
        "url": "https://www.rightmove.co.uk/properties/1",
        "epc_image_url": EPC_URL,
        "floorplan_url": PLAN_URL,
    }
    fields.update(overrides)
    return ListingData(**fields)


def reader(*answers: dict) -> FunctionModel:
    """A vision model that answers each successive call from `answers`.

    Reading a Listing's images takes up to three calls: the combined reading,
    then one focused compass read, then a second to confirm it. A test says
    what each of them returns, and the first entry is the combined read.
    """
    remaining = list(answers)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        answer = remaining.pop(0) if remaining else {}
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, answer)]
        )

    return FunctionModel(respond)


async def reading_for(*answers: dict, listing: ListingData | None = None) -> ImageReading:
    async with httpx.AsyncClient() as client:
        return await read_images(
            listing or a_listing(), settings_for(), client, model=reader(*answers)
        )


def mock_media() -> None:
    for url in (EPC_URL, PLAN_URL):
        respx.get(url).mock(
            return_value=httpx.Response(
                200, content=PNG_BYTES, headers={"content-type": "image/png"}
            )
        )


# --- the arithmetic, which is the whole point of doing it outside the model ---


@pytest.mark.parametrize(
    "north,windows,expected",
    [
        (0, 0, "N"),  # north up the page, windows facing the top
        (0, 90, "E"),
        (0, 180, "S"),
        (0, 270, "W"),
        (270, 180, "W"),  # Koa House: north points left, windows at the bottom
        (225, 45, "S"),  # Chronicle Tower: north down-left, windows up-right
        (90, 0, "W"),  # north to the right, so the top of the page is west
        (135, 135, "N"),  # windows along the north axis whatever the rotation
        (350, 10, "N"),  # 20 degrees apart across the wrap, still one point
        (10, 350, "N"),  # ...and the same the other way round
        (0, 22, "N"),  # rounds down to the nearest of eight points
        (0, 23, "NE"),  # ...and up
        (0, 359, "N"),
    ],
)
def test_the_aspect_is_the_difference_between_two_image_bearings(north, windows, expected):
    assert aspect_from_bearings(north, windows) == expected


def test_an_unread_compass_yields_no_aspect():
    assert aspect_from_bearings(None, 180) is None
    assert aspect_from_bearings(270, None) is None
    assert aspect_from_bearings(None, None) is None


def test_the_koa_house_plan_is_west_facing_not_south():
    """The regression this whole feature exists for.

    Asked directly, the model called this flat south-facing three times out of
    three, because it assumed north was up. North points left, the windows and
    balcony are along the bottom of the plan, and the flat faces west - the one
    aspect the couple would rather avoid.
    """
    assert aspect_from_bearings(270, 180) == "W"
    assert aspect_from_bearings(0, 180) == "S"  # what it said when it assumed


# --- the agreement check ---


@pytest.mark.asyncio
@respx.mock
async def test_two_agreeing_reads_are_not_enough_without_a_measurement():
    """Agreement between two readings is not evidence, and never was.

    Both reads say the same thing here and there is no compass to measure, so
    nothing is stored. On the annotated corpus the model agreeing with itself
    was right 11 times in 16 - good enough to log, not to surface as a fact.
    """
    mock_media()
    reading = await reading_for(
        {},
        {"north_clock": 270, "windows_clock": [180]},
        {"north_clock": 270, "windows_clock": [180]},
    )
    assert reading.window_aspects == []
    assert reading.north_clock == 270  # the reading is kept, just not trusted


@pytest.mark.asyncio
@respx.mock
async def test_two_reads_that_disagree_store_no_aspect():
    """Over eight sampled plans, two reads disagreed on three of them.

    A wrong aspect is invisible downstream - it looks exactly like a right one -
    and the couple cannot check it without opening the floorplan themselves. A
    null costs them nothing, because aspect is a preference and not a
    requirement.
    """
    mock_media()
    reading = await reading_for(
        {},
        {"north_clock": 0, "windows_clock": [0]},  # "north is up", the known failure
        {"north_clock": 270, "windows_clock": [0]},
    )
    assert reading.window_aspects == []
    # The first read's bearings survive, so a disagreement can be investigated.
    assert reading.north_clock == 0


@pytest.mark.asyncio
@respx.mock
async def test_two_reads_of_the_window_wall_may_differ_a_little():
    """A wall is still the same wall when two reads put it 15 degrees apart.

    The wall is the half of this that is still read rather than measured, so its
    tolerance is what decides how often an aspect survives at all.
    """
    box = {"compass_box": [8, 8, 20, 20]}
    reading = await reading_of(
        plan_with_compass(270),
        {},
        {**box, "windows_clock": [170]},
        {**box, "windows_clock": [185]},
    )
    assert reading.window_aspects == ["W"]


@pytest.mark.asyncio
@respx.mock
async def test_no_compass_means_no_second_call_at_all():
    """Most plans that fail, fail here - and the saving is not making the call."""
    calls: list[int] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(1)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {})])

    mock_media()
    async with httpx.AsyncClient() as client:
        reading = await read_images(
            a_listing(), settings_for(), client, model=FunctionModel(respond)
        )
    assert reading.window_aspects == []
    # The combined read, and one compass read that found nothing to confirm.
    assert len(calls) == 2


@pytest.mark.asyncio
@respx.mock
async def test_a_failed_second_read_costs_the_aspect_and_nothing_else():
    calls: list[int] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(1)
        if len(calls) == 1:
            return ModelResponse(
                parts=[ToolCallPart(info.output_tools[0].name, {"epc_band": "C"})]
            )
        if len(calls) == 2:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        info.output_tools[0].name,
                        {"north_clock": 270, "windows_clock": [180]},
                    )
                ]
            )
        raise RuntimeError("the provider fell over")

    mock_media()
    async with httpx.AsyncClient() as client:
        reading = await read_images(
            a_listing(), settings_for(), client, model=FunctionModel(respond)
        )
    assert reading.window_aspects == []
    assert reading.epc_band == "C"  # the rest of the reading survives


@pytest.mark.asyncio
@respx.mock
async def test_a_compass_reported_without_a_floorplan_is_discarded():
    """No plan was sent, so there was no compass to see."""
    respx.get(EPC_URL).mock(
        return_value=httpx.Response(
            200, content=PNG_BYTES, headers={"content-type": "image/png"}
        )
    )
    reading = await reading_for(
        {"north_clock": 270, "windows_clock": [180], "window_aspects": ["W"], "epc_band": "B"},
        listing=a_listing(floorplan_url=None),
    )
    # ...and no compass call is made at all, there being no plan to read.
    assert reading.window_aspects == []
    assert reading.north_clock is None
    assert reading.epc_band == "B"


@pytest.mark.asyncio
@respx.mock
async def test_the_models_own_aspect_is_never_trusted():
    """Same principle as `epc_band_source`: computed here, not asked for.

    The model answering the question directly is the failure that started this,
    so an answer it volunteers must not survive even when it looks right.
    """
    box = {"compass_box": [8, 8, 20, 20], "windows_clock": [180], "window_aspects": ["NE"]}
    reading = await reading_of(plan_with_compass(270), {"window_aspects": ["NE"]}, box, box)
    assert reading.window_aspects == ["W"]


# --- what the evaluator is shown ---


def test_only_the_aspect_reaches_the_evaluator_not_the_bearings():
    rendered = render_reading(
        ImageReading(north_clock=270, windows_clock=[180], window_aspects=["W"])
    )
    assert "window_aspects: W" in rendered
    # The bearings are about the picture, not the flat.
    assert "north_clock" not in rendered
    assert "windows_clock" not in rendered


def test_an_unconfirmed_aspect_is_simply_absent():
    assert "window_aspects" not in render_reading(ImageReading(north_clock=270))


# --- enlarging the plan, which is what makes a small compass legible ---


def _plan(width: int, height: int) -> object:
    """A JPEG of a given size, as a _Source carrying it."""
    import io

    from PIL import Image
    from pydantic_ai import BinaryContent

    from flat_scout.vision import _Source

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="JPEG")
    return _Source(BinaryContent(data=buffer.getvalue(), media_type="image/jpeg"))


def _size(content) -> tuple[int, int]:
    import io

    from PIL import Image

    return Image.open(io.BytesIO(content.data)).size


def test_a_small_plan_is_enlarged_before_the_compass_is_read():
    """The 434x378 plan whose 40-pixel compass rose was invisible to the model.

    It answered "north is up" twice on that plan, and agreeing with itself
    proved nothing. Enlarged, it read the rose correctly.
    """
    from flat_scout.vision import ASPECT_TARGET_PIXELS, _enlarged

    enlarged = _enlarged(_plan(434, 378))
    assert max(_size(enlarged)) == ASPECT_TARGET_PIXELS
    # The aspect ratio has to survive, or every bearing read off it is wrong.
    assert abs(_size(enlarged)[0] / _size(enlarged)[1] - 434 / 378) < 0.01


def test_a_plan_that_is_already_big_enough_is_sent_untouched():
    from flat_scout.vision import _enlarged

    plan = _plan(1409, 2048)
    assert _enlarged(plan) is plan.content


def test_an_enlargement_that_would_be_too_large_to_send_is_abandoned():
    """The provider's cap still applies, and the original is still readable."""
    from flat_scout import vision

    plan = _plan(400, 400)
    original = vision.MAX_IMAGE_BYTES
    vision.MAX_IMAGE_BYTES = 10
    try:
        assert vision._enlarged(plan) is plan.content
    finally:
        vision.MAX_IMAGE_BYTES = original


def test_an_unreadable_plan_is_sent_as_it_arrived():
    from pydantic_ai import BinaryContent

    from flat_scout.vision import _Source, _enlarged

    broken = _Source(BinaryContent(data=b"not an image", media_type="image/jpeg"))
    assert _enlarged(broken) is broken.content


def plan_with_arrow(north: float, size: int = 1400) -> bytes:
    """A plan-sized PNG whose indicator is a bare arrow, with no letter."""
    import io
    import math

    from PIL import ImageDraw

    canvas = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(canvas)
    centre = size // 8
    at = lambda bearing, distance: (  # noqa: E731 - a formula, not a function
        round(centre + distance * math.sin(math.radians(bearing))),
        round(centre - distance * math.cos(math.radians(bearing))),
    )
    draw.line([at(north + 180, size // 40), at(north, size // 60)], fill=0, width=4)
    draw.polygon(
        [at(north, size // 30), at(north + 140, size // 60), at(north - 140, size // 60)], fill=0
    )
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.asyncio
@respx.mock
async def test_both_compass_reads_see_the_same_picture():
    """Two reads of different images would agree by luck, not by evidence."""
    sent: list[bytes] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        part = messages[-1].parts[-1]
        if isinstance(part.content, list):
            for item in part.content:
                if hasattr(item, "data"):
                    sent.append(item.data)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {"compass_box": [8, 8, 20, 20], "windows_clock": [180]},
                )
            ]
        )

    mock_plan(plan_with_compass(270))
    async with httpx.AsyncClient() as client:
        reading = await read_images(
            a_listing(epc_image_url=None), settings_for(), client, model=FunctionModel(respond)
        )
    assert reading.window_aspects == ["W"]
    # The combined read gets the original; both compass reads get one enlargement.
    assert len(sent) == 3
    assert sent[1] == sent[2]


# --- the measured compass, which outranks anything the model says about north ---


def plan_with_compass(north: float, size: int = 1400) -> bytes:
    """A plan-sized PNG carrying a circle-and-letter indicator in its top left."""
    import io
    import math

    from PIL import ImageDraw

    canvas = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(canvas)
    centre, radius = size // 8, size // 40
    draw.ellipse(
        [centre - radius, centre - radius, centre + radius, centre + radius],
        outline=0,
        width=3,
    )
    at = lambda bearing, distance: (  # noqa: E731 - a formula, not a function
        round(centre + distance * math.sin(math.radians(bearing))),
        round(centre - distance * math.cos(math.radians(bearing))),
    )
    draw.line([(centre, centre), at(north, radius)], fill=0, width=3)
    letter = Image.new("L", (size // 24, size // 24), 255)
    mark = ImageDraw.Draw(letter)
    side, stroke = letter.width, max(2, letter.width // 9)
    inset = side // 6
    mark.line([(inset, side - inset), (inset, inset)], fill=0, width=stroke)
    mark.line([(inset, inset), (side - inset, side - inset)], fill=0, width=stroke)
    mark.line([(side - inset, side - inset), (side - inset, inset)], fill=0, width=stroke)
    letter = letter.rotate((360 - north) % 360, resample=Image.BICUBIC, fillcolor=255)
    spot = at(north, radius * 2.2)
    canvas.paste(letter, (spot[0] - letter.width // 2, spot[1] - letter.height // 2))
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


def mock_plan(content: bytes) -> None:
    respx.get(PLAN_URL).mock(
        return_value=httpx.Response(200, content=content, headers={"content-type": "image/png"})
    )


async def reading_of(plan: bytes, *answers: dict) -> ImageReading:
    mock_plan(plan)
    async with httpx.AsyncClient() as client:
        return await read_images(
            a_listing(epc_image_url=None), settings_for(), client, model=reader(*answers)
        )


@pytest.mark.asyncio
@respx.mock
async def test_a_measurement_the_model_confirms_is_the_one_kept():
    """Two estimates by unrelated methods, agreeing. The measured one is stored.

    The model's bearing is approximate and varies between runs; the measurement
    is exact and identical every time. Where they agree, precision costs
    nothing, so the measurement wins.
    """
    box = {"compass_box": [8, 8, 20, 20], "north_clock": 260, "windows_clock": [180]}
    reading = await reading_of(plan_with_compass(270), {}, box, box)
    assert reading.window_aspects == ["W"]
    assert abs((reading.north_clock - 270 + 180) % 360 - 180) <= 25
    assert reading.north_source == "letter-and-position"


@pytest.mark.asyncio
@respx.mock
async def test_the_measurement_stands_even_where_the_model_disagrees():
    """Scored on 24 annotated plans, the measurement is the better instrument.

    A lettered indicator measured off the pixels was right 8 times in 9; the
    model reading the same plans was right 11 times in 16. So a contradiction is
    logged and the measurement kept, rather than both being thrown away - which
    is what the first version of this did, abstaining on 13 plans of 17.
    """
    box = {"compass_box": [8, 8, 20, 20], "north_clock": 0, "windows_clock": [180]}
    reading = await reading_of(plan_with_compass(270), {}, box, box)
    assert reading.window_aspects == ["W"]
    assert reading.north_source == "letter-and-position"


@pytest.mark.asyncio
@respx.mock
async def test_a_bearing_measured_off_an_arrow_is_never_stored():
    """Two right out of seven on the corpus: a coin toss dressed as an instrument.

    An arrow carries no letter, so there is nothing to confirm the direction of
    its shaft, and the head-versus-tail judgement is the whole answer. It is
    still computed and still logged; it is simply not good enough to keep.
    """
    box = {"compass_box": [8, 8, 30, 30], "north_clock": 225, "windows_clock": [180]}
    reading = await reading_of(plan_with_arrow(225), {}, box, box)
    assert reading.window_aspects == []
    assert reading.north_source is None


@pytest.mark.asyncio
@respx.mock
async def test_disagreement_about_the_window_wall_still_loses_the_aspect():
    box = [8, 8, 20, 20]
    reading = await reading_of(
        plan_with_compass(270),
        {},
        {"compass_box": box, "north_clock": 270, "windows_clock": [0]},
        {"compass_box": box, "north_clock": 270, "windows_clock": [180]},
    )
    assert reading.window_aspects == []
    # North was settled; it is the wall that was not.
    assert reading.north_source == "letter-and-position"


@pytest.mark.asyncio
@respx.mock
async def test_a_box_the_wrong_way_round_is_still_usable():
    """Models give these in whatever order they please, and a swap is not a fault.

    Asked for percentages, the model returns pixels on nearly every real plan;
    asked for left-then-right, it sometimes gives them the other way about.
    Neither is worth losing a measurement over.
    """
    swapped = {"compass_box": [20, 20, 8, 8], "north_clock": 270, "windows_clock": [180]}
    reading = await reading_of(plan_with_compass(270), {}, swapped, swapped)
    assert reading.window_aspects == ["W"]


@pytest.mark.asyncio
@respx.mock
async def test_a_box_with_no_area_measures_nothing():
    empty = {"compass_box": [10, 10, 10, 10], "north_clock": 270, "windows_clock": [180]}
    reading = await reading_of(plan_with_compass(270), {}, empty, empty)
    assert reading.window_aspects == []
    assert reading.north_source is None


@pytest.mark.asyncio
@respx.mock
async def test_how_north_was_found_is_recorded_for_a_human_to_audit():
    """The aspect cannot be checked without opening the floorplan, so the
    provenance of the bearing is stored beside it."""
    box = {"compass_box": [8, 8, 20, 20], "north_clock": 225, "windows_clock": [180]}
    reading = await reading_of(plan_with_compass(225), {}, box, box)
    assert reading.north_source is not None
    assert reading.north_source in {
        "letter-and-position",
        "position",
        "arrow",
        "position (upright N)",
        "letter-and-position (upright N)",
    }


def test_one_confirming_wall_cannot_confirm_two():
    """Two reads are a set intersection, not a lookup.

    With walls either side of a single second-read wall, both used to match it,
    so one confirmation became two stored walls - and two aspects on the Listing.
    """
    from flat_scout.vision import _agreed_walls

    assert _agreed_walls([0, 90], [45]) == [22]
    assert _agreed_walls([0, 90], [0, 90]) == [0, 90]
    assert _agreed_walls([0], []) == []


def test_an_unconfirmed_floor_does_not_survive_the_guard():
    """That fabrication used to slip past by never being cleared."""
    from flat_scout.vision import _agreed_floor

    assert _agreed_floor("Ground Floor", None) is None


@pytest.mark.asyncio
@respx.mock
async def test_by_default_only_the_best_confirmed_wall_is_reported():
    """Off by default because it measures better, not because it is truer.

    Thirteen of the twenty-four annotated flats really are dual aspect, so this
    default states fewer true things on purpose - and fewer false ones with them.
    """
    box = [8, 8, 20, 20]
    reading = await reading_of(
        plan_with_compass(270),
        {},
        {"compass_box": box, "windows_clock": [180, 90]},
        {"compass_box": box, "windows_clock": [181, 130]},
    )
    # 180/181 agree to a degree; 90/130 agree to forty. The tighter pair wins.
    assert reading.windows_clock == [180]
    assert reading.window_aspects == ["W"]


@pytest.mark.asyncio
@respx.mock
async def test_the_flag_reports_every_wall_a_corner_flat_has():
    box = [8, 8, 20, 20]
    settings = settings_for()
    settings.features.multi_aspect = True
    mock_plan(plan_with_compass(270))
    async with httpx.AsyncClient() as client:
        reading = await read_images(
            a_listing(epc_image_url=None),
            settings,
            client,
            model=reader(
                {},
                {"compass_box": box, "windows_clock": [180, 90]},
                {"compass_box": box, "windows_clock": [181, 130]},
            ),
        )
    assert reading.windows_clock == [180, 110]
    assert len(reading.window_aspects) == 2
