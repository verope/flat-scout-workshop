from pathlib import Path

import httpx
import pytest
import respx
from pydantic_ai import BinaryContent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from flat_scout.config import Settings
from flat_scout.models import ImageReading, ListingData
from flat_scout.vision import read_images, render_reading

EPC_URL = "https://media.rightmove.co.uk/property-epc/a/1/a.png"
PLAN_URL = "https://media.rightmove.co.uk/property-floorplan/b/1/b.jpeg"

# Page 1 of a real EPC certificate, as Rightmove serves it for roughly a fifth
# of the Listings that publish an EPC at all. Trimmed to the one page we read.
EPC_PDF_BYTES = (Path(__file__).parent / "fixtures" / "epc_certificate.pdf").read_bytes()

# The smallest thing a media host can plausibly serve. The bytes never reach a
# real model in these tests, so their content does not matter - only that they
# are carried, typed and counted correctly.
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 64
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"0" * 64


def a_listing(**overrides) -> ListingData:
    base = dict(
        portal="rightmove",
        portal_id="92053296",
        url="https://www.rightmove.co.uk/properties/92053296",
        epc_image_url=EPC_URL,
        floorplan_url=PLAN_URL,
    )
    base.update(overrides)
    return ListingData(**base)


def settings_for() -> Settings:
    settings = Settings()
    settings.fetch.delay_seconds = 0
    return settings


def mock_media() -> dict[str, respx.Route]:
    """The two media URLs, answering as the Portal's host does.

    The routes are returned so a caller can assert on them. Asking respx for
    `respx.get(EPC_URL)` afterwards would not do: an identical pattern replaces
    the route rather than reading it back, and the replacement answers 200 with
    an empty body - which reads downstream as an absent image.
    """
    return {
        "epc": respx.get(EPC_URL).mock(
            return_value=httpx.Response(
                200, content=PNG_BYTES, headers={"content-type": "image/png"}
            )
        ),
        "floorplan": respx.get(PLAN_URL).mock(
            return_value=httpx.Response(
                200, content=JPEG_BYTES, headers={"content-type": "image/jpeg"}
            )
        ),
    }


def mock_certificate() -> None:
    """The EPC arrives as the official certificate rather than as a graph."""
    respx.get(EPC_URL).mock(
        return_value=httpx.Response(
            200, content=EPC_PDF_BYTES, headers={"content-type": "application/pdf"}
        )
    )


def reading_model(**fields) -> TestModel:
    """A model that answers with exactly these fields, and nothing invented."""
    base = dict(
        epc_band=None,
        epc_band_source=None,
        epc_floor_area_sqm=None,
        epc_property_type=None,
        layout_verdict=None,
        layout_notes=[],
        reception_is_separate=None,
        desk_space=None,
    )
    base.update(fields)
    return TestModel(custom_output_args=base)


@pytest.mark.asyncio
@respx.mock
async def test_a_readable_pair_of_images_becomes_a_structured_reading():
    mock_media()
    async with httpx.AsyncClient() as client:
        reading = await read_images(
            a_listing(),
            settings_for(),
            client,
            model=reading_model(
                epc_band="C",
                layout_verdict="good",
                layout_notes=["reception 4.2m x 3.6m", "desk wall by the window"],
                reception_is_separate=True,
                desk_space=True,
            ),
        )
    assert isinstance(reading, ImageReading)
    assert reading.epc_band == "C"
    assert reading.layout_verdict == "good"
    assert reading.reception_is_separate is True
    assert reading.desk_space is True
    assert len(reading.layout_notes) == 2


@pytest.mark.asyncio
@respx.mock
async def test_both_images_are_sent_to_the_model_as_binary_content():
    """The band is inside the PNG, so the PNG itself has to reach the model."""
    mock_media()
    seen: list[object] = []

    def capture(messages, info: AgentInfo) -> ModelResponse:
        # The first call only. The compass is read in its own call afterwards,
        # which sends the floorplan a second time and would double the list.
        if not seen:
            seen.extend(messages[-1].parts[-1].content)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "epc_band": "D",
                        "layout_verdict": "adequate",
                        "layout_notes": [],
                        "reception_is_separate": None,
                        "desk_space": None,
                    },
                )
            ]
        )

    async with httpx.AsyncClient() as client:
        await read_images(a_listing(), settings_for(), client, model=FunctionModel(capture))

    images = [part for part in seen if isinstance(part, BinaryContent)]
    assert [image.media_type for image in images] == ["image/png", "image/jpeg"]
    assert images[0].data == PNG_BYTES
    assert images[1].data == JPEG_BYTES


@pytest.mark.asyncio
@respx.mock
async def test_an_unreadable_graph_yields_no_band_rather_than_a_guess():
    """Null is the required answer. An invented band is the worst failure here."""
    mock_media()
    async with httpx.AsyncClient() as client:
        reading = await read_images(
            a_listing(), settings_for(), client, model=reading_model()
        )
    assert reading is not None
    assert reading.epc_band is None
    assert reading.layout_verdict is None
    assert reading.layout_notes == []


@pytest.mark.asyncio
@respx.mock
async def test_a_band_is_discarded_when_no_epc_image_was_ever_sent():
    """A band for an image the model never saw is fabricated by construction."""
    respx.get(PLAN_URL).mock(
        return_value=httpx.Response(
            200, content=JPEG_BYTES, headers={"content-type": "image/jpeg"}
        )
    )
    async with httpx.AsyncClient() as client:
        reading = await read_images(
            a_listing(epc_image_url=None),
            settings_for(),
            client,
            model=reading_model(epc_band="A", layout_verdict="good"),
        )
    assert reading.epc_band is None
    assert reading.layout_verdict == "good"


@pytest.mark.asyncio
@respx.mock
async def test_a_layout_verdict_is_discarded_when_no_floorplan_was_ever_sent():
    respx.get(EPC_URL).mock(
        return_value=httpx.Response(
            200, content=PNG_BYTES, headers={"content-type": "image/png"}
        )
    )
    async with httpx.AsyncClient() as client:
        reading = await read_images(
            a_listing(floorplan_url=None),
            settings_for(),
            client,
            model=reading_model(
                epc_band="B",
                layout_verdict="good",
                layout_notes=["invented"],
                reception_is_separate=True,
                desk_space=True,
                bedroom_has_window=False,
            ),
        )
    assert reading.epc_band == "B"
    assert reading.layout_verdict is None
    assert reading.layout_notes == []
    assert reading.reception_is_separate is None
    assert reading.desk_space is None
    # The costliest field to invent: a false answer is a fact about the flat,
    # and without a floorplan there was nothing to see it in.
    assert reading.bedroom_has_window is None


@pytest.mark.asyncio
@respx.mock
async def test_a_listing_with_no_images_never_calls_the_model():
    """Vision costs money per Listing, so nothing to read must cost nothing."""
    calls: list[int] = []

    def explode(messages, info: AgentInfo) -> ModelResponse:
        calls.append(1)
        return ModelResponse(parts=[TextPart("should never happen")])

    async with httpx.AsyncClient() as client:
        reading = await read_images(
            a_listing(epc_image_url=None, floorplan_url=None),
            settings_for(),
            client,
            model=FunctionModel(explode),
        )
    assert reading is None
    assert calls == []


@pytest.mark.asyncio
@respx.mock
async def test_a_dead_media_url_is_treated_as_an_absent_image_not_a_failure():
    """A 404 on the graph must not cost us the floorplan reading too."""
    respx.get(EPC_URL).mock(return_value=httpx.Response(404))
    respx.get(PLAN_URL).mock(
        return_value=httpx.Response(
            200, content=JPEG_BYTES, headers={"content-type": "image/jpeg"}
        )
    )
    async with httpx.AsyncClient() as client:
        reading = await read_images(
            a_listing(),
            settings_for(),
            client,
            model=reading_model(epc_band="C", layout_verdict="poor"),
        )
    assert reading.epc_band is None  # the graph was never seen
    assert reading.layout_verdict == "poor"


@pytest.mark.asyncio
@respx.mock
async def test_a_pdf_epc_is_skipped_because_it_cannot_be_read_as_an_image():
    respx.get(EPC_URL).mock(
        return_value=httpx.Response(
            200, content=b"%PDF-1.4", headers={"content-type": "application/pdf"}
        )
    )
    respx.get(PLAN_URL).mock(
        return_value=httpx.Response(
            200, content=JPEG_BYTES, headers={"content-type": "image/jpeg"}
        )
    )
    async with httpx.AsyncClient() as client:
        reading = await read_images(
            a_listing(), settings_for(), client, model=reading_model(epc_band="E")
        )
    assert reading.epc_band is None


@pytest.mark.asyncio
@respx.mock
async def test_a_vision_outage_raises_so_the_caller_can_fall_back_to_text():
    mock_media()

    def explode(messages, info: AgentInfo) -> ModelResponse:
        raise RuntimeError("provider outage")

    async with httpx.AsyncClient() as client:
        with pytest.raises(Exception):
            await read_images(
                a_listing(), settings_for(), client, model=FunctionModel(explode)
            )


def test_a_reading_renders_only_what_was_actually_read():
    rendered = render_reading(
        ImageReading(epc_band="C", layout_verdict="good", layout_notes=["reception 4.2m"])
    )
    assert "epc_band: C" in rendered
    assert "layout_verdict: good" in rendered
    assert "reception 4.2m" in rendered
    assert "reception_is_separate" not in rendered
    assert "None" not in rendered


def test_a_window_reading_reaches_the_evaluator_both_ways(tmp_path):
    """Both answers are facts about the flat, so neither may be dropped."""
    assert "bedroom_has_window: False" in render_reading(
        ImageReading(bedroom_has_window=False)
    )
    assert "bedroom_has_window: True" in render_reading(
        ImageReading(bedroom_has_window=True)
    )
    assert "bedroom_has_window" not in render_reading(ImageReading(epc_band="C"))


def test_an_empty_reading_says_so_rather_than_rendering_nothing():
    """Silence would let the evaluator assume the images were never there."""
    rendered = render_reading(ImageReading())
    assert "Nothing in the EPC graph or the floorplan could be read." == rendered
    assert "None" not in rendered


# --- The EPC served as a certificate rather than as a graph -------------------


@pytest.mark.asyncio
@respx.mock
async def test_a_pdf_certificate_is_rendered_to_a_png_and_sent_as_an_image():
    """Rasterised locally, so there is one image path and no document handling."""
    mock_certificate()
    respx.get(PLAN_URL).mock(
        return_value=httpx.Response(
            200, content=JPEG_BYTES, headers={"content-type": "image/jpeg"}
        )
    )
    seen: list = []

    def capture(messages, info: AgentInfo) -> ModelResponse:
        # The first call only. The compass is read in its own call afterwards,
        # which sends the floorplan a second time and would double the list.
        if not seen:
            seen.extend(messages[-1].parts[-1].content)
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"epc_band": "B"})]
        )

    async with httpx.AsyncClient() as client:
        await read_images(a_listing(), settings_for(), client, model=FunctionModel(capture))

    images = [part for part in seen if isinstance(part, BinaryContent)]
    assert [image.media_type for image in images] == ["image/png", "image/jpeg"]
    assert images[0].data.startswith(b"\x89PNG\r\n\x1a\n")  # a real rendered PNG
    assert len(images[0].data) > 10_000  # page 1 at a legible scale, not a stub

    # The reader must be told it is a certificate, or it will look for bars.
    said = " ".join(part for part in seen if isinstance(part, str)).lower()
    assert "certificate" in said


@pytest.mark.asyncio
@respx.mock
async def test_a_certificate_carries_the_floor_area_and_the_property_type():
    """Both are printed on page 1, and square footage is the brief's best signal."""
    mock_certificate()
    async with httpx.AsyncClient() as client:
        reading = await read_images(
            a_listing(floorplan_url=None),
            settings_for(),
            client,
            model=reading_model(
                epc_band="B", epc_floor_area_sqm=51.0, epc_property_type="Mid-floor flat"
            ),
        )
    assert reading.epc_band == "B"
    assert reading.epc_floor_area_sqm == 51.0
    assert reading.epc_property_type == "Mid-floor flat"
    assert reading.epc_band_source == "certificate"


@pytest.mark.asyncio
@respx.mock
async def test_a_graph_can_never_yield_a_floor_area_or_a_property_type():
    """Neither is printed on a graph, so either one would have been invented."""
    mock_media()
    async with httpx.AsyncClient() as client:
        reading = await read_images(
            a_listing(),
            settings_for(),
            client,
            model=reading_model(
                epc_band="C", epc_floor_area_sqm=51.0, epc_property_type="Mid-floor flat"
            ),
        )
    assert reading.epc_band == "C"
    assert reading.epc_floor_area_sqm is None
    assert reading.epc_property_type is None
    assert reading.epc_band_source == "graph"


@pytest.mark.asyncio
@respx.mock
async def test_the_band_source_is_derived_from_what_we_sent_not_from_the_model():
    """We know which image we sent; asking the model to report it invites a lie."""
    mock_media()
    async with httpx.AsyncClient() as client:
        reading = await read_images(
            a_listing(),
            settings_for(),
            client,
            model=reading_model(epc_band="C", epc_band_source="certificate"),
        )
    assert reading.epc_band_source == "graph"


@pytest.mark.asyncio
@respx.mock
async def test_no_band_means_no_band_source():
    mock_certificate()
    async with httpx.AsyncClient() as client:
        reading = await read_images(
            a_listing(floorplan_url=None), settings_for(), client, model=reading_model()
        )
    assert reading.epc_band is None
    assert reading.epc_band_source is None


@pytest.mark.asyncio
@respx.mock
async def test_an_unrenderable_pdf_falls_back_to_skipping_the_epc():
    """The fallback the rasteriser replaced must still hold when it fails."""
    respx.get(EPC_URL).mock(
        return_value=httpx.Response(
            200, content=b"%PDF-1.4 not really a pdf", headers={"content-type": "application/pdf"}
        )
    )
    respx.get(PLAN_URL).mock(
        return_value=httpx.Response(
            200, content=JPEG_BYTES, headers={"content-type": "image/jpeg"}
        )
    )
    async with httpx.AsyncClient() as client:
        reading = await read_images(
            a_listing(),
            settings_for(),
            client,
            model=reading_model(epc_band="E", layout_verdict="good"),
        )
    assert reading.epc_band is None  # the certificate was never seen
    assert reading.epc_band_source is None
    assert reading.layout_verdict == "good"  # and the floorplan still read


def test_a_floor_area_is_rendered_in_both_units():
    """criteria.md judges space in square feet; the certificate prints metres."""
    rendered = render_reading(
        ImageReading(
            epc_band="B",
            epc_band_source="certificate",
            epc_floor_area_sqm=51.0,
            epc_property_type="Mid-floor flat",
        )
    )
    assert "51" in rendered and "549 sq ft" in rendered
    assert "epc_property_type: Mid-floor flat" in rendered
    assert "epc_band_source: certificate" in rendered
    assert "None" not in rendered


# --- the local image cache ---------------------------------------------------
#
# `check` downloads every picture a Listing links into data/images and stores
# the path beside the URL. From then on this module reads the file: the media
# host is asked once in the life of a Listing rather than on every reading,
# and a host that has since died costs nothing at all.

CACHED_PNG = b"\x89PNG\r\n\x1a\n" + b"cached epc" * 8
CACHED_JPEG = b"\xff\xd8\xff\xe0" + b"cached plan" * 8


def cached_pair(tmp_path) -> tuple[str, str]:
    """The two images on disk, as `images.cache_images` would have left them."""
    directory = tmp_path / "rightmove_92053296"
    directory.mkdir(parents=True)
    (directory / "epc.png").write_bytes(CACHED_PNG)
    (directory / "floorplan.jpg").write_bytes(CACHED_JPEG)
    return str(directory / "epc.png"), str(directory / "floorplan.jpg")


@pytest.mark.asyncio
@respx.mock
async def test_a_cached_pair_of_images_is_read_off_disk_and_never_downloaded(tmp_path):
    """The bytes the model sees come from the file, and the host is not asked.

    The routes are mocked to fail rather than left unmocked: `_download` catches
    everything, so a leaked request would otherwise look like an absent image
    instead of like the mistake it is.
    """
    epc_path, plan_path = cached_pair(tmp_path)
    routes = [
        respx.get(EPC_URL).mock(return_value=httpx.Response(500)),
        respx.get(PLAN_URL).mock(return_value=httpx.Response(500)),
    ]
    seen: list[object] = []

    def capture(messages, info: AgentInfo) -> ModelResponse:
        if not seen:  # the first call only; the compass read sends the plan again
            seen.extend(messages[-1].parts[-1].content)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "epc_band": "C",
                        "layout_verdict": "good",
                        "layout_notes": [],
                        "reception_is_separate": None,
                        "desk_space": None,
                    },
                )
            ]
        )

    async with httpx.AsyncClient() as client:
        reading = await read_images(
            a_listing(epc_image_path=epc_path, floorplan_path=plan_path),
            settings_for(),
            client,
            model=FunctionModel(capture),
        )

    assert not any(route.called for route in routes)
    images = [part for part in seen if isinstance(part, BinaryContent)]
    assert [image.media_type for image in images] == ["image/png", "image/jpeg"]
    assert [image.data for image in images] == [CACHED_PNG, CACHED_JPEG]
    assert reading.epc_band == "C"
    assert reading.layout_verdict == "good"


@pytest.mark.asyncio
@respx.mock
async def test_a_path_with_no_file_behind_it_falls_back_to_the_url(tmp_path):
    """A `data/` copied without its images must still produce a reading."""
    routes = mock_media()
    async with httpx.AsyncClient() as client:
        reading = await read_images(
            a_listing(
                epc_image_path=str(tmp_path / "gone.png"),
                floorplan_path=str(tmp_path / "also-gone.jpg"),
            ),
            settings_for(),
            client,
            model=reading_model(epc_band="D"),
        )
    assert routes["epc"].called and routes["floorplan"].called
    assert reading.epc_band == "D"


@pytest.mark.asyncio
@respx.mock
async def test_a_cached_certificate_is_rasterised_exactly_as_a_downloaded_one(tmp_path):
    """The path is a source of bytes and nothing else: everything downstream of
    the fetch - the PDF rendering, the size cap, the readable-type check - has
    to behave identically whichever way the bytes arrived."""
    certificate = tmp_path / "epc.pdf"
    certificate.write_bytes(EPC_PDF_BYTES)
    route = respx.get(EPC_URL).mock(return_value=httpx.Response(500))
    async with httpx.AsyncClient() as client:
        reading = await read_images(
            a_listing(floorplan_url=None, epc_image_path=str(certificate)),
            settings_for(),
            client,
            model=reading_model(epc_band="B", epc_floor_area_sqm=51.0),
        )
    assert not route.called
    assert reading.epc_band == "B"
    assert reading.epc_band_source == "certificate"
    assert reading.epc_floor_area_sqm == 51.0


def test_enlarging_a_plan_drops_a_colour_profile_pillow_cannot_read_back():
    """A large embedded ICC profile made the enlarged plan unreadable.

    Pillow carries a JPEG's colour profile through `convert` and `resize` and
    writes it into the PNG as a compressed iCCP chunk, and its own guard then
    refuses to inflate anything over `MAX_TEXT_CHUNK` on the way back in. A
    real floorplan carried a 1.5 MB profile and cost that listing both its
    compass measurement and its wall bearings. The first half below is that
    failure, so the assertions after it guard something that actually happens.
    """
    import io

    from PIL import Image

    from flat_scout.vision import _enlarged, _Source

    buffer = io.BytesIO()
    Image.new("RGB", (400, 300)).save(
        buffer, format="JPEG", icc_profile=b"\x00" * 1_200_000
    )
    plan = _Source(BinaryContent(data=buffer.getvalue(), media_type="image/jpeg"))

    # Exactly what the save used to do, profile and all. These bytes are a PNG
    # that Pillow itself will not re-open.
    source = Image.open(io.BytesIO(plan.content.data))
    unreadable = io.BytesIO()
    source.convert("RGB").resize((1400, 1050), Image.LANCZOS).save(
        unreadable, format="PNG", optimize=True
    )
    with pytest.raises(ValueError, match="MAX_TEXT_CHUNK"):
        Image.open(io.BytesIO(unreadable.getvalue())).load()

    result = _enlarged(plan)
    # Not the original handed back by the `except`: the enlargement succeeded.
    assert result.media_type == "image/png"
    Image.open(io.BytesIO(result.data)).load()
    assert "icc_profile" not in Image.open(io.BytesIO(result.data)).info
