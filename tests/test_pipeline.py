from pathlib import Path

import httpx
import pytest
import respx
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from flat_scout.config import Settings
from flat_scout.db import Database
from flat_scout.pipeline import (
    image_reading_from_row,
    listing_from_row,
    process_url,
)

FIXTURES = Path(__file__).parent / "fixtures"
URL = "https://www.rightmove.co.uk/properties/92053296"
GOOD_HTML = (FIXTURES / "rightmove_listing.html").read_text()
OVER_CAP_HTML = GOOD_HTML.replace("£2,400 pcm", "£4,900 pcm")

# The EPC graph the fixture points at. The fixture has no floorplan, which is
# the ordinary case: most Listings do not publish one.
EPC_URL = (
    "https://media.rightmove.co.uk/property-epc/91bbba5f4/92053296/"
    "91bbba5f4a43d9ec8f92c054d5156a0d.png"
)
PHOTO_URL = (
    "https://media.rightmove.co.uk/property-photo/0be722e28/92053296/"
    "0be722e28dfae7b2eb5b59ab61ccb1a6.jpeg"
)
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 64
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"0" * 64


def mock_epc():
    return respx.get(EPC_URL).mock(
        return_value=httpx.Response(
            200, content=PNG_BYTES, headers={"content-type": "image/png"}
        )
    )


def mock_photo():
    """The headline photograph. Only the image cache ever asks for it."""
    return respx.get(PHOTO_URL).mock(
        return_value=httpx.Response(
            200, content=JPEG_BYTES, headers={"content-type": "image/jpeg"}
        )
    )


def band_model(band: str | None) -> TestModel:
    return TestModel(
        custom_output_args={
            "epc_band": band,
            "layout_verdict": None,
            "layout_notes": [],
            "reception_is_separate": None,
            "desk_space": None,
        }
    )


def verdict_model(seen: list[str]) -> FunctionModel:
    """A text evaluator that records the prompt it was handed."""

    def run(messages, info: AgentInfo) -> ModelResponse:
        seen.append(messages[-1].parts[-1].content)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {"verdict": "borderline", "score": 5.0, "reasons": []},
                )
            ]
        )

    return FunctionModel(run)


def settings_for(tmp_path) -> Settings:
    settings = Settings()
    settings.fetch.delay_seconds = 0
    settings.filters.postcodes = []  # the fixture is in E9, outside the shortlist
    # The shipped defaults are deliberately wide. These tests exercise the
    # filters, so they pin the values they were written against rather than
    # inheriting whatever config.py says today.
    settings.filters.max_price_pcm = 3500
    settings.filters.exclude_ground_floor = True
    # These exercise the holistic evaluator, which is no longer the default.
    # Pinned rather than inherited: a test that reaches a path by omission
    # starts testing a different path the day the default moves. The weighted
    # dispatch has its own test in `test_pipeline_weighted.py`.
    settings.features.weighted_criteria = False
    settings.evaluation.criteria_path = str(tmp_path / "criteria.md")
    (tmp_path / "criteria.md").write_text("Bike storage matters.")
    return settings


@pytest.mark.asyncio
@respx.mock
async def test_a_good_listing_reaches_evaluated(tmp_path):
    respx.get(URL).mock(return_value=httpx.Response(200, html=GOOD_HTML))
    db = Database(tmp_path / "flats.db")
    async with httpx.AsyncClient() as client:
        listing_id = await process_url(
            URL, db, settings_for(tmp_path), "manual", client, model=TestModel()
        )
    row = db.get(listing_id)
    assert row["status"] == "evaluated"
    assert row["price_pcm"] == 2400
    assert row["verdict"] in {"hopeful", "borderline", "reject"}


@pytest.mark.asyncio
@respx.mock
async def test_over_cap_listing_is_prefiltered_out(tmp_path):
    respx.get(URL).mock(return_value=httpx.Response(200, html=OVER_CAP_HTML))
    db = Database(tmp_path / "flats.db")
    async with httpx.AsyncClient() as client:
        listing_id = await process_url(
            URL, db, settings_for(tmp_path), "manual", client, model=TestModel()
        )
    assert db.get(listing_id)["status"] == "prefiltered_out"


@pytest.mark.asyncio
@respx.mock
async def test_alert_price_over_cap_skips_the_download_entirely(tmp_path):
    route = respx.get(URL).mock(return_value=httpx.Response(200, html=GOOD_HTML))
    db = Database(tmp_path / "flats.db")
    async with httpx.AsyncClient() as client:
        listing_id = await process_url(
            URL, db, settings_for(tmp_path), "alert", client,
            model=TestModel(), fields={"price_pcm": 9000},
        )
    assert db.get(listing_id)["status"] == "prefiltered_out"
    assert not route.called


@pytest.mark.asyncio
@respx.mock
async def test_fetch_failure_retries_then_marks_failed(tmp_path):
    respx.get(URL).mock(return_value=httpx.Response(503))
    db = Database(tmp_path / "flats.db")
    settings = settings_for(tmp_path)
    settings.fetch.max_attempts = 1
    async with httpx.AsyncClient() as client:
        listing_id = await process_url(
            URL, db, settings, "manual", client, model=TestModel()
        )
    assert db.get(listing_id)["status"] == "fetch_failed"


@pytest.mark.asyncio
@respx.mock
async def test_a_bot_challenge_counts_as_a_fetch_failure(tmp_path):
    respx.get(URL).mock(
        return_value=httpx.Response(200, html="<html>Just a moment...</html>")
    )
    db = Database(tmp_path / "flats.db")
    settings = settings_for(tmp_path)
    settings.fetch.max_attempts = 1
    async with httpx.AsyncClient() as client:
        listing_id = await process_url(
            URL, db, settings, "manual", client, model=TestModel()
        )
    assert db.get(listing_id)["status"] == "fetch_failed"


@pytest.mark.asyncio
@respx.mock
async def test_a_bot_challenge_does_not_spend_the_retry_budget(tmp_path):
    """The block is aimed at this host, so a second attempt fails too."""
    route = respx.get(URL).mock(
        return_value=httpx.Response(200, html="<html>Just a moment...</html>")
    )
    db = Database(tmp_path / "flats.db")
    settings = settings_for(tmp_path)
    settings.fetch.max_attempts = 4
    async with httpx.AsyncClient() as client:
        listing_id = await process_url(
            URL, db, settings, "manual", client, model=TestModel()
        )
    row = db.get(listing_id)
    assert row["status"] == "fetch_failed"
    assert row["fetch_attempts"] == 1
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_an_openrent_waf_challenge_yields_a_link_only_result_at_once(tmp_path):
    """The 405 + "Human Verification" page listing #107 met, four times over."""
    openrent_url = (
        "https://www.openrent.co.uk/property-to-rent/london/1-bed-flat-london-sw16/1687221"
    )
    challenge = (FIXTURES / "openrent_waf_challenge.html").read_text()
    respx.get(openrent_url).mock(return_value=httpx.Response(405, html=challenge))
    db = Database(tmp_path / "flats.db")
    async with httpx.AsyncClient() as client:
        listing_id = await process_url(
            openrent_url, db, settings_for(tmp_path), "alert", client,
            model=TestModel(), fields={"price_pcm": 2200},
        )
    row = db.get(listing_id)
    # `fetch_failed`, not `unavailable`: the Listing is alive, and a result
    # carrying the link still reaches the couple.
    assert row["status"] == "fetch_failed"
    assert row["fetch_attempts"] == 1
    kinds = [
        event["kind"]
        for event in db.conn.execute(
            "SELECT kind FROM events WHERE listing_id = ?", (listing_id,)
        )
    ]
    assert "bot_challenge" in kinds
    assert "fetch_failed" not in kinds


@pytest.mark.asyncio
@respx.mock
async def test_zoopla_is_evaluated_from_seed_fields_without_any_download(tmp_path):
    """No Zoopla page is ever downloaded."""
    zoopla_url = "https://www.zoopla.co.uk/to-rent/details/73991508"
    route = respx.get(zoopla_url).mock(return_value=httpx.Response(200, html="<html/>"))
    db = Database(tmp_path / "flats.db")
    async with httpx.AsyncClient() as client:
        listing_id = await process_url(
            zoopla_url, db, settings_for(tmp_path), "alert", client,
            model=TestModel(),
            fields={"price_pcm": 3100, "address": "Riverlight Quay, Nine Elms"},
        )
    row = db.get(listing_id)
    assert not route.called
    assert row["status"] == "evaluated"
    assert row["price_pcm"] == 3100


@pytest.mark.asyncio
@respx.mock
async def test_an_already_processed_url_is_never_downloaded_twice(tmp_path):
    route = respx.get(URL).mock(return_value=httpx.Response(200, html=GOOD_HTML))
    db = Database(tmp_path / "flats.db")
    settings = settings_for(tmp_path)
    async with httpx.AsyncClient() as client:
        await process_url(URL, db, settings, "manual", client, model=TestModel())
        await process_url(URL, db, settings, "manual", client, model=TestModel())
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_non_listing_url_returns_none(tmp_path):
    db = Database(tmp_path / "flats.db")
    async with httpx.AsyncClient() as client:
        assert (
            await process_url(
                "https://example.com", db, settings_for(tmp_path), "manual",
                client, model=TestModel(),
            )
            is None
        )


@pytest.mark.asyncio
@respx.mock
async def test_a_let_agreed_page_becomes_terminal_not_a_retry(tmp_path):
    """OpenRent serves 200 for a let flat, so status codes alone miss it."""
    url = "https://www.openrent.co.uk/property-to-rent/london/1-bed-flat-london-sw16/1687221"
    respx.get(url).mock(
        return_value=httpx.Response(
            200, html=(FIXTURES / "openrent_listing.html").read_text()
        )
    )
    db = Database(tmp_path / "flats.db")
    async with httpx.AsyncClient() as client:
        listing_id = await process_url(
            url, db, settings_for(tmp_path), "alert", client, model=TestModel()
        )
    assert db.get(listing_id)["status"] == "unavailable"


@pytest.mark.asyncio
@respx.mock
async def test_evaluation_failure_is_retryable_and_does_not_refetch(tmp_path, monkeypatch):
    """A model outage must not strand the Listing or waste a second download."""
    route = respx.get(URL).mock(return_value=httpx.Response(200, html=GOOD_HTML))
    db = Database(tmp_path / "flats.db")
    settings = settings_for(tmp_path)

    async def explode(*args, **kwargs):
        raise RuntimeError("provider outage")

    monkeypatch.setattr("flat_scout.pipeline.evaluate_listing", explode)
    async with httpx.AsyncClient() as client:
        listing_id = await process_url(
            URL, db, settings, "alert", client, model=TestModel()
        )
    row = db.get(listing_id)
    assert row["status"] == "fetch_pending"
    assert row["fetch_status"] == "ok"
    assert any(e["kind"] == "evaluation_failed" for e in db.events_for(listing_id))

    # The retry reuses the downloaded data instead of hitting the Portal again.
    monkeypatch.undo()
    async with httpx.AsyncClient() as client:
        await process_url(URL, db, settings, "alert", client, model=TestModel())
    assert route.call_count == 1
    assert db.get(listing_id)["status"] == "evaluated"


@pytest.mark.asyncio
@respx.mock
async def test_the_downloaded_signals_survive_the_round_trip_through_the_row(tmp_path):
    respx.get(URL).mock(return_value=httpx.Response(200, html=GOOD_HTML))
    db = Database(tmp_path / "flats.db")
    async with httpx.AsyncClient() as client:
        listing_id = await process_url(
            URL, db, settings_for(tmp_path), "manual", client, model=TestModel()
        )
    listing = listing_from_row(db.get(listing_id))
    assert listing.epc_caption == "EPC 1"
    assert listing.council_tax_band == "C"
    assert listing.nearest_station == "Hackney Wick Station"
    assert listing.nearest_station_miles == 0.3
    assert listing.image_url == (
        "https://media.rightmove.co.uk/property-photo/0be722e28/92053296/"
        "0be722e28dfae7b2eb5b59ab61ccb1a6.jpeg"
    )
    assert listing.latitude == 51.54538
    assert listing.longitude == -0.030554


@pytest.mark.asyncio
@respx.mock
async def test_the_band_read_off_the_graph_is_stored_and_reaches_the_evaluator(tmp_path):
    respx.get(URL).mock(return_value=httpx.Response(200, html=GOOD_HTML))
    mock_epc()
    db = Database(tmp_path / "flats.db")
    prompts: list[str] = []
    async with httpx.AsyncClient() as client:
        listing_id = await process_url(
            URL, db, settings_for(tmp_path), "manual", client,
            model=verdict_model(prompts), vision_model=band_model("C"),
        )
    row = db.get(listing_id)
    assert row["epc_band"] == "C"
    assert row["images_read_at"]
    assert row["vision_model"] == "openrouter:z-ai/glm-5.3-flash"
    assert row["status"] == "evaluated"
    assert "epc_band: C" in prompts[0]


@pytest.mark.asyncio
@respx.mock
async def test_check_caches_the_images_and_then_reads_them_off_disk(tmp_path, monkeypatch):
    """One download per picture, and the vision step never asks the host again.

    The paths are stored beside the URLs, so a re-run - or `annotate --db`, or a
    copy of `data/` on another machine - works with the media host unreachable.
    """
    from flat_scout import images

    monkeypatch.setattr(images, "IMAGES_DIR", tmp_path / "images")
    respx.get(URL).mock(return_value=httpx.Response(200, html=GOOD_HTML))
    epc = mock_epc()
    photo = mock_photo()
    db = Database(tmp_path / "flats.db")
    async with httpx.AsyncClient() as client:
        listing_id = await process_url(
            URL, db, settings_for(tmp_path), "manual", client,
            model=TestModel(), vision_model=band_model("C"),
        )
    directory = tmp_path / "images" / "rightmove_92053296"
    row = db.get(listing_id)
    assert row["image_path"] == (directory / "photo.jpg").as_posix()
    assert row["epc_image_path"] == (directory / "epc.png").as_posix()
    # The fixture publishes no floorplan, and nothing is invented for one.
    assert row["floorplan_path"] is None
    assert (directory / "photo.jpg").read_bytes() == JPEG_BYTES
    assert (directory / "epc.png").read_bytes() == PNG_BYTES
    # Once each: the cache fetched them, and `read_images` read the file.
    assert photo.call_count == 1
    assert epc.call_count == 1
    assert row["epc_band"] == "C"
    assert row["status"] == "evaluated"
    # And the row rebuilds with the paths on it, which is what the readers use.
    assert listing_from_row(row).epc_image_path == (directory / "epc.png").as_posix()


@pytest.mark.asyncio
@respx.mock
async def test_a_media_host_that_will_not_answer_costs_nothing_but_the_cache(tmp_path):
    """Neither the Listing nor the Verdict may depend on the pictures arriving."""
    respx.get(URL).mock(return_value=httpx.Response(200, html=GOOD_HTML))
    respx.get(PHOTO_URL).mock(return_value=httpx.Response(500))
    respx.get(EPC_URL).mock(return_value=httpx.Response(404))
    db = Database(tmp_path / "flats.db")
    async with httpx.AsyncClient() as client:
        listing_id = await process_url(
            URL, db, settings_for(tmp_path), "manual", client,
            model=TestModel(), vision_model=band_model("C"),
        )
    row = db.get(listing_id)
    assert row["status"] == "evaluated"
    assert row["image_path"] is None
    assert row["epc_image_path"] is None


@pytest.mark.asyncio
@respx.mock
async def test_an_epc_served_as_a_certificate_stores_the_printed_facts(tmp_path):
    """About a fifth of live EPCs are the certificate PDF, not a graph."""
    respx.get(URL).mock(return_value=httpx.Response(200, html=GOOD_HTML))
    respx.get(EPC_URL).mock(
        return_value=httpx.Response(
            200,
            content=(FIXTURES / "epc_certificate.pdf").read_bytes(),
            headers={"content-type": "application/pdf"},
        )
    )
    db = Database(tmp_path / "flats.db")
    prompts: list[str] = []
    certificate = TestModel(
        custom_output_args={
            "epc_band": "B",
            "epc_band_source": None,  # derived from what was sent, not from here
            "epc_floor_area_sqm": 51.0,
            "epc_property_type": "Mid-floor flat",
            "layout_verdict": None,
            "layout_notes": [],
            "reception_is_separate": None,
            "desk_space": None,
        }
    )
    async with httpx.AsyncClient() as client:
        listing_id = await process_url(
            URL, db, settings_for(tmp_path), "manual", client,
            model=verdict_model(prompts), vision_model=certificate,
        )
    row = db.get(listing_id)
    assert row["epc_band"] == "B"
    assert row["epc_band_source"] == "certificate"
    assert row["epc_floor_area_sqm"] == 51.0
    assert row["epc_property_type"] == "Mid-floor flat"
    assert row["status"] == "evaluated"
    assert "549 sq ft" in prompts[0]
    assert image_reading_from_row(row).epc_property_type == "Mid-floor flat"


@pytest.mark.asyncio
@respx.mock
async def test_image_signals_off_reads_no_images_and_spends_nothing(tmp_path):
    """The flag gates the vision model, which is what costs money.

    It does not gate the image cache: caching is a download and no model call,
    and having the pictures on disk is what makes turning the flag back on cost
    the media host nothing. So the EPC is fetched exactly once, by the cache,
    and nothing reads it.
    """
    respx.get(URL).mock(return_value=httpx.Response(200, html=GOOD_HTML))
    route = mock_epc()
    mock_photo()
    db = Database(tmp_path / "flats.db")
    settings = settings_for(tmp_path)
    settings.features.image_signals = False
    prompts: list[str] = []
    async with httpx.AsyncClient() as client:
        listing_id = await process_url(
            URL, db, settings, "manual", client,
            model=verdict_model(prompts), vision_model=band_model("C"),
        )
    row = db.get(listing_id)
    assert route.call_count == 1  # cached, never read
    assert row["epc_image_path"]
    assert row["images_read_at"] is None
    assert row["epc_band"] is None
    assert row["status"] == "evaluated"
    assert "epc_band" not in prompts[0]


@pytest.mark.asyncio
@respx.mock
async def test_an_unreadable_graph_is_recorded_as_read_with_no_band(tmp_path):
    """The couple's one hard requirement: no band beats a guessed band."""
    respx.get(URL).mock(return_value=httpx.Response(200, html=GOOD_HTML))
    mock_epc()
    db = Database(tmp_path / "flats.db")
    async with httpx.AsyncClient() as client:
        listing_id = await process_url(
            URL, db, settings_for(tmp_path), "manual", client,
            model=TestModel(), vision_model=band_model(None),
        )
    row = db.get(listing_id)
    assert row["epc_band"] is None
    assert row["images_read_at"]  # it was read; it just said nothing
    assert row["status"] == "evaluated"
    assert image_reading_from_row(row).epc_band is None


@pytest.mark.asyncio
@respx.mock
async def test_a_vision_outage_still_produces_a_verdict_from_the_text(tmp_path):
    """Requirement: a failed vision call must never cost us the Listing."""
    respx.get(URL).mock(return_value=httpx.Response(200, html=GOOD_HTML))
    mock_epc()
    db = Database(tmp_path / "flats.db")

    def explode(messages, info: AgentInfo) -> ModelResponse:
        raise RuntimeError("vision provider outage")

    async with httpx.AsyncClient() as client:
        listing_id = await process_url(
            URL, db, settings_for(tmp_path), "manual", client,
            model=TestModel(), vision_model=FunctionModel(explode),
        )
    row = db.get(listing_id)
    assert row["status"] == "evaluated"
    assert row["verdict"] in {"hopeful", "borderline", "reject"}
    assert row["images_read_at"] is None
    assert any(e["kind"] == "image_reading_failed" for e in db.events_for(listing_id))


@pytest.mark.asyncio
@respx.mock
async def test_a_retry_after_an_evaluation_failure_never_pays_for_the_images_twice(
    tmp_path, monkeypatch
):
    respx.get(URL).mock(return_value=httpx.Response(200, html=GOOD_HTML))
    route = mock_epc()
    db = Database(tmp_path / "flats.db")
    settings = settings_for(tmp_path)

    async def explode(*args, **kwargs):
        raise RuntimeError("provider outage")

    monkeypatch.setattr("flat_scout.pipeline.evaluate_listing", explode)
    async with httpx.AsyncClient() as client:
        listing_id = await process_url(
            URL, db, settings, "alert", client,
            model=TestModel(), vision_model=band_model("D"),
        )
    assert db.get(listing_id)["epc_band"] == "D"
    assert route.call_count == 1

    monkeypatch.undo()
    prompts: list[str] = []
    async with httpx.AsyncClient() as client:
        await process_url(
            URL, db, settings, "alert", client,
            model=verdict_model(prompts), vision_model=band_model("G"),
        )
    assert route.call_count == 1  # the stored reading was reused
    assert db.get(listing_id)["epc_band"] == "D"
    assert "epc_band: D" in prompts[0]


@pytest.mark.asyncio
@respx.mock
async def test_a_prefiltered_listing_never_reaches_the_vision_step(tmp_path):
    """Vision costs money per Listing, so a rejected one must not trigger it."""
    respx.get(URL).mock(return_value=httpx.Response(200, html=OVER_CAP_HTML))
    route = mock_epc()
    db = Database(tmp_path / "flats.db")
    async with httpx.AsyncClient() as client:
        listing_id = await process_url(
            URL, db, settings_for(tmp_path), "manual", client,
            model=TestModel(), vision_model=band_model("C"),
        )
    assert db.get(listing_id)["status"] == "prefiltered_out"
    assert not route.called


@pytest.mark.asyncio
@respx.mock
async def test_a_listing_with_no_images_is_never_marked_as_read(tmp_path):
    """A Zoopla Listing has no images at all, and must cost nothing."""
    zoopla_url = "https://www.zoopla.co.uk/to-rent/details/73991508"
    db = Database(tmp_path / "flats.db")
    async with httpx.AsyncClient() as client:
        listing_id = await process_url(
            zoopla_url, db, settings_for(tmp_path), "alert", client,
            model=TestModel(), vision_model=band_model("C"),
            fields={"price_pcm": 3100, "address": "Riverlight Quay, Nine Elms"},
        )
    row = db.get(listing_id)
    assert row["status"] == "evaluated"
    assert row["images_read_at"] is None
    assert image_reading_from_row(row) is None


@pytest.mark.asyncio
@respx.mock
async def test_an_odd_shaped_field_is_stored_rather_than_aborting_the_caller(tmp_path, monkeypatch):
    """Rightmove can send entranceFloor as an object.

    Raw sqlite3 refused to bind a dict, and the exception escaped past every
    guard - in a batch run that aborts every Listing after it. sqlite-utils
    stores it as JSON instead, so the Listing survives and the ground-floor
    hard filter still trips on it.
    """
    respx.get(URL).mock(return_value=httpx.Response(200, html=GOOD_HTML))
    db = Database(tmp_path / "flats.db")

    from flat_scout import pipeline as pipeline_module

    real = pipeline_module.fetch_listing

    async def odd_shape(*args, **kwargs):
        listing = await real(*args, **kwargs)
        listing.floor = {"alias": "ground", "displayText": "Ground floor"}
        return listing

    monkeypatch.setattr(pipeline_module, "fetch_listing", odd_shape)
    async with httpx.AsyncClient() as client:
        listing_id = await process_url(
            URL, db, settings_for(tmp_path), "alert", client, model=TestModel()
        )
    row = db.get(listing_id)
    assert row["status"] == "prefiltered_out"
    assert "ground" in row["floor"].lower()


@pytest.mark.asyncio
@respx.mock
async def test_a_stored_reading_is_ignored_once_image_signals_are_off(tmp_path, monkeypatch):
    """The flag governs what the evaluator sees, not just whether a call is made."""
    respx.get(URL).mock(return_value=httpx.Response(200, html=GOOD_HTML))
    db = Database(tmp_path / "flats.db")
    settings = settings_for(tmp_path)
    settings.features.image_signals = False

    seen: list = []

    async def capture(listing, criteria, s, model=None, reading=None):
        seen.append(reading)
        from flat_scout.models import Evaluation

        return Evaluation(verdict="borderline", score=5.0, reasons=["r"])

    from flat_scout import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "evaluate_listing", capture)
    monkeypatch.setattr(
        pipeline_module, "image_reading_from_row",
        lambda row: (_ for _ in ()).throw(AssertionError("must not load a stored reading")),
    )
    async with httpx.AsyncClient() as client:
        await process_url(URL, db, settings, "alert", client, model=TestModel())
    assert seen == [None]


@pytest.mark.asyncio
@respx.mock
async def test_a_repeat_ingest_leaves_an_evaluated_listing_alone(tmp_path):
    """The second run is a no-op: same stored mode, no second download."""
    route = respx.get(URL).mock(return_value=httpx.Response(200, html=GOOD_HTML))
    db = Database(tmp_path / "flats.db")
    settings = settings_for(tmp_path)
    async with httpx.AsyncClient() as client:
        listing_id = await process_url(URL, db, settings, "manual", client, model=TestModel())
        settings.features.weighted_criteria = True
        await process_url(URL, db, settings, "manual", client, model=TestModel())
    assert db.get(listing_id)["evaluation_mode"] == "holistic"
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_force_re_evaluates_an_evaluated_listing_without_downloading_again(tmp_path):
    """`check --force` is how the same Listing gets graded both ways to be
    compared. The page is already in the row, so only the evaluation runs."""
    route = respx.get(URL).mock(return_value=httpx.Response(200, html=GOOD_HTML))
    db = Database(tmp_path / "flats.db")
    settings = settings_for(tmp_path)
    async with httpx.AsyncClient() as client:
        listing_id = await process_url(URL, db, settings, "manual", client, model=TestModel())
        assert db.get(listing_id)["evaluation_mode"] == "holistic"
        settings.features.weighted_criteria = True
        graded = TestModel(custom_output_args={"grade": 7.0, "evidence": "seems fine"})
        again = await process_url(
            URL, db, settings, "manual", client, model=graded, force=True
        )
    assert again == listing_id
    row = db.get(listing_id)
    assert row["status"] == "evaluated"
    assert row["evaluation_mode"] == "weighted"
    assert db.criterion_grades(listing_id) != {}
    assert route.call_count == 1
