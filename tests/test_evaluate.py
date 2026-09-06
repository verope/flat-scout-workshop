import pytest
from pydantic_ai.models.test import TestModel

from flat_scout.config import Settings
from flat_scout.evaluate import build_prompt, evaluate_listing
from flat_scout.models import Evaluation, ImageReading, ListingData


def a_listing() -> ListingData:
    return ListingData(
        portal="zoopla",
        portal_id="1",
        url="u",
        address="Riverlight Quay, Nine Elms",
        postcode="SW8",
        price_pcm=3100,
        beds=1,
        sqft=560,
        floor="upper floor with lift",
        furnished="furnished",
        pet_notes="Pets allowed",
        description="Modern BTR flat.",
        listed_on="2026-08-16",
    )


def test_prompt_contains_the_criteria_and_the_listing_facts():
    prompt = build_prompt(a_listing(), criteria="# What we're looking for\nBike storage.")
    assert "Bike storage" in prompt
    assert "Riverlight Quay" in prompt
    assert "3100" in prompt
    assert "Pets allowed" in prompt


def test_prompt_carries_the_epc_tax_and_station_signals():
    listing = a_listing()
    listing.epc_caption = "EE Rating"
    listing.council_tax_band = "E"
    listing.nearest_station = "Elm Road Underground Station"
    listing.nearest_station_miles = 0.26
    prompt = build_prompt(listing, criteria="brief")
    assert "epc_caption: EE Rating" in prompt
    assert "council_tax_band: E" in prompt
    assert "nearest_station: Elm Road Underground Station" in prompt
    assert "nearest_station_miles: 0.26" in prompt


def test_prompt_carries_the_coordinates():
    """build_prompt serialises whatever is not None, so the new fields arrive free."""
    listing = a_listing()
    listing.latitude = 51.4818
    listing.longitude = -0.13687
    prompt = build_prompt(listing, criteria="brief")
    assert "latitude: 51.4818" in prompt
    assert "longitude: -0.13687" in prompt


def test_prompt_omits_a_location_the_pin_could_not_support():
    """An approximate pin leaves both coordinates None, and the model
    is told nothing rather than told a guess."""
    prompt = build_prompt(a_listing(), criteria="brief")
    assert "latitude" not in prompt
    assert "longitude" not in prompt


def test_prompt_omits_unknown_fields_rather_than_saying_none():
    listing = a_listing()
    listing.sqft = None
    prompt = build_prompt(listing, criteria="brief")
    assert "sqft" not in prompt
    assert "None" not in prompt


@pytest.mark.asyncio
async def test_evaluate_returns_a_structured_verdict():
    result = await evaluate_listing(a_listing(), "criteria", Settings(), model=TestModel())
    assert isinstance(result, Evaluation)
    assert result.verdict in {"hopeful", "borderline", "reject"}
    assert 0 <= result.score <= 10
    assert len(result.reasons) <= 3


def test_prompt_omits_media_urls_the_text_model_cannot_open():
    """The listing URL stays - it identifies the flat. Image URLs do not."""
    listing = a_listing()
    listing.image_url = "https://media.rightmove.co.uk/property-photo/x.jpeg"
    listing.epc_image_url = "https://media.rightmove.co.uk/property-epc/y.png"
    listing.floorplan_url = "https://media.rightmove.co.uk/property-floorplan/p.jpeg"
    prompt = build_prompt(listing, criteria="brief")
    assert "media.rightmove.co.uk" not in prompt
    assert "url: u" in prompt


def test_prompt_omits_the_cached_image_paths_as_well_as_the_urls():
    """A path is no more openable than a URL, and it names a directory on a
    machine the evaluator is not running on. Same rule, same block."""
    listing = a_listing()
    listing.image_path = "data/images/rightmove_92053296/photo.jpg"
    listing.epc_image_path = "data/images/rightmove_92053296/epc.pdf"
    listing.floorplan_path = "data/images/rightmove_92053296/floorplan.png"
    prompt = build_prompt(listing, criteria="brief")
    assert "data/images" not in prompt
    for field in ("image_path", "epc_image_path", "floorplan_path"):
        assert field not in prompt


def test_the_image_reading_reaches_the_text_evaluator():
    """The band and the layout are the whole point of having read the images."""
    prompt = build_prompt(
        a_listing(),
        criteria="brief",
        reading=ImageReading(
            epc_band="C",
            layout_verdict="good",
            layout_notes=["reception 4.2m x 3.6m"],
            reception_is_separate=True,
            desk_space=True,
        ),
    )
    assert "epc_band: C" in prompt
    assert "layout_verdict: good" in prompt
    assert "reception 4.2m x 3.6m" in prompt
    assert "desk_space: True" in prompt


def test_the_certificate_facts_reach_the_text_evaluator():
    """Floor area is the brief's best signal, and Rightmove often omits sqft."""
    prompt = build_prompt(
        a_listing(),
        criteria="brief",
        reading=ImageReading(
            epc_band="B",
            epc_band_source="certificate",
            epc_floor_area_sqm=51.0,
            epc_property_type="Mid-floor flat",
        ),
    )
    assert "epc_band: B" in prompt
    assert "549 sq ft" in prompt  # converted here, never by the model
    assert "Mid-floor flat" in prompt


def test_a_prompt_without_a_reading_has_no_image_section_at_all():
    """image_signals off must leave the prompt exactly as it was before."""
    prompt = build_prompt(a_listing(), criteria="brief")
    assert "epc_band" not in prompt
    assert "images" not in prompt.lower()


def test_an_unread_field_is_named_as_unknown_not_left_to_be_guessed():
    """A missing band must read as unknown, never as a bad band."""
    prompt = build_prompt(
        a_listing(), criteria="brief", reading=ImageReading(layout_verdict="poor")
    )
    assert "layout_verdict: poor" in prompt
    assert "epc_band" not in prompt
    assert "unknown" in prompt.lower()


def test_a_wholly_unreadable_pair_of_images_says_so_in_the_prompt():
    prompt = build_prompt(a_listing(), criteria="brief", reading=ImageReading())
    assert "could be read" in prompt


@pytest.mark.asyncio
async def test_evaluate_passes_the_reading_through_to_the_model():
    listing = a_listing()
    result = await evaluate_listing(
        listing,
        "criteria",
        Settings(),
        model=TestModel(),
        reading=ImageReading(epc_band="B"),
    )
    assert isinstance(result, Evaluation)


def test_the_prompt_says_an_absent_field_is_not_the_listing_being_silent():
    """The seam that produced a day of redundant questions.

    Rightmove sends `entranceFloor` as null on every Listing and states no pet
    policy at all, while the advert says which floor it is on and whether pets
    are welcome. Without this caveat the evaluator read the missing field as
    "the listing does not say" and asked the agent to confirm a fact it had
    just quoted from the description.
    """
    prompt = build_prompt(a_listing(), criteria="brief")
    assert "best-effort" in prompt
    assert "NOT the listing declining" in prompt
    # And it has to arrive before the facts it is a caveat about.
    assert prompt.index("best-effort") < prompt.index("portal:")


def test_a_window_reading_reaches_the_evaluator():
    prompt = build_prompt(
        a_listing(), criteria="brief", reading=ImageReading(bedroom_has_window=False)
    )
    assert "bedroom_has_window: False" in prompt


def test_the_model_is_asked_for_a_score_it_cannot_omit():
    """ModelVerdict is the live path's schema, and it must not gain fields.

    The couple are calibrating against this model now. A new optional field in
    its output schema is a change to the question it is being asked.
    """
    from flat_scout.models import ModelVerdict

    fields = set(ModelVerdict.model_fields)
    assert fields == {"verdict", "score", "reasons", "red_flags", "agent_questions"}
    assert ModelVerdict.model_fields["score"].is_required()


def test_the_record_may_carry_no_score_and_a_coverage():
    from flat_scout.models import Evaluation

    record = Evaluation(verdict="borderline", score=None, coverage=0.4, reasons=[])
    assert record.score is None
    assert record.coverage == 0.4
