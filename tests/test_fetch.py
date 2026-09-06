from pathlib import Path

import httpx
import pytest
import respx

from flat_scout.config import Filters, Settings
from flat_scout.evaluate import prefilter
from flat_scout.fetch import (
    BotChallenge,
    FetchError,
    ListingUnavailable,
    _epc_caption,
    _floor,
    _floorplan_url,
    _headline_image,
    _location,
    _nearest_station,
    bot_challenge,
    fetch_listing,
    is_fetchable,
    is_unavailable,
    parse_listing,
)

FIXTURES = Path(__file__).parent / "fixtures"

RIGHTMOVE_URL = "https://www.rightmove.co.uk/properties/92053296"
OPENRENT_URL = (
    "https://www.openrent.co.uk/property-to-rent/london/1-bed-flat-london-sw16/1687221"
)
# A genuine saved two-bed. The one-bed fixture above has one bathroom too, so it
# cannot tell a bed count from a bathroom count; this one can.
OPENRENT_TWO_BED_URL = (
    "https://www.openrent.co.uk/property-to-rent/farnworth-bolton/"
    "2-bed-terraced-house-charles-street-bl4/2278562"
)


# A genuine studio, and a genuine one-bed whose blurb advertises the block's
# studios. OpenRent calls a studio "1 bedrooms" in its summary row, so the two
# are indistinguishable there and only the headline separates them.
OPENRENT_STUDIO_URL = (
    "https://www.openrent.co.uk/property-to-rent/london/"
    "studio-flat-craven-street-wc2n/3007606"
)
OPENRENT_STUDIO_MENTION_URL = (
    "https://www.openrent.co.uk/property-to-rent/london/"
    "1-bed-flat-casson-square-se1/3008014"
)


def openrent_page(h1: str, summary: str, body: str = "") -> str:
    """A minimal OpenRent page: a headline, a summary row and a price."""
    return (
        f"<html><body><h1>{h1}</h1><p>{summary}</p>"
        f"<p>£1,500 per month</p><p>{body}</p></body></html>"
    )


def _settings() -> Settings:
    settings = Settings()
    settings.fetch.delay_seconds = 0
    return settings


def test_parses_rightmove_listing():
    html = (FIXTURES / "rightmove_listing.html").read_text()
    listing = parse_listing("rightmove", html, RIGHTMOVE_URL)
    assert listing.portal == "rightmove"
    assert listing.portal_id == "92053296"
    assert listing.price_pcm == 2400
    assert listing.address == "Eastway, Vergers Apartments Eastway, E9"
    assert listing.postcode == "E9"
    assert listing.beds == 1
    assert listing.sqft == 441
    assert listing.furnished == "Furnished"
    assert listing.listed_on == "Added yesterday"
    assert listing.description and "one-bedroom" in listing.description


def test_rightmove_description_has_no_html_tags():
    html = (FIXTURES / "rightmove_listing.html").read_text()
    listing = parse_listing("rightmove", html, RIGHTMOVE_URL)
    assert "<p>" not in listing.description
    assert "<strong>" not in listing.description


def test_rightmove_key_features_reach_the_description():
    html = (FIXTURES / "rightmove_listing.html").read_text()
    listing = parse_listing("rightmove", html, RIGHTMOVE_URL)
    assert "Spacious one bedroom apartment" in listing.description


def test_parses_openrent_listing():
    html = (FIXTURES / "openrent_listing.html").read_text()
    listing = parse_listing("openrent", html, OPENRENT_URL)
    assert listing.portal == "openrent"
    assert listing.portal_id == "1687221"
    assert listing.price_pcm and 500 < listing.price_pcm < 10000
    assert listing.address
    assert listing.description


def test_unparseable_html_raises_rather_than_returning_empty():
    with pytest.raises(FetchError):
        parse_listing(
            "rightmove", "<html><body>nope</body></html>",
            "https://www.rightmove.co.uk/properties/1",
        )


def test_zoopla_is_not_fetchable():
    """Zoopla serves a Cloudflare challenge, so we never download it."""
    assert is_fetchable("rightmove") is True
    assert is_fetchable("openrent") is True
    assert is_fetchable("zoopla") is False


def test_parsing_a_zoopla_page_is_refused():
    with pytest.raises(FetchError):
        parse_listing("zoopla", "<html></html>", "https://www.zoopla.co.uk/to-rent/details/1")


def test_openrent_pet_policy_reads_the_icon_not_the_label():
    """The fixture's Pets Allowed row carries a red cross: the flat forbids pets.

    The row holds two SVGs - an info icon inside the popover button, then the
    value icon - so reading the first one reports the opposite of the truth.
    """
    html = (FIXTURES / "openrent_listing.html").read_text()
    listing = parse_listing("openrent", html, OPENRENT_URL)
    assert listing.pet_notes is not None
    assert "not allowed" in listing.pet_notes.lower()


def test_text_from_two_elements_does_not_weld_into_one_word():
    """Found while checking why a negation guard missed a real page.

    The chunks were concatenated with nothing between them, so
    `<h1>...SW11</h1><div>No furnished...` came out as "SW11No furnished".
    Every extractor in this module is a regex over that string - the postcode
    is a HARD FILTER - and a weld hides the word boundary they all depend on.
    The real fixtures carried 101 of these on the Rightmove page.
    """
    from flat_scout.fetch import html_text

    welded = "<h1>A flat, London SW11</h1><div>No furnished option here.</div>"
    assert "SW11 No" in html_text(welded)


def test_an_inline_tag_does_not_split_a_word():
    """The other direction, and the reason this is not just "join with a
    space": `<b>Furn</b>ished` is one word and has to stay one."""
    from flat_scout.fetch import html_text

    assert html_text("<p>Furn<b>ish</b>ed flat</p>") == "Furnished flat"


def test_a_weld_cannot_hide_a_denied_furnishing():
    """The two fixes meeting: the guard reads `\\bno\\b`, which cannot match
    inside "sw11no", so the extractor had to stop making that string."""
    html = (
        "<html><body><h1>A flat, London SW11</h1>"
        "<div>No furnished option is available. £2000 per month</div></body></html>"
    )
    listing = parse_listing("openrent", html, OPENRENT_URL)
    assert listing.furnished is None
    assert listing.postcode == "SW11"


def test_a_denied_furnishing_is_extracted_as_unknown_not_as_its_opposite():
    """Found in review of the furnishing Criterion.

    OpenRent's furnishing is read out of free listing copy, and `_phrase`
    matches by substring: none of "fully furnished", "part furnished" or
    "unfurnished" appears inside "not furnished", so the search fell through
    to the bare "furnished" and stored the opposite of what the advert said.
    Harmless while nothing read the field; worth four weight now.

    Unknown rather than "unfurnished", because "not furnished to the standard
    you would expect" says nothing about whether there is a sofa in it.
    """
    from flat_scout.fetch import FURNISHED_PHRASES, _unnegated_phrase

    assert _unnegated_phrase("The flat is furnished.", FURNISHED_PHRASES) == "furnished"
    assert _unnegated_phrase("The flat is not furnished.", FURNISHED_PHRASES) is None
    assert _unnegated_phrase("It isn't furnished, sadly.", FURNISHED_PHRASES) is None
    assert _unnegated_phrase("Unfurnished throughout.", FURNISHED_PHRASES) == "unfurnished"
    assert _unnegated_phrase("Fully furnished.", FURNISHED_PHRASES) == "fully furnished"
    # The attached forms, which a word boundary does not catch: the hyphen is a
    # non-word character, so `\bfurnished\b` matches inside "non-furnished".
    assert _unnegated_phrase("It is non-furnished.", FURNISHED_PHRASES) is None
    assert _unnegated_phrase("A un-furnished flat.", FURNISHED_PHRASES) is None
    # "unfurnished" must survive that check - it is a real answer, not a denial.
    assert _unnegated_phrase("Let unfurnished.", FURNISHED_PHRASES) == "unfurnished"


def test_openrent_description_is_the_listing_copy_not_boilerplate():
    html = (FIXTURES / "openrent_listing.html").read_text()
    listing = parse_listing("openrent", html, OPENRENT_URL)
    assert "Javascript disabled" not in listing.description
    assert "period building" in listing.description
    assert len(listing.description) < 4000


def test_openrent_let_agreed_page_is_recognised_as_unavailable():
    html = (FIXTURES / "openrent_listing.html").read_text()
    assert is_unavailable("openrent", html) is True


def test_a_live_page_is_not_reported_unavailable():
    html = (FIXTURES / "rightmove_listing.html").read_text()
    assert is_unavailable("rightmove", html) is False


def test_rightmove_captures_the_running_cost_and_transport_signals():
    html = (FIXTURES / "rightmove_listing.html").read_text()
    listing = parse_listing("rightmove", html, RIGHTMOVE_URL)
    assert listing.council_tax_band == "C"
    assert listing.nearest_station == "Hackney Wick Station"
    assert listing.nearest_station_miles == 0.3


def test_rightmove_epc_keeps_the_portal_caption_verbatim():
    """The caption names the graph, not the band, so no letter is derived from it."""
    html = (FIXTURES / "rightmove_listing.html").read_text()
    listing = parse_listing("rightmove", html, RIGHTMOVE_URL)
    assert listing.epc_caption == "EPC 1"


def test_rightmove_captures_the_coordinates():
    html = (FIXTURES / "rightmove_listing.html").read_text()
    listing = parse_listing("rightmove", html, RIGHTMOVE_URL)
    assert listing.latitude == 51.54538
    assert listing.longitude == -0.030554


def test_an_approximate_pin_yields_no_coordinates_and_no_distances():
    """An approximate pin is the Portal withholding the address. A distance
    derived from one would look exact and be wrong by a quarter of a mile."""
    assert _location({"location": {
        "latitude": 51.59085, "longitude": -0.06141, "pinType": "APPROXIMATE_POINT"
    }}) == (None, None)


def test_a_listing_with_no_location_block_stays_unknown():
    assert _location({}) == (None, None)
    assert _location({"location": {"pinType": "ACCURATE_POINT"}}) == (None, None)


def test_an_accurate_pin_is_read_out():
    assert _location({"location": {
        "latitude": 51.4818, "longitude": -0.13687, "pinType": "ACCURATE_POINT"
    }}) == (51.4818, -0.13687)


def test_the_nearest_station_is_the_closest_one_not_the_first_listed():
    stations = [
        {"name": "Homerton Station", "distance": 0.55, "unit": "miles"},
        {"name": "Hackney Wick Station", "distance": 0.29, "unit": "miles"},
    ]
    assert _nearest_station(stations) == ("Hackney Wick Station", 0.29)


def test_a_station_distance_in_other_units_is_not_read_as_miles():
    stations = [{"name": "Somewhere Station", "distance": 300.0, "unit": "metres"}]
    assert _nearest_station(stations) == (None, None)


def test_missing_epc_and_stations_stay_unknown():
    assert _epc_caption([]) is None
    assert _nearest_station(None) == (None, None)


def test_rightmove_keeps_the_epc_image_url_for_later_reading():
    """The band is legible only inside the image, so the URL is the real signal."""
    html = (FIXTURES / "rightmove_listing.html").read_text()
    listing = parse_listing("rightmove", html, RIGHTMOVE_URL)
    assert listing.epc_image_url
    assert listing.epc_image_url.startswith("https://media.rightmove.co.uk/")


def test_rightmove_keeps_the_headline_photo_for_the_listing():
    """The first image is the shot the agent chose to lead with."""
    html = (FIXTURES / "rightmove_listing.html").read_text()
    listing = parse_listing("rightmove", html, RIGHTMOVE_URL)
    assert listing.image_url == (
        "https://media.rightmove.co.uk/property-photo/0be722e28/92053296/"
        "0be722e28dfae7b2eb5b59ab61ccb1a6.jpeg"
    )


def test_the_headline_photo_is_the_first_image_not_the_first_captioned_one():
    images = [
        {"url": "https://media.rightmove.co.uk/a.jpeg", "caption": None},
        {"url": "https://media.rightmove.co.uk/b.jpeg", "caption": "Reception"},
    ]
    assert _headline_image(images) == "https://media.rightmove.co.uk/a.jpeg"


def test_a_listing_with_no_photographs_has_no_headline_photo():
    assert _headline_image(None) is None
    assert _headline_image([]) is None
    assert _headline_image([{"caption": "Reception"}]) is None


def test_openrent_headline_image_is_the_property_photo():
    """OpenRent's og:image is a share graphic; the real photo is in image_src."""
    html = (FIXTURES / "openrent_listing.html").read_text()
    listing = parse_listing("openrent", html, OPENRENT_URL)
    assert listing.image_url == (
        "https://imagescdn.openrent.co.uk/listings/1687221/"
        "o_1gvbddj0p1hkeb9k2nt19udh06.JPG"
    )


def test_openrent_image_url_is_absolute():
    """OpenRent writes protocol-relative URLs; the stored URL needs a scheme."""
    html = (FIXTURES / "openrent_listing.html").read_text()
    listing = parse_listing("openrent", html, OPENRENT_URL)
    assert listing.image_url.startswith("https://")
    assert "share-graphic" not in listing.image_url


def test_an_entrance_floor_stated_as_an_object_becomes_its_display_text():
    """Rightmove sends entranceFloor either as a string or as an object.

    Found live on 92057625. Stored raw, the object breaks the sqlite bind and
    loses the Listing, and the ground-floor hard filter cannot match on it.
    """
    assert _floor({"alias": "higherWithLift", "displayText": "Higher than 2nd floor (with lift)"}) == (
        "Higher than 2nd floor (with lift)"
    )
    assert _floor({"alias": "ground", "displayText": "Ground floor"}) == "Ground floor"
    assert _floor("2nd") == "2nd"
    assert _floor(None) is None
    assert _floor({}) is None


def test_a_ground_floor_stated_as_an_object_still_trips_the_hard_filter():
    """The whole point of reading the floor is the ground-floor rule."""
    from flat_scout.config import Filters
    from flat_scout.evaluate import prefilter
    from flat_scout.models import ListingData

    listing = ListingData(
        portal="rightmove",
        portal_id="1",
        url="u",
        floor=_floor({"alias": "ground", "displayText": "Ground floor"}),
    )
    assert prefilter(listing, Filters(exclude_ground_floor=True)) == "ground floor"


def test_rightmove_keeps_the_floorplan_url_for_later_reading():
    """Whether the reception is a room or a corridor is only in the floorplan."""
    floorplans = [
        {
            "url": "https://media.rightmove.co.uk/property-floorplan/a/1/a.jpeg",
            "caption": "Floorplan 1",
            "type": "IMAGE",
            "resizedFloorplanUrls": {"size296x197": "https://media.rightmove.co.uk/dir/a.jpeg"},
        }
    ]
    assert _floorplan_url(floorplans) == (
        "https://media.rightmove.co.uk/property-floorplan/a/1/a.jpeg"
    )


def test_the_floorplan_is_the_full_size_one_not_the_thumbnail():
    """A 296x197 thumbnail cannot be read: room labels and dimensions vanish."""
    floorplans = [
        {
            "url": "https://media.rightmove.co.uk/property-floorplan/a/1/a.png",
            "type": "IMAGE",
            "resizedFloorplanUrls": {"size296x197": "https://media.rightmove.co.uk/dir/a.png"},
        }
    ]
    assert "_max_296x197" not in _floorplan_url(floorplans)
    assert "/dir/" not in _floorplan_url(floorplans)


def test_a_non_image_floorplan_is_skipped():
    """An interactive or PDF plan is not something the vision model can read."""
    floorplans = [
        {"url": "https://example.com/plan.html", "type": "INTERACTIVE"},
        {"url": "https://media.rightmove.co.uk/property-floorplan/b.jpeg", "type": "IMAGE"},
    ]
    assert _floorplan_url(floorplans) == (
        "https://media.rightmove.co.uk/property-floorplan/b.jpeg"
    )
    assert _floorplan_url([{"url": "https://example.com/plan.html", "type": "PDF"}]) is None


def test_a_listing_with_no_floorplan_stays_unknown():
    """Most Listings have none, and absence must never look like a bad layout."""
    assert _floorplan_url(None) is None
    assert _floorplan_url([]) is None
    assert _floorplan_url([{"caption": "Floorplan 1", "type": "IMAGE"}]) is None
    html = (FIXTURES / "rightmove_listing.html").read_text()
    assert parse_listing("rightmove", html, RIGHTMOVE_URL).floorplan_url is None


def test_an_environmental_impact_graph_is_never_offered_as_the_epc():
    """EI charts carry their own A-G scale, measuring CO2, not energy.

    Reading a band off one and storing it as epc_band would be a CO2 rating
    presented as evidence about warmth and bills.
    """
    from flat_scout.fetch import _epc_caption, _epc_image_url

    graphs = [
        {"url": "https://media.rightmove.co.uk/ei.png", "caption": "EI Rating"},
        {"url": "https://media.rightmove.co.uk/ee.png", "caption": "EE Rating"},
    ]
    assert _epc_image_url(graphs) == "https://media.rightmove.co.uk/ee.png"
    assert _epc_caption(graphs) == "EE Rating"


def test_an_ei_only_listing_yields_no_epc_image_at_all():
    from flat_scout.fetch import _epc_image_url

    graphs = [{"url": "https://media.rightmove.co.uk/ei.png", "caption": "EI Rating"}]
    assert _epc_image_url(graphs) is None


def test_captions_seen_in_the_wild_are_still_accepted():
    from flat_scout.fetch import _epc_image_url

    for caption in ("EPC", "EPC 1", "EPC Graph", "EPC Rating Graph", "EER", "EE Rating", ""):
        graphs = [{"url": "https://media.rightmove.co.uk/x.png", "caption": caption}]
        assert _epc_image_url(graphs) == "https://media.rightmove.co.uk/x.png", caption


def test_the_aws_waf_challenge_is_named_as_a_bot_challenge():
    """The block OpenRent serves to a blocked client, saved from a real request."""
    html = (FIXTURES / "openrent_waf_challenge.html").read_text()
    assert bot_challenge(html) == "AWS WAF"


def test_the_cloudflare_challenge_is_still_named():
    assert bot_challenge("<html><title>Just a moment...</title></html>") == "Cloudflare"


def test_a_real_listing_page_is_not_a_challenge():
    html = (FIXTURES / "openrent_listing.html").read_text()
    assert bot_challenge(html) is None


def test_one_aws_marker_alone_does_not_convict_a_page():
    """A Listing whose copy says "Human Verification" is still a Listing."""
    assert bot_challenge("<html>Human Verification of the entryphone</html>") is None


@pytest.mark.asyncio
@respx.mock
async def test_a_405_carrying_a_waf_challenge_is_reported_as_the_block_it_is():
    """The status code is a lie; the body is not."""
    html = (FIXTURES / "openrent_waf_challenge.html").read_text()
    respx.get(OPENRENT_URL).mock(return_value=httpx.Response(405, html=html))
    async with httpx.AsyncClient() as client:
        with pytest.raises(BotChallenge) as caught:
            await fetch_listing(OPENRENT_URL, _settings(), client)
    assert "AWS WAF" in str(caught.value)


@pytest.mark.asyncio
@respx.mock
async def test_a_405_from_anything_else_is_still_a_plain_405():
    respx.get(OPENRENT_URL).mock(return_value=httpx.Response(405, html="<html/>"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(FetchError) as caught:
            await fetch_listing(OPENRENT_URL, _settings(), client)
    assert not isinstance(caught.value, BotChallenge)
    assert "HTTP 405" in str(caught.value)


def test_a_bot_challenge_is_not_a_withdrawn_listing():
    """One is "gone for ever", the other is "we cannot see it from here"."""
    assert issubclass(BotChallenge, FetchError)
    assert not issubclass(BotChallenge, ListingUnavailable)
    assert not issubclass(ListingUnavailable, BotChallenge)


def test_openrent_bed_count_is_not_the_bathroom_count():
    """OpenRent writes "2 bedrooms 1 bathrooms" - the count precedes the label."""
    html = (FIXTURES / "openrent_two_bed.html").read_text()
    listing = parse_listing("openrent", html, OPENRENT_TWO_BED_URL)
    assert listing.beds == 2


def test_a_real_two_bed_page_is_stopped_by_the_hard_filter():
    """The consequence, and the reason this is worth more than a wrong label.

    A two-bed read as a one-bed clears `allowed_beds` and is surfaced to the
    couple, while the filter reports itself working.
    """
    html = (FIXTURES / "openrent_two_bed.html").read_text()
    listing = parse_listing("openrent", html, OPENRENT_TWO_BED_URL)
    filters = Filters(max_price_pcm=100000, allowed_beds=[0, 1], postcodes=[])
    assert "bed" in (prefilter(listing, filters) or "").lower()


def test_the_one_bed_fixture_still_reads_as_one_bed():
    html = (FIXTURES / "openrent_listing.html").read_text()
    assert parse_listing("openrent", html, OPENRENT_URL).beds == 1


def test_a_headline_disagreeing_with_the_summary_reads_no_bed_count():
    """Unknown passes the filter and gets judged; a confident guess does not."""
    html = openrent_page("2 Bed Flat, London, SW16", "3 bedrooms 1 bathrooms")
    assert parse_listing("openrent", html, OPENRENT_URL).beds is None


def test_a_headline_agreeing_with_the_summary_is_read_out():
    html = openrent_page("2 Bed Flat, London, SW16", "2 bedrooms 1 bathrooms")
    assert parse_listing("openrent", html, OPENRENT_URL).beds == 2


def test_a_page_stating_no_bed_count_reads_none_rather_than_guessing():
    html = openrent_page("Flat to rent in London", "1 bathrooms 2 tenants max.")
    assert parse_listing("openrent", html, OPENRENT_URL).beds is None


def test_a_summary_alone_is_still_read_out():
    """The headline is not always numbered; the summary row is the stated fact."""
    html = openrent_page("Flat to rent in London", "2 bedrooms 1 bathrooms")
    assert parse_listing("openrent", html, OPENRENT_URL).beds == 2


def test_a_studio_still_reads_as_zero_beds():
    html = openrent_page("Studio Flat, London, SW16", "1 bathrooms 1 tenants max.")
    assert parse_listing("openrent", html, OPENRENT_URL).beds == 0


def test_a_summary_row_of_inline_elements_still_reads_out():
    """`html_text` pads block tags but not inline ones (see `_TextExtractor`),
    so the two cells can flatten to "2 bedrooms1 bathrooms" with no space
    between them. The headline here carries no number, so the summary row is
    the only thing that can answer.
    """
    html = (
        "<html><body><h1>Flat to rent in London</h1>"
        "<span>2 bedrooms</span><span>1 bathrooms</span>"
        "<p>£1,500 per month</p></body></html>"
    )
    assert parse_listing("openrent", html, OPENRENT_URL).beds == 2


def test_a_one_bed_whose_blurb_mentions_studios_is_not_a_studio():
    """A real Listing, saved on 2026-08-21. Its headline says "1 Bed Flat" and
    its blurb says the block's properties "range from stylish studios to
    spacious one-, two- and three-bedroom apartments". The word is about the
    neighbours, not about this flat.
    """
    html = (FIXTURES / "openrent_one_bed_mentioning_studios.html").read_text()
    assert parse_listing("openrent", html, OPENRENT_STUDIO_MENTION_URL).beds == 1


def test_a_real_studio_still_reads_as_zero_beds():
    """The other direction, and the one that must not regress: studios are
    wanted. OpenRent's summary row calls this one "1 bedrooms"; only the
    headline says what it is.
    """
    html = (FIXTURES / "openrent_studio.html").read_text()
    assert parse_listing("openrent", html, OPENRENT_STUDIO_URL).beds == 0


def test_a_studio_in_the_address_does_not_make_a_two_bed_a_studio():
    """The headline is "<type>, <address>", and only the type is consulted."""
    html = openrent_page("2 Bed Flat, Studio Court, SW16", "2 bedrooms 1 bathrooms")
    assert parse_listing("openrent", html, OPENRENT_URL).beds == 2


def test_a_two_bed_talking_about_studios_is_still_stopped_by_the_hard_filter():
    """The consequence. A wrong `0` walks any size of flat through
    `allowed_beds` unread, because 0 is a size the couple want.
    """
    html = openrent_page(
        "2 Bed Flat, London, SW16",
        "2 bedrooms 1 bathrooms",
        body="A short walk from the yoga studio on the corner.",
    )
    listing = parse_listing("openrent", html, OPENRENT_URL)
    filters = Filters(max_price_pcm=100000, allowed_beds=[0, 1], postcodes=[])
    assert listing.beds == 2
    assert "bed" in (prefilter(listing, filters) or "").lower()


def test_a_studio_apartment_is_read_from_the_type_as_well():
    """"Studio Flat" is what OpenRent writes today; the type is not hard-coded."""
    html = openrent_page("Studio Apartment, London, SW16", "1 bedrooms 1 bathrooms")
    assert parse_listing("openrent", html, OPENRENT_URL).beds == 0
