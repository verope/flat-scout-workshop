from flat_scout.config import Filters
from flat_scout.evaluate import prefilter
from flat_scout.models import ListingData

FILTERS = Filters(
    max_price_pcm=3500,
    allowed_beds=[0, 1],
    exclude_ground_floor=True,
    postcodes=["SW8", "N1"],
)


def listing(**kw) -> ListingData:
    base = dict(
        portal="rightmove",
        portal_id="1",
        url="u",
        postcode="SW8",
        price_pcm=3100,
        beds=1,
        floor="6th",
    )
    base.update(kw)
    return ListingData(**base)


def test_passing_listing_returns_none():
    assert prefilter(listing(), FILTERS) is None


def test_rejects_over_price_cap():
    assert "price" in prefilter(listing(price_pcm=3600), FILTERS).lower()


def test_accepts_a_listing_exactly_at_the_cap():
    assert prefilter(listing(price_pcm=3500), FILTERS) is None


def test_rejects_postcode_outside_whitelist():
    assert "postcode" in prefilter(listing(postcode="E8"), FILTERS).lower()


def test_rejects_two_beds():
    assert "bed" in prefilter(listing(beds=2), FILTERS).lower()


def test_accepts_studio_as_zero_beds():
    assert prefilter(listing(beds=0), FILTERS) is None


def test_rejects_ground_floor():
    assert "ground" in prefilter(listing(floor="ground floor"), FILTERS).lower()


def test_unknown_values_pass_rather_than_reject():
    assert (
        prefilter(
            listing(price_pcm=None, beds=None, postcode=None, floor=None), FILTERS
        )
        is None
    )


def test_empty_postcode_whitelist_disables_the_postcode_filter():
    permissive = Filters(max_price_pcm=3500, allowed_beds=[0, 1], postcodes=[])
    assert prefilter(listing(postcode="E8"), permissive) is None


def test_a_non_string_floor_does_not_abort_the_pre_filter():
    """Rightmove can send entranceFloor as an object.

    Normalising happens in the parser, but the filter is a hard requirement and
    must not raise if an odd shape ever reaches it from another source.
    """
    assert prefilter(listing(floor={"displayText": "Ground floor"}), FILTERS) is not None
    assert prefilter(listing(floor=17), FILTERS) is None
