import pytest

from flat_scout.urls import canonicalise, detect_portal, extract_listing_urls


@pytest.mark.parametrize(
    "url,portal,portal_id",
    [
        ("https://www.rightmove.co.uk/properties/92053296", "rightmove", "92053296"),
        (
            "https://www.rightmove.co.uk/properties/92053296#/?channel=RES_LET",
            "rightmove",
            "92053296",
        ),
        ("https://www.zoopla.co.uk/to-rent/details/73991508/", "zoopla", "73991508"),
        (
            "https://www.zoopla.co.uk/to-rent/details/73991508/?search_identifier=abc",
            "zoopla",
            "73991508",
        ),
        (
            "https://www.openrent.co.uk/property-to-rent/london/1-bed-flat-london-sw16/1687221/",
            "openrent",
            "1687221",
        ),
    ],
)
def test_canonicalise_strips_tracking_and_extracts_id(url, portal, portal_id):
    result = canonicalise(url)
    assert result is not None
    assert result[0] == portal
    assert result[1] == portal_id
    assert "?" not in result[2] and "#" not in result[2]


def test_non_listing_urls_are_rejected():
    assert canonicalise("https://www.rightmove.co.uk/property-to-rent/Nine-Elms.html") is None
    assert canonicalise("https://example.com/properties/123") is None


def test_detect_portal_ignores_unknown_hosts():
    assert detect_portal("https://www.onthemarket.com/details/123/") is None


def test_extract_finds_urls_wrapped_in_percent_encoded_tracking_links():
    body = (
        '<a href="https://click.rightmove.co.uk/f/a/abc123/xyz/'
        'https%3A%2F%2Fwww.rightmove.co.uk%2Fproperties%2F92053296">See it</a>'
    )
    assert extract_listing_urls(body) == ["https://www.rightmove.co.uk/properties/92053296"]


def test_a_wrapped_and_a_direct_link_to_one_listing_yield_one_url():
    body = (
        '<a href="https://click.rightmove.co.uk/f/a/x/'
        'https%3A%2F%2Fwww.rightmove.co.uk%2Fproperties%2F92053296">Wrapped</a>'
        '<a href="https://www.rightmove.co.uk/properties/92053296">Direct</a>'
    )
    assert extract_listing_urls(body) == ["https://www.rightmove.co.uk/properties/92053296"]


def test_extract_listing_urls_finds_all_and_dedupes():
    body = """
      <a href="https://www.rightmove.co.uk/properties/92053296#/?utm=x">One</a>
      <a href="https://www.rightmove.co.uk/properties/92053296">One again</a>
      <a href="https://www.zoopla.co.uk/to-rent/details/73991508/">Two</a>
      <a href="https://www.rightmove.co.uk/unsubscribe.html">Not a listing</a>
    """
    assert extract_listing_urls(body) == [
        "https://www.rightmove.co.uk/properties/92053296",
        "https://www.zoopla.co.uk/to-rent/details/73991508",
    ]
