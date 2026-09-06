"""Listing-page download and per-Portal parsing.

The listing page is the primary source of truth. Zoopla is not downloaded at
all, because it serves a Cloudflare challenge to automated clients. Rightmove
and OpenRent are downloaded, but from a given machine only Rightmove may
answer: OpenRent puts an AWS WAF challenge in front of its listing pages for
the address the request comes from.
"""

from __future__ import annotations

import asyncio
import json
import re
from html.parser import HTMLParser

import httpx

from flat_scout import observe
from flat_scout.config import Settings
from flat_scout.models import ListingData
from flat_scout.observe import span
from flat_scout.urls import canonicalise

POSTCODE = re.compile(r"\b([A-Z]{1,2}\d{1,2}[A-Z]?)(?:\s*\d[A-Z]{2})?\b")
PRICE = re.compile(r"£\s?([\d,]+)")

# Rightmove's pin quality. Only ACCURATE_POINT places the pin on the building;
# every other value is the Portal blurring the address, and a distance derived
# from one would be a confident-looking guess. Across 31 live Listings sampled
# on 2026-08-17 exactly two values appeared - ACCURATE_POINT (26) and
# APPROXIMATE_POINT (5) - and the `location` block was never absent. The
# comparison is written as equality rather than as a blocklist so an unseen
# third value fails closed, into "unknown", rather than into a false distance.
ACCURATE_PIN = "ACCURATE_POINT"


def is_accurate_pin(pin_type: str | None) -> bool:
    """True only for the pin Rightmove places on the building itself."""
    return pin_type == ACCURATE_PIN


def build_async_client(settings: Settings) -> httpx.AsyncClient:
    """The one sanctioned way to build an `httpx.AsyncClient` in this app.

    Every client the app owns carries the fetch timeout from config here, which
    was silently lost twice before a single factory existed. Being the only
    sanctioned constructor is also what makes this the right place to trace
    from: `observe.instrument_client` is a no-op without a logfire token, and
    instruments every request the app makes once one is set.
    """
    client = httpx.AsyncClient(timeout=settings.fetch.timeout_seconds)
    observe.instrument_client(client)
    return client

# Zoopla is absent by design: it challenges every automated client.
FETCHABLE_PORTALS = frozenset({"rightmove", "openrent"})


class FetchError(Exception):
    pass


class ListingUnavailable(FetchError):
    """The Listing is gone. Retrying will not help, so the state is terminal."""


class BotChallenge(FetchError):
    """A Portal served a challenge page instead of the Listing.

    Retrying will not help either, but for a different reason, and the
    difference matters. `ListingUnavailable` means the flat is gone; this means
    the flat is fine and this host cannot see it. The block is aimed at our
    address, so it holds for as long as we ask from here - but the Listing is
    still worth surfacing with its link.
    """


def is_fetchable(portal: str) -> bool:
    return portal in FETCHABLE_PORTALS


# Tags that sit INSIDE a word or a sentence. Everything else ends a run of
# text, and a separator has to be inserted at the boundary: the chunks used to
# be concatenated with nothing between them, so `<h1>...SW11</h1><div>No
# furnished...` came out as "SW11No furnished" - two words welded into one.
#
# That is not cosmetic. Every extractor here is a regex over this string: the
# postcode (a hard filter), the price, the floor, the furnishing. A weld hides
# a word boundary, so `\bno\b` cannot see a denial and `\b[A-Z]{1,2}\d` can
# match across the join. The real fixtures carry these - "4Eastway", "wThe".
#
# The list is what may legitimately appear mid-word, so `<b>Furn</b>ished`
# still reads as one word. Splitting a word is the rarer harm and the visible
# one; welding two is common and silent.
INLINE_TAGS = frozenset(
    {"a", "b", "i", "em", "strong", "span", "u", "small", "sub", "sup", "abbr", "mark"}
)


class _TextExtractor(HTMLParser):
    """Visible text plus the first <h1>. Enough for our regex work."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.chunks: list[str] = []
        self.h1_chunks: list[str] = []
        self._suppress = 0
        self._in_h1 = False

    def _break(self, tag: str) -> None:
        if tag not in INLINE_TAGS:
            self.chunks.append(" ")
            if self._in_h1:
                self.h1_chunks.append(" ")

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("script", "style"):
            self._suppress += 1
        elif tag == "h1":
            self._in_h1 = True
        self._break(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._suppress:
            self._suppress -= 1
        elif tag == "h1":
            self._in_h1 = False
        self._break(tag)

    def handle_data(self, data: str) -> None:
        if self._suppress:
            return
        self.chunks.append(data)
        if self._in_h1:
            self.h1_chunks.append(data)


def _extract(html: str) -> _TextExtractor:
    parser = _TextExtractor()
    parser.feed(html)
    return parser


def html_text(html: str) -> str:
    return " ".join("".join(_extract(html).chunks).split())


def first_h1(html: str) -> str | None:
    text = " ".join("".join(_extract(html).h1_chunks).split())
    return text or None


def _outcode(text: str | None) -> str | None:
    if not text:
        return None
    match = POSTCODE.search(text.upper())
    return match.group(1) if match else None


def _price(text: str | None) -> int | None:
    if not text:
        return None
    match = PRICE.search(text)
    return int(match.group(1).replace(",", "")) if match else None


def _unflatten(table: list) -> object:
    """Resolve Rightmove's devalue-style payload.

    The page ships one shared table. Every value inside an object or array is
    an integer index into that table, which lets repeated strings appear once.
    A plain json.loads therefore yields a skeleton of integers, not data.
    """
    resolved: dict[int, object] = {}

    def walk(node):
        if not isinstance(node, int) or isinstance(node, bool):
            return node
        if node < 0:  # devalue's sentinels for undefined, NaN and friends
            return None
        if node in resolved:
            return resolved[node]
        value = table[node]
        if isinstance(value, dict):
            container: dict = {}
            resolved[node] = container
            for key, item in value.items():
                container[key] = walk(item)
            return container
        if isinstance(value, list):
            items: list = []
            resolved[node] = items
            for item in value:
                items.append(walk(item))
            return items
        resolved[node] = value
        return value

    return walk(0)


def _page_model(html: str) -> dict:
    index = html.find("window.__PAGE_MODEL")
    if index == -1:
        raise FetchError("window.__PAGE_MODEL absent")
    start = html.find("{", index)
    if start == -1:
        raise FetchError("window.__PAGE_MODEL has no object")
    try:
        wrapper, _ = json.JSONDecoder().raw_decode(html[start:])
    except json.JSONDecodeError as exc:
        raise FetchError(f"window.__PAGE_MODEL is not valid JSON: {exc}") from exc
    data = wrapper.get("data")
    if isinstance(data, str):
        model = _unflatten(json.loads(data))
    else:
        model = wrapper
    if not isinstance(model, dict):
        raise FetchError("window.__PAGE_MODEL did not resolve to an object")
    return model


def _sqft(sizings) -> int | None:
    for sizing in sizings or []:
        if sizing.get("unit") == "sqft" and sizing.get("minimumSize"):
            return int(sizing["minimumSize"])
    return None


def _headline_image(images) -> str | None:
    """The photograph the Listing leads with.

    Rightmove keeps the images in the order the agent arranged them, so the
    first one is the shot they chose to sell the Listing with. Captions are not
    consulted: they are frequently absent, and "Reception" is no more the
    headline than an uncaptioned exterior.
    """
    for image in images or []:
        url = image.get("url")
        if url:
            return url
    return None


# An Environmental Impact graph carries its own A-G scale, but it measures CO2,
# not energy efficiency. Reading a band off one and storing it as the EPC band
# would present a carbon rating as evidence about warmth and bills. Rightmove
# labels them "EI Rating"; the energy graph is "EE Rating", "EER" or "EPC ...".
ENVIRONMENTAL_IMPACT = re.compile(r"^EI\b|ENVIRONMENTAL", re.I)


def _energy_graph(graphs) -> dict | None:
    """The energy-efficiency graph, never the environmental-impact one."""
    for graph in graphs or []:
        caption = " ".join((graph.get("caption") or "").split())
        if ENVIRONMENTAL_IMPACT.search(caption):
            continue
        if graph.get("url"):
            return graph
    return None


def _epc_image_url(graphs) -> str | None:
    """The EPC graph image. The band is only legible inside it, not in the
    payload, so this is the input the image-reading step needs."""
    graph = _energy_graph(graphs)
    return graph.get("url") if graph else None


def _epc_caption(graphs) -> str | None:
    """Rightmove's caption for the energy graph, verbatim. No band is read from it.

    Across 30 live listings sampled on 2026-08-16 the caption named the graph,
    not the band: "EE Rating", "EER", "EPC", "EPC 1", "EPC Graph", "EPC Rating
    Graph". The band lives only inside the image. This describes the same graph
    `_epc_image_url` returns, so the two never disagree.
    """
    graph = _energy_graph(graphs)
    if graph is None:
        return None
    return " ".join((graph.get("caption") or "").split()) or None


def _floor(entrance_floor) -> str | None:
    """The entrance floor as text.

    Rightmove sends this either as a plain string or as an object carrying an
    `alias` and a `displayText`. Stored raw, the object raises in two places
    rather than passing quietly: sqlite refuses to bind a dict, and the
    ground-floor filter calls `.lower()` on it. Neither is a silent bypass, but
    both abort the Listing before it can be judged.
    """
    if isinstance(entrance_floor, dict):
        return entrance_floor.get("displayText") or entrance_floor.get("alias") or None
    return entrance_floor or None


def _floorplan_url(floorplans) -> str | None:
    """The full-size floorplan image, or None.

    `resizedFloorplanUrls` is deliberately ignored: the thumbnail Rightmove
    offers is 296x197, at which the room labels and dimensions the couple care
    about are gone. Anything Rightmove does not type as an IMAGE — an
    interactive plan, a PDF — is skipped, because the vision step cannot read it.
    """
    for floorplan in floorplans or []:
        if floorplan.get("type") not in (None, "IMAGE"):
            continue
        url = floorplan.get("url")
        if url:
            return url
    return None


def _nearest_station(stations) -> tuple[str | None, float | None]:
    """The closest station, by distance. Rightmove's order is not distance order."""
    closest = None
    for station in stations or []:
        # A distance in any other unit would be stored as miles, so skip it.
        if station.get("unit") != "miles" or station.get("distance") is None:
            continue
        if not station.get("name"):
            continue
        if closest is None or station["distance"] < closest["distance"]:
            closest = station
    if closest is None:
        return None, None
    return closest["name"], round(float(closest["distance"]), 2)


def _location(prop: dict) -> tuple[float | None, float | None]:
    """The Listing's coordinates, but only from a pin Rightmove calls accurate.

    Sampling 31 live Listings on 2026-08-17 turned up two pin types: 26 carried
    ACCURATE_POINT, and 5 - typically a scheme advertising many units from one
    address - carried APPROXIMATE_POINT. An approximate pin can sit a quarter of
    a mile from the building, which is more error than a coordinate is worth, so
    it is discarded and the Listing stays unlocated. The other signals on the
    row, `nearest_station_miles` among them, are unaffected.
    """
    location = prop.get("location") or {}
    if not is_accurate_pin(location.get("pinType")):
        return None, None
    latitude, longitude = location.get("latitude"), location.get("longitude")
    if latitude is None or longitude is None:
        return None, None
    return float(latitude), float(longitude)


def parse_rightmove(html: str, url: str) -> ListingData:
    model = _page_model(html)
    prop = model.get("propertyData") or {}
    if not prop:
        raise FetchError("propertyData absent from window.__PAGE_MODEL")
    address = prop.get("address") or {}
    lettings = prop.get("lettings") or {}

    description = html_text((prop.get("text") or {}).get("description") or "")
    features = prop.get("keyFeatures") or []
    if features:
        description = " ".join(["; ".join(features), description]).strip()

    station, station_miles = _nearest_station(prop.get("nearestStations"))
    latitude, longitude = _location(prop)

    _, portal_id, canonical = canonicalise(url)  # type: ignore[misc]
    return ListingData(
        portal="rightmove",
        portal_id=portal_id,
        url=canonical,
        address=address.get("displayAddress"),
        postcode=address.get("outcode") or _outcode(address.get("displayAddress")),
        price_pcm=_price((prop.get("prices") or {}).get("primaryPrice")),
        beds=prop.get("bedrooms"),
        sqft=_sqft(prop.get("sizings")),
        floor=_floor(prop.get("entranceFloor")),
        furnished=lettings.get("furnishType"),
        pet_notes=None,
        description=description or None,
        listed_on=(prop.get("listingHistory") or {}).get("listingUpdateReason"),
        image_url=_headline_image(prop.get("images")),
        epc_caption=_epc_caption(prop.get("epcGraphs")),
        epc_image_url=_epc_image_url(prop.get("epcGraphs")),
        floorplan_url=_floorplan_url(prop.get("floorplans")),
        council_tax_band=(prop.get("livingCosts") or {}).get("councilTaxBand"),
        nearest_station=station,
        nearest_station_miles=station_miles,
        latitude=latitude,
        longitude=longitude,
    )


FLOOR_PHRASES = ("ground floor", "top floor", "first floor", "second floor", "basement")
FURNISHED_PHRASES = ("fully furnished", "part furnished", "unfurnished", "furnished")

# "not furnished", "isn't furnished", "never furnished". `_phrase` matches by
# substring, and none of "fully furnished", "part furnished" or "unfurnished"
# is inside "not furnished" - so a plain search lands on the bare "furnished"
# and stores the exact opposite of what the advert said. Rightmove sends a
# closed enum and cannot reach this; OpenRent's furnishing is read out of the
# listing copy, where a sentence can say anything.
#
# The second alternative is the attached form, and it is why a word boundary
# is not enough on its own: "non-furnished" and "un-furnished" put a non-word
# character immediately before the match, so `\bfurnished\b` matches happily
# inside a word that means the opposite. `\bun\b` cannot be used either - it
# would have to not fire on "unfurnished", which is a real answer, so the
# prefix is only a negation when something separates it from the phrase.
NEGATED = re.compile(
    r"\b(not|isn't|is not|aren't|are not|never|no)\s+(\w+\s+){0,2}$"
    r"|\b(non|un)[\s-]+$",
    re.I,
)


def _unnegated_phrase(text: str, options: tuple[str, ...]) -> str | None:
    """`_phrase`, refusing a match that something just denied.

    Unknown rather than the opposite, deliberately. "Not furnished" plainly
    means unfurnished, but "not furnished to the standard you would expect"
    plainly does not, and this cannot tell them apart. A Criterion that comes
    back unknown asks the agent; one that guesses puts four weight behind a
    reading of a sentence it misread.
    """
    lowered = text.lower()
    for option in options:
        at = lowered.find(option)
        if at == -1:
            continue
        return None if NEGATED.search(lowered[:at]) else option
    return None


# OpenRent states a feature as a row: a label cell, then a value cell holding a
# tick or a cross. The label cell also contains an info button with its own
# icon, so the value is the LAST icon in the row, never the first.
OPENRENT_ROW = re.compile(
    r'<td class="fw-medium">\s*(?P<label>[^<]{2,40}?)\s*(?:<button|</td>)(?P<rest>.*?)</tr>',
    re.S,
)
OPENRENT_ICON = re.compile(r'<svg class="(?P<classes>[^"]*)"')
# The property photograph. og:image is OpenRent's own share graphic, so it is
# not usable; image_src carries the first listing photo, protocol-relative.
OPENRENT_IMAGE = re.compile(r'<link rel="image_src" href="(?P<url>[^"]+)"')
OPENRENT_DESCRIPTION = re.compile(r'id="descriptionText"[^>]*>(?P<body>.*?)</div>', re.S)

# What each challenge page is made of. Cloudflare's is the familiar "Just a
# moment..." interstitial. AWS WAF's is the "Human Verification" page OpenRent
# puts in front of a blocked client: a title, the cookie-domain list its script
# sets up, and the `gokuProps` blob carrying the puzzle token.
#
# All three AWS markers must be present. Any one of them could turn up in
# ordinary listing copy - a flat with an entryphone could plausibly mention
# human verification - and a Listing wrongly called a block would be dropped to
# its link alone, with no page ever read.
CHALLENGE_MARKERS = {
    "Cloudflare": ("Just a moment",),
    "AWS WAF": ("Human Verification", "awsWafCookieDomainList", "gokuProps"),
}


def bot_challenge(html: str) -> str | None:
    """Which bot challenge this body is, or None if it is an ordinary page.

    Read the body, never the status code. AWS WAF answers under a status of its
    own choosing, and OpenRent's arrives as HTTP 405 - which reads as "method
    not allowed" and is nothing of the sort. A 405 from anything else is still
    an honest 405 and must keep being reported as one.
    """
    for name, markers in CHALLENGE_MARKERS.items():
        if all(marker in html for marker in markers):
            return name
    return None


UNAVAILABLE_MARKERS = {
    "openrent": ("this property is no longer available for rent",),
    "rightmove": ("this property has been removed",),
}


def is_unavailable(portal: str, html: str) -> bool:
    """True when a Portal serves HTTP 200 for a Listing that is already gone.

    OpenRent returns an ordinary page reading "Let Agreed - This property is no
    longer available for rent", so status codes alone do not detect withdrawal.
    """
    lowered = html_text(html).lower()
    return any(marker in lowered for marker in UNAVAILABLE_MARKERS.get(portal, ()))


def _openrent_feature(html: str, label: str) -> bool | None:
    """True, False, or None when OpenRent does not state the feature at all."""
    flat = re.sub(r"\s+", " ", html)
    for row in OPENRENT_ROW.finditer(flat):
        if row.group("label").strip().lower() != label.lower():
            continue
        states = [
            icon.group("classes")
            for icon in OPENRENT_ICON.finditer(row.group("rest"))
            if "text-success" in icon.group("classes") or "text-danger" in icon.group("classes")
        ]
        if not states:
            return None
        return "text-success" in states[-1]
    return None


def _openrent_description(html: str) -> str | None:
    match = OPENRENT_DESCRIPTION.search(re.sub(r"\s+", " ", html))
    if match is None:
        return None
    return html_text(match.group("body")) or None


def _phrase(text: str, options: tuple[str, ...]) -> str | None:
    lowered = text.lower()
    for option in options:
        if option in lowered:
            return option
    return None


# OpenRent states the count BEFORE the label. The summary row reads "2 bedrooms
# 1 bathrooms 5 tenants max.", so a search for the digit AFTER "Bedrooms" finds
# the bathroom count instead, and a two-bed with one bathroom reads as a
# one-bed. The pair is matched together here rather than the bedrooms cell
# alone, so the bathroom count is consumed by the pattern rather than left
# sitting next to the label waiting to be mistaken for the answer.
#
# The gaps are `\s*` and not `\s+` because there need not be a gap at all:
# `html_text` pads block tags but not inline ones, so a summary row built from
# spans flattens to "2 bedrooms1 bathrooms". Requiring the space would make the
# count depend on OpenRent's choice of element, and fail silently the day it
# changed - back to whatever the headline says, or to nothing.
OPENRENT_SUMMARY_BEDS = re.compile(r"(\d+)\s*bedrooms?\s*(\d+)\s*bathrooms?", re.I)
# The <h1> states the same fact in its own words: "2 Bed Terraced House, Charles
# Street, BL4". Both "Bed" and "Bedroom" are seen in the wild.
OPENRENT_H1_BEDS = re.compile(r"(\d+)\s*bed(?:room)?s?\b", re.I)
# A studio has no number to state, so it is named instead: "Studio Flat, Craven
# Street, WC2N". The headline is "<type>, <address>", and only the type segment
# is read, so a flat on Studio Court is not turned into a studio by its address.
OPENRENT_STUDIO = re.compile(r"\bstudio\b", re.I)


def _openrent_is_studio(headline: str | None) -> bool:
    """Whether the headline calls this a studio.

    The headline is the only place on the page that says so, which is why this
    does not read the whole page as it used to. Sampling twenty live London
    Listings on 2026-08-21:

    - The summary row calls a studio "1 bedrooms 1 bathrooms", exactly as it
      calls a one-bed. It cannot tell them apart, so it is not consulted here.
      That is not a source disagreeing with the headline; it is a source that
      does not carry the fact.
    - The blurb is about the neighbourhood as much as the flat. One of the
      eighteen non-studios was a one-bed on Casson Square whose copy read
      "range from stylish studios to spacious one-, two- and three-bedroom
      apartments", and reading the whole page made it a studio.
    - No navigation, footer or cookie text mentioned the word on any of them.

    Getting this wrong towards "studio" is the expensive direction. `beds = 0`
    is a size the couple want, so it clears `allowed_beds` and a flat of any
    size travels through the hard filter unread, while the filter reports
    itself working. Reading one careful source beats reading every careless one.
    """
    return bool(OPENRENT_STUDIO.search((headline or "").split(",")[0]))


def _openrent_beds(text: str, headline: str | None) -> int | None:
    """The bed count, from two independent statements that have to agree.

    A studio is settled first and separately, by `_openrent_is_studio`, because
    OpenRent states it in words rather than in a number and the summary row
    would otherwise carry the day with "1 bedrooms".

    For everything else OpenRent says it twice, in the summary row and in the
    headline. Where both are legible they must match; where they disagree this
    returns None, and where only one of them is legible that one is taken.

    Requiring agreement is stricter than trusting either alone, and deliberately
    so, because `beds` is a hard filter and the two ways of being wrong are not
    equally bad. An unknown count passes `prefilter` and travels on to be judged
    by the evaluator and by the couple. A confident wrong count walks a two-bed
    through a one-bed shortlist, unread, while the filter reports itself
    working. Silence is much the cheaper mistake, so a page that cannot say the
    same thing twice says nothing.
    """
    if _openrent_is_studio(headline):
        return 0
    counts = {
        int(match.group(1))
        for match in (
            OPENRENT_SUMMARY_BEDS.search(text),
            OPENRENT_H1_BEDS.search(headline or ""),
        )
        if match is not None
    }
    # One agreed number, or nothing: an empty set is a page that never said,
    # and a set of two is a page that contradicted itself.
    return counts.pop() if len(counts) == 1 else None


PET_NOTES = {True: "Pets allowed", False: "Pets not allowed"}


def _openrent_image(html: str) -> str | None:
    match = OPENRENT_IMAGE.search(html)
    if match is None:
        return None
    found = match.group("url").strip()
    # The stored image URL must be absolute, and OpenRent omits the scheme.
    if found.startswith("//"):
        return "https:" + found
    return found or None


def parse_openrent(html: str, url: str) -> ListingData:
    text = html_text(html)
    headline = first_h1(html)
    description = _openrent_description(html)
    _, portal_id, canonical = canonicalise(url)  # type: ignore[misc]
    # Judge floor and furnishing from the listing copy, not the whole page:
    # OpenRent's boilerplate and its other adverts mention both.
    copy = description or text
    return ListingData(
        portal="openrent",
        portal_id=portal_id,
        url=canonical,
        address=headline,
        postcode=_outcode(text),
        price_pcm=_price(text),
        beds=_openrent_beds(text, headline),
        sqft=None,
        floor=_phrase(copy, FLOOR_PHRASES),
        furnished=_unnegated_phrase(copy, FURNISHED_PHRASES),
        pet_notes=PET_NOTES.get(_openrent_feature(html, "Pets Allowed")),
        image_url=_openrent_image(html),
        description=(description or text)[:4000] or None,
        listed_on=None,
    )


PARSERS = {"rightmove": parse_rightmove, "openrent": parse_openrent}


def parse_listing(portal: str, html: str, url: str) -> ListingData:
    parser = PARSERS.get(portal)
    if parser is None:
        raise FetchError(f"no parser for portal {portal!r}")
    listing = parser(html, url)
    if listing.price_pcm is None and listing.address is None:
        raise FetchError(f"parsed nothing usable from {url}")
    return listing


async def fetch_listing(
    url: str, settings: Settings, client: httpx.AsyncClient
) -> ListingData:
    parsed = canonicalise(url)
    if parsed is None:
        raise FetchError(f"not a listing URL: {url}")
    portal, _, canonical = parsed
    if not is_fetchable(portal):
        raise FetchError(
            f"{portal} pages are not downloaded: it challenges every automated client"
        )
    await asyncio.sleep(settings.fetch.delay_seconds)
    # The politeness sleep is outside the span and the request inside it, so
    # what this reports is the Portal's latency rather than our own manners.
    with span("fetch listing", portal=portal, url=canonical):
        response = await client.get(
            canonical,
            headers={"User-Agent": settings.fetch.user_agent},
            follow_redirects=True,
            timeout=settings.fetch.timeout_seconds,
        )
    # Ahead of every status check, because a challenge does not arrive under a
    # status that admits to being one. OpenRent's comes back as 405, which the
    # generic branch below filed away as a transport error; nothing stops the
    # next one arriving as 410 and being filed as a withdrawal, which is worse,
    # because that state is a dead end. Whatever the status, a challenge page is
    # not a Listing, so it is answered first.
    challenge = bot_challenge(response.text)
    if challenge:
        raise BotChallenge(
            f"{challenge} bot challenge (HTTP {response.status_code}) for {canonical}"
        )
    if response.status_code == 410:
        raise ListingUnavailable(f"listing withdrawn (410): {canonical}")
    if response.status_code != 200:
        raise FetchError(f"HTTP {response.status_code} for {canonical}")
    if is_unavailable(portal, response.text):
        raise ListingUnavailable(f"listing already let: {canonical}")
    return parse_listing(portal, response.text, canonical)
