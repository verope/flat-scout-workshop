"""Portal listing URLs.

This is the load-bearing ingestion layer: as long as URL extraction holds, no
Listing is ever lost, whatever the surrounding text looks like.
"""

from __future__ import annotations

import re
from urllib.parse import unquote

PATTERNS: dict[str, re.Pattern[str]] = {
    "rightmove": re.compile(r"https?://(?:www\.)?rightmove\.co\.uk/properties/(\d+)"),
    "zoopla": re.compile(r"https?://(?:www\.)?zoopla\.co\.uk/to-rent/details/(\d+)"),
    # The id is the final path segment and is all digits. A slug segment can
    # itself start with a digit ("1-bed-flat-london-sw16"), so the lookahead
    # is what stops the match landing inside the slug.
    "openrent": re.compile(
        r"https?://(?:www\.)?openrent\.co\.uk/property-to-rent/"
        r"(?:[^/\s\"'<>]+/){1,3}(\d+)(?=[/?#\s\"'<>]|$)"
    ),
}

CANONICAL: dict[str, str] = {
    "rightmove": "https://www.rightmove.co.uk/properties/{id}",
    "zoopla": "https://www.zoopla.co.uk/to-rent/details/{id}",
}

COMBINED = re.compile("|".join(f"(?:{pattern.pattern})" for pattern in PATTERNS.values()))


def detect_portal(url: str) -> str | None:
    for portal, pattern in PATTERNS.items():
        if pattern.search(url):
            return portal
    return None


def canonicalise(url: str) -> tuple[str, str, str] | None:
    """Return (portal, portal_id, canonical_url), or None if not a listing URL."""
    for portal, pattern in PATTERNS.items():
        match = pattern.search(url)
        if match is None:
            continue
        portal_id = match.group(1)
        template = CANONICAL.get(portal)
        if template is not None:
            return portal, portal_id, template.format(id=portal_id)
        # OpenRent's slug carries the human-readable address, so keep the
        # matched prefix rather than reconstructing it.
        canonical = url[: match.end()].split("?")[0].rstrip("/")
        return portal, portal_id, canonical
    return None


def extract_listing_urls(text: str) -> list[str]:
    """Every canonical listing URL in a blob of text, in order, deduplicated.

    Listing links usually arrive wrapped in a tracking redirect, with the
    destination percent-encoded, so the decoded forms are scanned too. Decoding
    costs nothing and avoids a network round trip per link to follow the
    redirect.
    """
    found: list[str] = []
    seen: set[tuple[str, str]] = set()
    once = unquote(text)
    twice = unquote(once)
    for candidate in (text, once, twice):
        for match in COMBINED.finditer(candidate):
            result = canonicalise(match.group(0))
            if result is None:
                continue
            portal, portal_id, canonical = result
            if (portal, portal_id) in seen:
                continue
            seen.add((portal, portal_id))
            found.append(canonical)
    return found
