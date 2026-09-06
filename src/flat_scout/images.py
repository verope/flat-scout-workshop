"""The local image cache: every picture a Listing links, downloaded once.

`check` fills it, so that nothing downstream - the vision reading, the
annotation corpus, a later re-run - depends on the Portal's media host still
answering. Paths are stored relative to the repository root, beside the URLs
they came from, so `data/` can be copied to another machine whole.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

from flat_scout.config import Settings
from flat_scout.models import ListingData
from flat_scout.observe import span

log = logging.getLogger(__name__)

# Relative, like `cli.DB_PATH`: the cache lives beside the database, and both
# are addressed from the repository root so `data/` moves as one directory.
IMAGES_DIR = Path("data/images")

# What a media host may serve that is worth keeping. The suffix is taken from
# the declared media type rather than from the URL, so a file on disk always
# says what it is - which is what `local_image` reads back out of it.
SUFFIXES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "application/pdf": ".pdf",
}
MEDIA_TYPES = {suffix: media for media, suffix in SUFFIXES.items()}

# Generous: a floorplan is a few hundred kilobytes and an EPC certificate a
# little more. Anything past this is not a picture of a flat, and the point of
# the cache is to hold what the readers will want, not whatever was served.
MAX_CACHED_BYTES = 20_000_000

# URL field, path field, file stem. One entry per picture a Listing links.
IMAGE_FIELDS = (
    ("image_url", "image_path", "photo"),
    ("floorplan_url", "floorplan_path", "floorplan"),
    ("epc_image_url", "epc_image_path", "epc"),
)


def cache_dir(listing: ListingData, root: Path = IMAGES_DIR) -> Path:
    """Where one Listing's pictures live. One directory per Listing."""
    return root / f"{listing.portal}_{listing.portal_id}"


async def cache_images(
    listing: ListingData,
    settings: Settings,
    client: httpx.AsyncClient,
    root: Path | None = None,
) -> dict[str, str]:
    """Download whatever this Listing links that is not already on disk.

    Returns only the fields cached on this call, ready for
    `Database.set_image_paths`. Never raises for one image: a media host is a
    third party, and every way it can disappoint us means the same thing - this
    picture is not in the cache, and the reader falls back to the URL.

    `root` is read from the module-level `IMAGES_DIR` at call time when it is
    not given, so a caller (or a test) can point the whole cache somewhere else
    without threading the path through every layer above.
    """
    root = IMAGES_DIR if root is None else root
    directory = cache_dir(listing, root)
    cached: dict[str, str] = {}
    for url_field, path_field, stem in IMAGE_FIELDS:
        url = getattr(listing, url_field)
        if not url:
            continue
        stored = getattr(listing, path_field, None)
        # Already here. The file's presence is the test, not the column's:
        # a path pointing at nothing is a cache that was emptied, and the
        # picture is worth fetching again.
        if stored and Path(stored).is_file():
            continue
        try:
            with span("cache image", portal_id=listing.portal_id, kind=stem):
                response = await client.get(
                    url,
                    headers={"User-Agent": settings.fetch.user_agent},
                    follow_redirects=True,
                    timeout=settings.fetch.timeout_seconds,
                )
                media_type = (
                    response.headers.get("content-type", "").split(";")[0].strip().lower()
                )
                if not response.is_success:
                    log.info(
                        "not caching the %s for %s: HTTP %s",
                        stem, listing.url, response.status_code,
                    )
                    continue
                suffix = SUFFIXES.get(media_type)
                if suffix is None:
                    log.info(
                        "not caching the %s for %s: %s is not an image",
                        stem, listing.url, media_type or "no content type",
                    )
                    continue
                data = response.content
                if not data or len(data) > MAX_CACHED_BYTES:
                    log.info(
                        "not caching the %s for %s: %d bytes",
                        stem, listing.url, len(data),
                    )
                    continue
                path = directory / f"{stem}{suffix}"
                directory.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        except Exception as exc:  # noqa: BLE001 - one picture, not the Listing
            log.warning("could not cache the %s for %s: %s", stem, listing.url, exc)
            continue
        # As written, and with forward slashes whatever platform wrote it:
        # the path goes in a database that is copied between machines.
        cached[path_field] = path.as_posix()
        log.info("cached the %s for %s at %s", stem, listing.url, cached[path_field])
    return cached


def local_image(path: str | None) -> tuple[bytes, str] | None:
    """The cached file's bytes and media type, or None if it is not there.

    Never raises. A missing file, an emptied cache, a path from a database
    copied without its `data/` directory - all of them mean the same thing to
    every caller: read the URL instead.
    """
    if not path:
        return None
    try:
        file = Path(path)
        if not file.is_file():
            return None
        media_type = MEDIA_TYPES.get(file.suffix.lower())
        if media_type is None:
            return None
        return file.read_bytes(), media_type
    except OSError:
        return None
