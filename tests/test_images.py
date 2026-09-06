"""The local image cache.

The property worth holding is that one Listing's pictures are fetched once and
then belong to us: the vision reading, `annotate --db` and every later re-run
read them off disk, and a media host that has since died costs nothing. The
other half is that a picture that will not download must never cost the
Listing - each one is caught on its own and the rest still land.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from flat_scout.config import Settings
from flat_scout.images import cache_images, cache_dir, local_image
from flat_scout.models import ListingData

PHOTO_URL = "https://media.example.com/photo.jpeg"
PLAN_URL = "https://media.example.com/plan.png"
EPC_URL = "https://media.example.com/epc.pdf"

JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"0" * 64
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"1" * 64
PDF_BYTES = b"%PDF-1.4\n" + b"2" * 64


def a_listing(**overrides) -> ListingData:
    base = dict(
        portal="rightmove",
        portal_id="92053296",
        url="https://www.rightmove.co.uk/properties/92053296",
        image_url=PHOTO_URL,
        floorplan_url=PLAN_URL,
        epc_image_url=EPC_URL,
    )
    base.update(overrides)
    return ListingData(**base)


def settings_for() -> Settings:
    settings = Settings()
    settings.fetch.delay_seconds = 0
    return settings


def mock_all() -> dict[str, respx.Route]:
    return {
        "photo": respx.get(PHOTO_URL).mock(
            return_value=httpx.Response(
                200, content=JPEG_BYTES, headers={"content-type": "image/jpeg"}
            )
        ),
        "floorplan": respx.get(PLAN_URL).mock(
            return_value=httpx.Response(
                200, content=PNG_BYTES, headers={"content-type": "image/png"}
            )
        ),
        "epc": respx.get(EPC_URL).mock(
            return_value=httpx.Response(
                200, content=PDF_BYTES, headers={"content-type": "application/pdf"}
            )
        ),
    }


@pytest.mark.asyncio
@respx.mock
async def test_every_picture_a_listing_links_is_written_once_into_its_own_directory(tmp_path):
    mock_all()
    async with httpx.AsyncClient() as client:
        cached = await cache_images(a_listing(), settings_for(), client, root=tmp_path)

    directory = tmp_path / "rightmove_92053296"
    assert cached == {
        "image_path": (directory / "photo.jpg").as_posix(),
        "floorplan_path": (directory / "floorplan.png").as_posix(),
        "epc_image_path": (directory / "epc.pdf").as_posix(),
    }
    assert (directory / "photo.jpg").read_bytes() == JPEG_BYTES
    assert (directory / "floorplan.png").read_bytes() == PNG_BYTES
    assert (directory / "epc.pdf").read_bytes() == PDF_BYTES
    assert cache_dir(a_listing(), tmp_path) == directory


@pytest.mark.asyncio
@respx.mock
async def test_the_stored_paths_use_forward_slashes(tmp_path):
    """The database is copied between machines, so a path in it must travel."""
    mock_all()
    async with httpx.AsyncClient() as client:
        cached = await cache_images(a_listing(), settings_for(), client, root=tmp_path)
    assert all("\\" not in path for path in cached.values())
    assert all(path.endswith(("photo.jpg", "floorplan.png", "epc.pdf")) for path in cached.values())


@pytest.mark.asyncio
@respx.mock
async def test_a_dead_media_url_costs_that_picture_and_not_the_other_two(tmp_path):
    mock_all()
    respx.get(PLAN_URL).mock(return_value=httpx.Response(404))
    async with httpx.AsyncClient() as client:
        cached = await cache_images(a_listing(), settings_for(), client, root=tmp_path)
    assert set(cached) == {"image_path", "epc_image_path"}
    assert not (tmp_path / "rightmove_92053296" / "floorplan.png").exists()


@pytest.mark.asyncio
@respx.mock
async def test_something_that_is_not_an_image_is_not_cached(tmp_path):
    """A media host answering with an HTML error page is not a floorplan."""
    mock_all()
    respx.get(PLAN_URL).mock(
        return_value=httpx.Response(
            200, content=b"<html>nope</html>", headers={"content-type": "text/html"}
        )
    )
    async with httpx.AsyncClient() as client:
        cached = await cache_images(a_listing(), settings_for(), client, root=tmp_path)
    assert "floorplan_path" not in cached
    assert list((tmp_path / "rightmove_92053296").iterdir()) != []


@pytest.mark.asyncio
@respx.mock
async def test_an_empty_body_is_not_cached(tmp_path):
    mock_all()
    respx.get(PLAN_URL).mock(
        return_value=httpx.Response(200, content=b"", headers={"content-type": "image/png"})
    )
    async with httpx.AsyncClient() as client:
        cached = await cache_images(a_listing(), settings_for(), client, root=tmp_path)
    assert "floorplan_path" not in cached


@pytest.mark.asyncio
@respx.mock
async def test_a_listing_with_no_floorplan_asks_for_none(tmp_path):
    """Most Listings publish no plan at all, and a None must cost no request."""
    routes = mock_all()
    async with httpx.AsyncClient() as client:
        cached = await cache_images(
            a_listing(floorplan_url=None), settings_for(), client, root=tmp_path
        )
    assert set(cached) == {"image_path", "epc_image_path"}
    assert not routes["floorplan"].called


@pytest.mark.asyncio
@respx.mock
async def test_a_picture_already_on_disk_is_never_fetched_again(tmp_path):
    """The whole point: a second `check` on the same Listing costs nothing."""
    routes = mock_all()
    directory = tmp_path / "rightmove_92053296"
    directory.mkdir(parents=True)
    (directory / "floorplan.png").write_bytes(PNG_BYTES)
    listing = a_listing(
        image_url=None,
        epc_image_url=None,
        floorplan_path=(directory / "floorplan.png").as_posix(),
    )
    async with httpx.AsyncClient() as client:
        cached = await cache_images(listing, settings_for(), client, root=tmp_path)
    assert cached == {}
    assert not routes["floorplan"].called


@pytest.mark.asyncio
@respx.mock
async def test_a_path_whose_file_has_gone_is_fetched_again(tmp_path):
    """A `data/` directory copied without its images is a cache to refill."""
    routes = mock_all()
    listing = a_listing(
        image_url=None,
        epc_image_url=None,
        floorplan_path="data/images/rightmove_92053296/floorplan.png",
    )
    async with httpx.AsyncClient() as client:
        cached = await cache_images(listing, settings_for(), client, root=tmp_path)
    assert cached == {
        "floorplan_path": (tmp_path / "rightmove_92053296" / "floorplan.png").as_posix()
    }
    assert routes["floorplan"].call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_the_cache_root_is_read_at_call_time(tmp_path, monkeypatch):
    """So that the whole cache can be moved without threading a path through
    every caller between the CLI and here."""
    from flat_scout import images

    mock_all()
    monkeypatch.setattr(images, "IMAGES_DIR", tmp_path / "elsewhere")
    async with httpx.AsyncClient() as client:
        cached = await images.cache_images(a_listing(), settings_for(), client)
    assert (tmp_path / "elsewhere" / "rightmove_92053296" / "photo.jpg").is_file()
    assert cached["image_path"].startswith((tmp_path / "elsewhere").as_posix())


@pytest.mark.parametrize(
    ("name", "content", "media_type"),
    [
        ("photo.jpg", JPEG_BYTES, "image/jpeg"),
        ("floorplan.png", PNG_BYTES, "image/png"),
        ("epc.pdf", PDF_BYTES, "application/pdf"),
    ],
)
def test_a_cached_file_reads_back_as_bytes_and_a_media_type(tmp_path, name, content, media_type):
    path = tmp_path / name
    path.write_bytes(content)
    assert local_image(str(path)) == (content, media_type)


def test_a_missing_file_reads_back_as_nothing(tmp_path):
    assert local_image(str(tmp_path / "gone.jpg")) is None


def test_no_path_at_all_reads_back_as_nothing():
    assert local_image(None) is None
    assert local_image("") is None


def test_a_file_of_an_unknown_kind_reads_back_as_nothing(tmp_path):
    """Only what the readers can handle. A .txt on disk is not an image."""
    path = tmp_path / "photo.txt"
    path.write_bytes(b"not an image")
    assert local_image(str(path)) is None


def test_a_directory_is_not_an_image(tmp_path):
    """`is_file` and not `exists`: an OSError here must never reach a caller."""
    directory = tmp_path / "photo.jpg"
    directory.mkdir()
    assert local_image(str(directory)) is None
