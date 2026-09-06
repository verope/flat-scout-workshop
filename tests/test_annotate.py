"""The floorplan corpus and its human answers.

The properties worth holding are all about not destroying work: someone spends
an evening reading bearings off drawings, and a command that is re-run when new
Listings arrive must add to that and never touch it.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from flat_scout.annotate import (
    Annotation,
    apply_update,
    download_plan,
    load_annotations,
    render_annotations,
    render_editor,
    render_sheet,
    serve,
)

PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
    b"\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05"
    b"\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)
URL = "https://media.example.com/plan.png"


def write(tmp_path, annotations: dict[str, Annotation]):
    path = tmp_path / "annotations.toml"
    path.write_text(render_annotations(annotations))
    return path


def test_an_answer_survives_being_written_and_read_back(tmp_path):
    entry = Annotation(
        portal_id="1",
        file="1.jpg",
        north_clock=225,
        windows_clock=(0, 90),
        compass_style="rose",
        notes='the "N" sits at the lower left',
        done=True,
    )
    assert load_annotations(write(tmp_path, {"1": entry}))["1"] == entry


def test_a_plan_with_no_compass_is_a_real_answer_not_a_gap(tmp_path):
    """"No indicator is drawn" is what the pipeline should return too, so it has
    to be sayable - and distinguishable from nobody having looked yet."""
    entry = Annotation(portal_id="1", file="1.jpg", compass_style="none", done=True)
    read = load_annotations(write(tmp_path, {"1": entry}))["1"]
    assert read.north_clock is None
    assert read.done is True


def test_an_unanswered_stub_is_left_visibly_blank(tmp_path):
    """Not pre-filled with the pipeline's own guess.

    Starting an annotator from the machine's answer measures their patience
    rather than the drawing, and the number under test would end up confirming
    itself.
    """
    text = render_annotations({"1": Annotation(portal_id="1", file="1.jpg")})
    assert "windows_clock = []" in text
    assert "# north_clock" in text  # commented out, so absence is deliberate
    assert "done = false" in text


def test_a_dual_aspect_flat_keeps_both_of_its_bearings(tmp_path):
    entry = Annotation(portal_id="1", file="1.jpg", windows_clock=(45, 135), done=True)
    assert load_annotations(write(tmp_path, {"1": entry}))["1"].windows_clock == (45, 135)


def test_no_annotations_file_yet_is_not_an_error(tmp_path):
    assert load_annotations(tmp_path / "absent.toml") == {}


@respx.mock
def test_a_plan_already_in_the_corpus_is_not_downloaded_again(tmp_path):
    """Re-running this on every new batch must not re-fetch the whole corpus."""
    route = respx.get(URL).mock(
        return_value=httpx.Response(200, content=PNG, headers={"content-type": "image/png"})
    )
    with httpx.Client() as client:
        first = download_plan("1", URL, client, "agent", tmp_path)
        second = download_plan("1", URL, client, "agent", tmp_path)
    assert first == second == "1.png"
    assert route.call_count == 1


@respx.mock
def test_a_media_host_that_fails_costs_one_plan_and_not_the_run(tmp_path):
    respx.get(URL).mock(return_value=httpx.Response(500))
    with httpx.Client() as client:
        assert download_plan("1", URL, client, "agent", tmp_path) is None


@respx.mock
def test_something_that_is_not_an_image_is_not_put_in_the_corpus(tmp_path):
    respx.get(URL).mock(
        return_value=httpx.Response(200, content=b"<html>", headers={"content-type": "text/html"})
    )
    with httpx.Client() as client:
        assert download_plan("1", URL, client, "agent", tmp_path) is None
    assert not list(tmp_path.iterdir())


def test_the_sheet_carries_its_images_inside_it():
    """It is opened from disk, and has to work with no network and no server."""
    entry = Annotation(portal_id="1", file="1.png", address="12 Elm Road")
    page = render_sheet([(entry, PNG, "image/png")])
    assert "data:image/png;base64," in page
    assert "https://" not in page
    assert "12 Elm Road" in page
    # The bearing convention, on the page rather than in someone's head.
    assert "0 = up the page" in page
    # ...and the block to paste an answer into.
    assert "[plan.1]" in page


def test_the_sheet_says_which_way_round_the_bearings_go():
    page = render_sheet([(Annotation(portal_id="1", file="1.png"), PNG, "image/png")])
    for label in ("0", "90", "180", "270"):
        assert f">{label}</text>" in page


@pytest.mark.parametrize("bad", ['a "quoted" note', "back\\slash"])
def test_a_note_with_awkward_characters_survives(tmp_path, bad):
    entry = Annotation(portal_id="1", file="1.jpg", notes=bad, done=True)
    assert load_annotations(write(tmp_path, {"1": entry}))["1"].notes == bad


# --- the editor that writes the file ---


def corpus(tmp_path, *ids: str) -> dict[str, Annotation]:
    (tmp_path / "plans").mkdir(exist_ok=True)
    entries = {}
    for portal_id in ids:
        (tmp_path / "plans" / f"{portal_id}.png").write_bytes(PNG)
        entries[portal_id] = Annotation(
            portal_id=portal_id, file=f"{portal_id}.png", address="12 Elm Road"
        )
    return entries


def test_saving_one_plan_leaves_every_other_answer_alone(tmp_path):
    """An evening's work is in this file by the time autosave is running."""
    entries = corpus(tmp_path, "1", "2")
    entries["2"] = Annotation("2", "2.png", "", north_clock=90, windows_clock=(180,), done=True)
    updated = apply_update(entries, "1", {"north_clock": 270, "windows_clock": [0]})
    assert updated["1"].north_clock == 270
    assert updated["2"] == entries["2"]


def test_a_cleared_field_can_become_empty_again(tmp_path):
    """The editor sends a whole plan, so an answer must be removable."""
    entries = {"1": Annotation("1", "1.png", "", north_clock=90, windows_clock=(180,))}
    updated = apply_update(entries, "1", {"north_clock": None, "windows_clock": []})
    assert updated["1"].north_clock is None
    assert updated["1"].windows_clock == ()


@pytest.mark.parametrize("north", [-1, 360, 999])
def test_a_bearing_outside_the_circle_is_refused(tmp_path, north):
    with pytest.raises(ValueError):
        apply_update({"1": Annotation("1", "1.png")}, "1", {"north_clock": north})


def test_a_window_bearing_is_wrapped_rather_than_refused(tmp_path):
    """Clicking twice near due north can produce 360 by rounding."""
    updated = apply_update({"1": Annotation("1", "1.png")}, "1", {"windows_clock": [360, 405]})
    assert updated["1"].windows_clock == (0, 45)


def test_a_plan_that_is_not_in_the_corpus_is_refused(tmp_path):
    with pytest.raises(KeyError):
        apply_update({"1": Annotation("1", "1.png")}, "9", {"north_clock": 0})


def test_the_editor_offers_the_click_to_pick_controls():
    page = render_editor([Annotation("1", "1.png", "12 Elm Road", north_clock=45)])
    assert "data-mode='north'" in page
    assert "data-mode='window'" in page
    assert "value='45'" in page  # existing answers come back for editing
    assert "/plans/1.png" in page


def test_the_editor_and_the_server_agree_end_to_end(tmp_path):
    """A real request over a real socket, because the wiring is the risk here."""
    import json
    import threading
    import urllib.request

    entries = corpus(tmp_path, "1", "2")
    path = tmp_path / "annotations.toml"
    path.write_text(render_annotations(entries))
    server = serve(path, tmp_path / "plans", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "12 Elm Road" in page

        body = json.dumps({"north_clock": 225, "windows_clock": [0, 90], "done": True}).encode()
        request = urllib.request.Request(
            base + "/annotations/1", data=body, headers={"Content-Type": "application/json"}
        )
        assert urllib.request.urlopen(request).status == 204
        saved = load_annotations(path)["1"]
        assert saved.north_clock == 225 and saved.windows_clock == (0, 90) and saved.done

        image = urllib.request.urlopen(base + "/plans/1.png").read()
        assert image == PNG
    finally:
        server.shutdown()
        server.server_close()


def test_the_server_will_not_serve_a_file_outside_the_corpus(tmp_path):
    import threading
    import urllib.error
    import urllib.request

    (tmp_path / "secret.png").write_bytes(b"not yours")
    entries = corpus(tmp_path, "1")
    path = tmp_path / "annotations.toml"
    path.write_text(render_annotations(entries))
    server = serve(path, tmp_path / "plans", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with pytest.raises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(base + "/plans/../secret.png")
        assert raised.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_the_corpus_is_enough_to_annotate_from(tmp_path):
    """No database, no network. The images and the answers are the whole thing.

    Annotating is reading a drawing and writing down what it says; the database
    is how new drawings arrive and nothing more. Making the two separable is
    what lets the corpus be checked out and worked on anywhere.
    """
    entries = corpus(tmp_path, "1")
    path = tmp_path / "annotations.toml"
    path.write_text(render_annotations(entries))
    reloaded = load_annotations(path)
    assert reloaded["1"].address == "12 Elm Road"
    # Enough to render the editor with, straight from what was on disk.
    page = render_editor(list(reloaded.values()))
    assert "12 Elm Road" in page


@pytest.mark.parametrize(
    "awkward",
    [
        "Northcote Road, \nBetween the Commons, SW11",  # a real Portal address
        "tab\there",
        'quo"te',
        "back\\slash",
        "bell\x07",
    ],
)
def test_an_awkward_string_does_not_break_the_whole_file(tmp_path, awkward):
    """One bad character used to cost every answer in the file, not just one.

    A Portal address arrived with a newline in it, the file stopped parsing, and
    the corpus would not load at all.
    """
    entries = {"1": Annotation("1", "1.png", address=awkward, notes=awkward, done=True)}
    read = load_annotations(write(tmp_path, entries))
    assert read["1"].address == awkward
    assert read["1"].notes == awkward


# --- plans that `check` already cached --------------------------------------


def test_a_cached_plan_is_copied_rather_than_downloaded(tmp_path):
    """`check` already paid for this picture, and the media host may well be
    gone by the time somebody sits down to annotate."""
    cached = tmp_path / "cache" / "floorplan.jpg"
    cached.parent.mkdir()
    cached.write_bytes(PNG)  # bytes, not a format assertion: the suffix decides
    into = tmp_path / "corpus"
    with respx.mock, httpx.Client() as client:
        route = respx.get(URL).mock(return_value=httpx.Response(500))
        name = download_plan("1", URL, client, "agent", into, local=str(cached))
    assert name == "1.jpg"
    assert (into / "1.jpg").read_bytes() == PNG
    assert not route.called


def test_a_cached_plan_the_corpus_cannot_hold_falls_back_to_the_url(tmp_path):
    """The sheet and the editor serve jpg, png and gif. A cached .webp or .pdf
    is a real file and still not a corpus file, so the URL is used as before."""
    for suffix in (".webp", ".pdf"):
        cached = tmp_path / f"floorplan{suffix}"
        cached.write_bytes(b"something else")
        into = tmp_path / f"corpus{suffix}"
        with respx.mock, httpx.Client() as client:
            route = respx.get(URL).mock(
                return_value=httpx.Response(
                    200, content=PNG, headers={"content-type": "image/png"}
                )
            )
            name = download_plan("1", URL, client, "agent", into, local=str(cached))
        assert name == "1.png"
        assert route.call_count == 1


def test_a_plan_with_a_path_and_no_url_still_enters_the_corpus(tmp_path):
    """A Listing whose media URL has died is exactly what the cache is for."""
    cached = tmp_path / "floorplan.png"
    cached.write_bytes(PNG)
    into = tmp_path / "corpus"
    with httpx.Client() as client:
        name = download_plan("1", None, client, "agent", into, local=str(cached))
    assert name == "1.png"
    assert (into / "1.png").read_bytes() == PNG


def test_a_plan_with_neither_a_url_nor_a_usable_file_is_skipped(tmp_path):
    with httpx.Client() as client:
        assert download_plan("1", None, client, "agent", tmp_path / "corpus") is None
        assert (
            download_plan("1", None, client, "agent", tmp_path / "corpus", local="gone.png")
            is None
        )


def test_plans_in_carries_the_cached_path_beside_the_url(tmp_path):
    """Including the rows a URL alone would miss."""
    from flat_scout.db import Database
    from flat_scout.models import ListingData

    from flat_scout.annotate import plans_in

    db = Database(tmp_path / "flats.db")
    with_both = db.upsert(
        ListingData(
            portal="rightmove", portal_id="1", url="u1", address="12 Elm Road",
            floorplan_url=URL,
        ),
        source="manual",
    )
    db.set_image_paths(with_both, {"floorplan_path": "data/images/rightmove_1/floorplan.png"})
    path_only = db.upsert(
        ListingData(portal="rightmove", portal_id="2", url="u2"), source="manual"
    )
    db.set_image_paths(path_only, {"floorplan_path": "data/images/rightmove_2/floorplan.png"})
    db.upsert(ListingData(portal="rightmove", portal_id="3", url="u3"), source="manual")
    db.conn.close()

    assert plans_in(tmp_path / "flats.db") == [
        ("1", "12 Elm Road", URL, "data/images/rightmove_1/floorplan.png"),
        ("2", "", None, "data/images/rightmove_2/floorplan.png"),
    ]
