"""Reading the facts that exist only inside a Listing's images.

Three things the couple care about are in pictures we already hold URLs for and
nowhere else in the payload:

- **The EPC band.** `epc_caption` holds Rightmove's label for its graph — "EE
  Rating", "EPC 1" — and never a band; 30 sampled Listings established that.
  The band is drawn inside the PNG.
- **Whether the reception is a room.** Marketing prose conceals that; a
  floorplan states it.
- **Which way the windows face.** Adverts almost never say, and most floorplans
  carry a north indicator. Asked for the aspect directly a model reads the
  compass and then rotates the drawing wrongly, so it is asked instead for the
  one thing it does better than any measurement - where on the page the compass
  is - and `compass.py` measures the angle from that crop. The two estimates
  have to agree before anything is stored: they fail differently, so a
  contradiction between them is visible in a way that two readings of the same
  kind never are.

The images are downloaded here and passed as `BinaryContent` rather than handed
to the provider as an `ImageUrl`. Three reasons: the Portal's media host is
fetched with the same browser user agent the listing pages use, so it answers
us the way it answers the rest of the app; we see the media type, so what
arrives can be handled on its merits; and a dead media URL becomes an ordinary
absent image rather than a provider error that would cost the whole reading.

Roughly a fifth of the EPCs Rightmove links are not graphs at all but the full
official certificate as a PDF — 9 of 41 in a live sample on 2026-08-17. Page 1
of one states the band as a printed letter, along with the floor area and the
property type. Those are rasterised here with pypdfium2 and sent down the same
image path, so there is one uniform code path and no provider-specific document
handling; if rendering fails the old skip still applies.

This module treats a wrong band as strictly worse than no band. Two mechanisms
enforce that: the prompt demands null whenever the image is doubtful, and
`_only_what_was_seen` erases any field the model answered about an image that
was never sent to it. The second is the load-bearing one, because it does not
depend on the model complying. `epc_band_source` follows the same principle
from the other direction — we know which image we sent, so it is derived here
rather than asked for.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import httpx
import pypdfium2
from PIL import Image
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent

from flat_scout.backoff import with_backoff
from flat_scout.config import MODEL_SETTINGS, Settings
from flat_scout.compass import (
    Bearing,
    outward_bearing,
    _angle_between,
    _mean_bearing,
    north_from_crop,
)
from flat_scout.images import local_image
from flat_scout.models import ImageReading, ListingData
from flat_scout.observe import span

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You read images from a rental listing and report only what \
you can see in them.

The EPC graph is a bar chart of lettered bands from A to G. Report the CURRENT
band — the marker on the current-rating column, not the potential one. Return
null for epc_band if the image is not an EPC graph, if it is too small or too
blurred to read, if no marker is visible, or if you have any doubt at all about
which band the marker sits on. A wrong band is far worse than no band: a guess
would be a fabricated fact reaching the people who read it. Never infer a band
from anything but the graph.

An Environmental Impact (CO2) chart looks almost identical: the same A to G
bars, the same layout. It is titled "Environmental Impact" or "CO2" rather than
"Energy Efficiency". Return null for epc_band if you are given one. Its rating
says nothing about warmth or bills, and reporting it as the EPC band would be a
wrong fact dressed as the right one.

The floorplan tells you whether the living space is a real room or a corridor
with a sofa in it. Judge:

- reception_is_separate: is the living space a distinct room, rather than open
  to the entrance hall or doubling as the route between other rooms?
- desk_space: is there a wall or a corner where a desk would fit without
  blocking a door or a walkway? One workable desk position is enough.
- layout_verdict: 'good' if the reception is a genuine room with somewhere for a
  desk; 'adequate' if it works but is tight or awkward; 'poor' if the reception
  is really circulation space, or there is nowhere a desk would go.
- layout_notes: at most three short phrases, each something visible on the plan.
  Quoted room dimensions are worth reporting.
- bedroom_has_window: does every bedroom have a window? Answer true only if you
  can see a window drawn on an external wall of each bedroom. Answer false ONLY
  where a bedroom is positively internal — enclosed by other rooms, with no
  external wall at all. If the plan does not draw windows, or you cannot tell
  which room is the bedroom, or a studio has no separate bedroom to judge,
  answer null. A false answer is a fact about the flat, so never give one from
  an undrawn window.

Return null for every layout field if the floorplan is missing or unreadable.


Some listings supply the official EPC certificate instead of a graph, rendered
here as a page image. It states the rating as a printed letter beside the words
"Energy rating" — read that letter, and prefer it over any bar chart elsewhere
on the page. The certificate also prints two things a graph never shows:

- epc_floor_area_sqm: the number beside "Total floor area", in square metres,
  as a number alone. 51 for "51 square metres".
- epc_property_type: the text beside "Property type", verbatim — "Mid-floor
  flat", "Ground-floor flat", "Top-floor flat".

Report those two ONLY from a certificate that states them in words. They are
never on a graph and never on a floorplan; if you were not given a certificate,
or it does not state them, return null. Do not convert units, do not estimate a
floor area from a floorplan's room dimensions, and do not infer a property type
from anything but the printed line.

Report nothing you cannot see. Null is always the right answer when unsure."""

# Eight points, not sixteen. The brief asks one question of this field - is it
# western - and a coarser scale is also what makes the agreement check below
# usable: two reads that put north 15 degrees apart still land on one answer,
# where sixteen points would have called that a disagreement and stored nothing.
COMPASS_POINTS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def aspect_from_bearings(north_clock: int | None, windows_clock: int | None) -> str | None:
    """Which way the windows face on the ground, from two bearings on the image.

    Both inputs are measured clockwise from the top of the picture, so the
    windows' bearing relative to north is simply the difference. Done here
    rather than asked for: a model reading one plan's arrow correctly at 222
    degrees still answered NW, NE and SE on three consecutive runs, while this
    is exact and free.
    """
    if north_clock is None or windows_clock is None:
        return None
    return COMPASS_POINTS[round(((windows_clock - north_clock) % 360) / 45) % 8]


class _AspectRead(BaseModel):
    """What a model can say about a floorplan's compass.

    `compass_box` is the one answer here that a model is genuinely better at
    than a measurement: finding a small symbol somewhere on a large drawing.
    Given the box, `compass.north_from_crop` measures the angle exactly, and the
    bearing the model offers is kept only as a fallback for when the measurement
    declines to answer.
    """

    compass_evidence: str | None = None
    # The floor as the title block writes it, asked here because this read
    # happens twice and can therefore be checked against itself.
    floor_text: str | None = None
    # Left, top, right, bottom as percentages of the image, which survive the
    # rescaling a pixel box would not.
    compass_box: list[int] | None = Field(default=None, min_length=4, max_length=4)
    north_clock: int | None = Field(default=None, ge=0, lt=360)
    windows_clock: list[int] = Field(default_factory=list, max_length=4)
    # Each glazed wall as [x1, y1, x2, y2], its two ends. Preferred over the
    # bearings above for the same reason `compass_box` is preferred over
    # `north_clock`: locating a thing is what a model is good at, and the
    # angle it faces is arithmetic that does not need one.
    window_walls: list[list[int]] = Field(default_factory=list, max_length=4)


ASPECT_PROMPT = """This is an estate agent's floorplan. Report only what is drawn.

Find the north indicator. It is drawn in one of three ways, and telling them
apart is most of the job:

1. A plain arrow. North is where the arrowhead points, which is often not up.
2. A small circle with a single radial line, and the letter N outside it. North
   is the direction from the circle's centre towards that letter. The letter is
   usually rotated so its own upright points north as well.
3. A star or rose with several points, labelled with two or more of N, E, S, W.
   North is the direction from the centre of the rose towards the letter N -
   and on these the letters frequently sit at the diagonals rather than at the
   top and bottom, so a rose whose N is at the lower left means north points
   down and to the left, at about 225 degrees.

Say which of the three you are looking at, and where the letters sit, BEFORE
giving any bearing. **Never assume north points up the page.** On many plans it
does not, and answering 0 out of habit is the most common way to get this wrong.
If there is genuinely no indicator drawn anywhere, say so and return nulls.

Give compass_box as well: the indicator's bounding box as [left, top, right,
bottom], in pixels of this image. Be generous - a box
with a little of the drawing around the symbol is much better than one that
clips it. This box matters more than your own bearing, because the angle is
measured from it afterwards rather than read.

Then find the glazed walls. Windows are drawn as breaks in the thick outer
wall, usually as thin parallel lines spanning its width. Give window_walls: one
entry per glazed wall, as [x1, y1, x2, y2] in pixels of this image, being the
two ends of the run of glazing along that wall. Most flats have more than one -
a corner flat faces two ways and both are worth having - so do not choose
between them. Leave the list empty if no glazing is legible.

Lastly, floor_text: many plans name the floor in a title block or heading -
"14th Floor", "First Floor", "Ground Floor", "Lower Ground". Copy those words
VERBATIM if they are printed on the drawing, and return null if they are not.
Do not work the floor out from the stairs, the entrance, the room names or
anything else: a floor invented here can remove the flat from the search
entirely, and on one plan with no such words anywhere on it the answer came
back "Ground Floor".

Do NOT work out which compass direction a wall faces, and do not guess a
bearing. Give the ends of the wall and nothing else; the direction it looks out
is calculated from that afterwards, against the shape of the whole drawing."""


# What a vision model can be sent as-is. A media host answering with an HTML
# error page is skipped rather than sent; a PDF is rendered first.
READABLE_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})
PDF_MEDIA_TYPE = "application/pdf"

# Providers cap image payloads. A Portal floorplan is far below this; anything
# above it is not a floorplan and is not worth the upload.
MAX_IMAGE_BYTES = 4_000_000

# 2x the PDF's own points, which puts an A4 certificate at about 1190x1684 —
# comfortably legible for the printed rating without a wasteful upload.
PDF_RENDER_SCALE = 2


@dataclass(frozen=True)
class _Source:
    """One image on its way to the reader, and where it came from.

    `is_certificate` is what lets the prompt name the thing correctly and what
    licenses the two certificate-only fields. It is set here, from what we
    actually fetched, and never taken from the model's word for it.
    """

    content: BinaryContent
    is_certificate: bool = False


def _render_first_page(data: bytes, label: str, url: str) -> _Source | None:
    """Page 1 of a PDF as a PNG, or None if it will not render.

    Rasterising rather than using a provider's native document support keeps
    one code path for everything the reader sees, and keeps the failure local:
    a PDF we cannot open is the same "unknown" the old skip produced.
    """
    try:
        pdf = pypdfium2.PdfDocument(io.BytesIO(data))
        if len(pdf) == 0:
            raise ValueError("no pages")
        image: Image.Image = pdf[0].render(scale=PDF_RENDER_SCALE).to_pil()
        buffer = io.BytesIO()
        # `icc_profile=None` for the same reason as in `_enlarged`: an embedded
        # colour profile buys a reader nothing and a big one makes the PNG
        # unreadable to Pillow.
        image.convert("RGB").save(buffer, format="PNG", optimize=True, icc_profile=None)
        rendered = buffer.getvalue()
    except Exception as exc:  # noqa: BLE001 - an unreadable PDF is not a failure
        log.warning("could not render the %s at %s: %s", label, url, exc)
        return None
    if len(rendered) > MAX_IMAGE_BYTES:
        log.info("skipping the rendered %s at %s: %d bytes", label, url, len(rendered))
        return None
    log.info("rendered page 1 of the %s at %s to %d bytes", label, url, len(rendered))
    return _Source(BinaryContent(data=rendered, media_type="image/png"), is_certificate=True)


async def _download(
    url: str | None,
    label: str,
    settings: Settings,
    client: httpx.AsyncClient,
    path: str | None = None,
) -> _Source | None:
    """The image at `path` if it was cached, else the one at `url`.

    Never raises. A Portal's media host is a third party, and every way it can
    disappoint us — a 404, a timeout, an HTML error page — means the same thing
    downstream: this image is unknown. `check` caches every picture a Listing
    links, so the ordinary case is that this reads a file and the media host is
    never asked at all; a path with nothing behind it falls back to the URL.
    """
    local = local_image(path)
    if local is not None:
        data, media_type = local
        log.info("read the %s from %s", label, path)
    else:
        if not url:
            return None
        try:
            response = await client.get(
                url,
                headers={"User-Agent": settings.fetch.user_agent},
                follow_redirects=True,
                timeout=settings.fetch.timeout_seconds,
            )
            response.raise_for_status()
            media_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
            data = response.content
        except Exception as exc:  # noqa: BLE001 - an absent image is not a failure
            log.warning("could not download the %s at %s: %s", label, url, exc)
            return None
    if not data or len(data) > MAX_IMAGE_BYTES:
        log.info("skipping the %s at %s: %d bytes", label, url, len(data))
        return None
    if media_type == PDF_MEDIA_TYPE:
        return _render_first_page(data, label, url)
    if media_type not in READABLE_MEDIA_TYPES:
        log.info("skipping the %s at %s: %s is not a readable image", label, url, media_type)
        return None
    return _Source(BinaryContent(data=data, media_type=media_type))


def _only_what_was_seen(
    reading: ImageReading, *, saw_epc: bool, saw_floorplan: bool, saw_certificate: bool
) -> ImageReading:
    """Erase any answer about an image the model was never given.

    This is the guarantee that does not rely on the model behaving. If no EPC
    graph was sent, an `epc_band` in the reply was invented, whatever the prompt
    asked for, and it must not reach the couple. The floor area and the property
    type are printed only on a certificate, so a graph can never yield either.

    `epc_band_source` is overwritten rather than validated: we know which image
    we sent, so the model's own claim about provenance carries no information
    and could only ever be wrong.
    """
    if not saw_epc and reading.epc_band is not None:
        log.warning("discarding an EPC band reported without an EPC image")
        reading = reading.model_copy(update={"epc_band": None})
    if not saw_certificate:
        stated = reading.epc_floor_area_sqm is not None or reading.epc_property_type is not None
        if stated:
            log.warning("discarding a floor area or property type reported without a certificate")
        reading = reading.model_copy(
            update={"epc_floor_area_sqm": None, "epc_property_type": None}
        )
    source = None
    if reading.epc_band is not None:
        source = "certificate" if saw_certificate else "graph"
    reading = reading.model_copy(update={"epc_band_source": source})
    if not saw_floorplan:
        invented = (
            reading.layout_verdict is not None
            or reading.layout_notes
            or reading.reception_is_separate is not None
            or reading.desk_space is not None
            or reading.bedroom_has_window is not None
            or reading.floor_text is not None
            or reading.north_clock is not None
            or reading.windows_clock is not None
        )
        if invented:
            log.warning("discarding a layout reading reported without a floorplan")
        reading = reading.model_copy(
            update={
                "layout_verdict": None,
                "layout_notes": [],
                "reception_is_separate": None,
                "desk_space": None,
                "bedroom_has_window": None,
                "floor_text": None,
                "north_clock": None,
                # A list, not None: `model_copy` does not validate, so a None
                # here reaches the database as the string "null" and comes back
                # as a type the model refuses to be built from.
                "windows_clock": [],
                "compass_evidence": None,
            }
        )
    # Never the model's own answer, whatever it put there: these are computed,
    # and the confirmation below is what licenses keeping them. `north_source`
    # goes with them - an audit field that reports how north was established
    # must not survive from a read that established nothing.
    return reading.model_copy(update={"window_aspects": [], "north_source": None})


# A north indicator is a small thing on a big drawing, and estate agents publish
# some very small drawings. One 434x378 plan carried a four-point rose about 40
# pixels across: the model could not see it, said north was up, and said it
# twice, so agreeing with itself proved nothing. Enlarged to 1400 it read the
# rose correctly - "E at upper-left, S at upper-right, N at lower-left" - and
# gave the right answer twice. Only the compass read is enlarged; the layout and
# the EPC band are legible at any size the Portal publishes.
# Two bearings this far apart are not the same wall or the same compass. It is
# a whole compass point, because the answer is bucketed into eight of them
# and a tighter cut would reject readings that mean the same thing.
AGREE_WITHIN = 45

ASPECT_MIN_PIXELS = 1200
ASPECT_TARGET_PIXELS = 1400


def _enlarged(floorplan: _Source) -> BinaryContent:
    """The floorplan, big enough to read a compass on. Never raises."""
    try:
        image = Image.open(io.BytesIO(floorplan.content.data))
        if max(image.size) >= ASPECT_MIN_PIXELS:
            return floorplan.content
        scale = ASPECT_TARGET_PIXELS / max(image.size)
        bigger = image.convert("RGB").resize(
            (round(image.width * scale), round(image.height * scale)), Image.LANCZOS
        )
        buffer = io.BytesIO()
        # `icc_profile=None` drops the source JPEG's colour profile. The
        # profile is irrelevant to measuring a compass, but Pillow would
        # otherwise copy it into the PNG as a compressed iCCP chunk, and a
        # profile over 1 MB then trips Pillow's own decompression guard on the
        # way back in - `Image.open` on those bytes raises `ValueError:
        # Decompressed data too large for PngImagePlugin.MAX_TEXT_CHUNK`. Seen
        # on a real floorplan whose JPEG carried a 1.5 MB profile, which cost
        # that listing its compass measurement and its wall bearings. The PNG
        # also comes out roughly ten times smaller without it.
        bigger.save(buffer, format="PNG", optimize=True, icc_profile=None)
        rendered = buffer.getvalue()
    except Exception as exc:  # noqa: BLE001 - the original is still readable
        log.warning("could not enlarge the floorplan: %s", exc)
        return floorplan.content
    if len(rendered) > MAX_IMAGE_BYTES:
        log.info("keeping the original floorplan: enlarged to %d bytes", len(rendered))
        return floorplan.content
    return BinaryContent(data=rendered, media_type="image/png")


async def _read_aspect(
    plan: BinaryContent, settings: Settings, model=None
) -> _AspectRead:
    """Read the compass, in a call that asks about nothing else. Never raises.

    Deliberately not folded into the main reading. Asked as one clause of the
    combined prompt - alongside the EPC band, the layout and the certificate -
    the compass reading fell apart: one plan whose focused reading put north at
    270 degrees three times running came back at 315 from the combined call, and
    over eight plans the combined read confirmed one aspect where the focused
    one confirmed five. The extra call buys the field back.

    A failure means the aspect is unconfirmed, which is the same outcome as a
    disagreement and must not cost the rest of the reading.
    """
    agent = Agent(
        model or settings.evaluation.vision_model,
        output_type=_AspectRead,
        system_prompt="You read floorplans and report only what is drawn on them.",
        name="floorplan-aspect",
        model_settings=MODEL_SETTINGS,
    )
    try:
        result = await with_backoff(lambda: agent.run([ASPECT_PROMPT, plan]))
    except Exception as exc:  # noqa: BLE001 - one unconfirmed field, not the reading
        log.warning("could not read the compass: %s", exc)
        return _AspectRead()
    return result.output


# A box the model gives is approximate, so it is opened out before cropping. Too
# much context is harmless - the measurement ignores everything that is not a
# letter or the indicator - while a clipped arrowhead is a wrong bearing.
BOX_MARGIN = 0.25


def _measure_north(plan: BinaryContent, box: list[int] | None) -> Bearing | None:
    """Measure north from the crop the model pointed at. Never raises."""
    if box is None:
        return None
    try:
        image = Image.open(io.BytesIO(plan.data))
        # Asked for pixels, and told to expect either. Models answer this kind of
        # question in whichever unit suits them and are not talked out of it: the
        # prompt asked for percentages and got [1830, 55, 1980, 200] on nearly
        # every plan, which divided by 100 put the left edge at 18.3 and the
        # right edge at 1.0. Every measurement was silently skipped for it.
        if max(box) <= 100:
            left, top, right, bottom = (value / 100 for value in box)
        else:
            left, top = box[0] / image.width, box[1] / image.height
            right, bottom = box[2] / image.width, box[3] / image.height
        left, right = sorted((max(0.0, left), min(1.0, right)))
        top, bottom = sorted((max(0.0, top), min(1.0, bottom)))
        if right - left < 0.001 or bottom - top < 0.001:
            return None
        pad_x = (right - left) * BOX_MARGIN
        pad_y = (bottom - top) * BOX_MARGIN
        crop = image.crop(
            (
                round(max(0.0, left - pad_x) * image.width),
                round(max(0.0, top - pad_y) * image.height),
                round(min(1.0, right + pad_x) * image.width),
                round(min(1.0, bottom + pad_y) * image.height),
            )
        )
        if min(crop.size) < 8:
            return None
        # The measurement wants strokes it can separate, and a compass on a plan
        # is small even after the plan itself has been enlarged.
        if max(crop.size) < ASPECT_MIN_PIXELS // 2:
            scale = (ASPECT_MIN_PIXELS // 2) / max(crop.size)
            crop = crop.resize(
                (round(crop.width * scale), round(crop.height * scale)), Image.LANCZOS
            )
        return north_from_crop(crop)
    except Exception as exc:  # noqa: BLE001 - the model's own bearing still stands
        log.warning("could not measure the compass: %s", exc)
        return None


def _wall_bearings(plan: BinaryContent, walls: list[list[int]]) -> list[int]:
    """Turn each located wall into the direction it looks out of the building."""
    if not walls:
        return []
    try:
        image = Image.open(io.BytesIO(plan.data))
    except Exception as exc:  # noqa: BLE001 - the model's own bearings still stand
        log.warning("could not open the plan to measure its walls: %s", exc)
        return []
    found: list[int] = []
    for wall in walls:
        if len(wall) != 4:
            continue
        # The model is asked for pixels and, as with the compass box, answers in
        # whatever unit it likes. Percentages are the only other thing it uses.
        scale_x, scale_y = (image.width / 100, image.height / 100) if max(wall) <= 100 else (1, 1)
        start = (wall[1] * scale_y, wall[0] * scale_x)
        end = (wall[3] * scale_y, wall[2] * scale_x)
        bearing = outward_bearing(image, start, end)
        if bearing is not None:
            found.append(round(bearing) % 360)
    return found


def _agreed_floor(first: str | None, second: str | None) -> str | None:
    """The floor, only if two reads of the same drawing printed the same words.

    Cheap here because the compass read already runs twice, and necessary
    because the alternative was a fabrication with teeth: a plan carrying the
    words "floor" and "ground" nowhere on it - confirmed by OCR over the whole
    image - was read as "Ground Floor", and a ground-floor answer is the one
    thing that removes a Listing outright.
    """
    if not first or not second:
        return None
    tidy = (" ".join(first.split()).lower(), " ".join(second.split()).lower())
    return first.strip() if tidy[0] == tidy[1] else None


def _agreed_walls(first: list[int], second: list[int]) -> list[int]:
    """The glazed walls both reads found, one bearing each.

    A set intersection rather than a single comparison, which is most of what
    listing the walls buys: two reads that name three walls between them and
    agree on two now keep those two, where a single-answer field had to throw
    the whole plan away whenever the reads named different real walls. Each wall
    in the first read survives if the second put one within a compass point of
    it, and is kept at the midpoint of the pair.
    """
    kept: list[tuple[float, int]] = []
    # One wall in the second read may confirm one wall in the first, and no
    # more. Without consuming it, two first-read walls either side of a single
    # second-read wall both match it and become two stored walls - so one
    # confirming wall turns into two stored aspects, which is the opposite
    # of what confirmation is for.
    unclaimed = list(second)
    for bearing in first:
        nearest = min(
            unclaimed,
            key=lambda other: _angle_between(float(bearing), float(other)),
            default=None,
        )
        if nearest is None or _angle_between(float(bearing), float(nearest)) > AGREE_WITHIN:
            continue
        unclaimed.remove(nearest)
        middle = round(_mean_bearing(float(bearing), float(nearest))) % 360
        if all(_angle_between(float(middle), float(other)) > AGREE_WITHIN for gap, other in kept):
            kept.append((_angle_between(float(bearing), float(nearest)), middle))
    # Tightest agreement first, so a caller taking only one wall takes the one
    # the two reads were surest about rather than whichever came back first.
    return [wall for _, wall in sorted(kept)]


def _agreed(first: int | None, second: int | None, within: float) -> float | None:
    """One bearing where two readings say nearly the same thing, else None."""
    if first is None or second is None:
        return None
    if _angle_between(float(first), float(second)) > within:
        return None
    return _mean_bearing(float(first), float(second))


async def _agreed_aspect(
    reading: ImageReading, floorplan: _Source | None, settings: Settings, model=None
) -> ImageReading:
    """Keep the aspect only where a second, independent read agrees with it.

    One read is not enough. Over eight sampled plans two reads of the same
    floorplan disagreed on three of them - once by 270 degrees - and the
    disagreements are invisible downstream: a wrong aspect looks exactly like a
    right one. Agreement is not proof either, but it is a cheap filter that the
    known failures do not survive, and a null costs the couple nothing, since
    aspect is a preference rather than a requirement.

    Skipped entirely when the first read found no compass, which is most of the
    saving: the confirming call is only made where there is something to
    confirm.
    """
    if floorplan is None:
        return reading
    # Enlarged once, then read twice: the two reads must see the same picture
    # for their agreement to mean anything.
    plan = _enlarged(floorplan)
    once = await _read_aspect(plan, settings, model)
    reading = reading.model_copy(
        update={
            "north_clock": once.north_clock,
            "windows_clock": once.windows_clock,
            "compass_evidence": once.compass_evidence,
        }
    )

    if (
        once.north_clock is None
        and once.compass_box is None
        and not once.windows_clock
        and not once.window_walls
    ):
        return reading  # no compass drawn: nothing to confirm, and no second call

    twice = await _read_aspect(plan, settings, model)
    # Measured from each read's box independently, and kept only where the two
    # measurements agree. Measuring one box was not enough: a model that sees no
    # compass still offers a box, the measurement dutifully measures whatever
    # glyph is inside it, and two plans with no indicator at all came back with
    # confident bearings. Two boxes drawn from two reads rarely land on the same
    # piece of noise, and never on the same angle of it.
    measured = _measure_north(plan, once.compass_box)
    confirm = _measure_north(plan, twice.compass_box)
    if measured is not None and (
        confirm is None or _angle_between(measured.degrees, confirm.degrees) > AGREE_WITHIN
    ):
        log.info(
            "compass not confirmed by a second box: %.0f then %s",
            measured.degrees,
            "nothing" if confirm is None else f"{confirm.degrees:.0f}",
        )
        measured = None
    read_north = _agreed(once.north_clock, twice.north_clock, AGREE_WITHIN)

    # Measured, or nothing. Scored against 24 annotated plans, a measurement
    # taken from a lettered indicator - a circle or a rose, where the letter N
    # says which way to look - was right 8 times out of 9. The model reading the
    # same plans was right 11 times out of 16, and an arrow measured off its own
    # ink was right twice out of seven, which is a coin toss dressed as an
    # instrument. So the arrow is computed and reported but never stored, and
    # there is no fall back to the model: on this corpus, falling back would buy
    # four more answers at the cost of three wrong ones, and a wrong aspect is a
    # fact the couple cannot check without opening the floorplan themselves.
    MEASURED_METHODS = ("position", "letter-and-position")
    north: float | None = None
    source: str | None = None
    if measured is not None and measured.method in MEASURED_METHODS:
        north = measured.degrees
        source = measured.method + (" (upright N)" if measured.upright_letter else "")
        if read_north is not None and _angle_between(north, read_north) > AGREE_WITHIN:
            # Kept, but said out loud. The two disagree about a third of the
            # time and the measurement is the more accurate of them, so this is
            # a note for whoever reads the logs rather than a veto.
            log.info(
                "north measured %.0f by %s, but read as %.0f", north, measured.method, read_north
            )
    elif measured is not None:
        log.info("north seen by %s only, which is not accurate enough to store",
                 measured.method)

    if north is not None:
        reading = reading.model_copy(
            update={"north_clock": round(north) % 360, "north_source": source}
        )
    # Which wall the windows are on stays a reading: finding the glazing on a
    # plan is perception, not measurement, and nothing here can do it.
    # Measured from where the model put the walls, falling back to the bearings
    # it offers only when it located nothing. Its bearings carry a pull towards
    # 0 - straight up the page - which is the same failure as "north is up" and
    # was costing an invented wall on four plans of twenty-four.
    once_walls = _wall_bearings(plan, once.window_walls) or once.windows_clock
    twice_walls = _wall_bearings(plan, twice.window_walls) or twice.windows_clock
    # The floor only where both reads printed the same words. It is the one
    # field here that can cost a flat rather than merely misdescribe it.
    # Assigned unconditionally, including None. Only setting it when the two
    # reads agreed left an unconfirmed value from the combined read in place -
    # which is exactly the "Ground Floor" that appears nowhere on one plan in
    # the corpus, surviving the guard built to catch it.
    reading = reading.model_copy(
        update={"floor_text": _agreed_floor(once.floor_text, twice.floor_text)}
    )

    walls = _agreed_walls(once_walls, twice_walls)
    if not settings.features.multi_aspect:
        # The best-confirmed wall alone. Reporting every wall states more true
        # things about a dual-aspect flat and more false ones alongside them,
        # and on the annotated corpus the trade came out against it.
        walls = walls[:1]
    reading = reading.model_copy(update={"windows_clock": walls})
    if north is None or not walls:
        return reading
    # One aspect per wall, deduplicated: two walls 30 degrees apart are the same
    # answer once the bearings are bucketed into eight compass points.
    aspects: list[str] = []
    for wall in walls:
        aspect = aspect_from_bearings(round(north), wall)
        if aspect is not None and aspect not in aspects:
            aspects.append(aspect)
    return reading.model_copy(update={"window_aspects": aspects})


async def read_images(
    listing: ListingData, settings: Settings, client: httpx.AsyncClient, model=None
) -> ImageReading | None:
    """Read the EPC band and the layout off the Listing's images.

    None means there was nothing to read — no image URLs, or none of them
    downloadable — and in that case no model is called and nothing is spent.
    A failure of the vision call itself is raised, for the caller to record and
    fall back on the text evaluation.
    """
    with span("download images", portal_id=listing.portal_id):
        epc = await _download(
            listing.epc_image_url, "EPC", settings, client, path=listing.epc_image_path
        )
        floorplan = await _download(
            listing.floorplan_url,
            "floorplan",
            settings,
            client,
            path=listing.floorplan_path,
        )
    if epc is None and floorplan is None:
        return None

    prompt: list = ["Read what you can from the images below."]
    if epc is not None:
        prompt += [
            "This is page 1 of the official EPC certificate:"
            if epc.is_certificate
            else "This is the EPC graph:",
            epc.content,
        ]
    if floorplan is not None:
        prompt += ["This is the floorplan:", floorplan.content]

    agent = Agent(
        model or settings.evaluation.vision_model,
        output_type=ImageReading,
        system_prompt=SYSTEM_PROMPT,
        name="epc-and-floorplan",
        model_settings=MODEL_SETTINGS,
    )
    with span(
        "read images",
        portal_id=listing.portal_id,
        epc=epc is not None,
        floorplan=floorplan is not None,
        certificate=epc is not None and epc.is_certificate,
    ):
        result = await with_backoff(lambda: agent.run(prompt))
    # Vision is the only per-Listing cost in the app that scales with images, so
    # the tokens it spent are worth having in the log when the bill arrives.
    usage = result.usage
    log.info(
        "read %d image(s) for %s: %s input, %s output tokens",
        (epc is not None) + (floorplan is not None),
        listing.url,
        usage.input_tokens,
        usage.output_tokens,
    )
    reading = _only_what_was_seen(
        result.output,
        saw_epc=epc is not None,
        saw_floorplan=floorplan is not None,
        saw_certificate=epc is not None and epc.is_certificate,
    )
    return await _agreed_aspect(reading, floorplan, settings, model)


SQFT_PER_SQM = 10.7639


def render_reading(reading: ImageReading) -> str:
    """The reading as prompt lines. Unknown fields are omitted, never sent as None."""
    area = None
    if reading.epc_floor_area_sqm is not None:
        # The certificate prints metres and the listing states feet; both are
        # rendered here so nothing downstream has to convert, which keeps the
        # arithmetic out of a model's hands.
        area = (
            f"{reading.epc_floor_area_sqm:g} sqm "
            f"({round(reading.epc_floor_area_sqm * SQFT_PER_SQM)} sq ft)"
        )
    lines = [
        f"{name}: {value}"
        for name, value in (
            ("epc_band", reading.epc_band),
            ("epc_band_source", reading.epc_band_source),
            ("epc_floor_area", area),
            ("epc_property_type", reading.epc_property_type),
            ("layout_verdict", reading.layout_verdict),
            ("reception_is_separate", reading.reception_is_separate),
            ("desk_space", reading.desk_space),
            ("bedroom_has_window", reading.bedroom_has_window),
            ("floor_from_the_plan", reading.floor_text),
            # The two raw bearings are deliberately not rendered: they are
            # about the picture, not about the flat, and an evaluator given
            # them would be invited to redo the arithmetic they exist to
            # keep out of a model's hands.
            ("window_aspects", ", ".join(reading.window_aspects) or None),
        )
        if value is not None
    ]
    lines += [f"layout_note: {note}" for note in reading.layout_notes]
    if not lines:
        return "Nothing in the EPC graph or the floorplan could be read."
    return "\n".join(lines)
