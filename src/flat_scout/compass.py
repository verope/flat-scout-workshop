"""Measuring a floorplan's north indicator, deterministically.

The model is asked where the compass is. Everything after that is arithmetic,
which is the division that has worked everywhere else in this app: `epc_band_source`
is derived from the image we sent rather than asked for, `aspect_from_bearings`
does the rotation in code, and the floor area is converted in `render_reading`.
Perception is the model's job. Measurement is not.

There is a reason to push it this far here rather than trusting a second reading.
A north indicator is drawn in one of three ways, and each carries the direction
in a different place:

1. **A plain arrow.** North is where the head points. The shaft gives an axis and
   the head says which end of it, so both come out of the ink itself.
2. **A circle with a radial line and an N beside it.** North is the direction
   from the circle to the letter. The letter is usually rotated so that its own
   upright points north as well, which gives two independent measurements of the
   same angle - and where they agree, that agreement is evidence no single
   reading of any kind can produce.
3. **A rose with letters at its points.** Same as (2), except the letters are
   often at the diagonals, which is the case that defeated the model: on a
   434x378 plan it read a rose whose N sits at the lower left as "north is up",
   twice, so agreeing with itself proved nothing.

**The upright N is the interesting case, and the reason the two estimators are
kept separate.** Where the letter is drawn upright, its rotation says nothing
about north - every plan with an upright N would measure zero - so a pipeline
that averaged the two would quietly answer "north is up" on exactly the drawings
that trapped the model. `glyph_rotation` therefore abstains rather than returning
zero, and the position estimator carries the answer alone.

Nothing here calls a model or touches the network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

# Ink is dark on white. Floorplans are line drawings, so anything but a very
# generous threshold loses thin strokes to antialiasing and JPEG ringing.
INK_BELOW = 160

# A letter is small. The indicator body - arrow, circle, rose - is the big thing
# in the crop, and these bound what may be considered a letter, as a fraction of
# the crop's larger side.
LETTER_MIN_SIDE = 0.04
LETTER_MAX_SIDE = 0.45

# Every 5 degrees. Finer buys nothing: the answer is rounded to one of eight
# compass points by `aspect_from_bearings`, which is a 45 degree bucket.
ROTATION_STEP = 5

# How far from upright a letter may sit before its rotation is taken as
# deliberate. Wider than it looks because it has to swallow the measurement's own
# error: the rotation search steps in 5 degrees and carries about 5 more of
# systematic bias, so a letter drawn 16 degrees off vertical measures 25. Erring
# wide is the safe direction - calling a turned letter upright costs only the
# confirmation, while calling an upright letter turned would invent a bearing
# out of a draughtsman's wobble.
UPRIGHT_WITHIN = 25

# Below this, the best-matching rotation is not a letter N at all. Drawn letters
# score 0.89 to 0.90 against the template and a whole arrow scores 0.72, so this
# sits in the gap rather than at either end of it. It is a real cut: raise it and
# genuine letters on scanned, JPEG-blurred plans start to fall through; lower it
# and an arrowhead is read as an N, which would put north 90 degrees out.
MATCH_FLOOR = 0.80

# Two estimators this far apart are not measuring the same indicator.
AGREEMENT_WITHIN = 30


@dataclass(frozen=True)
class Bearing:
    """Where north points, in degrees clockwise from the top of the image."""

    degrees: float
    # How it was arrived at, which is what a human needs to audit a wrong one.
    method: Literal["letter-and-position", "position", "letter", "arrow"]
    # True when the N is drawn upright, so its rotation carried no information
    # and the position estimator answered alone. This is the case that produced
    # the wrong readings the deterministic path exists to catch, and it is
    # reported rather than hidden.
    upright_letter: bool = False
    # Set only where two independent estimators ran; the angle between them.
    disagreement: float | None = None


def _ink(image: Image.Image) -> np.ndarray:
    """The crop as a boolean array, True where there is ink."""
    return np.asarray(image.convert("L"), dtype=np.uint8) < INK_BELOW


def _bearing_between(centre: tuple[float, float], point: tuple[float, float]) -> float:
    """Clockwise from the top of the image, which is how every bearing here reads.

    Rows increase downwards, so the vertical term is negated: a point directly
    above the centre must read 0 and not 180.
    """
    row_from, column_from = centre
    row_to, column_to = point
    return math.degrees(math.atan2(column_to - column_from, row_from - row_to)) % 360


def _angle_between(first: float, second: float) -> float:
    """The smaller angle between two bearings, 0 to 180."""
    return abs((first - second + 180) % 360 - 180)


def _mean_bearing(first: float, second: float) -> float:
    """The midpoint of two bearings, taken around the circle rather than along it.

    Averaging the numbers is wrong wherever they straddle the wrap, and wrong
    catastrophically rather than slightly: a position of 359.9 and an axis of 25
    average to 192, which points the opposite way to both of them. Caught by a
    drawn fixture whose north was straight up.
    """
    east = math.sin(math.radians(first)) + math.sin(math.radians(second))
    north = math.cos(math.radians(first)) + math.cos(math.radians(second))
    return math.degrees(math.atan2(east, north)) % 360


def _letter_n(size: int) -> np.ndarray:
    """An upright letter N, drawn rather than typeset.

    Three strokes, drawn with ImageDraw, so the template is identical on every
    machine. A font would be neither: another machine and the development laptop
    have different ones installed, and a template that changes with the host is
    a measurement that changes with the host.
    """
    image = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(image)
    width = max(1, size // 8)
    inset = max(1, size // 6)
    left, right = inset, size - inset
    top, bottom = inset, size - inset
    draw.line([(left, bottom), (left, top)], fill=0, width=width)
    draw.line([(left, top), (right, bottom)], fill=0, width=width)
    draw.line([(right, bottom), (right, top)], fill=0, width=width)
    return np.asarray(image, dtype=np.uint8) < INK_BELOW


def _correlation(candidate: np.ndarray, template: np.ndarray) -> float:
    """Agreement between two glyph masks of the same shape, 0 to 1.

    Both are blurred first, and the score is the cosine between them. Comparing
    the strokes directly does not work: a line drawing is mostly background, and
    two strokes three pixels apart - the same stroke, drawn by two draughtsmen -
    overlap barely a third of their area. Blurring turns each stroke into a
    ridge, so a small displacement costs a little similarity instead of almost
    all of it.
    """
    blur = max(1.0, candidate.shape[0] / 16)
    first = ndimage.gaussian_filter(candidate.astype(float), blur).ravel()
    second = ndimage.gaussian_filter(template.astype(float), blur).ravel()
    norm = float(np.linalg.norm(first) * np.linalg.norm(second))
    if not norm:
        return 0.0
    return float(first @ second / norm)


def _normalised(mask: np.ndarray, size: int) -> np.ndarray:
    """A component scaled into a fixed square, keeping its proportions.

    Stretching the bounding box to fill the square instead - the obvious
    implementation - is wrong twice over. It makes a long thin arrow and a
    squarish letter into the same blob, so the arrow matched the letter template
    as well as a letter did; and because a rotated glyph has a different
    bounding box, the stretch skews it by an amount that depends on the angle,
    which put a systematic 15 degree bias on every reading.
    """
    rows = np.any(mask, axis=1)
    columns = np.any(mask, axis=0)
    if not rows.any() or not columns.any():
        return np.zeros((size, size), dtype=bool)
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(columns)[0][[0, -1]]
    height, width = bottom - top + 1, right - left + 1
    scale = (size - 2) / max(height, width)
    cropped = Image.fromarray((mask[top : bottom + 1, left : right + 1] * 255).astype(np.uint8))
    shrunk = cropped.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.LANCZOS)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(shrunk, ((size - shrunk.width) // 2, (size - shrunk.height) // 2))
    return np.asarray(canvas) > 127


def glyph_rotation(mask: np.ndarray, size: int = 48) -> tuple[float, float]:
    """The AXIS a glyph is turned along, 0 to 179, and how well it matches an N.

    Not a direction, and this is a property of the letter rather than a
    limitation of the method: N has two-fold rotational symmetry. Turn one
    upside down and it is an N again, so no amount of looking at the glyph can
    distinguish north from south. An N lying on its side is on the 90 degree
    axis, and whether north is left or right is a question only its position
    beside the indicator can answer.

    Worth stating plainly because the obvious pipeline - read the letter, report
    where it points - is unfixable, not merely inaccurate. The letter confirms
    an axis; the position chooses the end of it.
    """
    best_angle, best_score = 0.0, 0.0
    template = _letter_n(size)
    glyph = _normalised(mask, size)
    picture = Image.fromarray((glyph * 255).astype(np.uint8))
    for angle in range(0, 180, ROTATION_STEP):
        # Rotating the glyph back by `angle` uprights a glyph whose own upright
        # points at `angle` clockwise. PIL turns anticlockwise for a positive
        # argument, which is that same direction once the row axis is flipped.
        turned = np.asarray(picture.rotate(angle, resample=Image.BILINEAR, fillcolor=0)) > 127
        score = _correlation(_normalised(turned, size), template)
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle, best_score


def _components(ink: np.ndarray) -> list[tuple[np.ndarray, tuple[float, float], float]]:
    """Each blob of ink as (mask, centroid, longest side in pixels)."""
    labels, count = ndimage.label(ink)
    found = []
    for index in range(1, count + 1):
        mask = labels == index
        rows, columns = np.where(mask)
        if not len(rows):
            continue
        side = max(rows.max() - rows.min(), columns.max() - columns.min()) + 1
        found.append((mask, (float(rows.mean()), float(columns.mean())), float(side)))
    return found


def _arrow_bearing(mask: np.ndarray, centroid: tuple[float, float]) -> float | None:
    """Which way an arrow points, from its own ink.

    The principal axis gives the shaft's line but not its sense - a line looks
    the same from both ends. The head decides it: it is the wide part, so the
    half of the ink lying towards it is heavier, and the centroid sits nearer
    that end than the midpoint of the axis does.
    """
    rows, columns = np.where(mask)
    if len(rows) < 8:
        return None
    points = np.stack([rows - rows.mean(), columns - columns.mean()])
    # The eigenvector of the largest eigenvalue is the direction of most spread.
    _, vectors = np.linalg.eigh(np.cov(points))
    axis = vectors[:, -1]
    projections = points.T @ axis
    # The end whose extreme is further from the centre of mass is the tail; the
    # head is blunt and heavy, so it pulls the mean towards itself.
    if abs(projections.max()) == abs(projections.min()):
        return None
    towards = axis if abs(projections.min()) > abs(projections.max()) else -axis
    return _bearing_between(
        centroid, (centroid[0] + float(towards[0]), centroid[1] + float(towards[1]))
    )


def north_from_crop(image: Image.Image) -> Bearing | None:
    """Where north points in this crop, or None if nothing here says.

    The crop is expected to hold a north indicator and little else - it comes
    from a bounding box the model gave us, which is the one part of this that a
    model does better than a measurement.
    """
    ink = _ink(image)
    if not ink.any():
        return None
    blobs = _components(ink)
    if not blobs:
        return None
    # Against the crop, not against the largest blob. A rose is drawn as four
    # separate points that touch only at a corner, so the biggest component is
    # one spike rather than the whole indicator, and sizing letters against it
    # rejected the letter for being nearly as large as a spike.
    span = max(image.size)

    best_letter_mask, best_letter, best_score, best_axis = None, None, 0.0, 0.0
    for mask, centroid, side in blobs:
        if not LETTER_MIN_SIDE * span <= side <= LETTER_MAX_SIDE * span:
            continue
        axis, score = glyph_rotation(mask)
        if score > best_score:
            best_letter_mask, best_letter, best_score, best_axis = mask, centroid, score, axis

    if best_letter is None or best_score < MATCH_FLOOR:
        # No letter worth trusting. An arrow carries the answer in its shape,
        # and the arrow is the largest thing here.
        body_mask, body_centroid, _ = max(blobs, key=lambda blob: blob[2])
        bearing = _arrow_bearing(body_mask, body_centroid)
        return None if bearing is None else Bearing(bearing, "arrow")

    # The indicator is everything that is not the letter, however many pieces it
    # is drawn in, and its centre is where the letter's bearing is measured from.
    body = np.logical_and(ink, np.logical_not(best_letter_mask))
    if not body.any():
        return None
    rows, columns = np.where(body)
    body_centroid = (float(rows.mean()), float(columns.mean()))

    from_position = _bearing_between(body_centroid, best_letter)
    # An axis of 0 is an upright letter, and an upright letter is how nearly
    # every rose is lettered. Its rotation is then a fact about the draughtsman
    # rather than about north, and position has to answer alone - which is
    # precisely the drawing that defeated the model, so it is reported.
    if min(best_axis, 180 - best_axis) <= UPRIGHT_WITHIN:
        return Bearing(from_position, "position", upright_letter=True)

    # The axis offers two bearings 180 apart. Position says which end.
    ends = (best_axis, (best_axis + 180) % 360)
    nearer = min(ends, key=lambda end: _angle_between(end, from_position))
    apart = _angle_between(nearer, from_position)
    if apart > AGREEMENT_WITHIN:
        # A turned letter that does not sit where it points. One of the two is
        # measuring something else - a stray glyph, or a body that is not the
        # indicator - so the weaker of them is dropped rather than averaged in.
        return Bearing(from_position, "position", disagreement=apart)
    return Bearing(
        _mean_bearing(from_position, nearer), "letter-and-position", disagreement=apart
    )


def outward_bearing(
    image: Image.Image, start: tuple[float, float], end: tuple[float, float]
) -> float | None:
    """Which way a wall faces, from its two endpoints and the drawing around it.

    A wall segment gives an axis and two possible normals; the building decides
    between them. The ink of a floorplan is overwhelmingly inside the envelope,
    so the normal pointing away from the drawing's centre of ink is the one that
    looks out of the flat.

    Here for the same reason `aspect_from_bearings` is: asked outright which way
    a wall faces, the model answers 0 - straight up the page - far more often
    than the plans warrant, exactly as it answered "north is up" on compasses it
    could not see. It locates the wall well enough; the angle is arithmetic.

    Coordinates are (row, column) on the image, matching everything else here.
    """
    rise, run = end[0] - start[0], end[1] - start[1]
    if abs(rise) < 1 and abs(run) < 1:
        return None  # a point is not a wall
    middle = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    # Both normals to the segment, in (row, column).
    normals = ((-run, rise), (run, -rise))
    ink = _ink(image)
    if not ink.any():
        return None
    rows, columns = np.where(ink)
    inside = (float(rows.mean()), float(columns.mean()))
    # The one that increases the distance from the middle of the ink.
    def outwards(normal: tuple[float, float]) -> float:
        moved = (middle[0] + normal[0], middle[1] + normal[1])
        return math.dist(moved, inside)

    # Compared directly rather than through `max`, which returns the first item
    # on a tie and so made the abstention below unreachable. A wall running
    # through the middle of the ink has no outside, and picking one of its two
    # normals arbitrarily gives an aspect pointing the opposite way.
    scores = (outwards(normals[0]), outwards(normals[1]))
    if abs(scores[0] - scores[1]) < 1e-6:
        return None
    best = normals[0] if scores[0] > scores[1] else normals[1]
    return _bearing_between(middle, (middle[0] + best[0], middle[1] + best[1]))
