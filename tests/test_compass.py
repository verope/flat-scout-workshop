"""Measuring a north indicator without asking a model.

Every fixture here is drawn, not sampled, so the ground truth is fixed by
construction: the test says "draw a rose whose N sits at the lower left" and the
answer is 225 by definition of having drawn it there. Where a bearing is
asserted, it is asserted because of how the picture was made.

The one convention worth stating once. A bearing is degrees clockwise from the
top of the image, so 0 is up the page, 90 is the right edge, 180 the bottom and
270 the left. PIL turns a picture anticlockwise for a positive angle, so a glyph
drawn by rotating an upright one by `t` has its own upright pointing at
`(360 - t) % 360`. Everything below follows from that.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from flat_scout.compass import (
    MATCH_FLOOR,
    Bearing,
    _angle_between,
    _bearing_between,
    _ink,
    _mean_bearing,
    _letter_n,
    glyph_rotation,
    north_from_crop,
)

SIZE = 400
CENTRE = SIZE // 2


def _offset(bearing: float, distance: float) -> tuple[int, int]:
    """A point `distance` from the centre, at `bearing` clockwise from up."""
    radians = math.radians(bearing)
    return (
        round(CENTRE + distance * math.sin(radians)),  # column
        round(CENTRE - distance * math.cos(radians)),  # row, which grows downwards
    )


def _letter_image(bearing: float, size: int = 70) -> Image.Image:
    """An N whose own upright points at `bearing`."""
    upright = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(upright)
    stroke = max(2, size // 9)
    inset = size // 6
    left, right, top, bottom = inset, size - inset, inset, size - inset
    draw.line([(left, bottom), (left, top)], fill=0, width=stroke)
    draw.line([(left, top), (right, bottom)], fill=0, width=stroke)
    draw.line([(right, bottom), (right, top)], fill=0, width=stroke)
    return upright.rotate((360 - bearing) % 360, resample=Image.BICUBIC, fillcolor=255)


def _paste(canvas: Image.Image, letter: Image.Image, at: tuple[int, int]) -> None:
    canvas.paste(letter, (at[0] - letter.width // 2, at[1] - letter.height // 2), None)


def circle_and_letter(north: float, letter_bearing: float | None = None) -> Image.Image:
    """Style 2: a circle with a radial line, and an N beside it.

    `letter_bearing` is how the letter itself is turned, which is normally the
    same as north but is separable here because the interesting cases are the
    ones where it is not.
    """
    canvas = Image.new("L", (SIZE, SIZE), 255)
    draw = ImageDraw.Draw(canvas)
    radius = 70
    draw.ellipse(
        [CENTRE - radius, CENTRE - radius, CENTRE + radius, CENTRE + radius],
        outline=0,
        width=4,
    )
    draw.line([(CENTRE, CENTRE), _offset(north, radius)], fill=0, width=4)
    turn = north if letter_bearing is None else letter_bearing
    _paste(canvas, _letter_image(turn), _offset(north, radius + 60))
    return canvas


def rose(north: float) -> Image.Image:
    """Style 3: a four-point star with upright letters at its points."""
    canvas = Image.new("L", (SIZE, SIZE), 255)
    draw = ImageDraw.Draw(canvas)
    for point in range(4):
        bearing = north + point * 90
        draw.polygon(
            [
                (CENTRE, CENTRE),
                _offset(bearing - 12, 60),
                _offset(bearing, 95),
                _offset(bearing + 12, 60),
            ],
            fill=0,
        )
    for point, letter in enumerate("NESW"):
        if letter == "N":
            _paste(canvas, _letter_image(0), _offset(north + point * 90, 140))
    return canvas


def _step(point: tuple[int, int], bearing: float, distance: float) -> tuple[int, int]:
    """`distance` from `point`, at `bearing` clockwise from the top of the image."""
    radians = math.radians(bearing)
    return (
        round(point[0] + distance * math.sin(radians)),
        round(point[1] - distance * math.cos(radians)),
    )


def arrow(north: float) -> Image.Image:
    """Style 1: a plain arrow, head towards north, no letter at all."""
    canvas = Image.new("L", (SIZE, SIZE), 255)
    draw = ImageDraw.Draw(canvas)
    tip = _offset(north, 130)
    base = _offset(north, 60)
    draw.line([_offset(north + 180, 120), base], fill=0, width=7)
    draw.polygon(
        [tip, _step(base, north + 90, 28), _step(base, north - 90, 28)], fill=0
    )
    return canvas


# --- the conventions everything else rests on ---


@pytest.mark.parametrize(
    "point,expected",
    [((0, 10), 0), ((10, 20), 90), ((20, 10), 180), ((10, 0), 270)],
)
def test_a_bearing_is_clockwise_from_the_top_of_the_picture(point, expected):
    assert _bearing_between((10, 10), point) == pytest.approx(expected)


@pytest.mark.parametrize(
    "first,second,expected",
    [(0, 10, 10), (10, 0, 10), (350, 10, 20), (10, 350, 20), (0, 180, 180), (90, 270, 180)],
)
def test_the_angle_between_two_bearings_wraps(first, second, expected):
    assert _angle_between(first, second) == pytest.approx(expected)


def test_the_letter_template_is_drawn_not_typeset():
    """A font would differ between the laptop and another machine."""
    template = _letter_n(48)
    assert template.any()
    assert template.shape == (48, 48)
    # An N is not symmetrical about the vertical axis; a template that were
    # would match its own mirror image and make 180 degrees unrecoverable.
    assert not np.array_equal(template, np.fliplr(template))


# --- reading a letter's own rotation ---


def _axes_apart(first: float, second: float) -> float:
    """The angle between two axes, which wrap at 180 rather than at 360."""
    return min(_angle_between(first, second), _angle_between(first, second + 180))


@pytest.mark.parametrize("bearing", [0, 45, 90, 180, 225, 270, 315])
def test_a_letters_rotation_gives_its_axis_not_its_direction(bearing):
    """N is centrally symmetric, so 225 and 45 are the same drawing.

    This is a fact about the alphabet, not a shortcoming of the measurement: an
    upside-down N is an N. It is why position, not the letter, decides which end
    of the axis north lies at.
    """
    axis, score = glyph_rotation(_ink(_letter_image(bearing, size=120)))
    assert 0 <= axis < 180
    # The search steps in 5 degrees and the answer is bucketed into 45 degree
    # compass points downstream, so this is precision to spare.
    assert _axes_apart(axis, bearing) <= 15
    assert score > MATCH_FLOOR


def test_a_letter_and_the_same_letter_upside_down_are_indistinguishable():
    upright, _ = glyph_rotation(_ink(_letter_image(30, size=120)))
    inverted, _ = glyph_rotation(_ink(_letter_image(210, size=120)))
    assert _axes_apart(upright, inverted) <= 15


def test_an_arrowhead_is_not_mistaken_for_a_letter():
    """Every shape has a best-matching rotation. The score is what says it is an N."""
    _, score = glyph_rotation(_ink(arrow(0)))
    assert score < MATCH_FLOOR


# --- the three indicator styles ---


def test_a_plain_arrow_is_read_from_its_own_ink():
    reading = north_from_crop(arrow(225))
    assert reading is not None
    assert reading.method == "arrow"
    assert _angle_between(reading.degrees, 225) <= 20


@pytest.mark.parametrize("north", [0, 90, 200, 270])
def test_an_arrow_is_read_whichever_way_it_points(north):
    reading = north_from_crop(arrow(north))
    assert reading is not None
    assert _angle_between(reading.degrees, north) <= 20


def test_a_circle_and_a_turned_letter_agree_with_each_other():
    """The Koa House case: north points left, and the N is turned to match.

    Two independent measurements - where the letter sits, and which way it is
    turned - and the value of the deterministic path is that they can be
    compared at all.
    """
    reading = north_from_crop(circle_and_letter(270))
    assert reading is not None
    assert reading.method == "letter-and-position"
    assert _angle_between(reading.degrees, 270) <= 15
    assert reading.upright_letter is False
    assert reading.disagreement is not None and reading.disagreement <= 15


def test_a_rose_with_upright_letters_is_read_from_where_the_n_sits():
    """The plan that beat the model, and the reason for reporting uprightness.

    A rose whose N is at the lower left means north points down and to the left.
    The letters are drawn the right way up, so their rotation says nothing, and
    a pipeline that used it would answer "north is up" - which is exactly what
    the model did, twice, on this drawing.
    """
    reading = north_from_crop(rose(225))
    assert reading is not None
    assert reading.method == "position"
    assert reading.upright_letter is True
    assert _angle_between(reading.degrees, 225) <= 20


@pytest.mark.parametrize("north", [45, 135, 225, 315])
def test_a_rose_is_read_at_any_rotation(north):
    reading = north_from_crop(rose(north))
    assert reading is not None
    assert _angle_between(reading.degrees, north) <= 20


def test_an_upright_letter_is_reported_as_upright_even_when_north_is_up():
    """The ambiguous case must not be silently indistinguishable from a reading.

    North genuinely is up here. The pipeline gets the right answer, and still
    has to say that the letter's rotation contributed nothing to it.
    """
    reading = north_from_crop(rose(0))
    assert reading is not None
    assert reading.upright_letter is True
    assert _angle_between(reading.degrees, 0) <= 20


def test_a_letter_turned_a_little_is_still_an_upright_letter():
    """Hand-drawn plans are not precise, and a few degrees is not a bearing."""
    reading = north_from_crop(circle_and_letter(0, letter_bearing=8))
    assert reading is not None
    assert reading.upright_letter is True


def test_a_letter_upside_down_from_where_it_sits_still_agrees():
    """Because N is symmetric, 90 and 270 are the same axis - so this is no
    disagreement at all, and treating it as one would throw away good readings."""
    reading = north_from_crop(circle_and_letter(90, letter_bearing=270))
    assert reading is not None
    assert reading.method == "letter-and-position"
    assert _angle_between(reading.degrees, 90) <= 20


def test_a_letter_across_from_where_it_sits_is_reported_as_a_disagreement():
    """Two estimators, a right angle apart. Position answers, the gap is kept."""
    reading = north_from_crop(circle_and_letter(0, letter_bearing=90))
    assert reading is not None
    assert reading.method == "position"
    assert reading.disagreement is not None and reading.disagreement > 30
    assert _angle_between(reading.degrees, 0) <= 20


# --- nothing to measure ---


def test_a_blank_crop_measures_nothing():
    assert north_from_crop(Image.new("L", (SIZE, SIZE), 255)) is None


def test_a_crop_with_no_indicator_in_it_measures_nothing_or_says_arrow():
    """A stray blob is not a compass, and must not be reported as a letter."""
    canvas = Image.new("L", (SIZE, SIZE), 255)
    ImageDraw.Draw(canvas).ellipse([180, 180, 220, 220], fill=0)
    reading = north_from_crop(canvas)
    assert reading is None or reading.method == "arrow"


def test_a_bearing_carries_how_it_was_reached():
    """A wrong answer has to be auditable, which means saying where it came from."""
    reading = Bearing(225.0, "position", upright_letter=True)
    assert reading.method == "position"
    assert reading.upright_letter is True


@pytest.mark.parametrize(
    "first,second,expected",
    [(0, 90, 45), (350, 10, 0), (10, 350, 0), (359, 25, 12), (270, 0, 315)],
)
def test_two_bearings_average_around_the_circle_not_along_it(first, second, expected):
    """The wrap is not a rounding detail: 359.9 and 25 average to 192 the naive
    way, which points the opposite direction to both of its inputs."""
    assert _angle_between(_mean_bearing(first, second), expected) <= 1


def test_north_straight_up_is_not_read_as_south():
    """The bug the averaging above exists to prevent, end to end.

    A circle with its letter directly above it puts the position estimator at
    359.99 rather than 0, and it used to be averaged with the letter's axis into
    a bearing pointing due south.
    """
    reading = north_from_crop(circle_and_letter(0, letter_bearing=40))
    assert reading is not None
    assert _angle_between(reading.degrees, 0) <= 25


def test_a_wall_through_the_middle_of_the_drawing_has_no_outside():
    """Neither normal points away from the building, so neither is reported.

    The abstention was unreachable before: `max` returns the first item on a
    tie, so the check for one could never fire, and a symmetric plan got an
    arbitrary normal - which is an aspect pointing the opposite way.
    """
    from flat_scout.compass import outward_bearing

    canvas = Image.new("L", (200, 200), 255)
    ImageDraw.Draw(canvas).rectangle([40, 40, 160, 160], outline=0, width=3)
    # A wall along the vertical centre line: the ink is symmetric about it.
    assert outward_bearing(canvas, (40, 100), (160, 100)) is None
