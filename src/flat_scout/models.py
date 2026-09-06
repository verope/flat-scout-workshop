from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field


@dataclass
class ListingData:
    """Everything we know about a Listing from the Portal."""

    portal: str
    portal_id: str
    url: str
    address: str | None = None
    postcode: str | None = None
    price_pcm: int | None = None
    beds: int | None = None
    sqft: int | None = None
    floor: str | None = None
    furnished: str | None = None
    pet_notes: str | None = None
    description: str | None = None
    # The photograph the Portal leads with. One shot tells the couple more at a
    # glance than the bullet points do, so the Listing carries it.
    image_url: str | None = None
    # Rightmove's caption for its EPC graph. It names the graph rather than the
    # band ("EE Rating", "EPC 1"), so it is stored verbatim and never decoded.
    epc_caption: str | None = None
    epc_image_url: str | None = None
    # The floorplan. Whether the reception is a room or a corridor with a sofa
    # in it is legible here and nowhere else — marketing prose conceals it.
    floorplan_url: str | None = None
    # Where `images.cache_images` put each picture, relative to the repository
    # root, or None if it was never cached. The URL stays beside it; the path
    # is what the readers prefer, so nothing depends on the Portal being up.
    image_path: str | None = None
    epc_image_path: str | None = None
    floorplan_path: str | None = None
    council_tax_band: str | None = None
    nearest_station: str | None = None
    nearest_station_miles: float | None = None
    # The Portal's pin, kept only when it is an accurate one. An approximate pin
    # leaves both coordinates None rather than placing the flat somewhere it
    # is not.
    latitude: float | None = None
    longitude: float | None = None
    listed_on: str | None = None


class ModelVerdict(BaseModel):
    """What the holistic evaluator is asked to return. Not a record.

    This is a pydantic-ai `output_type`, which means its fields are the
    question the model is asked. It is deliberately frozen in the shape the
    model has been answering during the calibration day: a field added here
    changes that question, and the Decisions already collected would then be
    answers to something else.

    Deleted with the holistic path once the weighted Criteria are switched on.
    """

    verdict: Literal["hopeful", "borderline", "reject"]
    score: float = Field(ge=0, le=10)
    reasons: list[str] = Field(max_length=3)
    red_flags: list[str] = Field(default_factory=list)
    agent_questions: list[str] = Field(default_factory=list)


class Evaluation(BaseModel):
    """The evaluator's Verdict on one Listing, as stored.

    Both paths produce this, and no model fills it in: the holistic path
    converts a `ModelVerdict` into one, and the weighted path builds one in
    `adjudicate`. Keeping a single record type is deliberate - two that
    differed by a field or two would only raise the question of which to use.
    """

    verdict: Literal["hopeful", "borderline", "reject"]
    # None where no Criterion could be determined at all. Absent is not zero:
    # a Listing nobody could read is not a bad Listing.
    score: float | None = Field(default=None, ge=0, le=10)
    # The share of the total Criterion weight that evidence actually answered.
    # 0.0 on the holistic path, which has no Criteria to count.
    coverage: float = Field(default=0.0, ge=0, le=1)
    # The 90% interval and the gate probability the result carries. None on the
    # point path - only the Bayesian adjudicator computes them.
    score_low: float | None = Field(default=None, ge=0, le=10)
    score_high: float | None = Field(default=None, ge=0, le=10)
    p_hopeful: float | None = Field(default=None, ge=0, le=1)
    # The immutable cache key for the Fit that produced the result. Without it,
    # a replay would silently audit a later `flat-scout fit` instead.
    posterior_fit_hash: str | None = None
    reasons: list[str] = Field(max_length=3)
    red_flags: list[str] = Field(default_factory=list)
    agent_questions: list[str] = Field(default_factory=list)


EpcBand = Literal["A", "B", "C", "D", "E", "F", "G"]


class ImageReading(BaseModel):
    """What a vision model could see in a Listing's EPC graph and floorplan.

    Every field may be unknown, and unknown is the answer whenever the image is
    absent, unreadable or ambiguous. See `vision.py` for why that matters.
    """

    epc_band: EpcBand | None = None
    # Where the band came from. Set from the image we actually sent, never from
    # the model's own claim: a band printed as a letter on the official
    # certificate is firmer evidence than one judged off a bar chart.
    epc_band_source: Literal["certificate", "graph"] | None = None
    # Printed on page 1 of the certificate and nowhere else. Never inferred
    # from a graph, from the floorplan, or from the listing prose.
    epc_floor_area_sqm: float | None = None
    epc_property_type: str | None = None
    layout_verdict: Literal["good", "adequate", "poor"] | None = None
    layout_notes: list[str] = Field(default_factory=list, max_length=3)
    reception_is_separate: bool | None = None
    desk_space: bool | None = None
    # A window in every bedroom was a hard requirement under the retired
    # `bedroom-window` Criterion, and a floorplan is the only place it is
    # legible - adverts never mention it either way. False is reserved for a
    # plan that positively shows an internal bedroom; a plan that merely does
    # not draw its windows is None, because rejecting a flat over an undrawn
    # window would be the costliest kind of wrong answer.
    bedroom_has_window: bool | None = None
    # Which way north points and which way the main windows face, both as
    # bearings on the IMAGE rather than on the ground: 0 is the top of the
    # picture, 90 the right edge. They are stored raw because they are the
    # working behind `window_aspects` - a wrong one is diagnosable from them
    # and from nothing else, since the two failures look identical downstream.
    north_clock: int | None = Field(default=None, ge=0, lt=360)
    # One bearing per glazed wall, because 13 of the 24 annotated plans face
    # more than one way. Asking for a single direction made the model choose
    # between two true answers, and it chose differently on every run: three
    # reads of one plan gave 45, 315 and 135 degrees, which looked like noise
    # and was mostly the question being unanswerable as put.
    windows_clock: list[int] = Field(default_factory=list, max_length=4)
    # What the couple actually asked for, and the only one of the three the
    # evaluator is shown. Always computed from the two bearings above and never
    # taken from the model's own answer, in the manner of `epc_band_source`:
    # asked directly, the model reads the compass correctly and then rotates
    # the plan wrongly, which is a confident wrong answer of exactly the kind
    # the couple cannot check. See `aspect_from_bearings`.
    window_aspects: list[Literal["N", "NE", "E", "SE", "S", "SW", "W", "NW"]] = Field(
        default_factory=list, max_length=4
    )
    # What the compass looks like, in the model's words. Purely a scaffold: on a
    # plan whose indicator is a 40-pixel circle with a rotated N, being made to
    # describe it first is what stopped the model answering "north is up" - it
    # did so three times out of three, on a plan where north points left.
    compass_evidence: str | None = None
    # What the plan's own title block says about which floor this is - "14th
    # Floor", "First Floor", "Ground Floor" - verbatim and never interpreted.
    # It is the only structured source there is: Rightmove sends `entranceFloor`
    # as null on every Listing seen, so the ground-floor hard filter has never
    # once fired, and the floor otherwise exists only in sales prose.
    floor_text: str | None = None
    # How north was arrived at: "letter-and-position", "position", "arrow",
    # "model", each optionally "(upright N)". Stored because the aspect is a
    # fact the couple cannot check without opening the floorplan themselves,
    # and this is the only thing that says how much to trust one. An upright N
    # in particular means the letter's own rotation said nothing and its
    # position answered alone - the case that produced the wrong readings.
    north_source: str | None = None
