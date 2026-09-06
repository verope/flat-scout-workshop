"""The Criteria whose answer needs more than one field.

A map or a bands block reads one field. These read several, or read one and
then prefer another - the floor area prefers the EPC certificate's measured
area to the advert's square footage.

The order inside each grader is the order of evidence in `criteria.md`: what
was read off this flat's own documents first, and unknown rather than a guess
when nothing answers.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from flat_scout.adjudicate import Grade
from flat_scout.criteria import Criterion
from flat_scout.grading import unknown_grade
from flat_scout.models import ImageReading, ListingData

SQFT_PER_SQM = 10.7639

GRADERS: dict[
    str, Callable[[Criterion, ListingData, ImageReading | None, Path], Grade]
] = {}


def register(name: str):
    def keep(function):
        GRADERS[name] = function
        return function

    return keep


# How far two measurements of one flat may sit apart and still be the same
# flat. criteria.md puts agreement at "within a few per cent"; the failure it
# describes is a certificate belonging to a different unit in the block, which
# is tens of per cent out - its own example, 529 sq ft advertised against a
# 74 m² certificate, is 34% apart. 10% is above anything rounding or a gross
# figure measured against a net one can produce, and far below a wrong unit.
SIZE_AGREEMENT = 0.10


def _measurements(
    listing: ListingData, reading: ImageReading | None
) -> tuple[float | None, float | None]:
    """The listing's own floor area and the certificate's, both in m².

    The advert's figure is converted; the certificate already publishes metres,
    so it is read as it stands. The Criterion's own bands are metric, and
    converting once here keeps the comparison and the grade in one unit.
    """
    stated = float(listing.sqft) / SQFT_PER_SQM if listing.sqft else None
    certified = (
        reading.epc_floor_area_sqm
        if reading is not None and reading.epc_floor_area_sqm
        else None
    )
    return stated, certified


def _disputed(stated: float | None, certified: float | None) -> bool:
    """Whether two present measurements disagree materially.

    Measured against the larger of the two, so the answer does not depend on
    which figure is called the baseline. One measurement is never a dispute.
    """
    if stated is None or certified is None:
        return False
    return abs(stated - certified) / max(stated, certified) > SIZE_AGREEMENT


def _size_dispute(
    listing: ListingData, reading: ImageReading | None
) -> tuple[str, str] | None:
    """What to flag and what to ask about a contradicted floor area, or None.

    Both, rather than either: the contradiction is worth a line in the result,
    because it is the kind of thing only a person can resolve, and criteria.md
    lists exactly this under what is worth asking an agent - the floor area,
    and which unit the EPC is for.
    """
    stated, certified = _measurements(listing, reading)
    if not _disputed(stated, certified):
        return None
    return (
        f"The advert says {listing.sqft} sq ft ({stated:.0f} m²) and the EPC "
        f"certificate measures {certified:.0f} m², so the floor area is unresolved.",
        "Could you confirm the flat's floor area, and which unit the EPC "
        "published with the listing is for? The advert and the certificate "
        "give different figures.",
    )


@register("floor_area")
def floor_area(criterion, listing, reading, directory) -> Grade:
    """Square metres, off the certificate first and the listing second.

    The certificate's own measured area often exists where the portal states no
    square footage at all, so it is worth looking before concluding the size is
    unknown. Where both exist and agree the certificate wins: it is a
    measurement taken for a document, not a number written to sell.

    Where they disagree materially neither is used. criteria.md is explicit
    that the size is then unresolved rather than one of them being right, so
    the Criterion goes unknown and the contradiction is raised as a concern,
    with the question to put to the agent alongside it.
    """
    stated, certified = _measurements(listing, reading)
    dispute = _size_dispute(listing, reading)
    if dispute is not None:
        concern, question = dispute
        return Grade(
            criterion=criterion.slug, value=None, determined=False,
            evidence=concern, graded_by="floor_area",
            concern=concern, question=question,
        )
    if certified is not None:
        area, source = certified, "the EPC certificate"
    elif stated is not None:
        area, source = stated, "the listing"
    else:
        return unknown_grade(
            criterion, "Neither the listing nor an EPC certificate states a floor area."
        )
    value = 10.0 if area >= 60 else 8.0 if area >= 50 else 0.0
    return Grade(
        criterion=criterion.slug, value=value, determined=True,
        evidence=f"{area:.0f} m², from {source}.", graded_by="floor_area",
    )


# A non-western aspect is worth real credit; west is a mild negative rather
# than a problem. The best wall of a corner flat is the one that counts.
ASPECT_GRADE = {
    "N": 10.0, "NE": 10.0, "E": 10.0, "SE": 10.0,
    "S": 10.0, "SW": 7.0, "NW": 7.0, "W": 4.0,
}


@register("best_aspect")
def best_aspect(criterion, listing, reading, directory) -> Grade:
    """The best wall a dual-aspect flat is glazed on.

    13 of the 24 annotated plans face more than one way, and a flat facing
    north and west is not a west-facing flat. Grading the worst wall would
    mark a corner flat down for having one extra window.
    """
    aspects = list(reading.window_aspects) if reading is not None else []
    if not aspects:
        return unknown_grade(criterion, "No aspect was read off the floorplan.")
    best = max(aspects, key=lambda aspect: ASPECT_GRADE[aspect])
    return Grade(
        criterion=criterion.slug, value=ASPECT_GRADE[best], determined=True,
        evidence=f"Windows face {', '.join(aspects)}, read off the plan's compass.",
        graded_by="best_aspect",
    )


# What the Portals actually write: "Furnished", "Furnished or unfurnished,
# landlord is flexible", "Unfurnished", "Part furnished", and sometimes nothing
# at all. Matched longest phrase first, because every one of them contains the
# word "furnished".
FURNISHING = (
    ("furnished or unfurnished", 8.0, "The landlord will furnish it or not, as asked."),
    ("part furnished", 5.0, "Part furnished."),
    ("unfurnished", 0.0, "Unfurnished: everything would have to be bought or moved."),
    ("fully furnished", 10.0, "Fully furnished."),
    ("furnished", 10.0, "Furnished."),
)


@register("furnishing")
def furnishing(criterion, listing, reading, directory) -> Grade:
    """Furnished or not, off the Portal's own lettings field.

    One of the few Criteria the Portals answer directly and nearly always:
    almost every Listing carries a value, which makes it among the best-covered
    evidence in the brief. It is a straight read of a stated field, so there is
    no model call and nothing to infer.

    "Furnished or unfurnished, landlord is flexible" grades 8 rather than 10.
    It is a good answer - the couple can have it furnished by asking - but it
    is an offer rather than a fact, and it is worth one question to the agent
    rather than a mark equal to a flat that is furnished today.
    """
    stated = (listing.furnished or "").strip()
    if not stated:
        return unknown_grade(criterion, "The listing does not say how it is furnished.")
    lowered = stated.lower()
    # Every phrase below contains the word "furnished", so a denial of one is a
    # substring of the thing denied: "not furnished" matches "furnished" and
    # would grade 10, reversing the advert and putting four weight behind it.
    # `fetch._unnegated_phrase` is the real defence, at the point the value is
    # read off the page. This is the second line, for a value that arrives from
    # anywhere else - a Portal widening its enum, a hand-edited row.
    # The attached forms - "non-furnished", "un furnished" - matter as much as
    # the loose ones. `\bun\b` would be wrong here: "unfurnished" is a real
    # answer and grades 0, so the prefix only denies when something separates
    # it from the word.
    if re.search(r"\b(not|no|never|isn't|aren't)\b|\b(non|un)[\s-]+", lowered):
        return unknown_grade(
            criterion, f"The listing's furnishing reads {stated!r}, which is not a plain answer."
        )
    for phrase, value, evidence in FURNISHING:
        if phrase in lowered:
            question = (
                "Is the flat furnished as standard, and can we see the inventory?"
                if value < 10.0
                else None
            )
            return Grade(
                criterion=criterion.slug, value=value, determined=True,
                evidence=evidence, graded_by="furnishing", question=question,
            )
    # A phrase no Portal has used yet. Unknown rather than a guess: this field
    # is free text, and inventing a reading of an unfamiliar one would put a
    # number in the result that nothing on the page supports.
    return unknown_grade(criterion, f"The listing calls the furnishing {stated!r}.")


# Everything a floor can be called that means "at or below the pavement".
# "Raised ground" is included deliberately: it is a few steps up, not a storey,
# and it is exactly the phrase an advert reaches for when it would rather not
# say ground floor.
AT_GROUND = ("ground", "basement")


def _is_ground(where: str) -> bool:
    return any(word in where.lower() for word in AT_GROUND)


@register("not_ground_floor")
def not_ground_floor(criterion, listing, reading, directory) -> Grade:
    """A hard-filter miss, caught wherever the floor is actually written down.

    Three sources, and the order is a claim about how much each can be
    trusted, because only the first may disqualify a flat outright.

    **The EPC certificate** prints the property type as a fact on an official
    document. A ground-floor answer there grades 0 and fires the veto.

    **The floorplan's title block** is a model reading a drawing. It grades 2 -
    low enough to hurt, above the `veto_at_or_below: 0` line so it can never
    reject on its own - and raises a concern, which surfaces it in front of a
    human. The restraint is not theoretical: one plan in the corpus was once
    read as "Ground Floor" when the drawing contains neither word, confirmed by
    OCR over the whole image. A flat removed on an invented fact is removed
    invisibly, and nobody can appeal what they never see. `vision._agreed_floor`
    now requires two independent reads to agree, which is what makes this source
    usable at all - that reading no longer survives - but "much better" is not
    "certain".

    **The Portal's own `floor` field** is the advert's word, and the pre-filter
    already rejects on it before a Listing ever reaches here. It is read last
    so that a flat which somehow arrives with one is still graded.

    The point of the widening: the floorplan names a floor far more often than
    the certificate names a property type, and ground-floor flats had been
    passing the pre-filter - some of them surfaced to the couple - because the
    plan knew and nothing asked it.
    """
    stated = reading.epc_property_type if reading is not None else None
    if stated:
        return Grade(
            criterion=criterion.slug,
            value=0.0 if _is_ground(stated) else 10.0,
            determined=True,
            evidence=f"The EPC certificate calls this a {stated}.",
            graded_by="not_ground_floor",
        )

    plan = reading.floor_text if reading is not None else None
    if plan:
        if not _is_ground(plan):
            return Grade(
                criterion=criterion.slug, value=10.0, determined=True,
                evidence=f"The floorplan's title block says {plan}.",
                graded_by="not_ground_floor",
            )
        return Grade(
            criterion=criterion.slug,
            # Never 0 from a drawing. See the docstring.
            value=2.0,
            determined=True,
            evidence=f"The floorplan's title block says {plan}.",
            concern=(
                f"The floorplan says {plan}, and the advert did not. Ground floor "
                "is meant to be filtered out, so check the plan before viewing."
            ),
            graded_by="not_ground_floor",
        )

    # Only with a reading in hand, and that guard is about provenance rather
    # than about needing the reading. `set_floor_from_plan` merges the plan's
    # floor into `listings.floor` when the Portal supplied none, so this field
    # holds either the advert's word or a drawing's, and nothing on it says
    # which. Reaching this line means the reading was consulted and named no
    # floor - so the plan cannot be where `floor` came from, and it is the
    # advert's. With `image_signals` off the reading is withheld, `floor` may
    # be a vision reading in disguise, and using it would let a flag that
    # promises to hide vision data grade a flat on exactly that.
    #
    # The cost is a Listing whose images were never read at all: its floor is
    # genuinely the Portal's and is skipped anyway, because from here the two
    # cases are indistinguishable. Rare, and one weight-2 Criterion going
    # unknown is the cheaper mistake.
    if reading is not None and listing.floor:
        where = str(listing.floor)
        if not _is_ground(where):
            return Grade(
                criterion=criterion.slug, value=10.0, determined=True,
                evidence=f"The listing says {where}.", graded_by="not_ground_floor",
            )
        return Grade(
            criterion=criterion.slug, value=2.0, determined=True,
            evidence=f"The listing says {where}.",
            concern=f"The listing says {where}, which the ground-floor filter should have caught.",
            graded_by="not_ground_floor",
        )

    return unknown_grade(
        criterion, "Neither the certificate, the floorplan nor the listing names a floor."
    )
