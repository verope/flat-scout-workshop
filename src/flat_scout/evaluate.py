"""Pre-filters and the LLM evaluation.

The hard filters live in config.toml and are executed here. criteria.md holds
only judgement material, so it never needs to restate them: a Listing that
fails a hard filter never reaches the model.
"""

from __future__ import annotations

from dataclasses import asdict

from pydantic_ai import Agent

from flat_scout.backoff import with_backoff
from flat_scout.config import Filters, Settings
from flat_scout.models import Evaluation, ImageReading, ListingData, ModelVerdict
from flat_scout.observe import span
from flat_scout.vision import render_reading

SYSTEM_PROMPT = """You are triaging London rental listings for a couple.

Judge the listing against their brief, which follows. Be decisive and be honest:
a listing that merely fails to disqualify itself is 'borderline', not 'hopeful'.
'hopeful' means you would spend a Saturday viewing it.

Give at most three reasons, each a short phrase, not a sentence. Put anything that
would be a problem into red_flags. Never invent facts the listing does not state.

agent_questions is sent to a letting agent, under the couple's names, by a human
who trusts you. Two rules follow from that, and they matter more than covering
every subject:

- Ask only what the listing has genuinely left unanswered. Read the description
  and the floorplan reading before you write a question: the extracted fields
  below are best-effort and are often empty for facts the advert states in
  plain words. A missing field means nobody extracted it, NOT that the listing
  is silent. A question whose answer is already in the advert wastes the one
  approach the couple get to make.
- Never ask about anything the agent cannot know. They have never heard of this
  system, its filters, its scores or its field names. Doubts about how the
  listing was handled belong in red_flags, described as what you actually saw.
"""


def prefilter(listing: ListingData, filters: Filters) -> str | None:
    """A rejection reason, or None if the Listing survives.

    An unknown value never rejects. A missing price among the fields we already
    have should send the Listing on to the download, not bin it.
    """
    if listing.price_pcm is not None and listing.price_pcm > filters.max_price_pcm:
        return f"price £{listing.price_pcm} over cap £{filters.max_price_pcm}"
    if listing.postcode and filters.postcodes and listing.postcode not in filters.postcodes:
        return f"postcode {listing.postcode} outside shortlist"
    if listing.beds is not None and listing.beds not in filters.allowed_beds:
        return f"{listing.beds} beds outside {filters.allowed_beds}"
    # str() rather than .lower() directly: a Portal can send the floor as an
    # object, and this is a hard filter. It must reach a verdict on an odd
    # shape rather than raise and abort the Listing.
    if filters.exclude_ground_floor and listing.floor and "ground" in str(listing.floor).lower():
        return "ground floor"
    return None


# Media URLs and the local paths beside them are for the eye, not the text
# evaluator: it cannot open either, and both are long. They stay on the row,
# where a reader and any image-reading step can reach them.
MEDIA_FIELDS = frozenset(
    {
        "image_url",
        "epc_image_url",
        "floorplan_url",
        "image_path",
        "epc_image_path",
        "floorplan_path",
    }
)


# The heading for what vision.py read off the images. The caveat is not
# decoration: an absent band must read as unknown, never as a bad band, and the
# model cannot tell the two apart unless it is told.
IMAGE_HEADING = """# Read from the images

A vision model read the listing's EPC graph and floorplan. Anything it could not
read with confidence is absent below, and absent means unknown - never treat a
missing line as a negative. Everything present is a direct reading of the image,
so it outranks the listing's prose where the two disagree."""


# Why the facts block needs a caveat over it. The fields are scraped from the
# Portal's payload, and several of the ones the couple ask about are simply not
# in it: Rightmove sends `entranceFloor` as null on every Listing seen so far,
# and states no pet policy at all, while the description says "set on the
# thirteenth floor" and advertises a pet spa. An omitted line therefore means
# "not extracted", and the first day of real results showed what happens without
# this paragraph - the evaluator asked agents which floor a flat was on while
# quoting the floor from the advert in the same sentence.
LISTING_HEADING = """# The listing

These fields are a best-effort extraction from the Portal, and an absent field
means only that nothing was extracted for it. It is NOT the listing declining to
say. The description below is the advert's own words and frequently answers what
a field leaves blank - read it before concluding that anything is unstated."""


def facts_block(listing: ListingData) -> str:
    """The Listing's fields, one per line, with the unknown ones omitted.

    Omitted rather than rendered as None, so the model never reasons about a
    null. Media URLs stay out: the text evaluator cannot open them, and they
    are long.
    """
    return "\n".join(
        f"{key}: {value}"
        for key, value in asdict(listing).items()
        if value is not None and key not in MEDIA_FIELDS
    )


def build_prompt(
    listing: ListingData, criteria: str, reading: ImageReading | None = None
) -> str:
    """Unknown fields are omitted, so the model never reasons about a None."""
    prompt = f"# Their brief\n\n{criteria}\n\n{LISTING_HEADING}\n\n{facts_block(listing)}\n"
    if reading is not None:
        prompt += f"\n{IMAGE_HEADING}\n\n{render_reading(reading)}\n"
    return prompt


async def evaluate_listing(
    listing: ListingData,
    criteria: str,
    settings: Settings,
    model=None,
    reading: ImageReading | None = None,
) -> Evaluation:
    agent = Agent(
        model or settings.evaluation.model,
        output_type=ModelVerdict,
        system_prompt=SYSTEM_PROMPT,
        name="holistic-evaluator",
    )
    with span(
        "evaluate listing",
        portal=listing.portal,
        portal_id=listing.portal_id,
        with_images=reading is not None,
    ):
        result = await with_backoff(lambda: agent.run(build_prompt(listing, criteria, reading)))
    # The holistic path counts no Criteria, so it reports no coverage. The
    # default of 0.0 would read as "nothing could be determined", which is why
    # the choice between this path and `pipeline.evaluate_weighted`, and the
    # reports downstream of it, gate on the `weighted_criteria` flag and not
    # on this number.
    return Evaluation(**result.output.model_dump())
