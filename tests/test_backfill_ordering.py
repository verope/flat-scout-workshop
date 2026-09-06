"""How a backfill interleaves one Listing's grader calls with every other's.

The warm call in `grade_listing` only pays for itself if the Listing's other
calls follow it while its cache entry is still alive. `backfill_grades` starts
every row in one `gather` and shares one FIFO call gate between them, so
whether that holds is a property of the backfill's scheduling and not of
`grade_listing` alone.
"""

import asyncio
import re

import pytest
from pydantic_ai.models.test import TestModel

from flat_scout.config import Settings
from flat_scout.db import Database
from flat_scout.models import ListingData
from flat_scout.pipeline import backfill_grades

LISTINGS = 8
CONCURRENCY = 2


def a_corpus(tmp_path, count: int = LISTINGS) -> Database:
    db = Database(tmp_path / "flats.db")
    for portal_id in range(count):
        db.upsert(
            ListingData(
                portal="rightmove", portal_id=str(portal_id), url=f"u{portal_id}",
                postcode="SW11", price_pcm=3000, sqft=600,
                description="A modern flat.",
            ),
            source="manual",
        )
    return db


def which_listing(messages) -> str:
    """The `portal_id` the prompt carries, so a call can be attributed."""
    from pydantic_ai.messages import UserPromptPart

    text = "".join(
        block
        for message in messages
        for part in message.parts
        if isinstance(part, UserPromptPart)
        for block in (part.content if isinstance(part.content, list) else [part.content])
        if isinstance(block, str)
    )
    found = re.search(r"portal_id: (\d+)", text)
    assert found is not None, "the prompt no longer names the Listing"
    return found.group(1)


@pytest.mark.asyncio
async def test_a_listings_calls_are_not_spread_across_the_whole_corpus(tmp_path):
    """The warm call's cache entry has a five-minute life.

    A Listing whose six siblings run only after every other Listing's warm call
    has gone through is a Listing whose entry expires before it is read - the
    siblings then miss and write concurrently, which is the arrangement the
    warm call exists to prevent. The span asserted here is in calls rather than
    seconds because the test cannot run a real clock, but on a corpus of
    several hundred Listings the two are the same statement.
    """
    order: list[str] = []

    class Ordered(TestModel):
        async def request(self, messages, *args, **kwargs):
            order.append(which_listing(messages))
            await asyncio.sleep(0.01)
            return await super().request(messages, *args, **kwargs)

    await backfill_grades(
        a_corpus(tmp_path),
        Settings(),
        model=Ordered(custom_output_args={"grade": 7.0, "evidence": "fine"}),
        concurrency=CONCURRENCY,
    )
    spans = {}
    for key in sorted(set(order)):
        seats = [i for i, seen in enumerate(order) if seen == key]
        spans[key] = seats[-1] - seats[0]
    # Six calls per Listing, at most CONCURRENCY Listings genuinely in flight,
    # so a Listing's calls should occupy a window of roughly 6 * CONCURRENCY.
    # Doubled, to leave the event loop room without admitting a corpus-wide gap.
    assert max(spans.values()) <= 6 * CONCURRENCY * 2, spans
