from pathlib import Path

from flat_scout.adjudicate import Grade
from flat_scout.criteria import Criterion, load_criteria
from flat_scout.db import Database
from flat_scout.graders import GRADERS
from flat_scout.models import ListingData
from flat_scout.report import (
    criterion_splits,
    weighted_distribution,
    weights_report,
)


def a_criterion(slug: str, weight: float = 1, veto_at_or_below: float | None = None) -> Criterion:
    return Criterion(
        slug=slug, name=slug.capitalize(), description="d", weight=weight,
        grade={"model": True}, body="b", rubric_hash="h",
        veto_at_or_below=veto_at_or_below,
    )


def stored(
    db: Database,
    portal_id: str,
    status: str,
    grades: dict[str, float],
    decided_at: str | None = None,
):
    """A graded Listing at any status, decided or not."""
    listing_id = db.upsert(
        ListingData(portal="rightmove", portal_id=portal_id, url=f"u{portal_id}"),
        source="manual",
    )
    db.conn.execute(
        "UPDATE listings SET status = ?, decided_at = ?, score = 5 WHERE id = ?",
        (status, decided_at, listing_id),
    )
    db.conn.commit()
    db.set_criterion_grades(
        listing_id,
        [
            (a_criterion(slug), Grade(slug, value, True, f"{slug} evidence"))
            for slug, value in grades.items()
        ],
    )
    return listing_id


def decided(db: Database, portal_id: str, status: str, grades: dict[str, float]):
    return stored(db, portal_id, status, grades, decided_at="2026-08-17")


def test_a_criterion_that_separates_the_decisions_shows_a_positive_gap(tmp_path):
    db = Database(tmp_path / "flats.db")
    decided(db, "1", "approved", {"layout": 9, "noise": 5})
    decided(db, "2", "rejected", {"layout": 2, "noise": 5})
    splits = {s.slug: s for s in criterion_splits(db, [a_criterion("layout"), a_criterion("noise")])}
    assert splits["layout"].separation == 7
    assert splits["noise"].separation == 0


def test_a_criterion_ranking_backwards_shows_a_negative_gap(tmp_path):
    """This is the finding no report can produce today."""
    db = Database(tmp_path / "flats.db")
    decided(db, "1", "approved", {"dark-run": 2})
    decided(db, "2", "rejected", {"dark-run": 9})
    [split] = criterion_splits(db, [a_criterion("dark-run")])
    assert split.separation == -7


def test_a_criterion_with_no_decided_grades_reports_no_separation(tmp_path):
    db = Database(tmp_path / "flats.db")
    decided(db, "1", "approved", {"layout": 9})
    [split] = criterion_splits(db, [a_criterion("aspect")])
    assert split.separation is None


def test_every_approval_counts_in_the_attribution(tmp_path):
    """The side comes off the Decision, not off a single status literal.

    The attribution selected `status IN ('approved', 'rejected')` while the
    threshold arithmetic beside it read the Decision, so the two halves of
    one report could disagree about who was approved and the attribution
    could lose approvals the other half kept. Both sides now read the same
    predicate.
    """
    db = Database(tmp_path / "flats.db")
    decided(db, "1", "approved", {"layout": 9})
    decided(db, "2", "approved", {"layout": 8})
    decided(db, "3", "rejected", {"layout": 2})

    [split] = criterion_splits(db, [a_criterion("layout")])
    assert split.approved_mean == 8.5
    assert split.rejected_mean == 2
    assert split.separation == 6.5


def test_an_undecided_listing_is_no_side_of_the_attribution(tmp_path):
    """The predicate is the Decision, not "anything that is not rejected".

    An evaluated Listing nobody has decided on yet has `decided_at IS NULL`,
    and counting it as an approval would fill the approved mean with flats
    nobody said yes to.
    """
    db = Database(tmp_path / "flats.db")
    decided(db, "1", "approved", {"layout": 9})
    stored(db, "2", "evaluated", {"layout": 1})
    stored(db, "3", "evaluated", {"layout": 1})

    [split] = criterion_splits(db, [a_criterion("layout")])
    assert split.approved_mean == 9


def test_the_weights_report_names_the_backwards_criterion(tmp_path):
    db = Database(tmp_path / "flats.db")
    decided(db, "1", "approved", {"dark-run": 2, "layout": 9})
    decided(db, "2", "rejected", {"dark-run": 9, "layout": 2})
    from flat_scout.config import Settings

    text = weights_report(db, [a_criterion("dark-run"), a_criterion("layout")], Settings())
    assert "Dark-run" in text
    assert "backwards" in text.lower()


def test_the_weights_report_says_when_there_is_nothing_to_replay(tmp_path):
    from flat_scout.config import Settings

    db = Database(tmp_path / "flats.db")
    text = weights_report(db, [a_criterion("layout")], Settings())
    assert "no decision" in text.lower()


def test_the_shipped_criteria_can_be_replayed_without_a_model(tmp_path):
    from flat_scout.config import Settings

    db = Database(tmp_path / "flats.db")
    criteria = load_criteria(Path("criteria"), known_graders=set(GRADERS))
    decided(db, "1", "approved", {"pets": 9})
    assert isinstance(weights_report(db, criteria, Settings()), str)


# Every module that can reach a model does `from pydantic_ai import Agent` at
# import time, so the name each call site resolves is its own module global.
# Patching `pydantic_ai.Agent` rebinds the package attribute and reaches none
# of them - `flat_scout.grading.Agent is RecordingAgent` was False, and the
# guard below passed against a path that constructed a real Agent.
_AGENT_SITES = (
    "flat_scout.grading.Agent",
    "flat_scout.evaluate.Agent",
    "flat_scout.vision.Agent",
)


def test_the_weights_report_never_constructs_a_model_agent(tmp_path, monkeypatch):
    """The tuning loop only exists if replaying a weighting is free. This
    proves it at the boundary rather than by threading a stand-in through a
    `model` parameter `weights_report` does not have anywhere in its call
    chain (`criterion_splits` is SQL, `load_criteria` is file reads).

    Two boundaries, because either alone has a hole. The four `Agent` names
    are what the call sites actually resolve, and they catch a construction
    that never runs. `Model.request` catches a call made through any path
    this test has not thought of, including one added later.

    Recording rather than raising: a stand-in that raised would be caught by
    any broad `except Exception` between here and the call site and pass for
    the wrong reason - the same defect class every other fix round on this
    branch has been.
    """
    from pydantic_ai.models import Model

    from flat_scout.config import Settings

    constructed: list[tuple] = []
    requested: list[tuple] = []

    class RecordingAgent:
        def __init__(self, *args, **kwargs):
            constructed.append((args, kwargs))

    for site in _AGENT_SITES:
        monkeypatch.setattr(site, RecordingAgent)
    monkeypatch.setattr(
        Model, "request", lambda self, *a, **kw: requested.append((a, kw))
    )

    db = Database(tmp_path / "flats.db")
    criteria = load_criteria(Path("criteria"), known_graders=set(GRADERS))
    decided(db, "1", "approved", {"pets": 9})
    decided(db, "2", "rejected", {"pets": 2})
    weights_report(db, criteria, Settings())
    assert constructed == []
    assert requested == []


def test_the_agent_sites_this_guard_patches_are_the_ones_that_are_resolved(monkeypatch):
    """The guard above is only worth anything if its patch lands.

    `monkeypatch.setattr` on a dotted path raises when the attribute is
    absent, so this fails loudly if a module is renamed or stops importing
    `Agent` - and it asserts identity, which is the check the earlier
    `pydantic_ai.Agent` patch would have failed.
    """
    import importlib

    sentinel = object()
    for site in _AGENT_SITES:
        monkeypatch.setattr(site, sentinel)
    for site in _AGENT_SITES:
        module, attribute = site.rsplit(".", 1)
        assert getattr(importlib.import_module(module), attribute) is sentinel


def test_the_split_table_leads_with_a_strongly_backward_criterion(tmp_path):
    """A Criterion ranking strongly backwards is not doing nothing - it is
    doing real work in the wrong direction, and it is the single most
    actionable row in the report. Sorting by raw separation would bury a -9
    below a +1; sorting by magnitude surfaces it instead.
    """
    db = Database(tmp_path / "flats.db")
    decided(db, "1", "approved", {"backward": 1, "weak": 9})
    decided(db, "2", "rejected", {"backward": 10, "weak": 8})
    splits = criterion_splits(db, [a_criterion("backward"), a_criterion("weak")])
    assert [s.separation for s in splits] == [-9, 1]
    assert splits[0].slug == "backward"


def _settings(floor: float = 0.5, threshold: float = 5.0):
    from flat_scout.config import Settings

    settings = Settings()
    settings.evaluation.coverage_floor = floor
    settings.evaluation.hopeful_threshold = threshold
    return settings


def test_editing_a_weight_changes_what_the_weights_report_says(tmp_path):
    """Rollout step 3 is unrunnable unless this holds.

    The report used to print `criterion_splits` alone - mean grade among
    approvals minus mean among rejections - which is computed from the
    stored grades and is entirely independent of the weights. Editing a
    weight changed only the printed `w` column, so there was no way to see
    what a candidate weighting would have done, and no evidence-based way to
    choose `hopeful_threshold` before flipping the flag.

    Two Criteria that disagree perfectly: level weights score every Listing
    at 5.0 and separate nothing, and shifting the weight onto the one that
    agrees with the couple pulls the two sides apart.
    """
    db = Database(tmp_path / "flats.db")
    decided(db, "1", "approved", {"layout": 10, "noise": 0})
    decided(db, "2", "rejected", {"layout": 0, "noise": 10})

    level = weights_report(
        db, [a_criterion("layout", 1), a_criterion("noise", 1)], _settings()
    )
    tilted = weights_report(
        db, [a_criterion("layout", 9), a_criterion("noise", 1)], _settings()
    )

    assert "approved   1   mean  5.0" in level
    assert "rejected   1   mean  5.0" in level
    assert "approved   1   mean  9.0" in tilted
    assert "rejected   1   mean  1.0" in tilted
    assert level != tilted


def test_the_weights_report_prints_the_new_scale_distribution(tmp_path):
    """Nothing printed the new-scale distribution at all, and `--decisions`
    still works off the holistic `listings.score`. So there was no evidence
    on which to set `hopeful_threshold`, which is what step 3 asks for."""
    db = Database(tmp_path / "flats.db")
    decided(db, "1", "approved", {"layout": 9})
    decided(db, "2", "rejected", {"layout": 2})

    text = weights_report(db, [a_criterion("layout", 1)], _settings(threshold=5.0))
    assert "Score on the new scale" in text
    # The holistic score on both rows is 5.0; the replayed scores are 9 and 2.
    assert "   9.0  " in text
    assert "   2.0  " in text
    assert "threshold 5.0" in text


def test_a_hard_filtered_listing_is_absent_from_the_distribution(tmp_path):
    """The distribution is the population a cut could surface.

    `backfill_grades` deliberately grades every stored row, and a
    `prefiltered_out` one still gets deterministic area and price grades off
    its postcode and its rent, so it carries a real score. `db.TRANSITIONS`
    gives it no onward move at all: it can never be surfaced, and counting it
    skews the very number `hopeful_threshold` is read off.
    """
    db = Database(tmp_path / "flats.db")
    stored(db, "1", "evaluated", {"layout": 9})
    stored(db, "2", "prefiltered_out", {"layout": 2})
    stored(db, "3", "unavailable", {"layout": 4})

    criteria = [a_criterion("layout", 1)]
    assert weighted_distribution(db, criteria, _settings()) == [(9.0, 1)]


def test_a_vetoed_listing_scoring_high_does_not_clear_the_threshold(tmp_path):
    """The count rollout step 3 reads is "how many would this cut surface".

    A determined veto - `pets = 0` - still leaves a score, and a heavily
    weighted Criterion elsewhere can put that score well above any cut.
    `adjudicate` returns `reject` regardless and the weighted path can never
    surface it, so counting it overstates the reach and calibrates
    `hopeful_threshold` against Listings that are permanently out.
    """
    db = Database(tmp_path / "flats.db")
    decided(db, "1", "approved", {"layout": 8, "pets": 8})
    # Nine parts layout at 10 against one part pets at 0 scores 9.0, which is
    # comfortably over the cut, and the veto rejects it anyway.
    stored(db, "2", "evaluated", {"layout": 10, "pets": 0})

    criteria = [a_criterion("layout", 9), a_criterion("pets", 1, veto_at_or_below=0)]
    text = weights_report(db, criteria, _settings(threshold=5.0))

    # Both are in the bars - the distribution is every graded Listing.
    assert "  9.0  " in text
    assert "1 of 2 clear" in text
    assert "veto" in text


def test_a_non_half_point_threshold_does_not_contradict_its_own_bars(tmp_path):
    """`bar_lines` draws the threshold line before the first *bucket* whose
    value clears it; the count printed beside it is score-exact, since
    `adjudicate` gates on `score >= hopeful_threshold` and not on a bucket. At
    a half-point threshold the two agree by construction, because a bucket
    boundary is also a score boundary - but a non-half-point cut is exactly
    what someone types on calibration day, and there the two used to disagree:
    a bucket sitting entirely below the drawn line could still hold a Listing
    whose own score cleared the cut, so the bars showed fewer Listings above
    the line than the sentence underneath them claimed.

    Three Listings share the 7.0-7.5 bucket - two below a 7.2 cut and one at
    or above it - and two more sit in buckets fully above it, so 3 of the 5
    clear the cut on score. The below/above split is asymmetric (2 against 1)
    deliberately, so a report that named the right bucket but swapped which
    side was which would still be caught.
    """
    db = Database(tmp_path / "flats.db")
    decided(db, "1", "approved", {"layout": 7.0})
    stored(db, "2", "evaluated", {"layout": 7.1})
    stored(db, "3", "evaluated", {"layout": 7.3})
    stored(db, "4", "evaluated", {"layout": 7.6})
    stored(db, "5", "evaluated", {"layout": 9.0})

    text = weights_report(db, [a_criterion("layout", 1)], _settings(threshold=7.2))

    assert "3 of 5 clear the configured 7.2" in text
    # The straddled bucket says so itself, rather than leaving a reader to
    # reconcile a bar that reads 3 against a sentence that reads 3 different
    # Listings.
    assert "2 below, 1 at or above" in text


def test_a_listing_under_the_coverage_floor_does_not_clear_the_threshold(tmp_path):
    """The other absolute gate, and it is absolute for the same reason.

    Coverage below the floor holds a Listing at borderline however high it
    scores, so no cut can surface it either. The caveat used to name this gate
    and imply the arithmetic below accounted for it; the count itself did not.
    """
    db = Database(tmp_path / "flats.db")
    decided(db, "1", "approved", {"layout": 8, "noise": 8})
    # `graded_pairs` returns an ungraded Criterion as undetermined, so a row
    # carrying only the light Criterion reads 2 parts in 10 of the brief.
    stored(db, "2", "evaluated", {"layout": 9})

    # The weights moved on 2026-08-20 and the reason is the finding itself:
    # unread weight now scores the 4.0 prior, so eight parts unread cannot sit
    # under a 9.0 any more. (9 x 2 + 4 x 8) / 10 = 5.0, which clears the cut
    # and is still held by the floor - which is what this test is about.
    criteria = [a_criterion("layout", 2), a_criterion("noise", 8)]
    text = weights_report(db, criteria, _settings(floor=0.5, threshold=5.0))

    assert "  5.0  " in text
    assert "1 of 2 clear" in text
    assert "50%" in text


def test_the_threshold_count_names_both_absolute_gates(tmp_path):
    """The caveat named the coverage floor alone, and said the arithmetic
    below counted it. A veto gates just as absolutely and was unmentioned."""
    db = Database(tmp_path / "flats.db")
    decided(db, "1", "approved", {"layout": 8, "pets": 8})

    criteria = [a_criterion("layout", 9), a_criterion("pets", 1, veto_at_or_below=0)]
    text = weights_report(db, criteria, _settings(threshold=5.0))

    assert "veto" in text
    assert "floor" in text


def test_an_undecided_evaluated_listing_is_most_of_the_distribution(tmp_path):
    """The docstring's intent, held: the corpus and not the decided part."""
    db = Database(tmp_path / "flats.db")
    stored(db, "1", "evaluated", {"layout": 9})
    stored(db, "2", "evaluated", {"layout": 8.5})
    stored(db, "3", "fetch_failed", {"layout": 8})
    stored(db, "4", "new", {"layout": 7.5})

    counts = dict(weighted_distribution(db, [a_criterion("layout", 1)], _settings()))
    assert sum(counts.values()) == 4


def test_the_weights_report_prints_the_overlap_band_and_the_misclassifications(tmp_path):
    """All three belong in one view: separation, band, misclassifications."""
    db = Database(tmp_path / "flats.db")
    decided(db, "1", "approved", {"layout": 8})
    decided(db, "2", "approved", {"layout": 4})
    decided(db, "3", "rejected", {"layout": 6})
    decided(db, "4", "rejected", {"layout": 2})

    text = weights_report(db, [a_criterion("layout", 1)], _settings(threshold=5.0))
    assert "Overlap" in text
    # Approvals run down to 4.0 and rejections up to 6.0, so the band is 4-6
    # and holds one Decision from each side.
    assert "4.0 - 6.0" in text
    assert "Misclassifications" in text
    # The configured cut of 5.0 loses the 4.0 approval and shows the 6.0
    # rejection - one of each, and both named.
    assert "#2" in text
    assert "#3" in text


def test_a_listing_no_criterion_could_read_is_skipped_not_scored_zero(tmp_path):
    """`adjudicate` returns `score=None` where nothing was determined.

    Treating that as 0.0 would rank a Listing nobody could read below one
    that read badly - the misreading the weighted design exists to prevent -
    and would drag the rejected mean down with a number off no evidence.
    """
    db = Database(tmp_path / "flats.db")
    decided(db, "1", "rejected", {"layout": 8})
    unreadable = decided(db, "2", "rejected", {})

    text = weights_report(db, [a_criterion("layout", 1)], _settings())
    assert "rejected   1   mean  8.0" in text
    assert f"#{unreadable}" not in text


def test_a_listing_under_the_coverage_floor_is_gated_at_every_cut(tmp_path):
    """The current weighted Verdict carries the coverage floor and vetoes.

    A Listing either gates out carries no information about where the
    threshold goes, and counting it in the threshold arithmetic would credit
    a cut with hiding a Listing that was never going to be surfaced.
    """
    db = Database(tmp_path / "flats.db")
    decided(db, "1", "approved", {"layout": 9, "noise": 9})
    decided(db, "2", "rejected", {"layout": 1})  # noise unread: coverage 1/2

    criteria = [a_criterion("layout", 1), a_criterion("noise", 1)]
    text = weights_report(db, criteria, _settings(floor=0.75))
    assert "gated" in text.lower()
    # And with the floor lowered the same Listing rejoins the arithmetic.
    assert "gated" not in weights_report(db, criteria, _settings(floor=0.25)).lower()
