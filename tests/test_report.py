from flat_scout.adjudicate import Grade
from flat_scout.criteria import Criterion
from flat_scout.db import Database
from flat_scout.models import Evaluation, ListingData
from flat_scout.report import (
    Decided,
    decisions_report,
    decisions_by_side,
    filter_families,
    hard_filtered,
    histogram,
    overlap_band,
    score_distribution,
    separates_nothing,
    status_counts,
    suggest_threshold,
    suppressed_by_threshold,
    suppressed_report,
    threshold_cost,
)


def seed(db: Database, portal_id: str, score: float, verdict: str = "borderline") -> int:
    listing_id = db.upsert(
        ListingData(portal="zoopla", portal_id=portal_id, url=f"u{portal_id}"),
        source="alert",
    )
    db.set_evaluation(
        listing_id,
        Evaluation(verdict=verdict, score=score, reasons=["r"]),
        "anthropic:claude-sonnet-5",
    )
    return listing_id


def test_distribution_buckets_scores_by_half_point(tmp_path):
    db = Database(tmp_path / "flats.db")
    for index, score in enumerate([7.1, 7.4, 8.6, 3.2]):
        seed(db, str(index), score)
    buckets = dict(score_distribution(db))
    assert buckets[7.0] == 2
    assert buckets[8.5] == 1
    assert buckets[3.0] == 1


def test_distribution_ignores_unevaluated_listings(tmp_path):
    db = Database(tmp_path / "flats.db")
    seed(db, "1", 7.5)
    db.upsert(ListingData(portal="zoopla", portal_id="2", url="u2"), source="alert")
    assert sum(count for _, count in score_distribution(db)) == 1


def test_histogram_marks_the_threshold_and_counts_what_lands_above(tmp_path):
    db = Database(tmp_path / "flats.db")
    for index, score in enumerate([5.0, 7.0, 8.0, 9.0]):
        seed(db, str(index), score)
    text = histogram(db, threshold=7.5)
    assert "7.5" in text
    assert "2 of 4" in text  # 8.0 and 9.0 clear a 7.5 threshold


def test_histogram_says_so_when_there_is_nothing_to_calibrate_on(tmp_path):
    assert "no evaluated" in histogram(Database(tmp_path / "flats.db"), threshold=7.5).lower()


def test_status_counts_are_ordered_by_size(tmp_path):
    db = Database(tmp_path / "flats.db")
    for index in range(3):
        listing_id = db.upsert(
            ListingData(portal="zoopla", portal_id=f"p{index}", url=f"u{index}"),
            source="alert",
        )
        if index < 2:
            db.transition(listing_id, "prefiltered_out")
    counts = status_counts(db)
    assert counts[0] == ("prefiltered_out", 2)
    assert ("new", 1) in counts


def store(
    db: Database,
    portal_id: str,
    score: float | None,
    *,
    address: str = "1 Test Street",
    price_pcm: int = 2000,
    reasons: list[str] | None = None,
    verdict: str = "borderline",
) -> int:
    """A Listing carried all the way to `evaluated`, ready for a Decision."""
    listing_id = db.upsert(
        ListingData(
            portal="zoopla",
            portal_id=portal_id,
            url=f"u{portal_id}",
            address=address,
            price_pcm=price_pcm,
        ),
        source="alert",
    )
    if score is not None:
        db.set_evaluation(
            listing_id,
            Evaluation(verdict=verdict, score=score, reasons=reasons or ["a reason"]),
            "anthropic:claude-sonnet-5",
        )
    db.transition(listing_id, "evaluated")
    return listing_id


def decide(
    db: Database, portal_id: str, score: float | None, decision: str, **fields
) -> int:
    listing_id = store(db, portal_id, score, **fields)
    db.transition(listing_id, decision)
    return listing_id


def test_decisions_split_by_what_the_humans_decided(tmp_path):
    db = Database(tmp_path / "flats.db")
    decide(db, "a", 8.0, "approved")
    decide(db, "r", 4.0, "rejected")
    store(db, "u", 6.0)  # a Listing nobody has decided on yet
    approved, rejected = decisions_by_side(db)
    assert [row["score"] for row in approved] == [8.0]
    assert [row["score"] for row in rejected] == [4.0]


def test_decisions_ignore_listings_that_were_never_scored(tmp_path):
    db = Database(tmp_path / "flats.db")
    decide(db, "a", 8.0, "approved")
    decide(db, "n", None, "approved")  # evaluated without a score
    approved, _ = decisions_by_side(db)
    assert len(approved) == 1
    assert (
        "1 decision was made on a Listing that was never scored"
        in decisions_report(db, 3.0)
    )


def test_report_says_so_when_nothing_has_been_decided(tmp_path):
    db = Database(tmp_path / "flats.db")
    store(db, "u", 6.0)
    assert "no decisions" in decisions_report(db, 3.0).lower()


def test_report_will_not_compare_one_sided_decisions(tmp_path):
    db = Database(tmp_path / "flats.db")
    decide(db, "a", 6.0, "approved")
    decide(db, "b", 5.5, "approved")
    text = decisions_report(db, 3.0)
    assert "approved   2" in text
    assert "rejected   0" in text
    assert "nothing to compare" in text.lower()


def test_report_leads_with_the_gap_between_the_means(tmp_path):
    db = Database(tmp_path / "flats.db")
    for index, score in enumerate([8.0, 7.0, 9.0, 8.0]):
        decide(db, f"a{index}", score, "approved")
    for index, score in enumerate([4.0, 3.0, 5.0, 4.0]):
        decide(db, f"r{index}", score, "rejected")
    text = decisions_report(db, 3.0)
    assert "8.0" in text  # approved mean
    assert "4.0" in text  # rejected mean
    assert "4.0 above" in text


def test_report_leaves_undecided_listings_out_of_the_arithmetic(tmp_path):
    db = Database(tmp_path / "flats.db")
    decide(db, "a", 8.0, "approved")
    decide(db, "r", 4.0, "rejected")
    store(db, "u1", 9.9)
    store(db, "u2", 0.1)
    text = decisions_report(db, 3.0)
    assert "9.9" not in text  # an undecided Listing must not move the arithmetic


def test_report_warns_that_a_handful_of_decisions_is_a_small_sample(tmp_path):
    db = Database(tmp_path / "flats.db")
    decide(db, "a", 8.0, "approved")
    decide(db, "r", 4.0, "rejected")
    assert "small sample" in decisions_report(db, 3.0).lower()


def test_overlap_band_is_where_the_two_ranges_cross(tmp_path):
    assert overlap_band([8.0, 5.5, 7.0], [4.0, 6.8]) == (5.5, 6.8)


def test_there_is_no_overlap_band_when_the_ranges_are_clean(tmp_path):
    assert overlap_band([8.0, 7.0], [4.0, 6.8]) is None


def test_report_names_the_overlap_band(tmp_path):
    db = Database(tmp_path / "flats.db")
    for index, score in enumerate([9.0, 8.0, 5.5]):
        decide(db, f"a{index}", score, "approved")
    for index, score in enumerate([2.0, 4.0, 6.8]):
        decide(db, f"r{index}", score, "rejected")
    text = decisions_report(db, 3.0)
    assert "5.5 - 6.8" in text


def test_report_says_when_the_ranges_never_cross(tmp_path):
    db = Database(tmp_path / "flats.db")
    for index, score in enumerate([9.0, 8.0, 7.5]):
        decide(db, f"a{index}", score, "approved")
    for index, score in enumerate([2.0, 4.0, 6.8]):
        decide(db, f"r{index}", score, "rejected")
    assert "No overlap" in decisions_report(db, 3.0)


def test_report_lists_the_disagreements_with_their_reasons(tmp_path):
    db = Database(tmp_path / "flats.db")
    clear_approval = decide(db, "a0", 9.0, "approved")
    low = decide(
        db,
        "a1",
        5.5,
        "approved",
        address="12 Elm Road",
        price_pcm=1800,
        reasons=["tiny, but the garden wins it"],
    )
    high = decide(
        db,
        "r0",
        6.8,
        "rejected",
        address="99 Loud Lane",
        price_pcm=2600,
        reasons=["scores well on paper"],
    )
    clear_rejection = decide(db, "r1", 2.0, "rejected")
    text = decisions_report(db, 3.0)
    assert f"#{high}" in text
    assert "99 Loud Lane" in text
    assert "£2,600" in text
    assert "scores well on paper" in text
    assert f"#{low}" in text
    assert "12 Elm Road" in text
    assert "tiny, but the garden wins it" in text
    # The two the model and the couple agreed on are not disagreements, and
    # listing them again would bury the four rows worth reading.
    assert f"#{clear_approval}" not in text
    assert f"#{clear_rejection}" not in text


def scored(*scores: float) -> list[Decided]:
    """Decided Listings with no hopeful Verdict, so only the cut acts on them."""
    return [Decided(score) for score in scores]


def test_suggested_threshold_is_the_cut_that_agrees_with_most_decisions(tmp_path):
    assert suggest_threshold(scored(8.0, 7.0, 6.0), scored(4.0, 3.0, 2.0)) == 6.0


def test_suggested_threshold_breaks_a_tie_towards_keeping_approvals(tmp_path):
    # 5.5 shows one rejection; 8.0 hides every rejection but costs an approval.
    # Both disagree with exactly one decision, and a flat they wanted and never
    # saw is the more expensive of the two mistakes.
    assert suggest_threshold(scored(9.0, 8.0, 5.5), scored(2.0, 4.0, 6.8)) == 5.5


def test_a_hopeful_verdict_is_never_lost_to_a_threshold(tmp_path):
    # The Listing is surfaced on its Verdict, so no cut can take it away.
    lost, _ = threshold_cost(9.0, [Decided(2.0, hopeful=True)], [])
    assert lost == 0


def test_a_hopeful_rejection_is_shown_however_high_the_cut(tmp_path):
    _, shown = threshold_cost(9.0, [], [Decided(2.0, hopeful=True)])
    assert shown == 1


def test_the_suggested_cut_does_not_pay_to_hide_what_it_cannot_hide(tmp_path):
    # Every rejection carries a hopeful Verdict and arrives whatever the cut,
    # so raising the cut buys nothing and only costs the low approval.
    approved = [Decided(8.0), Decided(5.0)]
    rejected = [Decided(7.0, hopeful=True), Decided(6.0, hopeful=True)]
    assert suggest_threshold(approved, rejected) == 5.0


def test_the_candidate_cuts_can_express_showing_nothing(tmp_path):
    # One approval under three rejections: any cut above 4.0 disagrees with a
    # single decision, and every cut inside the data disagrees with more.
    approved, rejected = scored(1.0), scored(2.0, 3.0, 4.0)
    assert suggest_threshold(approved, rejected) > 4.0
    assert separates_nothing(approved, rejected)


def test_nothing_is_separated_even_when_the_top_score_ties_with_hiding_it_all(tmp_path):
    # A cut at the highest score shows exactly one Listing and loses every
    # approval, so it ties with hiding the lot rather than losing to it. It is
    # no more of a recommendation for the tie.
    approved, rejected = scored(4.0, 3.0, 2.0), scored(8.0, 7.0, 6.0)
    assert suggest_threshold(approved, rejected) <= 8.0
    assert separates_nothing(approved, rejected)


def test_a_working_ranking_is_not_mistaken_for_an_inverted_one(tmp_path):
    assert not separates_nothing(scored(8.0, 7.0, 6.0), scored(4.0, 3.0, 2.0))


def test_report_refuses_to_recommend_a_cut_that_hides_everything(tmp_path):
    db = Database(tmp_path / "flats.db")
    for index, score in enumerate([4.0, 3.0, 2.0]):
        decide(db, f"a{index}", score, "approved")
    for index, score in enumerate([9.0, 8.0, 7.0]):
        decide(db, f"r{index}", score, "rejected")
    text = decisions_report(db, 3.0)
    assert "No cutoff separates these decisions" in text
    assert "the rubrics in criteria/" in text
    # The least-bad cut here is a finding, never a number for config.toml.
    assert "A cut at" not in text
    # ...but what today's setting does is still plain arithmetic.
    assert "The configured 3.0" in text


def test_a_threshold_deciding_nothing_is_not_reported_as_a_broken_brief(tmp_path):
    db = Database(tmp_path / "flats.db")
    # Every approval came in on its Verdict, so hiding every Listing on score
    # costs nothing and ties with any cut - but the ranking is not backwards.
    for index, score in enumerate([4.0, 2.0]):
        decide(db, f"a{index}", score, "approved", verdict="hopeful")
    for index, score in enumerate([9.0, 8.0]):
        decide(db, f"r{index}", score, "rejected")
    text = decisions_report(db, 3.0)
    assert "No cutoff separates these decisions" in text
    assert "arrived on its Verdict" in text
    assert "ranking backwards" not in text


def test_report_counts_the_listings_a_threshold_cannot_touch(tmp_path):
    db = Database(tmp_path / "flats.db")
    for index, score in enumerate([8.0, 5.0]):
        decide(db, f"a{index}", score, "approved")
    # Inside the overlap, so it is listed as a disagreement as well as counted.
    decide(db, "r0", 6.0, "rejected", verdict="hopeful")
    decide(db, "r1", 2.0, "rejected")
    text = decisions_report(db, 3.0)
    assert "1 of 4 decided Listings carry a hopeful Verdict" in text
    assert "(hopeful Verdict)" in text  # tagged on the row itself
    assert text.count("(hopeful Verdict)") == 1  # and only on that row


def test_report_is_quiet_about_verdicts_when_none_were_surfaced_on_one(tmp_path):
    db = Database(tmp_path / "flats.db")
    decide(db, "a", 8.0, "approved")
    decide(db, "r", 4.0, "rejected")
    assert "hopeful Verdict" not in decisions_report(db, 3.0)


def test_the_arithmetic_does_not_charge_a_cut_for_a_hopeful_approval(tmp_path):
    db = Database(tmp_path / "flats.db")
    decide(db, "a0", 8.0, "approved")
    decide(db, "a1", 7.0, "approved")
    # Scores below every candidate cut, but surfaced on its Verdict regardless.
    decide(db, "a2", 1.0, "approved", verdict="hopeful")
    for index, score in enumerate([4.0, 3.0, 2.0]):
        decide(db, f"r{index}", score, "rejected")
    text = decisions_report(db, 3.0)
    assert "loses 0 of 3 approvals" in text


def test_suggested_threshold_shows_what_it_costs(tmp_path):
    db = Database(tmp_path / "flats.db")
    for index, score in enumerate([9.0, 8.0, 5.5]):
        decide(db, f"a{index}", score, "approved")
    for index, score in enumerate([2.0, 4.0, 6.8]):
        decide(db, f"r{index}", score, "rejected")
    text = decisions_report(db, 3.0)
    assert "cut at 5.5" in text
    assert "loses 0 of 3 approvals" in text
    assert "spares 2 of 3 rejections" in text


def test_report_says_what_the_configured_threshold_does_today(tmp_path):
    db = Database(tmp_path / "flats.db")
    for index, score in enumerate([9.0, 8.0, 5.5]):
        decide(db, f"a{index}", score, "approved")
    for index, score in enumerate([2.0, 4.0, 6.8]):
        decide(db, f"r{index}", score, "rejected")
    text = decisions_report(db, threshold=3.0)
    assert "configured 3.0" in text
    assert "loses 0 of 3 approvals" in text
    assert "spares 1 of 3 rejections" in text


def test_the_report_command_defaults_to_the_histogram(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from flat_scout import cli

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text("[evaluation]\nhopeful_threshold = 3.0\n")
    db = Database(tmp_path / "data" / "flats.db")
    decide(db, "a", 8.0, "approved")
    result = CliRunner().invoke(cli.app, ["report"])
    assert result.exit_code == 0, result.output
    assert "threshold 3.0" in result.output


def test_the_report_command_switches_to_the_decisions_view(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from flat_scout import cli

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text("[evaluation]\nhopeful_threshold = 3.0\n")
    db = Database(tmp_path / "data" / "flats.db")
    decide(db, "a", 8.0, "approved")
    decide(db, "r", 4.0, "rejected")
    result = CliRunner().invoke(cli.app, ["report", "--decisions"])
    assert result.exit_code == 0, result.output
    assert "approved   1" in result.output
    assert "threshold 3.0" not in result.output  # the histogram, not this view


def test_disagreements_lead_with_the_worst_offender_on_each_side(tmp_path):
    db = Database(tmp_path / "flats.db")
    for index, score in enumerate([9.0, 6.0, 5.0]):
        decide(db, f"a{index}", score, "approved")
    for index, score in enumerate([8.0, 7.0, 1.0]):
        decide(db, f"r{index}", score, "rejected")
    text = decisions_report(db, 3.0)
    rejected_first = text.index("Rejected, but scored high")
    approved_first = text.index("Approved, but scored low")
    # The highest-scoring rejection heads its section, the lowest-scoring
    # approval heads its own.
    assert text[rejected_first:approved_first].index("8.0") < text[
        rejected_first:approved_first
    ].index("7.0")
    assert text[approved_first:].index("5.0") < text[approved_first:].index("6.0")


def test_disagreements_are_capped_and_count_what_they_leave_out(tmp_path):
    db = Database(tmp_path / "flats.db")
    # A scale that is entirely out: every Decision lands inside the overlap.
    for index in range(14):
        decide(db, f"a{index}", 5.0 + index * 0.1, "approved")
    for index in range(14):
        decide(db, f"r{index}", 5.0 + index * 0.1, "rejected")
    text = decisions_report(db, 3.0)
    assert "...and 4 more" in text
    assert text.count("...and 4 more") == 2  # both sides are trimmed


def a_criterion(slug: str, weight: float = 1) -> Criterion:
    return Criterion(
        slug=slug, name=slug.capitalize(), description="d", weight=weight,
        grade={"model": True}, body="b", rubric_hash="h",
    )


def grade(db: Database, listing_id: int, slug: str, value: float) -> None:
    db.set_criterion_grades(
        listing_id, [(a_criterion(slug), Grade(slug, value, True, f"{slug} evidence"))]
    )


# `decisions_report` grew a fourth argument, `criteria`, whose attribution is
# appended before every return the function can take - not only the final one
# the brief showed. These three prove each of the return points the brief
# left out still carries the finding, using the weighted-mode shape where a
# Listing can be decided with criterion_grades and no `listings.score` at all:
# `criterion_splits` reads status, not the holistic score, so the two data
# sets disagree about which Listings are "in".
def test_decisions_report_names_a_backwards_criterion_with_no_scored_decisions(tmp_path):
    """The `not total` return: nothing here has a holistic score at all."""
    db = Database(tmp_path / "flats.db")
    a_id = decide(db, "a", None, "approved")
    r_id = decide(db, "r", None, "rejected")
    grade(db, a_id, "dark-run", 2)
    grade(db, r_id, "dark-run", 9)
    text = decisions_report(db, 3.0, [a_criterion("dark-run")])
    assert "No decisions on scored Listings yet" in text
    assert "backwards" in text.lower()


def test_decisions_report_names_a_backwards_criterion_with_only_one_side_scored(tmp_path):
    """The `not approved_scores or not rejected_scores` return."""
    db = Database(tmp_path / "flats.db")
    a_id = decide(db, "a", 8.0, "approved")
    r_id = decide(db, "r", None, "rejected")
    grade(db, a_id, "dark-run", 2)
    grade(db, r_id, "dark-run", 9)
    text = decisions_report(db, 3.0, [a_criterion("dark-run")])
    assert "Only approvals so far" in text
    assert "backwards" in text.lower()


def test_decisions_report_names_a_backwards_criterion_when_the_suggested_cut_matches(tmp_path):
    """The `suggested == threshold` return, mid-function."""
    db = Database(tmp_path / "flats.db")
    a_id = decide(db, "a", 8.0, "approved")
    r_id = decide(db, "r", 2.0, "rejected")
    grade(db, a_id, "dark-run", 2)
    grade(db, r_id, "dark-run", 9)
    text = decisions_report(db, 8.0, [a_criterion("dark-run")])
    assert "That is the configured 8.0" in text
    assert "backwards" in text.lower()


def hidden(
    db: Database,
    portal_id: str,
    score: float,
    *,
    verdict: str = "borderline",
    reasons: list[str] | None = None,
    **fields,
) -> int:
    """An evaluated Listing sitting in the queue, waiting on the threshold."""
    listing_id = db.upsert(
        ListingData(portal="zoopla", portal_id=portal_id, url=f"u{portal_id}", **fields),
        source="alert",
    )
    db.set_evaluation(
        listing_id,
        Evaluation(verdict=verdict, score=score, reasons=reasons or ["a reason"]),
        "anthropic:claude-sonnet-5",
    )
    db.transition(listing_id, "evaluated")
    return listing_id


def binned(db: Database, portal_id: str, why: str, **fields) -> int:
    """A Listing a hard filter stopped before it ever reached the evaluator."""
    listing_id = db.upsert(
        ListingData(portal="zoopla", portal_id=portal_id, url=f"u{portal_id}", **fields),
        source="alert",
    )
    db.transition(listing_id, "prefiltered_out")
    db.record_event(listing_id, "prefiltered_out", why)
    return listing_id


def test_suppressed_holds_the_evaluated_listings_never_surfaced(tmp_path):
    db = Database(tmp_path / "flats.db")
    low = hidden(db, "low", 2.0)
    hidden(db, "high", 8.0)  # clears the cut, so it is queued rather than hidden
    assert [row["id"] for row in suppressed_by_threshold(db, 3.0)] == [low]


def test_suppressed_never_counts_a_listing_the_verdict_surfaces_anyway(tmp_path):
    db = Database(tmp_path / "flats.db")
    # `evaluate_weighted` surfaces a hopeful Verdict whatever the score, so no
    # cut ever suppressed this one and the report must not offer it as one the
    # couple were denied.
    hidden(db, "hopeful", 1.0, verdict="hopeful")
    assert suppressed_by_threshold(db, 3.0) == []


def test_suppressed_leaves_out_listings_that_were_decided(tmp_path):
    db = Database(tmp_path / "flats.db")
    decide(db, "r", 1.0, "rejected")  # low-scoring, but they saw it and said no
    assert suppressed_by_threshold(db, 3.0) == []


def test_hard_filtered_listings_carry_the_filter_that_stopped_them(tmp_path):
    db = Database(tmp_path / "flats.db")
    binned(db, "p", "price £3800 over cap £3500", price_pcm=3800)
    rows = hard_filtered(db)
    assert len(rows) == 1
    assert rows[0]["why"] == "price £3800 over cap £3500"


def test_a_hard_filtered_listing_with_no_event_is_not_given_a_reason(tmp_path):
    db = Database(tmp_path / "flats.db")
    listing_id = db.upsert(
        ListingData(portal="zoopla", portal_id="q", url="uq"), source="alert"
    )
    db.transition(listing_id, "prefiltered_out")
    assert hard_filtered(db)[0]["why"] is None
    assert "not recorded" in suppressed_report(db, 3.0)


def test_the_suppressed_report_lists_near_misses_first_with_their_reasons(tmp_path):
    db = Database(tmp_path / "flats.db")
    far = hidden(db, "far", 0.5, address="1 Far Road", price_pcm=1200)
    near = hidden(
        db, "near", 2.9, address="2 Near Road", price_pcm=2600, reasons=["no second bedroom"]
    )
    text = suppressed_report(db, 3.0)
    assert "2 Near Road" in text
    assert "£2,600" in text
    assert "no second bedroom" in text
    # The one that nearly cleared the cut is the one worth reading first.
    assert text.index(f"#{near}") < text.index(f"#{far}")


def test_the_hard_filters_are_grouped_by_which_one_fired(tmp_path):
    db = Database(tmp_path / "flats.db")
    for index in range(3):
        binned(
            db, f"p{index}", "on the fields already on the row: price £3800 over cap £3500"
        )
    binned(db, "c", "postcode E14 outside shortlist")
    binned(db, "b", "3 beds outside [1, 2]")
    binned(db, "g", "ground floor")
    assert filter_families(hard_filtered(db)) == [
        ("price over the cap", 3),
        ("beds", 1),
        ("ground floor", 1),
        ("postcode outside the shortlist", 1),
    ]
    text = suppressed_report(db, 3.0)
    assert "price over the cap" in text
    assert "postcode outside the shortlist" in text


def test_an_unrecognised_filter_reason_is_counted_honestly(tmp_path):
    db = Database(tmp_path / "flats.db")
    # A reason a later prefilter() might return. Guessing which bucket it
    # belongs in would be worse than admitting it does not fit one.
    binned(db, "x", "no working boiler")
    assert filter_families(hard_filtered(db)) == [("other", 1)]
    # Whatever the bucket says, the reason itself is printed on the row.
    assert "no working boiler" in suppressed_report(db, 3.0)


def test_the_suppressed_report_says_lowering_the_cut_releases_them(tmp_path):
    db = Database(tmp_path / "flats.db")
    hidden(db, "a", 2.9)
    hidden(db, "b", 1.0)
    text = suppressed_report(db, 3.0)
    # The rows stay in `evaluated`, and every run re-reads that status against
    # the cut in force. Nobody should discover that by lowering the threshold
    # and watching a batch of them arrive at once.
    assert "next run" in text
    assert "2.9" in text  # how far the cut has to drop to release the first one


def test_the_suppressed_report_says_a_cut_cannot_release_an_unscored_listing(tmp_path):
    """A Listing where no Criterion could be determined stores `score = NULL`,
    and it belongs in this section - it was seen and shown to nobody. But no
    value of `hopeful_threshold` releases it, so counting it silently among
    the rows a lower cut "releases in one burst" overstates what the knob
    does, in the number somebody reads before turning it.
    """
    db = Database(tmp_path / "flats.db")
    hidden(db, "scored", 2.9)
    hidden(db, "unread", None)
    text = suppressed_report(db, 3.0)
    assert "Below the threshold (2)" in text
    # Still listed, and still honest about having no score.
    assert "#2" in text
    assert "1 of them" in text
    assert "no cut" in text.lower()


def test_the_suppressed_report_says_nothing_of_the_sort_when_every_row_scored(tmp_path):
    """The discrimination: the caveat must not appear over a corpus a lower
    cut really would release in full."""
    db = Database(tmp_path / "flats.db")
    hidden(db, "a", 2.9)
    hidden(db, "b", 1.0)
    assert "no cut" not in suppressed_report(db, 3.0).lower()


def test_the_suppressed_report_caps_the_rows_and_counts_the_rest(tmp_path):
    db = Database(tmp_path / "flats.db")
    for index in range(14):
        hidden(db, f"h{index}", 0.1 * index)
    for index in range(13):
        binned(db, f"p{index}", "price £3800 over cap £3500")
    text = suppressed_report(db, 3.0)
    assert "...and 4 more" in text
    assert "...and 3 more" in text


def test_the_suppressed_report_says_so_when_nothing_was_suppressed(tmp_path):
    db = Database(tmp_path / "flats.db")
    hidden(db, "shown", 8.0)
    assert "Nothing has been suppressed" in suppressed_report(db, 3.0)


def test_the_suppressed_report_counts_the_listings_that_were_let_agreed(tmp_path):
    db = Database(tmp_path / "flats.db")
    listing_id = db.upsert(
        ListingData(portal="rightmove", portal_id="g", url="ug"), source="alert"
    )
    db.transition(listing_id, "unavailable")
    text = suppressed_report(db, 3.0)
    # Not suppressed by anything this system chose - but it is the other half of
    # "where did the rest go", and it is one line.
    assert "let agreed" in text


def test_the_report_command_switches_to_the_suppressed_view(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from flat_scout import cli

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text("[evaluation]\nhopeful_threshold = 3.0\n")
    db = Database(tmp_path / "data" / "flats.db")
    hidden(db, "low", 2.0, address="9 Quiet Street")
    result = CliRunner().invoke(cli.app, ["report", "--suppressed"])
    assert result.exit_code == 0, result.output
    assert "9 Quiet Street" in result.output
    assert "threshold 3.0" not in result.output  # the histogram, not this view


def test_the_report_command_will_not_show_two_views_at_once(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from flat_scout import cli

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text("[evaluation]\nhopeful_threshold = 3.0\n")
    Database(tmp_path / "data" / "flats.db")
    result = CliRunner().invoke(cli.app, ["report", "--decisions", "--suppressed"])
    assert result.exit_code != 0
    assert "one" in result.output.lower()
