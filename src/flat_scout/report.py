"""Read-only views over the database.

The histogram exists for one job: setting `hopeful_threshold` from the shadow
day's real score distribution instead of from a guess.

The decisions report answers the question the histogram cannot. A distribution
of numbers says nothing about the ranking underneath it: it counts how many
Listings scored above a line, never whether the ones above it are the ones the
couple would have wanted. Holding score against Decision does, and the answer
decides what to fix - a threshold, or the brief in criteria/criteria.md.

Both of those read `listings.score`, which is the holistic number and is never
rewritten. The weights report asks the same two questions on the new scale, and
computes its own score from `criterion_grades` and the weights on disk. That is
arithmetic over stored rows, so a candidate weighting can be edited and re-read
as often as it takes without a model call - which is what makes step 3 of the
rollout a measurement rather than a guess.

The shift report asks the deployment question the other three cannot. All of
them describe a corpus; this one describes what the switch would have DONE to
the Decisions already taken - and in particular which flats the couple wanted
would never have been shown to them at all. That failure is invisible
everywhere else here, because every other view reads Listings that were
surfaced. It is arithmetic over stored rows too, and calls no model.

The posterior report is the weights report's question asked of the Bayesian
gate. That gate is two numbers rather than one - a threshold and a confidence
- and a weighted mean has nowhere to put the second, so nothing above can
calibrate it. It replays candidate pairs of the two over the same stored
grades and the same Decisions, and calls no model either.
"""

from __future__ import annotations

import json
import sqlite3
import textwrap
from dataclasses import dataclass

from flat_scout.adjudicate import adjudicate, vetoes_fired
from flat_scout.db import Database

BUCKET = 0.5
# Read in a terminal, and often piped from another machine. Prose is wrapped;
# the rows are not, because an id, a score and a price want to line up.
WIDTH = 78

# Below this many Decisions the arithmetic is arranged coincidence. The rows in
# the disagreements section still mean something; the means do not.
SMALL_SAMPLE = 20
# ...and a lopsided set is thin however large it is: forty approvals against
# two rejections says nothing about where the line goes.
THIN_SIDE = 5
# How many disagreements to print per side, worst first. If the scale is badly
# out, every Decision falls inside the overlap, and forty rows of it is the
# same finding repeated - the count that follows carries the rest.
DISAGREEMENTS_SHOWN = 10


def score_distribution(db: Database) -> list[tuple[float, int]]:
    """Evaluated Listings counted into half-point score buckets."""
    rows = db.conn.execute("SELECT score FROM listings WHERE score IS NOT NULL").fetchall()
    buckets: dict[float, int] = {}
    for row in rows:
        bucket = int(row["score"] / BUCKET) * BUCKET
        buckets[bucket] = buckets.get(bucket, 0) + 1
    return sorted(buckets.items())


def status_counts(db: Database) -> list[tuple[str, int]]:
    return [
        (row["status"], row["count"])
        for row in db.conn.execute(
            "SELECT status, COUNT(*) AS count FROM listings "
            "GROUP BY status ORDER BY count DESC, status"
        )
    ]


def histogram(db: Database, threshold: float) -> str:
    """The score distribution as text, with the threshold drawn in."""
    distribution = score_distribution(db)
    if not distribution:
        return "No evaluated listings yet - nothing to calibrate on."

    lines = bar_lines(distribution, threshold)
    total = sum(count for _, count in distribution)
    above = sum(count for bucket, count in distribution if bucket >= threshold)
    lines.append("")
    lines.append(f"  {above} of {total} would clear the threshold.")
    return "\n".join(lines)


def bar_lines(
    distribution: list[tuple[float, int]],
    threshold: float,
    straddled: tuple[int, int] | None = None,
) -> list[str]:
    """One bar per bucket, with the threshold drawn in where it falls.

    Shared by the holistic histogram and the weighted replay, which is the
    point: the two scales are read against each other on the calibration
    day, and a difference in how they are drawn would read as a difference
    in the numbers.

    `straddled` is (below, above): how the bucket holding a non-boundary
    threshold splits between Listings that miss the cut and Listings that
    clear it. Only a caller holding the exact scores behind a bucket can
    compute it - the weighted report, never the holistic histogram, which
    counts by bucket and has nothing finer to offer. Passed, the threshold
    line is drawn against that bucket's own row instead of a whole bucket
    beyond it, and the row says how it splits - so a bucket that reads as
    sitting cleanly below the cut never disagrees with a score-exact count
    that puts some of its Listings above. Omitted, the line is drawn before
    the first bucket whose value clears the threshold, same as always.
    """
    widest = max(count for _, count in distribution)
    lines: list[str] = []
    drawn = False
    for bucket, count in distribution:
        bar = "#" * max(1, round(count * 30 / widest))
        if not drawn and straddled is not None and bucket <= threshold < bucket + BUCKET:
            below, above = straddled
            lines.append(f"  {bucket:>4.1f}  {bar} {count}  ({below} below, {above} at or above)")
            lines.append(f"  ----- threshold {threshold} " + "-" * 24)
            drawn = True
            continue
        if not drawn and bucket >= threshold:
            lines.append(f"  ----- threshold {threshold} " + "-" * 24)
            drawn = True
        lines.append(f"  {bucket:>4.1f}  {bar} {count}")
    if not drawn:
        lines.append(f"  ----- threshold {threshold} " + "-" * 24)
    return lines


# What the humans decided, read off the Decision rather than off the status.
# An approval is anything carrying a Decision that is not a rejection, so a
# Listing whose status moves on afterwards is still counted - dropping those
# as the search progresses would quietly bias the comparison towards the
# flats nobody chased.
_DECIDED_COLUMNS = "id, score, verdict, price_pcm, address, postcode, reasons, status"
_APPROVED = "decided_at IS NOT NULL AND status != 'rejected'"
_REJECTED = "status = 'rejected'"

# Statuses with an empty transition set in `db.TRANSITIONS` that were never
# surfaced. A Listing here is terminal by construction, so no threshold can
# reach it and no distribution should count it.
NEVER_CARDED = ("prefiltered_out", "unavailable")
_NEVER_CARDED_SQL = ", ".join(f"'{status}'" for status in NEVER_CARDED)


def decisions_by_side(db: Database) -> tuple[list[sqlite3.Row], list[sqlite3.Row]]:
    """Scored Listings that carry a Decision, split into (approved, rejected).

    Highest score first, because the head of each list is where a disagreement
    with the other side lives.
    """

    def fetch(where: str) -> list[sqlite3.Row]:
        return db.conn.execute(
            f"SELECT {_DECIDED_COLUMNS} FROM listings "
            f"WHERE score IS NOT NULL AND {where} ORDER BY score DESC, id"
        ).fetchall()

    return fetch(_APPROVED), fetch(_REJECTED)


def decided_rows(db: Database, columns: str) -> list[sqlite3.Row]:
    """Every Listing carrying a Decision, either side, lowest id first.

    One home for the two predicates above, so a report cannot quietly drift
    onto `status = 'approved'` and drop a Listing whose status has moved on
    since. The side is read off `status == 'rejected'` by every caller.
    """
    return db.conn.execute(
        f"SELECT {columns} FROM listings "
        f"WHERE ({_APPROVED}) OR ({_REJECTED}) ORDER BY id"
    ).fetchall()


def overlap_band(
    approved_scores: list[float], rejected_scores: list[float]
) -> tuple[float, float] | None:
    """The band where approvals and rejections coexist.

    Below it everything was rejected and above it everything approved, so a
    threshold anywhere outside the band agrees with every Decision. Inside it,
    no threshold can: that is what makes naming the band more useful than any
    summary statistic. None means the two ranges never cross.

    Scores only, deliberately. A hopeful Verdict is also unseparable by any
    threshold, but folding those in would stretch the band to wherever the
    lowest of them happens to sit and destroy the one thing it says clearly -
    where the two score ranges cross. That failure is a different one, and it
    is reported on its own: counted under the threshold arithmetic, and tagged
    on the rows themselves.
    """
    if not approved_scores or not rejected_scores:
        return None
    low, high = min(approved_scores), max(rejected_scores)
    return (low, high) if low <= high else None


@dataclass(frozen=True)
class Decided:
    """A decided Listing, reduced to the two fields the adjudication reads.

    `hopeful` is the Verdict, and it is not a restatement of the score. The two
    are independent fields of the evaluator's output, and `evaluate_weighted`
    in pipeline.py surfaces a hopeful Verdict whatever the score - so a
    threshold has no purchase on one at all. Arithmetic over the score alone
    would credit a cut with hiding Listings it could never have hidden, on the
    very number the threshold gets set from.
    """

    score: float
    hopeful: bool = False


def _decided(rows: list[sqlite3.Row]) -> list[Decided]:
    return [Decided(row["score"], row["verdict"] == "hopeful") for row in rows]


def would_be_shown(item: Decided, cut: float) -> bool:
    """The adjudication, replayed against a hypothetical threshold."""
    return item.hopeful or item.score >= cut


def _candidates(items: list[Decided]) -> list[float]:
    """Every cut worth trying, lowest first.

    The observed scores, because a cut between two of them behaves exactly like
    the lower one - plus a sentinel above them all, meaning "show nothing on
    score". Without that last one the search cannot express the answer that no
    cutoff helps, and settles for the least-bad cut inside the data instead.
    """
    scores = sorted({item.score for item in items})
    return [*scores, scores[-1] + BUCKET]


def suggest_threshold(approved: list[Decided], rejected: list[Decided]) -> float:
    """The cut that disagrees with the fewest Decisions.

    Ties break towards the lower cut, i.e. towards surfacing a Listing they
    would have rejected rather than hiding one they would have wanted.

    Ask `separates_nothing` before printing the answer as a recommendation: a
    set nothing separates still has a least-bad cut, and it is not one anybody
    should paste into config.toml.

    Both sides are needed: a cut fitted to approvals alone is just their lowest
    score. Callers check that first.
    """
    best, fewest = 0.0, None
    for cut in _candidates(approved + rejected):
        wrong = _disagreements(cut, approved, rejected)
        if fewest is None or wrong < fewest:
            fewest, best = wrong, cut
    return best


def _disagreements(cut: float, approved: list[Decided], rejected: list[Decided]) -> int:
    return sum(threshold_cost(cut, approved, rejected))


def separates_nothing(approved: list[Decided], rejected: list[Decided]) -> bool:
    """Whether hiding every Listing on score does as well as any real cut.

    Not "did the sentinel win the search": it can tie with a cut at the very
    top score, which shows nothing either and would then be printed as though
    it were a recommendation. Ties count, so the question is asked directly.
    """
    candidates = _candidates(approved + rejected)
    costs = [_disagreements(cut, approved, rejected) for cut in candidates]
    return costs[-1] <= min(costs)  # the sentinel is the last candidate


def threshold_cost(
    cut: float, approved: list[Decided], rejected: list[Decided]
) -> tuple[int, int]:
    """(approvals lost, rejections still shown) had this cut been in force."""
    return (
        sum(1 for item in approved if not would_be_shown(item, cut)),
        sum(1 for item in rejected if would_be_shown(item, cut)),
    )


def _count(db: Database, where: str) -> int:
    sql = f"SELECT COUNT(*) AS n FROM listings WHERE {where}"
    return int(db.conn.execute(sql).fetchone()["n"])


def _reasons(row: sqlite3.Row) -> list[str]:
    try:
        parsed = json.loads(row["reasons"] or "[]")
    except (TypeError, ValueError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _listing_lines(row: sqlite3.Row) -> list[str]:
    where = row["address"] or row["postcode"] or "?"
    price = f"£{row['price_pcm']:,}" if row["price_pcm"] else "£?"
    # A hopeful Verdict is flagged on the row itself: no cut would have changed
    # what happened to this one, so it is not evidence about the threshold.
    tag = "  (hopeful Verdict)" if row["verdict"] == "hopeful" else ""
    # The decisions view selects on `score IS NOT NULL`; the suppressed view
    # cannot, because a Listing with no score is exactly one nobody was shown.
    score = f"{row['score']:.1f}" if row["score"] is not None else " ? "
    lines = [f"    #{row['id']}  {score}  {price}  {where}{tag}"]
    lines.extend(f"        {reason}" for reason in _reasons(row))
    return lines


def _disagreement_lines(heading: str, rows: list[sqlite3.Row]) -> list[str]:
    lines = [f"  {heading}"]
    for row in rows[:DISAGREEMENTS_SHOWN]:
        lines.extend(_listing_lines(row))
    hidden = len(rows) - DISAGREEMENTS_SHOWN
    if hidden > 0:
        lines.append(f"    ...and {hidden} more")
    return lines


def _side_line(name: str, scores: list[float]) -> str:
    if not scores:
        return f"  {name:<8} {0:>3}"
    mean = sum(scores) / len(scores)
    return (
        f"  {name:<8} {len(scores):>3}   mean {mean:>4.1f}   "
        f"range {min(scores):.1f} - {max(scores):.1f}"
    )


def _append_attribution(lines: list[str], db: Database, criteria: list | None) -> None:
    """The per-Criterion finding, tacked onto a decisions report in place.

    Independent of the holistic score: a weighted-mode Listing can carry
    criterion_grades with no `listings.score` at all, so this is checked at
    every exit from `decisions_report` rather than folded into the score
    arithmetic above it.
    """
    if not criteria:
        return
    splits = criterion_splits(db, criteria)
    if any(split.separation is not None for split in splits):
        lines += attribution_lines(splits)


def decisions_report(db: Database, threshold: float, criteria: list | None = None) -> str:
    """Score against Decision: whether the ranking agrees with the couple.

    `threshold` is the configured `hopeful_threshold`, shown only so the cut in
    force can be read in the same arithmetic as the suggested one. Nothing here
    writes, and nothing here advises: the numbers are theirs to act on.

    `criteria`, when given, appends the per-Criterion attribution - which
    Criterion is separating the Decisions and which is ranking backwards. The
    holistic path passes nothing and prints exactly what it prints today.
    """
    approved, rejected = decisions_by_side(db)
    approved_decided, rejected_decided = _decided(approved), _decided(rejected)
    approved_scores = [row["score"] for row in approved]
    rejected_scores = [row["score"] for row in rejected]
    total = len(approved) + len(rejected)

    unscored = _count(db, "decided_at IS NOT NULL AND score IS NULL")

    lines: list[str] = []

    def say(text: str, indent: str = "  ") -> None:
        """A sentence, wrapped to a terminal rather than run off the edge."""
        lines.append(textwrap.fill(text, WIDTH, initial_indent=indent, subsequent_indent=indent))

    if not total:
        say(
            "No decisions on scored Listings yet - nothing to compare scores "
            "against.",
            "",
        )
    else:
        say(f"Score against Decision, over {total} decisions.", "")
    if unscored:
        say(
            f"{unscored} decision{'s were' if unscored != 1 else ' was'} made on a "
            "Listing that was never scored, and left out."
        )
    if not total:
        _append_attribution(lines, db, criteria)
        return "\n".join(lines)

    lines.append("")
    lines.append(_side_line("approved", approved_scores))
    lines.append(_side_line("rejected", rejected_scores))
    lines.append("")

    if not approved_scores or not rejected_scores:
        side = "approvals" if approved_scores else "rejections"
        say(
            f"Only {side} so far, so there is nothing to compare. Decisions have to "
            "land on both sides of the same run of Listings before any of this "
            "means anything."
        )
        _append_attribution(lines, db, criteria)
        return "\n".join(lines)

    # The one line worth reading first. A ranking that works puts this well
    # above zero; a negative gap means the scale is pointing the wrong way and
    # no threshold will fix it.
    gap = sum(approved_scores) / len(approved_scores) - sum(rejected_scores) / len(rejected_scores)
    if gap > 0:
        say(f"The approved mean sits {gap:.1f} above the rejected mean.")
    elif gap < 0:
        say(
            f"The approved mean sits {abs(gap):.1f} BELOW the rejected mean. The ranking "
            "is upside down - that is a brief problem, not a threshold one."
        )
    else:
        say("The two means are identical. The score is not ranking anything.")

    if total < SMALL_SAMPLE or min(len(approved), len(rejected)) < THIN_SIDE:
        say(
            f"Small sample: {len(approved)} approved, {len(rejected)} rejected. "
            "Read the rows below, not the averages."
        )

    band = overlap_band(approved_scores, rejected_scores)
    lines.append("")
    lines.append("Overlap")
    if band is None:
        say("No overlap: every approval scored above every rejection.")
    else:
        # The band's ends are themselves a Decision each - its floor is an
        # approval and its ceiling a rejection - so neither list is ever empty
        # and neither heading is ever printed over nothing.
        low, high = band
        inside_approved = [row for row in approved if low <= row["score"] <= high]
        inside_rejected = [row for row in rejected if low <= row["score"] <= high]
        say(
            f"{low:.1f} - {high:.1f}, holding {len(inside_approved)} "
            f"approval{'s' if len(inside_approved) != 1 else ''} and "
            f"{len(inside_rejected)} rejection{'s' if len(inside_rejected) != 1 else ''}. "
            "No threshold can separate those."
        )
        lines.append("")
        lines.append("Disagreements")
        # Worst first on each side: the rejection that scored highest and the
        # approval that scored lowest are the two rows most likely to name
        # something the brief never asked about.
        lines.extend(_disagreement_lines("Rejected, but scored high", inside_rejected))
        lines.extend(
            _disagreement_lines("Approved, but scored low", inside_approved[::-1])
        )

    lines.append("")
    lines.append("Threshold arithmetic")

    # A Listing is surfaced on a hopeful Verdict OR a score over the line, so
    # any Listing carrying that Verdict arrives whatever the cut is. Where
    # there are many of them the threshold has far less leverage than the sums
    # below suggest, and the reader has to be told before reading them.
    unaffected = sum(
        1 for item in approved_decided + rejected_decided if item.hopeful
    )
    if unaffected:
        say(
            f"{unaffected} of {total} decided Listings carry a hopeful Verdict and "
            "are surfaced whatever the cut is. The sums below account for that."
        )

    if separates_nothing(approved_decided, rejected_decided):
        # Hiding every Listing did at least as well as any cut inside the data.
        # Printing the least-bad cut here would read as "set the threshold to
        # 8.6" - advice to switch the system off, dressed as calibration.
        say(
            "No cutoff separates these decisions: hiding every Listing on score "
            "would disagree with no more of them than any cut inside the data."
        )
        # Which of the two failures it is. Hiding everything costs approvals
        # only when the score put them low, and that is the inverted ranking.
        # Costing none means every approval came in on its Verdict instead,
        # and the threshold was never deciding anything.
        sentinel = _candidates(approved_decided + rejected_decided)[-1]
        lost, _ = threshold_cost(sentinel, approved_decided, rejected_decided)
        if lost:
            say(
                "The score is ranking backwards, so the rubrics in criteria/ are "
                "what need rewriting - no value of hopeful_threshold rescues this."
            )
        else:
            say(
                "Every approval arrived on its Verdict rather than its score, so "
                "the threshold has nothing to decide here either way."
            )
    else:
        suggested = suggest_threshold(approved_decided, rejected_decided)
        lost, shown = threshold_cost(suggested, approved_decided, rejected_decided)
        say(
            f"A cut at {suggested:.1f} loses {lost} of {len(approved)} approvals and "
            f"spares {len(rejected) - shown} of {len(rejected)} rejections."
        )
        if suggested == threshold:
            say(f"That is the configured {threshold:.1f}.")
            _append_attribution(lines, db, criteria)
            return "\n".join(lines)
    lost, shown = threshold_cost(threshold, approved_decided, rejected_decided)
    say(
        f"The configured {threshold:.1f} loses {lost} of {len(approved)} approvals and "
        f"spares {len(rejected) - shown} of {len(rejected)} rejections."
    )
    _append_attribution(lines, db, criteria)
    return "\n".join(lines)


# How many suppressed Listings to print per section, and no more: a cap that is
# too low hides evidence, and a report nobody reads to the end hides all of it.
SUPPRESSED_SHOWN = 10

# The prefilter reasons in evaluate.py, each reduced to the filter that fired.
# The reason itself embeds the Listing's own values ("price £3800 over cap
# £3500") and cannot be grouped as it stands, so these match on the fixed part
# of it. A reason none of them match is counted as "other" rather than forced
# into the nearest-looking bucket: prefilter() can grow a rule this list does
# not know about, and a wrong count is worse than an honest unknown. The whole
# reason is printed on the Listing's own row either way.
_FILTER_FAMILIES = (
    ("price over the cap", "over cap"),
    ("postcode outside the shortlist", "outside shortlist"),
    ("beds", "beds outside"),
    ("ground floor", "ground floor"),
)
UNRECOGNISED_FILTER = "other"


def _filter_family(why: str | None) -> str:
    for name, fixed_part in _FILTER_FAMILIES:
        if why and fixed_part in why:
            return name
    return UNRECOGNISED_FILTER


def filter_families(rows: list[sqlite3.Row]) -> list[tuple[str, int]]:
    """Hard-filtered Listings counted by the filter that stopped them.

    Biggest first: which filter is doing the work is the finding. A price cap
    that accounts for nearly every row is a cap set below the market, and that
    is a config.toml problem no amount of reading the rows will show.
    """
    counts: dict[str, int] = {}
    for row in rows:
        family = _filter_family(row["why"])
        counts[family] = counts.get(family, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


# The prefilter is terminal and fires once, but the reason is read as the last
# such event rather than the only one: a report must not raise on a row whose
# history is odd.
_HARD_FILTERED_SQL = """
SELECT l.id, l.url, l.address, l.postcode, l.price_pcm, l.beds,
       (SELECT e.detail FROM events e
         WHERE e.listing_id = l.id AND e.kind = 'prefiltered_out'
         ORDER BY e.id DESC LIMIT 1) AS why
FROM listings l WHERE l.status = 'prefiltered_out' ORDER BY l.id DESC
"""


def hard_filtered(db: Database) -> list[sqlite3.Row]:
    """Listings a hard filter stopped, newest first, with the reason."""
    return db.conn.execute(_HARD_FILTERED_SQL).fetchall()


def suppressed_by_threshold(db: Database, threshold: float) -> list[sqlite3.Row]:
    """Evaluated Listings that were never surfaced, nearest miss first.

    `evaluated` is the whole set to look at: a Listing leaves that status only
    when a Decision is taken on it, so anything still in it was seen by the
    evaluator and by nobody else. Which of them the cut is responsible for is
    then decided by `would_be_shown` - the same rule the pipeline surfaces on
    - so a hopeful Verdict is never reported as something a threshold withheld.
    """
    rows = db.conn.execute(
        f"SELECT {_DECIDED_COLUMNS} FROM listings WHERE status = 'evaluated' "
        "ORDER BY score DESC, id"
    ).fetchall()
    return [
        row
        for row in rows
        if not would_be_shown(
            Decided(row["score"] or 0.0, row["verdict"] == "hopeful"), threshold
        )
    ]


def _hard_filtered_lines(row: sqlite3.Row) -> list[str]:
    where = row["address"] or row["postcode"] or "?"
    price = f"£{row['price_pcm']:,}" if row["price_pcm"] else "£?"
    why = row["why"] or "reason not recorded"
    return [f"    #{row['id']}  {price}  {where}", f"        {why}", f"        {row['url']}"]


def _capped(lines: list[str], rows: list[sqlite3.Row], render) -> list[str]:
    for row in rows[:SUPPRESSED_SHOWN]:
        lines.extend(render(row))
    hidden = len(rows) - SUPPRESSED_SHOWN
    if hidden > 0:
        lines.append(f"    ...and {hidden} more")
    return lines


def suppressed_report(db: Database, threshold: float) -> str:
    """Everything the system saw and never showed anybody.

    The counterpart to the histogram, which counts what got through. A triage
    tool is judged on what it threw away, and until this view existed the only
    way to read that was SQL against the database by hand.
    """
    below = suppressed_by_threshold(db, threshold)
    filtered = hard_filtered(db)
    unavailable = _count(db, "status = 'unavailable'")

    lines: list[str] = []

    def say(text: str, indent: str = "  ") -> None:
        lines.append(textwrap.fill(text, WIDTH, initial_indent=indent, subsequent_indent=indent))

    say("Suppressed Listings: seen by the system, shown to nobody.", "")

    if not below and not filtered:
        say("Nothing has been suppressed - every Listing was surfaced.")
    if below:
        lines.append("")
        lines.append(f"Below the threshold ({len(below)})")
        # These rows are not history. They sit in `evaluated`, and every run
        # re-reads that status against the cut in force, so lowering
        # hopeful_threshold surfaces the ones that clear the new line - all of
        # them, at once, however old. Discovering that by watching a batch of
        # them arrive at once would be a nasty surprise.
        released = (
            f" Lowering it to {below[0]['score']:.1f} releases the first of them, and "
            "a cut well under that releases the lot in one burst."
            if below[0]["score"] is not None
            else ""
        )
        # The ones no cut can reach. On the weighted path a Listing where no
        # Criterion could be determined stores `score = NULL`, and it belongs
        # in this section - it was seen and shown to nobody - but the threshold
        # is not what is holding it. Counting it silently among the rows "a cut
        # well under that releases in one burst" overstates what the knob does,
        # in the number somebody reads immediately before turning it. Their
        # remedy is the rubrics or the Portal's data, never the cut.
        unscored = sum(1 for row in below if row["score"] is None)
        if unscored:
            released += (
                f" {unscored} of them carr{'y' if unscored != 1 else 'ies'} no score "
                "at all - no Criterion could be determined - so no cut releases "
                f"{'those' if unscored != 1 else 'that one'}."
            )
        say(
            "Still live, and still queued: every run re-reads this status against "
            "the threshold in force, so a lower one surfaces these on the next "
            "run, however old they are." + released
        )
        lines.append("")
        _capped(lines, below, _listing_lines)
    if filtered:
        lines.append("")
        lines.append(f"Stopped by a hard filter ({len(filtered)})")
        say("Never evaluated, so they carry no score. The filters live in config.toml.")
        lines.append("")
        for family, count in filter_families(filtered):
            lines.append(f"    {family:<34}{count:>4}")
        lines.append("")
        _capped(lines, filtered, _hard_filtered_lines)
    if unavailable:
        lines.append("")
        say(
            f"{unavailable} further Listing{'s' if unavailable != 1 else ''} went "
            "unavailable: let agreed before the page could be read. Nothing here "
            f"suppressed {'them' if unavailable != 1 else 'it'}.",
            "",
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class CriterionSplit:
    """How one Criterion graded the Listings on each side of a Decision."""

    slug: str
    name: str
    weight: float
    approved_mean: float | None
    rejected_mean: float | None
    separation: float | None


def criterion_splits(db: Database, criteria: list) -> list[CriterionSplit]:
    """Per Criterion, the mean grade among approvals against rejections.

    The separation is the whole point. A large positive one means the Criterion
    is doing the work. A negative one means it is ranking backwards, and it
    names the file to edit - which is the question `report --decisions` has
    never been able to answer, because a single holistic score has no parts.
    """
    # The sides come off the Decision, through the same two predicates
    # `decisions_by_side` and `replay_decisions` read. Selecting
    # `status IN ('approved', 'rejected')` here instead would let the two
    # halves of one report disagree about who was approved - the attribution
    # reading the status while the threshold arithmetic printed beside it read
    # the Decision. Both sides read the same predicate now, and it is applied
    # to `listings` alone, in a subquery, so its bare column names can never
    # bind to a join partner.
    rows = db.conn.execute(
        f"SELECT g.criterion, g.grade, "
        f"CASE WHEN d.status = 'rejected' THEN 'rejected' ELSE 'approved' END AS side "
        f"FROM criterion_grades g JOIN ("
        f"  SELECT id, status FROM listings WHERE ({_APPROVED}) OR ({_REJECTED})"
        f") d ON d.id = g.listing_id "
        f"WHERE g.determined = 1 AND g.grade IS NOT NULL"
    ).fetchall()
    sides: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        side = sides.setdefault(row["criterion"], {"approved": [], "rejected": []})
        side[row["side"]].append(row["grade"])

    splits = []
    for criterion in criteria:
        side = sides.get(criterion.slug, {"approved": [], "rejected": []})
        approved = _mean(side["approved"])
        rejected = _mean(side["rejected"])
        gap = None if approved is None or rejected is None else approved - rejected
        splits.append(
            CriterionSplit(
                slug=criterion.slug, name=criterion.name, weight=criterion.weight,
                approved_mean=approved, rejected_mean=rejected, separation=gap,
            )
        )
    # Most informative first, by magnitude rather than raw value: a Criterion
    # ranking strongly backwards is not doing nothing, it is doing real work in
    # the wrong direction, and it is the single most actionable row in the
    # report. Sorting on the raw separation would bury it at the foot of the
    # table below a Criterion that separates almost nothing - the footnote
    # position that finding must never end up in. `None` (not graded on both
    # sides) sorts last regardless, since there is nothing to rank at all.
    return sorted(
        splits,
        key=lambda s: (
            s.separation is not None,
            abs(s.separation) if s.separation is not None else 0,
        ),
        reverse=True,
    )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def attribution_lines(splits: list[CriterionSplit]) -> list[str]:
    lines = ["", "Per Criterion, mean grade by Decision:", ""]
    for split in splits:
        if split.separation is None:
            lines.append(f"  {split.name:<32} w{split.weight:<3.0f}  not graded on both sides")
            continue
        lines.append(
            f"  {split.name:<32} w{split.weight:<3.0f}  "
            f"approved {split.approved_mean:>4.1f}   rejected {split.rejected_mean:>4.1f}   "
            f"gap {split.separation:>+5.1f}"
        )
    backwards = [split for split in splits if split.separation is not None and split.separation < 0]
    if backwards:
        named = ", ".join(split.name for split in backwards)
        lines += [
            "",
            textwrap.fill(
                f"{named} ranks backwards: the Listings they graded well are the ones "
                "the couple rejected. No weight rescues that - the rubric in "
                "criteria/ is what needs work.",
                WIDTH,
            ),
        ]
    return lines


@dataclass(frozen=True)
class Replayed:
    """One Listing, judged again by the weights and rubrics in force now.

    This is the second score - what a candidate weighting would have said. It
    is arithmetic over the stored grade vector, so it costs nothing to
    produce and nothing to produce again with a weight moved, which is the
    whole reason the vector is stored per Criterion.

    It is never written anywhere. `listings.score` is what was surfaced and
    what the Decision answered.
    """

    listing_id: int
    score: float
    coverage: float
    # Hidden at every cut. The current weighted adjudication carries the
    # coverage floor and the vetoes, so no value of `hopeful_threshold` brings
    # one of these back, and counting it in the threshold arithmetic would
    # credit a cut with hiding a Listing that was never going to be surfaced.
    gated: bool
    address: str
    price_pcm: int | None


def _adjudicated(db: Database, listing_id: int, criteria: list, settings):
    """The stored grade vector, weighed by the Criteria as they are now.

    No model is called and none can be: `graded_pairs` is SQL over
    `criterion_grades`, and `adjudicate` is arithmetic with no I/O at all.
    """
    return adjudicate(
        db.graded_pairs(listing_id, criteria),
        hopeful_threshold=settings.evaluation.hopeful_threshold,
        coverage_floor=settings.evaluation.coverage_floor,
        unknown_prior=settings.evaluation.unknown_prior,
    )


def _replayed(db: Database, row: sqlite3.Row, criteria: list, settings) -> Replayed | None:
    result = _adjudicated(db, row["id"], criteria, settings)
    # No score means no Criterion could be determined and none was defaulted.
    # Counting that as 0.0 would rank a Listing nobody could read below one
    # that read badly, and would drag a mean down with a number off no
    # evidence - the misreading the whole weighted design exists to prevent.
    if result.score is None:
        return None
    return Replayed(
        listing_id=row["id"],
        score=result.score,
        coverage=result.coverage,
        gated=(
            result.verdict == "reject"
            or result.coverage < settings.evaluation.coverage_floor
        ),
        address=row["address"] or row["postcode"] or "?",
        price_pcm=row["price_pcm"],
    )


def replay_decisions(
    db: Database, criteria: list, settings
) -> tuple[list[Replayed], list[Replayed]]:
    """Every decided Listing, re-adjudicated, split into (approved, rejected).

    Highest score first, matching `decisions_by_side`: the head of each list
    is where a disagreement with the other side lives.

    The sides come off the Decision and not off the score, the same way
    `decisions_by_side` reads them - anything decided that was not rejected
    is a Listing somebody once said yes to.
    """
    rows = decided_rows(db, "id, address, postcode, price_pcm, status")
    approved: list[Replayed] = []
    rejected: list[Replayed] = []
    for row in rows:
        replayed = _replayed(db, row, criteria, settings)
        if replayed is None:
            continue
        (rejected if row["status"] == "rejected" else approved).append(replayed)

    def order(item: Replayed) -> tuple[float, int]:
        return (-item.score, item.listing_id)

    return sorted(approved, key=order), sorted(rejected, key=order)


@dataclass(frozen=True)
class Scored:
    """One replayed Listing: its score, and whether any cut could surface it.

    `gated` is the two ABSOLUTE gates and only those. A determined veto makes
    `adjudicate` return `reject`, and coverage under the floor holds the
    Verdict at borderline, whatever the score - so neither depends on the cut
    being considered, and a Listing carrying either can never be surfaced at
    any threshold. That is what disqualifies it from a count describing what
    a cut would surface, and what distinguishes it from a Listing merely
    below the line, which a lower line would surface.
    """

    score: float
    gated: bool


def weighted_scores(db: Database, criteria: list, settings) -> list[Scored]:
    """Every gradable Listing, replayed against the weights in force now.

    The corpus, not the decided part of it. `hopeful_threshold` is set from
    how many Listings a cut would surface, and the ones nobody has decided on
    are most of that count.

    Every stored row is graded by `backfill_grades`, including the ones a hard
    filter stopped, and those still pick up deterministic area and price grades
    off a postcode and a rent - so they carry a real score and used to be
    counted here. `db.TRANSITIONS` gives `prefiltered_out` and `unavailable` no
    onward move whatsoever: neither can ever be surfaced, and counting them
    inflates the very number rollout step 3 reads off. Nothing else is
    excluded: `new` and `fetch_pending` are simply early, and `fetch_failed`
    carries stored grades like any other row, so all three are still counted.
    """
    ids = [
        row["listing_id"]
        for row in db.conn.execute(
            "SELECT DISTINCT g.listing_id FROM criterion_grades g "
            "JOIN listings l ON l.id = g.listing_id "
            f"WHERE l.status NOT IN ({_NEVER_CARDED_SQL}) "
            "ORDER BY g.listing_id"
        )
    ]
    scored = []
    for listing_id in ids:
        result = _adjudicated(db, listing_id, criteria, settings)
        if result.score is None:
            continue
        scored.append(
            Scored(
                score=result.score,
                gated=(
                    result.verdict == "reject"
                    or result.coverage < settings.evaluation.coverage_floor
                ),
            )
        )
    return scored


def _straddle(scored: list[Scored], threshold: float) -> tuple[int, int] | None:
    """How the bucket holding a non-boundary threshold splits, for `bar_lines`.

    None when the threshold sits on a bucket boundary - the bars and the
    score-exact count already agree there, since a bucket boundary is also a
    score boundary - or when nothing was graded into that bucket, so it never
    printed a row for the line to be drawn against.
    """
    ratio = threshold / BUCKET
    if abs(ratio - round(ratio)) < 1e-9:
        return None
    bucket = int(threshold / BUCKET) * BUCKET
    in_bucket = [item.score for item in scored if bucket <= item.score < bucket + BUCKET]
    if not in_bucket:
        return None
    below = sum(1 for score in in_bucket if score < threshold)
    return (below, len(in_bucket) - below)


def weighted_distribution(
    db: Database, criteria: list, settings
) -> list[tuple[float, int]]:
    """The replayed score of every gradable Listing, in the histogram's buckets.

    Every one of them, gated or not: the bars are the shape of the corpus on
    the new scale, and hiding the vetoed Listings would misdescribe that. It
    is the sentence under the bars that has to be about what a cut can
    surface.
    """
    buckets: dict[float, int] = {}
    for item in weighted_scores(db, criteria, settings):
        bucket = int(item.score / BUCKET) * BUCKET
        buckets[bucket] = buckets.get(bucket, 0) + 1
    return sorted(buckets.items())


def _replayed_lines(heading: str, items: list[Replayed]) -> list[str]:
    lines = [f"  {heading}"]
    for item in items[:DISAGREEMENTS_SHOWN]:
        price = f"£{item.price_pcm:,}" if item.price_pcm else "£?"
        lines.append(
            f"    #{item.listing_id}  {item.score:>4.1f}  {item.coverage:>4.0%} read  "
            f"{price}  {item.address}"
        )
    hidden = len(items) - DISAGREEMENTS_SHOWN
    if hidden > 0:
        lines.append(f"    ...and {hidden} more")
    return lines


def weights_report(db: Database, criteria: list, settings) -> str:
    """What the weights in force now would have said. No model call.

    This is the instrument rollout step 3 depends on: it replays a candidate
    weighting over the stored grades and prints the separation, the overlap
    band and the misclassifications.

    Printing `criterion_splits` alone was not enough to run step 3 with.
    That table is computed from the grades and is independent of the weights,
    so editing a weight moved only the printed `w` column - and nothing
    anywhere printed a score on the new scale, while `--decisions` still
    reads the holistic `listings.score`. There was no evidence on which to
    choose `hopeful_threshold` before flipping the flag.

    The attribution stays. It answers a different question - which Criterion
    is doing the work, and which is ranking backwards - and no threshold
    arithmetic can answer that one.
    """
    threshold = settings.evaluation.hopeful_threshold
    floor = settings.evaluation.coverage_floor
    approved, rejected = replay_decisions(db, criteria, settings)
    if not approved and not rejected:
        return (
            "No decision has been made on a graded Listing yet, so there is "
            "nothing to replay a weighting against. Run `flat-scout grade "
            "--backfill` first, then approve or reject a few Listings."
        )

    lines = [
        "Replayed over the stored grades. No model was called.",
        f"Weights total {sum(c.weight for c in criteria):.0f}; "
        f"coverage floor {floor:.0%}.",
    ]

    def say(text: str, indent: str = "  ") -> None:
        lines.append(textwrap.fill(text, WIDTH, initial_indent=indent, subsequent_indent=indent))

    scored = weighted_scores(db, criteria, settings)
    distribution = weighted_distribution(db, criteria, settings)
    if distribution:
        total = len(scored)
        # Counted off the score and not off the bucket it fell in: a cut of
        # 5.3 is cleared by a Listing scoring 5.4, which buckets at 5.0.
        clearing = [item for item in scored if item.score >= threshold]
        above = sum(1 for item in clearing if not item.gated)
        gated = len(clearing) - above
        lines += ["", "Score on the new scale, over every graded Listing:", ""]
        lines += bar_lines(distribution, threshold, _straddle(scored, threshold))
        lines.append("")
        # The bars are every graded Listing; this sentence is only the ones a
        # cut could actually surface. A determined veto rejects whatever the
        # score, and coverage under the floor holds the Verdict at borderline
        # whatever the score - both gate absolutely, independently of the cut,
        # so counting them here would calibrate `hopeful_threshold` against
        # Listings no threshold can reach.
        say(
            f"{above} of {total} clear the configured {threshold:.1f} and could be "
            "surfaced on it."
        )
        if gated:
            say(
                f"A further {gated} score at or above the cut and can never be "
                f"surfaced at any cut: a veto, or coverage under the {floor:.0%} "
                "floor. Both are in the bars and neither is in the count above."
            )
        else:
            say(
                f"A veto, or coverage under the {floor:.0%} floor, would gate a "
                "Listing at any cut. Neither is counted above; no graded Listing "
                "over the cut carries either."
            )

    lines += ["", "Replayed against Decision", ""]
    lines.append(_side_line("approved", [item.score for item in approved]))
    lines.append(_side_line("rejected", [item.score for item in rejected]))
    lines.append("")

    if not approved or not rejected:
        side = "approvals" if approved else "rejections"
        say(
            f"Only {side} so far, so there is nothing to separate. Decisions have "
            "to land on both sides of the same run of Listings before any of this "
            "means anything."
        )
        lines += attribution_lines(criterion_splits(db, criteria))
        return "\n".join(lines)

    gap = _mean([item.score for item in approved]) - _mean(
        [item.score for item in rejected]
    )
    if gap > 0:
        say(f"The approved mean sits {gap:.1f} above the rejected mean.")
    elif gap < 0:
        say(
            f"The approved mean sits {abs(gap):.1f} BELOW the rejected mean. These "
            "weights rank upside down - that is a rubric problem, not a threshold one."
        )
    else:
        say("The two means are identical. These weights are not ranking anything.")

    total_decided = len(approved) + len(rejected)
    if total_decided < SMALL_SAMPLE or min(len(approved), len(rejected)) < THIN_SIDE:
        say(
            f"Small sample: {len(approved)} approved, {len(rejected)} rejected. "
            "Read the rows below, not the averages."
        )

    band = overlap_band(
        [item.score for item in approved], [item.score for item in rejected]
    )
    lines += ["", "Overlap"]
    if band is None:
        say("No overlap: every approval replayed above every rejection.")
    else:
        low, high = band
        inside_approved = [item for item in approved if low <= item.score <= high]
        inside_rejected = [item for item in rejected if low <= item.score <= high]
        say(
            f"{low:.1f} - {high:.1f}, holding {len(inside_approved)} "
            f"approval{'s' if len(inside_approved) != 1 else ''} and "
            f"{len(inside_rejected)} rejection{'s' if len(inside_rejected) != 1 else ''}. "
            "No threshold can separate those under these weights."
        )

    # The gate is not a threshold question. A Listing the coverage floor or a
    # veto removes is hidden at every cut, so it carries no information about
    # where the cut goes - but it is a cost of the floor, and step 3 sets the
    # floor off this report too, so it is counted rather than dropped.
    gated_approved = [item for item in approved if item.gated]
    gated_rejected = [item for item in rejected if item.gated]
    open_approved = [item for item in approved if not item.gated]
    open_rejected = [item for item in rejected if not item.gated]

    lines += ["", "Threshold arithmetic"]
    if gated_approved or gated_rejected:
        say(
            f"{len(gated_approved)} approval{'s' if len(gated_approved) != 1 else ''} and "
            f"{len(gated_rejected)} rejection{'s' if len(gated_rejected) != 1 else ''} are "
            f"gated by the {floor:.0%} coverage floor or a veto, so no cut shows them. "
            "The sums below are over the rest."
        )
    if not open_approved or not open_rejected:
        say(
            "What is left has Decisions on one side only, so no cut can be fitted "
            "to it. Lower the coverage floor, or grade more of the corpus."
        )
        lines += attribution_lines(criterion_splits(db, criteria))
        return "\n".join(lines)

    # `hopeful=False` throughout, and deliberately. `Decided.hopeful` exists
    # because the holistic path surfaces a hopeful Verdict whatever the score,
    # so a threshold has no purchase on one. The weighted path removes that
    # second route: `evaluate_weighted` re-adjudicates the stored Grade vector,
    # and the resulting Verdict is derived from the score, so the only thing
    # standing between a replayed score and being surfaced is the cut and the
    # gate above.
    approved_decided = [Decided(item.score) for item in open_approved]
    rejected_decided = [Decided(item.score) for item in open_rejected]

    suggested = None
    if separates_nothing(approved_decided, rejected_decided):
        say(
            "No cutoff separates these decisions under these weights: hiding every "
            "Listing on score would disagree with no more of them than any cut "
            "inside the data. Try a different weighting before trying a different "
            "cut."
        )
    else:
        suggested = suggest_threshold(approved_decided, rejected_decided)
        lost, shown = threshold_cost(suggested, approved_decided, rejected_decided)
        say(
            f"A cut at {suggested:.1f} loses {lost} of {len(open_approved)} approvals "
            f"and spares {len(open_rejected) - shown} of {len(open_rejected)} rejections."
        )
        if suggested == threshold:
            say(f"That is the configured {threshold:.1f}.")
    if suggested != threshold:
        lost, shown = threshold_cost(threshold, approved_decided, rejected_decided)
        say(
            f"The configured {threshold:.1f} loses {lost} of {len(open_approved)} "
            f"approvals and spares {len(open_rejected) - shown} of "
            f"{len(open_rejected)} rejections."
        )

    # Named, not only counted. A count says how wrong the cut is; the rows say
    # what it is wrong about, and that is what sends a reader to a rubric.
    missed = [item for item in open_approved if item.score < threshold]
    let_through = [item for item in open_rejected if item.score >= threshold]
    if missed or let_through:
        lines += ["", f"Misclassifications at the configured {threshold:.1f}"]
        if let_through:
            lines += _replayed_lines("Rejected, but replayed above the cut", let_through)
        if missed:
            lines += _replayed_lines(
                "Approved, but replayed below the cut", missed[::-1]
            )

    lines += attribution_lines(criterion_splits(db, criteria))
    return "\n".join(lines)


@dataclass(frozen=True)
class Shift:
    """One decided Listing, as it was scored then and as the weights say now.

    Two numbers on two scales. `was_score` is the holistic model's judgement of
    the whole flat, frozen on the row and never rewritten; `now_score` is a
    weighted mean of stored Grades. Nothing subtracts them, and nothing should:
    the difference between a judgement and a mean is not a movement, and a
    reader given one would act on it.

    The comparison that does mean something is `surfaced` against the fact
    that every Listing here was surfaced once - that is how it got a Decision
    at all.
    """

    listing_id: int
    was_score: float | None
    was_verdict: str | None
    now_score: float | None
    now_coverage: float
    now_verdict: str
    # The Criteria whose veto fired, by name. A Verdict of `reject` says a flat
    # was disqualified and not which rubric did it, and those are different
    # edits.
    vetoed_by: tuple[str, ...]
    address: str
    price_pcm: int | None

    @property
    def surfaced(self) -> bool:
        """Whether this would reach them at all under the weighted path.

        `pipeline.evaluate_weighted` derives the current Verdict from the
        stored Grade vector when `weighted_criteria` is on. There is no second
        route on the score, the way the holistic path has one, so this is the
        whole rule.
        """
        return self.now_verdict == "hopeful"


def blockers(shift: Shift, threshold: float, floor: float) -> list[str]:
    """Why this Listing would not be surfaced, in the reader's own terms.

    Empty when it would be. Three different problems live in one Verdict and
    they have three different remedies: a veto sends you to the rubric that
    fired, a thin read to the coverage floor or to what the Portal published,
    and a low score to the weights. All of them are listed, not just the first
    - a Listing under the floor AND under the cut is not rescued by moving the
    floor, and a report naming one blocker at a time would have somebody find
    that out by re-running it.
    """
    if shift.surfaced:
        return []
    reasons = []
    if shift.vetoed_by:
        reasons.append(f"a veto: {', '.join(shift.vetoed_by)}")
    if shift.now_coverage < floor:
        reasons.append(
            f"coverage {shift.now_coverage:.0%} under the {floor:.0%} floor"
        )
    if shift.now_score is None:
        # Not a bad flat: a flat nobody could read. Printed as such, because
        # 0.0 would file it beside a genuine failure.
        reasons.append("nothing could be graded")
    elif shift.now_score < threshold:
        reasons.append(f"score {shift.now_score:.1f} under the {threshold:.1f} cut")
    return reasons


def decided_shifts(
    db: Database, criteria: list, settings
) -> tuple[list[Shift], list[Shift], int]:
    """Every decided Listing replayed: (approved, passed, ungraded count).

    The sides come off the Decision, not the status.

    A Listing with no stored Grades at all cannot be replayed and is counted
    rather than dropped: a reader has to know the report is short before
    trusting anything in it.

    No model is called. `graded_pairs` is SQL and `adjudicate` is arithmetic.
    """
    rows = decided_rows(db, "id, score, verdict, address, postcode, price_pcm, status")

    approved: list[Shift] = []
    passed: list[Shift] = []
    ungraded = 0
    for row in rows:
        if not db.criterion_grades(row["id"]):
            ungraded += 1
            continue
        pairs = db.graded_pairs(row["id"], criteria)
        result = adjudicate(
            pairs,
            hopeful_threshold=settings.evaluation.hopeful_threshold,
            coverage_floor=settings.evaluation.coverage_floor,
            unknown_prior=settings.evaluation.unknown_prior,
        )
        shift = Shift(
            listing_id=row["id"],
            was_score=row["score"],
            was_verdict=row["verdict"],
            now_score=result.score,
            now_coverage=result.coverage,
            now_verdict=result.verdict,
            vetoed_by=tuple(c.name for c, _ in vetoes_fired(pairs)),
            address=row["address"] or row["postcode"] or "?",
            price_pcm=row["price_pcm"],
        )
        (passed if row["status"] == "rejected" else approved).append(shift)

    # Highest stored score first: the strongest flat they lost sits at the top.
    # Deliberately NOT by how far the score travelled - see `Shift`.
    def order(shift: Shift) -> tuple[float, int]:
        return (-(shift.was_score if shift.was_score is not None else -1), shift.listing_id)

    return sorted(approved, key=order), sorted(passed, key=order), ungraded


def _shift_lines(shift: Shift, threshold: float, floor: float) -> list[str]:
    price = f"£{shift.price_pcm:,}" if shift.price_pcm else "£?"
    was = f"{shift.was_score:.1f}" if shift.was_score is not None else " ? "
    now = f"{shift.now_score:.1f}" if shift.now_score is not None else " ? "
    lines = [
        f"    #{shift.listing_id}  was {was} {shift.was_verdict or '?':<10}  "
        f"now {now} {shift.now_coverage:>4.0%} read {shift.now_verdict:<10}  "
        f"{price}  {shift.address}"
    ]
    reasons = blockers(shift, threshold, floor)
    if reasons:
        lines.append(f"        held back by {'; '.join(reasons)}")
    return lines


def shift_report(db: Database, criteria: list, settings) -> str:
    """What the weighted brief would do to the flats already decided on.

    The one question worth asking before the flag is deployed to people who
    have been deciding on Listings for a fortnight: would a flat they wanted
    still have reached them, and would a flat they threw away still be
    filtered out.

    It leads with the first, because the two errors are not symmetrical. A
    Listing that should have been hidden costs three seconds. A Listing that
    should have been surfaced and was not is never seen at all, so nobody can
    correct it - and it is invisible in every other view in this module, all
    of which read the Listings that WERE surfaced.
    """
    threshold = settings.evaluation.hopeful_threshold
    floor = settings.evaluation.coverage_floor
    approved, passed, ungraded = decided_shifts(db, criteria, settings)

    lines: list[str] = []

    def say(text: str, indent: str = "  ") -> None:
        lines.append(textwrap.fill(text, WIDTH, initial_indent=indent, subsequent_indent=indent))

    def note_ungraded() -> None:
        """How much of the decided corpus this report cannot speak for.

        Printed before the findings rather than after them. A count that
        arrives at the foot of a report has already been acted on.
        """
        if not ungraded:
            return
        lines.append("")
        say(
            f"{ungraded} decided Listing{'s carry' if ungraded != 1 else ' carries'} "
            "no stored Grades and cannot be replayed at all, so nothing below "
            f"speaks for {'them' if ungraded != 1 else 'it'}. Run `flat-scout "
            "grade --backfill` first if this report has to be complete.",
            "",
        )

    if not approved and not passed:
        lines.append(
            textwrap.fill(
                "No decision has been made on a graded Listing yet, so there is "
                "nothing to replay the weighted brief against. Run `flat-scout "
                "grade --backfill` first, then approve or reject a few Listings.",
                WIDTH,
            )
        )
        note_ungraded()
        return "\n".join(lines)

    total = len(approved) + len(passed)
    lines += [
        "What the weighted brief would do to the flats already decided on.",
        "Replayed over the stored Grades. No model was called.",
        f"{total} Decisions replayed; cut {threshold:.1f}; "
        f"coverage floor {floor:.0%}.",
    ]
    note_ungraded()
    lines.append("")
    # Said once, plainly, and before any number is read. The old Score came off
    # a model asked to judge a whole flat; the new one is a weighted mean of
    # Grades. Subtracting them would produce a tidy figure measuring nothing,
    # so none is printed and nothing is sorted by one.
    say(
        "The two scores are on different scales - the old one is the model's "
        "holistic judgement, the new one a weighted mean of Grades - so the "
        "distance between them is not a movement and none is printed. The "
        "comparison to trust is whether the flat is in or out.",
        "",
    )

    hidden = [shift for shift in approved if not shift.surfaced]
    lines += ["", f"Approved, but the weighted brief would hide them "
              f"({len(hidden)} of {len(approved)})"]
    if not approved:
        say("Nothing has been approved yet, so there is none of this to find.")
    elif hidden:
        say(
            "They wanted these. Under the weights in force now none of them would "
            "have been surfaced, so there would have been nothing to say yes to - "
            "and nothing to notice was missing."
        )
        lines.append("")
        # Uncapped, unlike every other list in this module. This is the finding
        # the report exists for, and a count with the rows trimmed off it is
        # exactly the summary the report is meant to replace.
        for shift in hidden:
            lines += _shift_lines(shift, threshold, floor)
    else:
        say("None. Every flat they approved would still be surfaced.")

    shown = [shift for shift in passed if shift.surfaced]
    lines += ["", f"Passed, but the weighted brief would still show them "
              f"({len(shown)} of {len(passed)})"]
    if not passed:
        say("Nothing has been passed yet, so there is none of this to count.")
    elif shown:
        say(
            "Noise, not a failure: a Listing they would dismiss again in three "
            "seconds. Worth knowing the size of, not worth acting on before the "
            "list above is empty."
        )
        lines.append("")
        _capped(lines, shown, lambda shift: _shift_lines(shift, threshold, floor))
    else:
        say("None. Every flat they passed would stay filtered out.")

    if approved and passed:
        lines.append("")
        say(
            f"{len(approved) - len(hidden)} of {len(approved)} approvals would "
            f"still be shown, and {len(passed) - len(shown)} of {len(passed)} "
            "passes would still be hidden.",
            "",
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class Rescored:
    """One Listing, judged twice: by the stored brief and by the current one."""

    listing_id: int
    was: float | None
    now: float | None
    was_verdict: str | None
    now_verdict: str
    address: str
    postcode: str | None
    price_pcm: int | None
    reasons: list[str]

    @property
    def moved(self) -> float | None:
        """How far the score travelled, or None where there is now no score.

        `now` is `float | None`: the weighted path can regrade a Listing down
        to no score at all when no Criterion could be determined.
        """
        return None if self.now is None else self.now - (self.was or 0.0)


# Below this, a difference is not worth a reader's attention: half a point is
# the smallest step the model appears to use, so anything under it is a tie.
#
# Not a noise floor. Six runs of one Listing against one brief returned the
# same score six times, so repeated evaluation looks close to deterministic
# here - which makes this cut conservative rather than corrective. It is a
# reading aid, and a rescore that reports "0 moved" has genuinely changed
# nothing rather than merely hidden the wobble.
MOVED = 0.5


def crossings(changes: list[Rescored], threshold: float) -> tuple[int, int]:
    """(Listings that would now be shown, Listings that would now be hidden).

    `change.now` can be None - the weighted path can regrade a Listing down to
    no score at all - and a Listing that crossed nothing cannot be counted on
    either side, so those are skipped rather than compared against None.
    """
    return (
        sum(
            1
            for change in changes
            if change.now is not None
            and not would_be_shown(Decided(change.was or 0.0, change.was_verdict == "hopeful"), threshold)
            and would_be_shown(Decided(change.now, change.now_verdict == "hopeful"), threshold)
        ),
        sum(
            1
            for change in changes
            if change.now is not None
            and would_be_shown(Decided(change.was or 0.0, change.was_verdict == "hopeful"), threshold)
            and not would_be_shown(Decided(change.now, change.now_verdict == "hopeful"), threshold)
        ),
    )


def rescore_report(changes: list[Rescored], threshold: float) -> str:
    """What the brief in force now would have made of the Listings on hand.

    Sorted by how far each moved, because the question a brief edit asks is
    which Listings it acted on - not which of them scores highest.
    """
    lines: list[str] = []

    def say(text: str, indent: str = "  ") -> None:
        lines.append(textwrap.fill(text, WIDTH, initial_indent=indent, subsequent_indent=indent))

    if not changes:
        return "Nothing to rescore - no Listing in the database carries a score."

    # `now` can be None on the weighted path - a regrade that lands on no
    # scored Criterion at all - and `.moved` subtracts from it, so those are
    # excluded here rather than left to crash the sort below.
    moved = sorted(
        (change for change in changes if change.now is not None and abs(change.moved) >= MOVED),
        key=lambda change: -abs(change.moved),
    )
    say(f"Rescored {len(changes)} Listings against the current brief. Nothing was written.", "")
    say(
        f"{len(moved)} moved by {MOVED} or more. Anything smaller is left out: two "
        "runs of the same model over the same brief do not agree to the decimal, "
        "and reporting that as a change would credit the edit with noise."
    )

    gained, lost = crossings(changes, threshold)
    if gained or lost:
        say(
            f"At the configured threshold of {threshold:.1f}, {gained} would newly "
            f"be surfaced and {lost} would newly be held back."
        )
    else:
        say(f"No Listing crosses the configured threshold of {threshold:.1f} either way.")

    if moved:
        lines.append("")
        for change in moved:
            price = f"£{change.price_pcm:,}" if change.price_pcm else "£?"
            was = f"{change.was:.1f}" if change.was is not None else " ? "
            arrow = "up" if change.moved > 0 else "down"
            lines.append(
                f"    #{change.listing_id}  {was} -> {change.now:.1f}  "
                f"({arrow} {abs(change.moved):.1f})  {price}  {change.address}"
            )
            lines.extend(f"        {reason}" for reason in change.reasons)
    return "\n".join(lines)


@dataclass(frozen=True)
class Backfilled:
    listing_id: int
    coverage: float
    calls: int
    regraded: list[str]


def backfill_report(done: list[Backfilled]) -> str:
    """What the backfill did, and what it spent.

    The coverage distribution is the useful half: it is what decides
    `coverage_floor`, and a corpus where most Listings sit at 30 per cent says
    the floor has to be low or the Criteria have to lean on fewer fields.
    """
    if not done:
        return "Nothing to grade - no Listing in the database."
    lines = [
        f"Graded {len(done)} Listings. {sum(row.calls for row in done)} model calls.",
        "",
        "Coverage:",
    ]
    for low in (0.0, 0.25, 0.5, 0.75):
        count = sum(1 for row in done if low <= row.coverage < low + 0.25)
        lines.append(f"  {low:.0%}-{low + 0.25:.0%}  {'#' * count} {count}")
    unchanged = sum(1 for row in done if not row.regraded)
    if unchanged:
        lines += ["", f"{unchanged} were already graded against the current rubrics."]
    return "\n".join(lines)


@dataclass(frozen=True)
class CriterionSpread:
    slug: str
    name: str
    runs: int
    # How many Listings this measurement was taken over. The same for every
    # row in one `measure_criteria` call - carried per-row rather than
    # threaded through as a second return value, so a caller with one row in
    # hand already knows the size of the corpus behind it.
    sampled: int
    unknown_rate: float
    # The widest gap between the highest and lowest grade any one Listing got
    # across the runs. Anchors the model can find produce a small number here.
    worst_spread: float


def measure_report(
    spreads: list[CriterionSpread], aspect_agreement: tuple[int, int] | None
) -> str:
    sampled = spreads[0].sampled if spreads else 0
    runs = spreads[0].runs if spreads else 0
    lines = [
        f"{sampled} Listings sampled, {runs} runs each, over the Criteria that "
        "can reach a model.",
        "",
    ]
    if sampled == 0:
        lines.append(
            "No Listings have reached the evaluator yet, so there is nothing to measure."
        )
    else:
        for row in sorted(spreads, key=lambda r: r.worst_spread, reverse=True):
            lines.append(
                f"  {row.name:<32} unknown {row.unknown_rate:>4.0%}   "
                f"widest disagreement {row.worst_spread:>4.1f}"
            )
        shaky = [row for row in spreads if row.worst_spread >= 3]
        if shaky:
            lines += ["", textwrap.fill(
                f"{', '.join(row.name for row in shaky)} disagreed with itself by 3 points "
                "or more on the same Listing. The rubric has no anchor the model can find, "
                "so the weight is buying noise. Rewrite the rubric before changing the weight.",
                WIDTH,
            )]
        empty = [row for row in spreads if row.unknown_rate >= 0.9]
        if empty:
            lines += ["", textwrap.fill(
                f"{', '.join(row.name for row in empty)} was unknown on nearly every Listing. "
                "Its weight is being redistributed to everything else, quietly. Either the "
                "rubric asks for something adverts never state, or it wants a deterministic "
                "field it is not reading.",
                WIDTH,
            )]

    lines.append("")
    if aspect_agreement is None:
        lines.append("Aspect: no annotated plan overlaps the graded Listings.")
    else:
        agreed, total = aspect_agreement
        lines.append(
            f"Aspect: the Criterion graded {agreed} of {total} annotated plans the "
            "same off the human's reading as off the pipeline's."
        )
    return "\n".join(lines)


def fit_report(fit, criteria) -> str:
    """The `flat-scout fit` table: sigma, categorical prior and data per Criterion."""
    from flat_scout.fit import GRADE_BINS

    lines = [f"Fitted {fit.fitted_at} over hash {fit.input_hash}.", ""]
    by_slug = {criterion.slug: criterion for criterion in criteria}
    for slug in sorted(fit.criteria):
        entry = fit.criteria[slug]
        name = by_slug[slug].name if slug in by_slug else slug
        prior_mean = sum(b * w for b, w in zip(GRADE_BINS, entry.prior_weights))
        lines.append(
            f"  {name:<32} sigma {entry.sigma:>4.1f} "
            f"({entry.n_runs:>3} runs)   prior mean {prior_mean:>4.1f} "
            f"({entry.n_grades:>3} grades)"
        )
        lines.append(
            "    prior "
            + " ".join(
                f"{grade:g}:{weight:.0%}"
                for grade, weight in zip(GRADE_BINS, entry.prior_weights)
                if weight > 0
            )
        )
    lines += ["", f"Pooled sigma {fit.pooled_sigma:.1f} - the answer for an unmeasured Criterion."]
    return "\n".join(lines)


# The gates the grid replays. `hopeful_threshold` and `hopeful_confidence`
# are set off this table, so it brackets the plausible range rather than
# searching it: a reader picks a row. The configured pair is folded in at
# read time, so there is always the row in force to compare the rest against.
GRID_THRESHOLDS = (6.0, 6.5, 7.0, 7.5)
GRID_CONFIDENCES = (0.5, 0.6, 0.7, 0.8)


@dataclass(frozen=True)
class Chance:
    """One decided Listing's P(S >= t) at one candidate threshold.

    A different threshold is a different question, so a Chance is only good
    for the `t` it was drawn at - which is why the grid re-scores every
    Listing per threshold instead of reading one posterior at four cuts.
    """

    listing_id: int
    approved: bool
    # A veto rejects whatever the posterior says, so a vetoed Listing is
    # hidden at every pair in the grid. Carried beside `p_hopeful` rather
    # than folded into it, because the two are not the same failure: a low
    # probability is an argument about the gate, and a veto is not.
    vetoed: bool
    p_hopeful: float

    def would_surface(self, confidence: float) -> bool:
        """`adjudicate_bayesian`, reduced to the question surfacing turns on."""
        return not self.vetoed and self.p_hopeful >= confidence


def decided_chances(
    db: Database,
    rows: list[sqlite3.Row],
    criteria: list,
    fit,
    threshold: float,
    settings,
) -> list[Chance]:
    """Every decided Listing's posterior at one candidate threshold.

    The seed is the Listing id, the same one `adjudicate_current` draws with,
    so a row here reproduces the number the Listing was surfaced on.
    """
    from flat_scout.posterior import score_posterior

    chances = []
    for row in rows:
        pairs = db.graded_pairs(row["id"], criteria)
        posterior = score_posterior(pairs, fit, threshold=threshold, seed=row["id"])
        chances.append(
            Chance(
                listing_id=row["id"],
                approved=row["status"] != "rejected",
                vetoed=bool(vetoes_fired(pairs)),
                p_hopeful=posterior.p_hopeful,
            )
        )
    return chances


def gate_cost(chances: list[Chance], confidence: float) -> tuple[int, int, int]:
    """(surfaced, approvals missed, rejections surfaced) under this pair.

    The two errors are never summed here. A rejection surfaced costs three
    seconds; an approval missed is a flat they wanted that was never surfaced
    at all, so nobody could correct it. The report adds them for one column
    and prints both beside it.
    """
    surfaced = [chance for chance in chances if chance.would_surface(confidence)]
    missed = sum(
        1
        for chance in chances
        if chance.approved and not chance.would_surface(confidence)
    )
    return len(surfaced), missed, sum(1 for chance in surfaced if not chance.approved)


def _gate_rows(
    chances_by_threshold: dict[float, list[Chance]],
    confidences: list[float],
    live: tuple[float, float],
) -> list[str]:
    """The grid, one row per candidate pair, blank line between thresholds."""
    lines = [
        f"  {'gate':<14}{'surfaced':>8}   {'approved missed':>15}   "
        f"{'rejected surfaced':>17}   {'misclassified':>13}"
    ]
    for threshold in sorted(chances_by_threshold):
        for confidence in confidences:
            surfaced, missed, wasted = gate_cost(
                chances_by_threshold[threshold], confidence
            )
            gate = f"t={threshold:.1f} p={confidence:.2f}"
            lines.append(
                f"{'* ' if (threshold, confidence) == live else '  '}{gate:<14}"
                f"{surfaced:>8}   {missed:>15}   {wasted:>17}   {missed + wasted:>13}"
            )
        lines.append("")
    return lines


def best_gate(
    chances_by_threshold: dict[float, list[Chance]], confidences: list[float]
) -> tuple[float, float, int]:
    """The pair disagreeing with the fewest Decisions, and by how many.

    Ties break towards the lower confidence and then the lower threshold -
    towards surfacing a flat they would have rejected rather than hiding one
    they wanted, the same way `suggest_threshold` breaks them.
    """
    scored = [
        (sum(gate_cost(chances, confidence)[1:]), threshold, confidence)
        for threshold, chances in sorted(chances_by_threshold.items())
        for confidence in confidences
    ]
    wrong, threshold, confidence = min(scored, key=lambda row: (row[0], row[2], row[1]))
    return threshold, confidence, wrong


def posterior_report(db: Database, settings, criteria: list) -> str:
    """The calibration instrument for the Bayesian gate. No model call.

    `hopeful_threshold` and `hopeful_confidence` are a pair, and neither
    means anything alone: a high cut read at low confidence surfaces the same
    flats as a low cut read at high confidence. `report --weights` can only
    speak for the first of them, because a weighted mean has nowhere to put
    the second - so both constants are set off this grid, which replays every
    candidate pair over the stored Grades and the Decisions already taken and
    prints what each pair would have surfaced.

    Arithmetic over stored rows throughout. `graded_pairs` is SQL and
    `score_posterior` is numpy over the last completed Fit. `flat-scout fit`
    owns refitting, so a pair can be tried, read and tried again as often as
    it takes without starting PyMC here.
    """
    from flat_scout.fit import load_fit

    rows = decided_rows(db, "id, status")
    graded = [row for row in rows if db.criterion_grades(row["id"])]
    if not graded:
        return (
            "No decision has been made on a graded Listing yet, so there is "
            "nothing to replay a gate against. Run `flat-scout grade "
            "--backfill` first, then approve or reject a few Listings."
        )

    threshold = settings.evaluation.hopeful_threshold
    confidence = settings.evaluation.hopeful_confidence
    fit = load_fit(db, criteria)
    chances_by_threshold = {
        candidate: decided_chances(db, graded, criteria, fit, candidate, settings)
        for candidate in sorted({*GRID_THRESHOLDS, threshold})
    }
    confidences = sorted({*GRID_CONFIDENCES, confidence})
    live = chances_by_threshold[threshold]
    approved = [chance for chance in live if chance.approved]
    rejected = [chance for chance in live if not chance.approved]

    lines = ["Replayed over the stored grades. No model was called."]

    def say(text: str, indent: str = "  ") -> None:
        lines.append(textwrap.fill(text, WIDTH, initial_indent=indent, subsequent_indent=indent))

    # The effective sample size, printed before the findings rather than
    # after them. A count that arrives at the foot of a report has already
    # been acted on.
    lines.append(
        f"Decisions behind this: {len(live)} "
        f"({len(approved)} approved, {len(rejected)} rejected)."
    )
    ungraded = len(rows) - len(graded)
    if ungraded:
        say(
            f"A further {ungraded} decided Listing"
            f"{'s have' if ungraded != 1 else ' has'} no stored Grades and cannot be "
            "replayed at all. Run `flat-scout grade --backfill` to bring "
            f"{'them' if ungraded != 1 else 'it'} in."
        )
    if len(live) < SMALL_SAMPLE or min(len(approved), len(rejected)) < THIN_SIDE:
        say(
            "Small sample: read the counts as a direction and not as a rate. Two "
            "pairs one Decision apart are not told apart by this much evidence."
        )

    lines += ["", "Every candidate gate, replayed", ""]
    lines += _gate_rows(chances_by_threshold, confidences, (threshold, confidence))
    say(f"* the pair in force: t={threshold:.1f}, p={confidence:.2f}.")

    # A veto is absolute, so it is a floor under the columns above: no pair
    # lifts it, and a reader comparing rows has to know which part of the
    # count cannot move.
    vetoed_approved = sum(1 for chance in approved if chance.vetoed)
    vetoed_rejected = sum(1 for chance in rejected if chance.vetoed)
    if vetoed_approved:
        say(
            f"{vetoed_approved} of the approvals carr"
            f"{'y' if vetoed_approved != 1 else 'ies'} a veto and "
            f"{'are' if vetoed_approved != 1 else 'is'} hidden at every pair above. "
            "That count is a floor under `approved missed`, and only a rubric "
            "moves it."
        )
    if vetoed_rejected:
        say(
            f"{vetoed_rejected} of the rejections carr"
            f"{'y' if vetoed_rejected != 1 else 'ies'} a veto, so no pair above "
            "surfaces "
            f"{'them' if vetoed_rejected != 1 else 'it'}. Those are agreeing with the "
            "couple at every gate, and none of them is evidence about the gate."
        )

    if approved and rejected:
        pick_threshold, pick_confidence, wrong = best_gate(
            chances_by_threshold, confidences
        )
        say(
            f"Fewest disagreements at t={pick_threshold:.1f} p={pick_confidence:.2f}: "
            f"{wrong} of {len(live)}. Ties break towards surfacing a flat they would "
            "have rejected rather than hiding one they wanted."
        )

    lines += ["", "Calibration at the live gate"]
    if not approved or not rejected:
        side = "approvals" if approved else "rejections"
        say(
            f"Only {side} so far, so there is nothing to hold the posterior "
            "against. Decisions have to land on both sides of the same run of "
            "Listings before any of this means anything."
        )
        return "\n".join(lines)

    def mean_p(chances: list[Chance]) -> float:
        return sum(chance.p_hopeful for chance in chances) / len(chances)

    say(
        f"At t={threshold:.1f}, approved flats sat at p(hopeful) "
        f"{mean_p(approved):.0%} on average, rejected at {mean_p(rejected):.0%}."
    )
    say(
        f"{sum(1 for c in approved if c.would_surface(confidence))} of {len(approved)} "
        f"approvals clear the configured p={confidence:.2f}; "
        f"{sum(1 for c in rejected if c.would_surface(confidence))} of {len(rejected)} "
        "rejections do."
    )
    # Named and diagnosed under this gate. `report --shift` deliberately
    # replays the point-score threshold and coverage floor, so sending a
    # Bayesian miss there would explain it with rules this gate does not use.
    missed = [
        chance.listing_id
        for chance in approved
        if not chance.would_surface(confidence)
    ]
    if missed:
        by_id = {chance.listing_id: chance for chance in approved}
        diagnoses = []
        for listing_id in missed[:DISAGREEMENTS_SHOWN]:
            chance = by_id[listing_id]
            reason = (
                "veto"
                if chance.vetoed
                else f"p(hopeful) {chance.p_hopeful:.0%} below the {confidence:.0%} gate"
            )
            diagnoses.append(f"#{listing_id} ({reason})")
        shown = ", ".join(diagnoses)
        more = len(missed) - DISAGREEMENTS_SHOWN
        say(
            f"The approvals it would miss: {shown}"
            f"{f', and {more} more' if more > 0 else ''}."
        )
    if mean_p(approved) <= mean_p(rejected):
        say(
            "The rejections sit at least as high as the approvals. No pair of "
            "constants fixes that: the posterior is ranking upside down, which is "
            "a rubric or a weight problem."
        )
    return "\n".join(lines)
