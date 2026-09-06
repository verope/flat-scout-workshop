"""`report --shift`: what the weighted brief would do to settled Decisions.

The switch to weighted Criteria changes which flats are surfaced. The couple
have already decided on a run of flats, and the only honest way to ask whether
the new brief is safe to deploy is to replay it over those Decisions and see
who it would have hidden.

The dangerous direction is one-way. A flat they APPROVED that the weights
would hide never reaches them at all - they cannot recover from it, because
they never see it. A flat they PASSED that the weights still show costs them
three seconds. So the first is named, and the second is counted.
"""

from __future__ import annotations

from pathlib import Path

from flat_scout.adjudicate import Grade
from flat_scout.config import Settings
from flat_scout.criteria import Criterion, load_criteria
from flat_scout.db import Database
from flat_scout.graders import GRADERS
from flat_scout.models import ListingData
from flat_scout.report import decided_shifts, shift_report

CRITERIA_DIR = Path(__file__).resolve().parent.parent / "criteria"


def a_criterion(slug: str, weight: float = 1, veto_at_or_below: float | None = None):
    return Criterion(
        slug=slug, name=slug.capitalize(), description="d", weight=weight,
        grade={"model": True}, body="b", rubric_hash="h",
        veto_at_or_below=veto_at_or_below,
    )


PAIR = [a_criterion("layout"), a_criterion("light")]


def a_listing(
    db: Database,
    portal_id: str,
    *,
    status: str,
    score: float | None = 6.0,
    verdict: str | None = "hopeful",
    grades: dict[str, float | None] | None = None,
    decided: bool = True,
    price_pcm: int | None = 2400,
) -> int:
    """One Listing with a frozen holistic score and an optional grade vector.

    `grades` maps slug to value; a None value stores an undetermined Grade,
    which is how a Criterion that read nothing is spelled.
    """
    listing_id = db.upsert(
        ListingData(portal="rightmove", portal_id=portal_id, url=f"u{portal_id}"),
        source="manual",
    )
    db.conn.execute(
        "UPDATE listings SET status = ?, decided_at = ?, score = ?, verdict = ?, "
        "address = ?, price_pcm = ? WHERE id = ?",
        (
            status,
            "2026-08-17" if decided else None,
            score,
            verdict,
            f"{portal_id} Somewhere Street",
            price_pcm,
            listing_id,
        ),
    )
    db.conn.commit()
    if grades:
        db.set_criterion_grades(
            listing_id,
            [
                (a_criterion(slug), Grade(slug, value, value is not None, f"{slug} read"))
                for slug, value in grades.items()
            ],
        )
    return listing_id


def settings_with(floor: float = 0.3, threshold: float = 5.0) -> Settings:
    settings = Settings()
    settings.evaluation.coverage_floor = floor
    settings.evaluation.hopeful_threshold = threshold
    return settings


def row_for(text: str, listing_id: int) -> str:
    """The row for one Listing plus everything indented under it."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith(f"#{listing_id} "):
            block = [line]
            for following in lines[index + 1 :]:
                if not following.startswith("        "):
                    break
                block.append(following)
            return "\n".join(block)
    return ""


def test_an_approved_listing_the_weights_would_hide_leads_the_report(tmp_path):
    """The false negative is the finding, and it is named, not counted.

    They wanted this flat. Under the new brief it would never have been
    surfaced, so no Decision on it would ever have been possible.
    """
    db = Database(tmp_path / "flats.db")
    a_listing(db, "1", status="approved", score=8.0, grades={"layout": 2, "light": 2})
    a_listing(
        db, "2", status="rejected", score=3.0, verdict="borderline",
        grades={"layout": 9, "light": 9},
    )
    text = shift_report(db, PAIR, settings_with())

    assert "1 Somewhere Street" in text
    would_hide = text.index("Approved, but")
    still_shows = text.index("Passed, but")
    assert would_hide < still_shows, text
    assert text.index("#1") < text.index("#2"), text


def test_a_veto_a_thin_read_and_a_low_score_are_told_apart(tmp_path):
    """Three different problems with three different remedies.

    A veto sends you to the rubric that fired it, a thin read sends you to the
    coverage floor or to what the Portal published, and a low score sends you
    to the weights. Reporting all three as "not hopeful" would tell a reader
    to fix nothing in particular.
    """
    criteria = [
        a_criterion("layout", weight=6),
        a_criterion("pets", weight=2, veto_at_or_below=1),
        a_criterion("quiet", weight=2),
    ]
    db = Database(tmp_path / "flats.db")
    a_listing(db, "1", status="approved", grades={"layout": 9, "pets": 0, "quiet": 8})
    a_listing(db, "2", status="approved", grades={"layout": 9})
    a_listing(db, "3", status="approved", grades={"layout": 2, "pets": 8, "quiet": 2})
    text = shift_report(db, criteria, settings_with(floor=0.7, threshold=5.0))

    assert "veto" in row_for(text, 1).lower()
    assert "Pets" in row_for(text, 1)
    assert "coverage" in row_for(text, 2).lower()
    assert "veto" not in row_for(text, 2).lower()
    assert "score" in row_for(text, 3).lower()
    assert "coverage" not in row_for(text, 3).lower()


def test_a_listing_gated_by_both_the_floor_and_the_cut_says_both(tmp_path):
    """Lowering the floor alone would not bring this one back.

    Naming only the first blocker would have a reader drop the floor, re-run
    the report and find the flat still hidden.
    """
    criteria = [a_criterion("layout", weight=6), a_criterion("quiet", weight=4)]
    db = Database(tmp_path / "flats.db")
    a_listing(db, "1", status="approved", grades={"layout": 2})
    text = shift_report(db, criteria, settings_with(floor=0.7, threshold=5.0))

    row = row_for(text, 1).lower()
    assert "coverage" in row
    assert "score" in row


def test_a_passed_listing_the_weights_would_still_show_is_counted(tmp_path):
    """Noise, not a failure. Counted, and never above the false negatives."""
    db = Database(tmp_path / "flats.db")
    a_listing(
        db, "1", status="rejected", score=3.0, verdict="borderline",
        grades={"layout": 9, "light": 9},
    )
    a_listing(
        db, "2", status="rejected", score=2.0, verdict="borderline",
        grades={"layout": 1, "light": 1},
    )
    text = shift_report(db, PAIR, settings_with())

    assert "Passed, but the weighted brief would still show them (1 of 2)" in text
    assert "#1" in text


def test_a_decided_listing_with_no_grades_is_counted_and_sends_you_to_the_backfill(
    tmp_path,
):
    """Silently reporting on the graded subset would understate the risk.

    A Listing with no stored Grades cannot be replayed at all, so it is
    neither shown nor hidden - it is simply missing, and a reader has to know
    the report is short before trusting the counts in it.
    """
    db = Database(tmp_path / "flats.db")
    a_listing(db, "1", status="approved", grades={"layout": 8, "light": 8})
    a_listing(db, "2", status="approved", grades=None)
    a_listing(db, "3", status="rejected", grades=None)
    text = shift_report(db, PAIR, settings_with())

    assert "2 decided Listings carry no stored Grades" in text
    assert "grade --backfill" in text


def test_a_listing_nothing_could_be_determined_on_is_not_surfaced_and_not_zero(
    tmp_path,
):
    """No score is not a bad score.

    Printing 0.0 would rank a flat nobody could read below one that read
    badly, and would put it in the same bucket as a genuine failure.
    """
    db = Database(tmp_path / "flats.db")
    a_listing(db, "1", status="approved", score=8.0, grades={"layout": None, "light": None})
    text = shift_report(db, PAIR, settings_with())

    row = row_for(text, 1)
    assert "0.0" not in row, row
    assert "nothing could be graded" in row.lower()
    approved, _, _ = decided_shifts(db, PAIR, settings_with())
    assert approved[0].now_score is None
    assert approved[0].surfaced is False


def test_every_approval_lands_on_the_approved_side(tmp_path):
    """The side comes off the Decision, not off a single status literal.

    `decided_shifts` reads `_APPROVED` and `_REJECTED`, so anything carrying a
    Decision that is not a rejection is an approval - and none of them is
    quietly dropped on the way into the replay.
    """
    db = Database(tmp_path / "flats.db")
    a_listing(db, "1", status="approved", grades={"layout": 1, "light": 1})
    a_listing(db, "2", status="approved", grades={"layout": 1, "light": 1})
    approved, passed, _ = decided_shifts(db, PAIR, settings_with())

    assert [shift.listing_id for shift in approved] == [1, 2]
    assert passed == []


def test_an_undecided_listing_is_on_neither_side(tmp_path):
    """The report is about Decisions. A Listing nobody decided on is not one."""
    db = Database(tmp_path / "flats.db")
    a_listing(
        db, "1", status="evaluated", decided=False, grades={"layout": 1, "light": 1}
    )
    approved, passed, ungraded = decided_shifts(db, PAIR, settings_with())

    assert (approved, passed, ungraded) == ([], [], 0)


def test_no_decisions_at_all_says_so_rather_than_printing_zeroes(tmp_path):
    db = Database(tmp_path / "flats.db")
    text = shift_report(db, PAIR, settings_with())
    assert "no decision" in text.lower()


def test_decisions_on_one_side_only_says_so(tmp_path):
    db = Database(tmp_path / "flats.db")
    a_listing(db, "1", status="approved", grades={"layout": 8, "light": 8})
    text = shift_report(db, PAIR, settings_with())
    assert "nothing has been passed" in text.lower()


def test_the_two_scores_are_shown_side_by_side_and_never_subtracted(tmp_path):
    """A holistic judgement and a weighted mean of Grades are different scales.

    Their difference is not a movement, and printing one would be a
    plausible-looking wrong number - the kind a reader acts on.
    """
    db = Database(tmp_path / "flats.db")
    a_listing(db, "1", status="approved", score=9.0, grades={"layout": 1, "light": 1})
    a_listing(db, "2", status="approved", score=6.0, grades={"layout": 2, "light": 2})
    text = shift_report(db, PAIR, settings_with())

    assert "was 9.0" in text and "now 1.0" in text
    assert "different scales" in text
    # The two would sort the other way round on the size of the difference
    # (-8.0 against -4.0 is the same order here, so use the report's own claim
    # instead): nothing anywhere prints a delta.
    for forbidden in ("8.0 lower", "down 8.0", "-8.0", "delta"):
        assert forbidden not in text, forbidden


def test_the_rows_are_ordered_by_the_old_score_and_not_by_the_gap(tmp_path):
    """Sorting by the difference would rank on the number that means nothing.

    #1 dropped by 4 and #2 by 7, so a delta sort puts #2 first. The report
    reads highest stored score first: the strongest flat they lost, at the top.
    """
    db = Database(tmp_path / "flats.db")
    a_listing(db, "1", status="approved", score=8.0, grades={"layout": 4, "light": 4})
    a_listing(db, "2", status="approved", score=7.0, grades={"layout": 0, "light": 0})
    text = shift_report(db, PAIR, settings_with())
    assert text.index("#1") < text.index("#2"), text


def test_a_veto_is_reported_even_when_the_score_clears_the_cut(tmp_path):
    """A vetoed Listing can score well. Reporting it as below the cut would
    send a reader to lower the threshold, which would change nothing."""
    criteria = [
        a_criterion("layout", weight=8),
        a_criterion("pets", weight=2, veto_at_or_below=1),
    ]
    db = Database(tmp_path / "flats.db")
    a_listing(db, "1", status="approved", grades={"layout": 10, "pets": 0})
    text = shift_report(db, criteria, settings_with(floor=0.3, threshold=5.0))

    row = row_for(text, 1)
    assert "now 8.0" in row
    assert "veto" in row.lower()
    assert "under the" not in row.lower()


# Every module that can reach a model does `from pydantic_ai import Agent` at
# import time, so the name each call site resolves is its own module global.
# Patching `pydantic_ai.Agent` rebinds the package attribute and reaches none
# of them.
_AGENT_SITES = (
    "flat_scout.grading.Agent",
    "flat_scout.evaluate.Agent",
    "flat_scout.vision.Agent",
)


def test_the_shift_report_never_constructs_a_model_agent(tmp_path, monkeypatch):
    """This replays stored Grades. Re-grading would cost money, and worse,
    would answer a different question - what the model says today rather than
    what the stored evidence says.

    Recording rather than raising: a stand-in that raised would be swallowed
    by any broad `except Exception` between here and the call site and pass
    for the wrong reason.
    """
    from pydantic_ai.models import Model

    constructed: list[tuple] = []
    requested: list[tuple] = []

    class RecordingAgent:
        def __init__(self, *args, **kwargs):
            constructed.append((args, kwargs))

    for site in _AGENT_SITES:
        monkeypatch.setattr(site, RecordingAgent)
    monkeypatch.setattr(Model, "request", lambda self, *a, **kw: requested.append((a, kw)))

    db = Database(tmp_path / "flats.db")
    criteria = load_criteria(CRITERIA_DIR, known_graders=set(GRADERS))
    a_listing(db, "1", status="approved", grades={"pets": 9})
    a_listing(db, "2", status="rejected", grades={"pets": 2})
    a_listing(db, "3", status="approved", grades=None)
    shift_report(db, criteria, Settings())

    assert constructed == []
    assert requested == []


def test_the_shipped_criteria_can_be_shifted(tmp_path):
    db = Database(tmp_path / "flats.db")
    criteria = load_criteria(CRITERIA_DIR, known_graders=set(GRADERS))
    a_listing(db, "1", status="approved", grades={"pets": 9})
    assert isinstance(shift_report(db, criteria, Settings()), str)


def _cli(tmp_path, monkeypatch, args: list[str]):
    from typer.testing import CliRunner

    from flat_scout import cli

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(
        "[evaluation]\nhopeful_threshold = 5.0\n"
        f'criteria_dir = "{CRITERIA_DIR}"\n'
    )
    return CliRunner().invoke(cli.app, args)


def test_the_report_command_switches_to_the_shift_view(tmp_path, monkeypatch):
    db = Database(tmp_path / "data" / "flats.db")
    a_listing(db, "1", status="approved", grades={"pets": 1})
    result = _cli(tmp_path, monkeypatch, ["report", "--shift"])

    assert result.exit_code == 0, result.output
    assert "1 Somewhere Street" in result.output
    # The shift view, not the histogram the bare command prints.
    assert "would clear the threshold" not in result.output


def test_the_shift_view_will_not_run_beside_another_view(tmp_path, monkeypatch):
    Database(tmp_path / "data" / "flats.db")
    result = _cli(tmp_path, monkeypatch, ["report", "--shift", "--weights"])

    assert result.exit_code != 0
    assert "one" in result.output.lower()
