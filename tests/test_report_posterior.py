"""`report --posterior`: the two Bayesian constants, set from evidence.

`hopeful_threshold` and `hopeful_confidence` are a pair, and neither means
anything alone: a high cut with a low confidence surfaces the same flats as
a low cut with a high one. The grid replays every candidate pair over the
Grades already stored and the Decisions already taken, so the two are chosen
off what they would have done rather than off a guess.

No model is called and no sampler runs: `score_posterior` is numpy, and
`load_fit` has nothing to sample with no measure runs stored.
"""

from __future__ import annotations

import re
from pathlib import Path

from flat_scout.adjudicate import Grade
from flat_scout.config import Settings
from flat_scout.criteria import Criterion, load_criteria
from flat_scout.db import Database
from flat_scout.graders import GRADERS
from flat_scout.models import ListingData
from flat_scout.report import posterior_report

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
    grades: dict[str, float | None] | None = None,
    graded_by: str = "model:test",
    decided: bool = True,
    criteria: list[Criterion] | None = None,
) -> int:
    """One decided Listing with a grade vector.

    `graded_by` matters here in a way it does not in the point path: a model
    grade is widened by the fit's sigma, and a deterministic one is not.
    """
    listing_id = db.upsert(
        ListingData(portal="rightmove", portal_id=portal_id, url=f"u{portal_id}"),
        source="manual",
    )
    db.conn.execute(
        "UPDATE listings SET status = ?, decided_at = ?, score = 6.0, verdict = ?, "
        "address = ?, price_pcm = 2400 WHERE id = ?",
        (
            status,
            "2026-08-17" if decided else None,
            "hopeful",
            f"{portal_id} Somewhere Street",
            listing_id,
        ),
    )
    db.conn.commit()
    if grades:
        by_slug = {c.slug: c for c in (criteria or PAIR)}
        db.set_criterion_grades(
            listing_id,
            [
                (
                    by_slug[slug],
                    Grade(slug, value, value is not None, f"{slug} read", graded_by),
                )
                for slug, value in grades.items()
            ],
        )
    return listing_id


def settings_with(threshold: float = 7.5, confidence: float = 0.6) -> Settings:
    settings = Settings()
    settings.evaluation.hopeful_threshold = threshold
    settings.evaluation.hopeful_confidence = confidence
    return settings


def counts(text: str, gate: str) -> list[int]:
    """The counts on the grid row for one gate, left to right."""
    for line in text.splitlines():
        if gate in line:
            return [int(n) for n in re.findall(r"\d+", line.split(gate)[1])]
    raise AssertionError(f"no grid row for {gate}\n{text}")


def separated(db: Database) -> None:
    """Three flats they wanted, graded high; two they passed, graded low."""
    for portal_id in ("1", "2", "3"):
        a_listing(db, portal_id, status="approved", grades={"layout": 9, "light": 9})
    for portal_id in ("4", "5"):
        a_listing(db, portal_id, status="rejected", grades={"layout": 2, "light": 2})


def test_the_grid_replays_every_candidate_pair(tmp_path):
    db = Database(tmp_path / "flats.db")
    separated(db)
    text = posterior_report(db, settings_with(), PAIR)

    assert "Decisions behind this: 5" in text
    assert "t=" in text and "p=" in text
    assert "misclassified" in text
    # Both ends of both axes are on the table, not just the configured pair.
    assert counts(text, "t=6.0 p=0.50")
    assert counts(text, "t=7.5 p=0.80")


def test_a_pair_that_agrees_with_every_decision_misclassifies_nothing(tmp_path):
    """The whole point of the table: a row a reader can paste into config."""
    db = Database(tmp_path / "flats.db")
    separated(db)
    text = posterior_report(db, settings_with(), PAIR)

    surfaced, missed, wasted, wrong = counts(text, "t=7.5 p=0.60")
    assert (surfaced, missed, wasted, wrong) == (3, 0, 0, 0)


def test_raising_the_confidence_trades_rejections_surfaced_for_approvals_missed(tmp_path):
    """Which is the trade the pair is chosen on, and it has to be visible.

    Both flats graded the same and were decided differently, so no gate tells
    them apart: raising `p` can only swap one error for the other.
    """
    db = Database(tmp_path / "flats.db")
    a_listing(db, "1", status="approved", grades={"layout": 8, "light": 8})
    a_listing(db, "2", status="rejected", grades={"layout": 8, "light": 8})
    text = posterior_report(db, settings_with(), PAIR)

    _, low_missed, low_wasted, _ = counts(text, "t=7.5 p=0.50")
    _, high_missed, high_wasted, _ = counts(text, "t=7.5 p=0.80")
    assert high_missed > low_missed
    assert high_wasted < low_wasted


def test_the_configured_pair_is_in_the_grid_and_marked(tmp_path):
    """A pair off the grid would leave the reader nothing to compare against."""
    db = Database(tmp_path / "flats.db")
    separated(db)
    text = posterior_report(db, settings_with(threshold=6.8, confidence=0.55), PAIR)

    row = next(line for line in text.splitlines() if "t=6.8 p=0.55" in line)
    assert row.startswith("*")
    assert "in force" in text


def test_calibration_holds_the_posterior_against_the_decision(tmp_path):
    db = Database(tmp_path / "flats.db")
    separated(db)
    text = posterior_report(db, settings_with(), PAIR)

    line = next(line for line in text.splitlines() if "p(hopeful)" in line)
    approved, rejected = [int(n) for n in re.findall(r"(\d+)%", line)]
    assert approved > rejected


def test_a_missed_approval_is_diagnosed_by_the_bayesian_gate(tmp_path):
    """Do not send a probability miss to the point-score shift report."""
    db = Database(tmp_path / "flats.db")
    a_listing(
        db, "1", status="approved", grades={"layout": 6, "light": 6}, graded_by="map"
    )
    a_listing(
        db, "2", status="rejected", grades={"layout": 2, "light": 2}, graded_by="map"
    )

    text = posterior_report(db, settings_with(), PAIR)

    assert "#1 (p(hopeful) 0% below the 60% gate)" in text
    assert "report --shift" not in text


def test_a_veto_is_named_as_a_floor_no_pair_can_lift(tmp_path):
    """A vetoed approval is missed on every row, and no gate recovers it.

    Counted rather than dropped, because it is a real disagreement with the
    couple - but a reader comparing rows has to know it moves in none of them.
    """
    criteria = [a_criterion("layout"), a_criterion("pets", veto_at_or_below=1)]
    db = Database(tmp_path / "flats.db")
    a_listing(
        db, "1", status="approved", grades={"layout": 9, "pets": 0},
        graded_by="map", criteria=criteria,
    )
    a_listing(
        db, "2", status="rejected", grades={"layout": 2, "pets": 5},
        graded_by="map", criteria=criteria,
    )
    text = posterior_report(db, settings_with(), criteria)

    assert "veto" in text.lower()
    # Named as well as counted, so a reader can go and read the flat.
    assert "#1" in text
    for gate in ("t=6.0 p=0.50", "t=7.5 p=0.80"):
        surfaced, missed, _, _ = counts(text, gate)
        assert (surfaced, missed) == (0, 1)


def test_a_decided_listing_with_no_grades_is_counted_and_not_replayed(tmp_path):
    db = Database(tmp_path / "flats.db")
    separated(db)
    a_listing(db, "9", status="approved", grades=None)
    text = posterior_report(db, settings_with(), PAIR)

    assert "Decisions behind this: 5" in text
    assert "A further 1 decided Listing" in text
    assert "--backfill" in text


def test_no_decided_grades_says_so_instead_of_printing_an_empty_grid(tmp_path):
    db = Database(tmp_path / "flats.db")
    a_listing(
        db, "1", status="evaluated", grades={"layout": 9, "light": 9}, decided=False
    )
    text = posterior_report(db, settings_with(), PAIR)

    assert "t=" not in text
    assert "backfill" in text


def test_the_shipped_criteria_can_be_replayed(tmp_path):
    """The real brief, not a two-Criterion fixture. No model, no sampler."""
    db = Database(tmp_path / "flats.db")
    criteria = load_criteria(CRITERIA_DIR, known_graders=set(GRADERS))
    a_listing(db, "1", status="approved", grades={"pets": 9}, criteria=criteria)
    a_listing(db, "2", status="rejected", grades={"pets": 2}, criteria=criteria)
    assert "misclassified" in posterior_report(db, Settings(), criteria)


def _cli(tmp_path, monkeypatch, args: list[str]):
    from typer.testing import CliRunner

    from flat_scout import cli

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(
        "[evaluation]\nhopeful_threshold = 5.0\n"
        f'criteria_dir = "{CRITERIA_DIR}"\n'
    )
    return CliRunner().invoke(cli.app, args)


def test_the_report_command_switches_to_the_posterior_view(tmp_path, monkeypatch):
    db = Database(tmp_path / "data" / "flats.db")
    criteria = load_criteria(CRITERIA_DIR, known_graders=set(GRADERS))
    a_listing(db, "1", status="approved", grades={"pets": 9}, criteria=criteria)
    result = _cli(tmp_path, monkeypatch, ["report", "--posterior"])

    assert result.exit_code == 0, result.output
    assert "Decisions behind this: 1" in result.output
    # The grid, not the histogram the bare command prints.
    assert "would clear the threshold" not in result.output


def test_the_posterior_view_will_not_run_beside_another_view(tmp_path, monkeypatch):
    Database(tmp_path / "data" / "flats.db")
    result = _cli(tmp_path, monkeypatch, ["report", "--posterior", "--weights"])

    assert result.exit_code != 0
    assert "one" in result.output.lower()
