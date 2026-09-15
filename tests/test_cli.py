"""The commands in cli.py, and the client they build."""

from __future__ import annotations

import httpx
from typer.testing import CliRunner

from flat_scout import cli

runner = CliRunner()


def test_check_command_builds_its_client_with_the_configured_timeout(tmp_path, monkeypatch):
    """cli.py used to build `httpx.AsyncClient()` bare, so `check` saw httpx's
    5-second default instead of the configured timeout. The URL is
    deliberately not a recognised Portal listing: it makes `process_url`
    return before any network call, so only the client construction is under
    test here."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text("[fetch]\ntimeout_seconds = 17\n")

    captured: dict = {}
    real_init = httpx.AsyncClient.__init__

    def spy_init(self, *args, **kwargs):
        captured.update(kwargs)
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", spy_init)

    runner.invoke(cli.app, ["check", "https://example.com/not-a-listing"])

    assert captured.get("timeout") == 17.0


def _capture_process_url(monkeypatch, tmp_path):
    """Stub the pipeline and keep what `check` handed it. The stub returns
    None, so `check` exits on the "not a listing" branch and touches nothing."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text("[features]\nweighted_criteria = true\n")
    captured: dict = {}

    async def fake_process_url(url, db, settings, source, client, **kwargs):
        captured["weighted"] = settings.features.weighted_criteria
        captured["force"] = kwargs.get("force", False)
        return None

    monkeypatch.setattr(cli, "process_url", fake_process_url)
    return captured


def test_check_holistic_turns_the_weighted_flag_off_for_this_run(tmp_path, monkeypatch):
    captured = _capture_process_url(monkeypatch, tmp_path)
    runner.invoke(cli.app, ["check", "https://example.com/x", "--holistic"])
    assert captured["weighted"] is False


def test_check_weighted_turns_the_weighted_flag_on_for_this_run(tmp_path, monkeypatch):
    captured = _capture_process_url(monkeypatch, tmp_path)
    (tmp_path / "config.toml").write_text("[features]\nweighted_criteria = false\n")
    runner.invoke(cli.app, ["check", "https://example.com/x", "--weighted"])
    assert captured["weighted"] is True


def test_check_without_a_mode_flag_follows_the_config(tmp_path, monkeypatch):
    captured = _capture_process_url(monkeypatch, tmp_path)
    runner.invoke(cli.app, ["check", "https://example.com/x"])
    assert captured["weighted"] is True


def test_check_refuses_both_mode_flags(tmp_path, monkeypatch):
    captured = _capture_process_url(monkeypatch, tmp_path)
    result = runner.invoke(
        cli.app, ["check", "https://example.com/x", "--holistic", "--weighted"]
    )
    assert result.exit_code == 2
    assert "Pass one" in result.output
    assert captured == {}


def test_check_force_reaches_the_pipeline(tmp_path, monkeypatch):
    captured = _capture_process_url(monkeypatch, tmp_path)
    runner.invoke(cli.app, ["check", "https://example.com/x", "--force"])
    assert captured["force"] is True


def test_check_prints_the_grade_vector_after_a_weighted_evaluation(tmp_path, monkeypatch):
    """The row alone hides what the weighted evaluator actually did. The
    grades live in their own table and the weights in the Criterion files,
    and the demo had to reach for SQL to show either."""
    from flat_scout.db import Database
    from flat_scout.criteria import Criterion
    from flat_scout.grading import Grade
    from flat_scout.models import ListingData

    monkeypatch.chdir(tmp_path)
    (tmp_path / "criteria").mkdir()
    (tmp_path / "criteria" / "lift.md").write_text(
        "---\nname: A lift above the second floor\ndescription: d\nweight: 2\n"
        "grade:\n  model: true\nunknown: skip\n---\n\n## Rubric\n**10** — lift.\n"
    )
    (tmp_path / "config.toml").write_text(
        '[features]\nweighted_criteria = true\n[evaluation]\ncriteria_dir = "criteria"\n'
    )
    (tmp_path / "data").mkdir()
    db = Database(tmp_path / "data" / "flats.db")
    listing_id = db.upsert(
        ListingData(portal="rightmove", portal_id="1", url="https://www.rightmove.co.uk/properties/1"),
        "manual",
    )
    criterion = Criterion(
        slug="lift", name="A lift above the second floor", description="d", weight=2,
        grade={"model": True}, body="## Rubric\n**10** — lift.",
    )
    db.set_criterion_grades(
        listing_id,
        [(criterion, Grade(criterion="lift", value=10.0, determined=True,
                           evidence="Third floor with a lift.", graded_by="model:x"))],
    )
    db.conn.execute(
        "UPDATE listings SET evaluation_mode = 'weighted' WHERE id = ?", (listing_id,)
    )
    db.conn.commit()

    async def fake_process_url(url, db, settings, source, client, **kwargs):
        return listing_id

    monkeypatch.setattr(cli, "process_url", fake_process_url)
    result = runner.invoke(cli.app, ["check", "https://www.rightmove.co.uk/properties/1"])

    assert result.exit_code == 0, result.output
    import json

    printed = json.loads(result.output)
    assert printed["id"] == listing_id
    [lift] = printed["grades"]
    assert lift["criterion"] == "lift"
    assert lift["name"] == "A lift above the second floor"
    assert lift["weight"] == 2
    assert lift["grade"] == 10.0
    assert lift["determined"] is True
    assert lift["evidence"] == "Third floor with a lift."
