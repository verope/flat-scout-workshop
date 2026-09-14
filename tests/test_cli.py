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
