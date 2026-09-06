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
