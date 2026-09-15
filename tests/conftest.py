"""Two guards, for the whole suite.

`images.cache_images` writes into `data/images` relative to the working
directory, and under pytest that directory is the repository itself. Every test
gets its own throwaway cache root instead, so a test that reaches the cache -
directly, or through `process_url` - can never leave files in the tree. A test
that cares where the files went sets the same attribute itself, which overrides
this one.

The second guard is the secrets. `load_settings` reads `.env`, and observability
is switched on by the token alone, so one test that runs a CLI command with a
real `.env` present configured logfire for the whole process and exported every
later test's spans to a real project. All of them are pinned empty here;
`load_dotenv` never overrides a variable that is already set.
"""

from __future__ import annotations

import pytest

from flat_scout import images


@pytest.fixture(autouse=True)
def _cache_images_outside_the_repository(tmp_path, monkeypatch):
    monkeypatch.setattr(images, "IMAGES_DIR", tmp_path / "image-cache")


@pytest.fixture(autouse=True)
def _no_real_secrets(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    # The model slugs too: a developer's `.env` may name a different model,
    # and a test asserting the default must not read theirs.
    monkeypatch.setenv("OPENROUTER_TEXT_MODEL", "")
    monkeypatch.setenv("OPENROUTER_IMAGE_MODEL", "")
    monkeypatch.setenv("LOGFIRE_TOKEN", "")
    monkeypatch.setenv("LOGFIRE_SEND_TO_LOGFIRE", "false")
