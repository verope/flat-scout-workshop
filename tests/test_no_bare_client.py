"""Enforces the one-factory rule for `httpx.AsyncClient`.

`fetch.build_async_client` is the only sanctioned way to build one: it carries
the fetch timeout (silently lost twice before - see PR #17). A second
construction site is exactly how that happened the first time, so this fails
the build the moment a new one appears, rather than waiting for someone to
notice in production.

A plain grep over the source tree, deliberately: the point is to catch a
literal `httpx.AsyncClient(` call anywhere it should not be, not to reason
about control flow.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "flat_scout"
FACTORY_FILE = SRC / "fetch.py"

CONSTRUCTION = re.compile(r"httpx\.AsyncClient\(")


def test_no_bare_async_client_outside_the_factory():
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        if path == FACTORY_FILE:
            continue
        text = path.read_text()
        if CONSTRUCTION.search(text):
            offenders.append(str(path.relative_to(SRC)))
    assert offenders == [], (
        "httpx.AsyncClient( constructed outside fetch.build_async_client: "
        f"{offenders}. Use flat_scout.fetch.build_async_client(settings) "
        "instead, so the fetch timeout is never skipped."
    )


def test_the_factory_itself_still_builds_the_client_directly():
    """The grep above passes trivially if `build_async_client` is ever
    rewritten to stop constructing a client at all - this pins that it still
    does, so a passing suite always means a real, working factory exists."""
    text = FACTORY_FILE.read_text()
    assert CONSTRUCTION.search(text), (
        "fetch.py no longer constructs httpx.AsyncClient( itself - "
        "build_async_client must be the one place that does."
    )
