import io
import logging
import time

import httpx
import pytest

from flat_scout import observe
from flat_scout.config import Settings


class Clock:
    """A hand-wound clock, so a cadence measured in seconds is testable."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


class Pipe(io.StringIO):
    """Not a terminal - what running the command through a pipe gives."""

    def isatty(self) -> bool:
        return False


class Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_a_run_nobody_can_see_still_says_it_has_started(caplog):
    """The first line is the point of the whole module.

    `grade --backfill` printed nothing at all until it finished, so a run that
    was working and a run that was wedged looked exactly alike for ten minutes.
    """
    with caplog.at_level(logging.INFO, logger="flat_scout.observe"):
        observe.Progress(97, "grading Listings", stream=Pipe()).close()
    assert "97" in caplog.text
    assert "grading Listings" in caplog.text


def test_a_pipe_gets_lines_on_a_clock_rather_than_a_bar(caplog):
    """Ticks alone must not drive the output.

    Through a pipe the bar is unreadable, and one line per Listing is a wall of
    text. Time is what a watcher actually cares about: a line every half
    minute says the run is alive, whether that was one Listing or thirty.
    """
    clock = Clock()
    progress = observe.Progress(
        10, "grading", stream=Pipe(), every_seconds=30.0, clock=clock
    )
    with caplog.at_level(logging.INFO, logger="flat_scout.observe"):
        for _ in range(3):
            clock.tick(5.0)
            progress.tick()
        assert caplog.text == ""      # 15 seconds in, still quiet
        clock.tick(20.0)
        progress.tick()
        assert "4/10" in caplog.text


def test_the_line_says_how_long_is_left(caplog):
    clock = Clock()
    progress = observe.Progress(10, "grading", stream=Pipe(), every_seconds=1.0, clock=clock)
    with caplog.at_level(logging.INFO, logger="flat_scout.observe"):
        for _ in range(5):
            clock.tick(60.0)
            progress.tick()
    # Five in five minutes; five to go is five minutes more.
    assert "5m" in caplog.text


def test_failures_are_counted_where_they_can_be_seen(caplog):
    """A backfill that grades 87 of 97 and says nothing is the bug.

    The ten that failed were logged one by one at ERROR, into a logger no CLI
    command had configured, so they went nowhere at all.
    """
    clock = Clock()
    progress = observe.Progress(3, "grading", stream=Pipe(), every_seconds=1.0, clock=clock)
    with caplog.at_level(logging.INFO, logger="flat_scout.observe"):
        progress.tick()
        progress.tick(failed=True)
        progress.tick(failed=True)
        progress.close()
    assert "2 failed" in caplog.text


def test_a_finished_run_says_so_once(caplog):
    clock = Clock()
    progress = observe.Progress(2, "grading", stream=Pipe(), every_seconds=1.0, clock=clock)
    with caplog.at_level(logging.INFO, logger="flat_scout.observe"):
        progress.tick()
        progress.tick()
        progress.close()
        progress.close()
    assert caplog.text.count("done") == 1


def test_a_run_where_nothing_finishes_still_says_it_is_alive(caplog):
    """The case the whole module is for, and the one ticks cannot cover.

    `tick` fires when a unit *finishes*. A run taking one wedged Listing, or
    a `--limit 1` that hangs on its only row, finishes nothing at all - so a
    tick-driven cadence prints "1 to do" and then goes silent for exactly as
    long as the run is in trouble. Found in review of the first version, which
    had this defect.

    Real time, real thread, and a hundredth of a second of it: the beat cannot
    be driven by the hand-wound clock the other tests use, because a thread
    racing an assertion is the thing those tests avoid.
    """
    progress = observe.Progress(1, "grading", stream=Pipe(), every_seconds=0.05)
    with caplog.at_level(logging.INFO, logger="flat_scout.observe"):
        time.sleep(0.2)
        assert "0/1" in caplog.text
        assert "elapsed" in caplog.text
    progress.close()


def test_the_beat_stops_when_the_run_does():
    progress = observe.Progress(1, "grading", stream=Pipe(), every_seconds=0.01)
    progress.tick()
    progress.close()
    assert progress._beat is not None
    assert not progress._beat.is_alive()


def test_a_run_that_stopped_early_does_not_call_itself_done(caplog):
    """Found in review. Every caller is a context manager now, so an exception
    reaches `close` as surely as a completion does - and "done, 1 in 0s" over a
    benchmark that aborted at 1 of 40 reads as a whole corpus. The numbers off
    a partial run would then be read as numbers off a full one."""
    with caplog.at_level(logging.INFO, logger="flat_scout.observe"):
        try:
            with observe.Progress(40, "reading floorplans", stream=Pipe()) as progress:
                progress.tick()
                raise RuntimeError("the provider gave up")
        except RuntimeError:
            pass
    assert "done" not in caplog.text
    assert "stopped at 1/40" in caplog.text
    assert "RuntimeError" in caplog.text


def test_a_run_cut_short_without_an_exception_says_so_too(caplog):
    """`--limit` and a break both leave the count short of the total."""
    with caplog.at_level(logging.INFO, logger="flat_scout.observe"):
        with observe.Progress(10, "grading", stream=Pipe()) as progress:
            progress.tick()
    assert "stopped at 1/10" in caplog.text


def test_a_terminal_gets_a_bar():
    progress = observe.Progress(4, "grading", stream=Terminal())
    assert progress.bar is not None
    progress.tick()
    assert progress.bar.n == 1
    progress.close()


def test_it_is_a_context_manager_that_closes_itself():
    with observe.Progress(1, "grading", stream=Pipe()) as progress:
        progress.tick()
    assert progress.closed


class FakeLogfire:
    def __init__(self) -> None:
        self.configured: dict | None = None
        self.instrumented: list[str] = []
        self.instrumented_clients: list = []

    def configure(self, **kwargs) -> None:
        self.configured = kwargs

    def instrument_pydantic_ai(self) -> None:
        self.instrumented.append("pydantic_ai")

    def instrument_httpx(self, client=None) -> None:
        self.instrumented.append("httpx")
        self.instrumented_clients.append(client)

    class LogfireLoggingHandler(logging.Handler):
        def emit(self, record) -> None:
            pass


@pytest.fixture(autouse=True)
def _observability_off_between_tests():
    yield
    observe.stop()


def test_no_token_means_nothing_is_sent_anywhere():
    """Off is the default, and off must not need a network to decide it.

    A test suite, a laptop and a clone by anyone else all run without a token,
    and none of them should be configuring an exporter or reaching out to say
    hello.
    """
    fake = FakeLogfire()
    assert observe.configure_observability(Settings(), service="test", module=fake) is False
    assert fake.configured is None
    assert observe.sending() is False


def test_a_token_turns_it_on_and_instruments_the_model_calls():
    """The model calls are the thing worth seeing.

    Cost, latency and the retry behaviour of five Criteria per Listing are
    invisible in a log line, and pydantic-ai reports all three to logfire the
    moment it is instrumented.
    """
    fake = FakeLogfire()
    settings = Settings()
    settings.secrets.logfire_token = "pylf_v1_eu_secret"
    assert observe.configure_observability(settings, service="backfill", module=fake) is True
    assert fake.configured["token"] == "pylf_v1_eu_secret"
    assert fake.configured["service_name"] == "backfill"
    assert "pydantic_ai" in fake.instrumented
    assert observe.sending() is True


def test_the_console_is_left_to_the_logging_we_already_have():
    """Two renderers on one stream is worse than one.

    The commands already print a timestamped line per event, and logfire's
    console exporter would print every span beside it.
    """
    fake = FakeLogfire()
    settings = Settings()
    settings.secrets.logfire_token = "pylf_v1_eu_secret"
    observe.configure_observability(settings, service="check", module=fake)
    assert fake.configured["console"] is False


def test_a_span_is_a_no_op_when_nothing_is_listening():
    with observe.span("grading a Listing", listing_id=4):
        pass


def test_a_span_reaches_logfire_when_something_is():
    seen = []

    class Spanning(FakeLogfire):
        def span(self, name, **attributes):
            seen.append((name, attributes))

            class Nothing:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, *exc):
                    return False

            return Nothing()

    settings = Settings()
    settings.secrets.logfire_token = "pylf_v1_eu_secret"
    observe.configure_observability(settings, service="x", module=Spanning())
    with observe.span("grading a Listing", listing_id=4):
        pass
    assert seen == [("grading a Listing", {"listing_id": 4})]


def test_logging_is_configured_once_however_often_it_is_asked():
    """Every command calls this, and one of them used to call `basicConfig`
    itself.

    Two handlers on the root logger means every line twice.
    """
    before = list(logging.getLogger().handlers)
    observe.configure_logging()
    observe.configure_logging()
    after = list(logging.getLogger().handlers)
    assert len(after) <= len(before) + 1


@pytest.mark.asyncio
async def test_a_provider_credential_never_reaches_an_exported_span():
    """The P1 this module exists to close.

    A provider credential can travel as a `key` or an `app_key` query
    parameter, and `configure_observability` turns on
    `logfire.instrument_httpx()`, which records the request URL as a span
    attribute. Whenever LOGFIRE_TOKEN is set, such a key would otherwise be
    exported to logfire on every request, unredacted, before it ever reaches
    an exception message or a log record.

    This drives requests carrying each parameter through clients logfire
    actually instruments, with logfire pointed at a `TestExporter` instead of the
    network (`send_to_logfire=False`), and inspects every attribute of every
    exported span. That includes the "pending span" logfire exports the
    instant a request starts - the one a bare `request_hook` cannot reach in
    time, which is why the fix lives in `_redact_credential_query_params`
    and not in a hook. No test elsewhere in this suite drives a real
    request through real logfire instrumentation, so this is the one place
    the leak this module fixes can actually be observed happening - or not.
    """
    import logfire
    import opentelemetry.util.http as otel_http
    from logfire.testing import TestExporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    sentinel = "SENTINEL-live-provider-key-do-not-export"
    exporter = TestExporter()
    logfire.configure(
        send_to_logfire=False,
        additional_span_processors=[SimpleSpanProcessor(exporter)],
    )
    try:
        observe._redact_credential_query_params()

        async def api_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"items": []})

        async with httpx.AsyncClient(transport=httpx.MockTransport(api_handler)) as client:
            logfire.instrument_httpx(client)
            await client.get(
                "https://api.example.com/v1",
                params={"app_key": sentinel},
            )

        async def media_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"items": []})

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(media_handler)
        ) as client:
            logfire.instrument_httpx(client)
            await client.get(
                "https://media.example.co.uk/photo.jpg",
                params={"key": sentinel},
            )
    finally:
        for param in ("key", "app_key"):
            if param in otel_http.PARAMS_TO_REDACT:
                otel_http.PARAMS_TO_REDACT.remove(param)

    spans = exporter.exported_spans
    # If instrumentation never ran, the loop below would pass on an empty
    # list and prove nothing - the exact failure mode this test must not
    # have. `main` added this instrumentation deliberately; two spans per
    # request (a pending one and a final one) times two providers is four.
    assert len(spans) == 4

    for span in spans:
        for attr_name, attr_value in (span.attributes or {}).items():
            assert sentinel not in str(attr_value), f"{attr_name} leaked the credential"

    urls = [v for span in spans for k, v in (span.attributes or {}).items() if k == "http.url"]
    assert any("api.example.com/v1" in u for u in urls), urls
    assert any("media.example.co.uk/photo.jpg" in u for u in urls), urls


# --- what gets traced, and what is left alone ---------------------------


def test_turning_it_on_does_not_instrument_every_httpx_client():
    """A global hook traces every client in the process, not only ours.

    OpenRouter's transport is the case that matters: the pydantic-ai span
    already carries the tokens and the latency, so a second HTTP span inside
    every one of them is a duplicate.
    """
    fake = FakeLogfire()
    settings = Settings()
    settings.secrets.logfire_token = "pylf_v1_eu_secret"
    observe.configure_observability(settings, service="check", module=fake)
    assert "httpx" not in fake.instrumented
    assert "pydantic_ai" in fake.instrumented


def test_our_own_clients_are_traced_one_at_a_time():
    """What replaces it: the clients this app builds, and only those.

    `fetch.build_async_client` is the one sanctioned constructor, so tracing
    there covers the Portal fetches without reaching anything a library
    opened for itself.
    """
    fake = FakeLogfire()
    settings = Settings()
    settings.secrets.logfire_token = "pylf_v1_eu_secret"
    observe.configure_observability(settings, service="check", module=fake)
    client = httpx.AsyncClient()
    observe.instrument_client(client)
    assert fake.instrumented_clients == [client]


def test_instrumenting_a_client_while_nothing_is_sending_does_nothing():
    """The no-token path must not touch the client it is handed."""
    assert observe.sending() is False
    observe.instrument_client(httpx.AsyncClient())  # must not raise


@pytest.mark.parametrize("noisy", ["httpx", "httpcore", "urllib3"])
def test_the_chatty_libraries_are_quieted_to_warning(noisy):
    """They were louder than the pipeline in its own logs.

    The root handler still carries their warnings and errors; only the running
    commentary goes.
    """
    observe._logging_configured = False
    try:
        observe.configure_logging()
        assert logging.getLogger(noisy).level == logging.WARNING, noisy
    finally:
        observe._logging_configured = False


# --- naming the model calls ---------------------------------------------


AGENTS = {
    "flat_scout.evaluate": "holistic-evaluator",
    "flat_scout.grading": "criterion-grader",
}


def test_every_agent_is_named_so_a_span_says_which_call_it_was():
    """Unnamed, logfire renders all five identically as `agent run`.

    Nothing in the span then says whether the call weighed the whole brief,
    graded one Criterion, or read a floorplan - and a Listing makes five of
    the middle kind. Our own parent spans carry the identity, but a flat list
    of spans is the view this is read in.
    """
    import ast
    from pathlib import Path

    unnamed = []
    for path in Path("src/flat_scout").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if getattr(func, "id", None) != "Agent":
                continue
            if not any(kw.arg == "name" for kw in node.keywords):
                unnamed.append(f"{path.name}:{node.lineno}")
    assert unnamed == [], f"pydantic-ai Agents built without name=: {unnamed}"
