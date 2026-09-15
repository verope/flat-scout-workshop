"""Two ways to watch a run that used to be silent.

`grade --backfill` runs over a corpus of Listings and five Criteria each,
several hundred model calls, and roughly ten minutes. It printed nothing
whatever until the last one landed. A run that was working and a run that was
wedged were indistinguishable, and the only way to find out which you had was
to wait for it or kill it.

Worse, the module logging was real and thrown away. `pipeline.py` names every
Listing that fails at ERROR - and nothing ever called `basicConfig`, so those
records went to the root logger's handler of last resort and vanished. The
backfill graded 87 of 97 Listings and said so nowhere.

So there are two things here, for two different watchers:

- `Progress`, for the person waiting at a terminal or a pipe. A bar when
  something can draw one, a line every half minute when nothing can.
- `configure_observability`, for the question a bar cannot answer: what did the
  model calls cost, which Criterion is slow, what did the retries do. That is
  logfire, and it is off unless a token says otherwise.

`configure_logging` is the unglamorous third, and is what actually recovers the
error lines the pipeline was already writing.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from contextlib import contextmanager
from typing import IO, Any, Callable, Iterator

log = logging.getLogger(__name__)

# Long enough that a ten-minute run says a dozen things rather than a hundred,
# short enough that a watcher never sits in silence wondering.
EVERY_SECONDS = 30.0

_sending = False
_logfire: Any = None
_logging_configured = False


def configure_logging(verbose: bool = False) -> None:
    """Give the module loggers somewhere to write. Safe to call repeatedly.

    Every command should call this. The pipeline logs a named ERROR for each
    Listing it drops, and until this runs those lines are written to nowhere
    at all.
    """
    global _logging_configured
    if _logging_configured:
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        return
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # All chatty at INFO, and none of it says anything this project needs -
    # a library narrating itself louder than the work it does, in the log and
    # in logfire both, since the root handler forwards these too. Warnings and
    # errors from all of them still come through; only the running commentary
    # goes.
    for chatty in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(chatty).setLevel(logging.WARNING)
    _logging_configured = True


# Query-string parameter names that must never appear in an exported span.
# `key` and `app_key` both travel as query parameters, and a provider added
# later needs only its parameter name added here, not its hostname: the rule
# is generic on purpose.
_CREDENTIAL_QUERY_PARAMS = ("key", "app_key")


def _redact_credential_query_params() -> None:
    """Stop the URL `instrument_httpx` records on a span from ever holding
    one of `_CREDENTIAL_QUERY_PARAMS`.

    Not a `request_hook`. `opentelemetry-instrumentation-httpx` builds the
    `http.url` span attribute - and the human-readable message logfire
    derives from it - by calling `opentelemetry.util.http.redact_url`
    *before* the span exists, so a hook only ever sees the attribute already
    set. That would be enough for the span a request finishes with, but
    logfire also exports a "pending span" the instant the request starts,
    stamped with whatever `redact_url` produced at creation - a hook runs
    inside that span's `with` block, which is already too late to reach the
    copy that already left. Extending the list `redact_url` reads, which
    `opentelemetry-instrumentation-httpx` already calls unconditionally on
    every URL for exactly this purpose (it ships redacting AWS and Google
    signed-URL parameters the same way), is the one point early enough to
    catch both. Verified against `logfire.testing.TestExporter`, not assumed
    - see test_observe.py.
    """
    import opentelemetry.util.http as otel_http  # noqa: PLC0415

    for param in _CREDENTIAL_QUERY_PARAMS:
        if param not in otel_http.PARAMS_TO_REDACT:
            otel_http.PARAMS_TO_REDACT.append(param)


def configure_observability(settings, *, service: str, module: Any = None) -> bool:
    """Turn on logfire if a token exists. Returns whether anything is sending.

    Deliberately token-gated rather than flag-gated. A clone with no token, the
    test suite, and a laptop debugging session must all do nothing at all - not
    configure an exporter, not open a connection, not warn. There is therefore
    no way to have this on and not know it: the token is the switch.

    `module` is for the tests, which must never import an exporter to prove one
    was not started.
    """
    global _sending, _logfire
    token = getattr(settings.secrets, "logfire_token", "")
    if not token:
        return False
    if module is None:  # pragma: no cover - the import is the thing being avoided
        import logfire as module  # noqa: PLC0415
    module.configure(
        token=token,
        service_name=service,
        # Belt and braces. The token above is already the switch; this makes
        # certain that a misread setting cannot start an exporter with nothing
        # to send to.
        send_to_logfire="if-token-present",
        # The commands already print a timestamped line per event and
        # `Progress` prints the rest. A second renderer on the same stream is
        # noise.
        console=False,
    )
    # The reason this dependency is here at all. Every model call gets a span
    # with its latency, its token counts and its retries - every Criterion of
    # the brief per Listing, none of which a log line can tell apart.
    module.instrument_pydantic_ai()
    # NOT `instrument_httpx()`. Unscoped, it hooks every client in the
    # process, and one of them is not ours: pydantic-ai's own span already
    # carries a model call's tokens, latency and retries, so tracing
    # OpenRouter's transport underneath it adds a duplicate child to the very
    # thing being looked at.
    #
    # The Portal downloads are still worth a span - a slow or 403-ing fetch
    # looks identical to a slow model from outside - so they are instrumented
    # one client at a time by `instrument_client`, called from the one
    # sanctioned constructor in `fetch.py`.
    #
    # The redaction stays here rather than moving with them: it patches the
    # list `opentelemetry.util.http.redact_url` reads, which has to be in
    # place before any instrumentation runs, and is not per-client. See
    # `_redact_credential_query_params` for why a request hook is too late.
    _redact_credential_query_params()
    # The ERROR lines the pipeline already writes, kept together with the spans
    # they happened inside rather than only on stdout.
    logging.getLogger().addHandler(module.LogfireLoggingHandler())
    _sending = True
    _logfire = module
    return True


def start(settings, *, service: str, verbose: bool = False) -> bool:
    """Both of the above, in the one line every command should open with.

    `service` names the command in logfire, so a wedged backfill and a wedged
    benchmark are told apart without reading the spans.
    """
    configure_logging(verbose)
    return configure_observability(settings, service=service)


def instrument_client(client: Any) -> None:
    """Trace one httpx client, if anything is listening. Safe to call always.

    The scoped half of what `instrument_httpx()` used to do globally. Called
    from `fetch.build_async_client`, which `fetch.py` declares the only
    sanctioned way to build an `httpx.AsyncClient` in this app - so every
    client this codebase owns is traced and nothing a library opened for
    itself is.

    A no-op without a token, like `span` and `event`, so the caller never has
    to ask whether anyone is watching.
    """
    if not _sending or _logfire is None:
        return
    _logfire.instrument_httpx(client)


def sending() -> bool:
    """Whether anything is being exported. Read by `stop`, and by the tests."""
    return _sending


def stop() -> None:
    """Forget the exporter. For the tests; nothing in the app needs it."""
    global _sending, _logfire
    _sending = False
    _logfire = None


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    """A logfire span, or nothing at all.

    Written as a no-op rather than as an `if` at each call site so the pipeline
    can be marked up once and stay readable whether or not anyone is watching.
    """
    if not _sending or _logfire is None:
        yield
        return
    with _logfire.span(name, **attributes):
        yield


def event(message: str, **attributes: Any) -> None:
    """One thing that happened, with its numbers attached.

    For the facts a span cannot carry because they are only known at the end -
    what a run did, what a backfill cost. The call sites already write a log
    line; this is the structured twin of it, and does nothing when nothing is
    listening.
    """
    if not _sending or _logfire is None:
        return
    _logfire.info(message, **attributes)


def _clipped(seconds: float) -> str:
    """A duration a person reads at a glance: 40s, 6m20s, 1h04m."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


class Progress:
    """How far along a long run is, told to whoever can hear it.

    Two audiences, one interface. At a terminal this is a tqdm bar. Through a
    pipe - which is how every one of these commands is really run on another
    machine - a bar is thousands of carriage returns, so it becomes a line on
    a clock instead.

    The cadence is time, not ticks. A line per Listing is a wall of text on a
    fast run and still silence on a slow one; the watcher's question is "is it
    alive", and only the clock answers that.

    Which is why the clock runs in a thread rather than only inside `tick`.
    Ticking is what a *finished* unit does, and the run this module was written
    for is the one where nothing finishes: a backfill taking one wedged
    Listing, or a `--limit 1` that hangs on its only row, would otherwise
    print "1 to do" and go silent for ever - the precise failure the bar
    exists to rule out. The beat says "0/1, 4m10s elapsed" until something
    changes, which is both the honest answer and the useful one.
    """

    def __init__(
        self,
        total: int,
        what: str,
        *,
        stream: IO[str] | None = None,
        every_seconds: float = EVERY_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.total = total
        self.what = what
        self.every_seconds = every_seconds
        self.clock = clock
        self.done = 0
        self.failed = 0
        self.closed = False
        stream = sys.stderr if stream is None else stream
        self.started = clock()
        self._said_at = self.started
        self._saying = threading.Lock()
        self._finished = threading.Event()
        self.bar = None
        if stream.isatty():
            from tqdm import tqdm  # noqa: PLC0415 - only needed where it draws

            self.bar = tqdm(total=total, desc=what, unit="", file=stream, leave=True)
        else:
            # The line that was missing. Everything else here is a refinement
            # of knowing that the run began at all.
            log.info("%s: %d to do", what, total)
        # A hand-wound clock has no heartbeat: a test that controls time wants
        # every line to come from something it drove, not from a thread racing
        # its assertions. Real time gets the beat, which is every caller in the
        # application.
        self._beat: threading.Thread | None = None
        if clock is time.monotonic:
            self._beat = threading.Thread(
                target=self._beating, name=f"progress-{what}", daemon=True
            )
            self._beat.start()

    def _beating(self) -> None:
        """Say the same thing the ticks say, on the clock, whatever is running.

        A daemon thread, so a crashing command is never held open by it, and it
        only ever reads counters and writes a log line.
        """
        while not self._finished.wait(self.every_seconds):
            self._say()

    def _say(self) -> None:
        """One line, if one is due. Called from both the worker and the beat."""
        with self._saying:
            now = self.clock()
            if now - self._said_at < self.every_seconds:
                return
            # The last of them says nothing `close` is not about to say better.
            if self.done >= self.total:
                return
            self._said_at = now
            line = self._line(now)
        if self.bar is not None:
            # tqdm redraws on update, so a bar over a stalled unit freezes with
            # its elapsed clock stopped. Refreshing keeps that honest.
            self.bar.refresh()
            return
        log.info("%s: %s", self.what, line)

    def tick(self, failed: bool = False) -> None:
        self.done += 1
        if failed:
            self.failed += 1
        if self.bar is not None:
            self.bar.update(1)
            if self.failed:
                self.bar.set_postfix_str(f"{self.failed} failed")
            return
        self._say()

    def _line(self, now: float) -> str:
        elapsed = now - self.started
        share = f"{self.done}/{self.total}"
        if self.total:
            share += f" ({self.done * 100 // self.total}%)"
        parts = [share]
        if self.failed:
            parts.append(f"{self.failed} failed")
        # Only once something has finished: an ETA off a sample of zero is a
        # division by zero dressed up as information.
        if self.done and elapsed > 0 and self.done < self.total:
            each = elapsed / self.done
            parts.append(f"~{_clipped(each * (self.total - self.done))} left")
        elif not self.done:
            # Nothing to estimate from, so say the one true thing instead: the
            # clock is moving even though the count is not. "0/1, 4m10s
            # elapsed" is what a wedged run looks like, and it looks wedged.
            parts.append(f"{_clipped(elapsed)} elapsed")
        return ", ".join(parts)

    def close(self, aborted: BaseException | None = None) -> None:
        """The last word: what finished, how long it took, what did not.

        Said once. `close` is called by the context manager and again by hand
        in places, and a run that reports itself finished twice reads as two
        runs.

        A run that stopped early says so. Since every caller became a context
        manager, an exception reaches here as surely as a completion does, and
        "done, 40 in 3m02s" over a benchmark that aborted at 40 of 97 is a
        lie of exactly the kind this module exists to stop telling - it reads
        as a whole corpus, and the numbers off it would be read as one.
        """
        if self.closed:
            return
        self.closed = True
        # Before the last line, so the beat cannot slip one in behind it.
        self._finished.set()
        if self._beat is not None:
            self._beat.join(timeout=1.0)
        if self.bar is not None:
            self.bar.close()
            return
        taken = _clipped(self.clock() - self.started)
        failed = f", {self.failed} failed" if self.failed else ""
        if aborted is None and self.done >= self.total:
            log.info("%s: done, %d in %s%s", self.what, self.done, taken, failed)
            return
        why = f": {type(aborted).__name__}" if aborted is not None else ""
        log.warning(
            "%s: stopped at %d/%d after %s%s%s",
            self.what, self.done, self.total, taken, failed, why,
        )

    def __enter__(self) -> Progress:
        return self

    def __exit__(self, kind: object, exc: BaseException | None, trace: object) -> bool:
        self.close(aborted=exc)
        return False
