"""The floorplan corpus, and the answers a human gave for it.

Everything read off a floorplan is a fact the couple cannot check without
opening the plan themselves, so the only way to know whether a change helped is
to hold it against answers a person wrote down while looking at the drawing.
Twice now, self-consistency has said a reading was reliable when it was
confidently wrong: a model answered "north is up" three times running on a plan
where north points left.

Annotations are in IMAGE terms - which way north points on the page, which way
the windows face on the page - and never in compass terms. Two reasons. The
person annotating does no arithmetic, so there is nothing for them to get wrong
beyond what they can see; and the two stages can then be scored separately,
which matters because they fail independently and are fixed by different work.

The stubs this writes are deliberately empty. Pre-filling them with what the
pipeline currently says would be asking someone to agree with the answer under
test, and an annotator who starts from the machine's guess is measuring their
own patience rather than the drawing.
"""

from __future__ import annotations

import base64
import sqlite3
import tomllib
from dataclasses import dataclass
from pathlib import Path

import httpx

CORPUS = Path("tests/fixtures/floorplans")
ANNOTATIONS = CORPUS / "annotations.toml"

# Read straight off the page: 0 is towards the top, 90 the right edge. The sheet
# draws this ring over every plan so nobody has to hold it in their head.
COMPASS_HINT = "0 = up the page, 90 = right, 180 = down, 270 = left"


@dataclass(frozen=True)
class Annotation:
    """One human's answers about one floorplan."""

    portal_id: str
    file: str
    # Captured when the plan enters the corpus, so that annotating needs nothing
    # but the corpus itself. Without it the editor would have to reach into a
    # database for a caption, which would make a self-contained set of images
    # and answers depend on a file that is not part of it.
    address: str = ""
    # None means "no north indicator is drawn on this plan", which is a real
    # answer and not a gap: it is what the pipeline should return too.
    north_clock: int | None = None
    # One entry per direction the main living space's windows face. More than
    # one is a dual-aspect flat, which is common and worth knowing about rather
    # than collapsing into whichever wall happened to be looked at first.
    windows_clock: tuple[int, ...] = ()
    compass_style: str = ""  # arrow | circle | rose | none
    notes: str = ""
    done: bool = False


def load_annotations(path: Path = ANNOTATIONS) -> dict[str, Annotation]:
    if not path.exists():
        return {}
    raw = tomllib.loads(path.read_text())
    found = {}
    for portal_id, fields in (raw.get("plan") or {}).items():
        found[portal_id] = Annotation(
            portal_id=portal_id,
            file=fields.get("file", ""),
            address=fields.get("address", ""),
            north_clock=fields.get("north_clock"),
            windows_clock=tuple(fields.get("windows_clock", ())),
            compass_style=fields.get("compass_style", ""),
            notes=fields.get("notes", ""),
            done=bool(fields.get("done", False)),
        )
    return found


# What a TOML basic string may not hold literally. The newline is the one that
# actually turned up: a Portal address arrived as "Northcote Road,\nBetween the
# Commons, SW11", which wrote a file that would not parse and took the whole
# corpus down with it - every answer in it, not just that plan's.
_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
    "\b": "\\b",
    "\f": "\\f",
}


def _quote(text: str) -> str:
    """A TOML basic string, safe for anything a Portal or a human puts in it."""
    out = []
    for char in text:
        if char in _ESCAPES:
            out.append(_ESCAPES[char])
        elif ord(char) < 0x20 or ord(char) == 0x7F:
            out.append(f"\\u{ord(char):04X}")
        else:
            out.append(char)
    return '"' + "".join(out) + '"'


def render_annotations(annotations: dict[str, Annotation]) -> str:
    """The file as text, with the unanswered entries left plainly unanswered."""
    lines = [
        "# Ground truth for the floorplan corpus, written by a human looking at",
        "# each drawing. Bearings are ON THE IMAGE: " + COMPASS_HINT + ".",
        "#",
        "# north_clock    which way the drawn north indicator points.",
        "#                Delete the line entirely if no indicator is drawn.",
        "# windows_clock  which way the main living space's windows face. One",
        "#                entry per direction; a dual-aspect flat has two.",
        "# compass_style  arrow | circle | rose | none",
        "# done           set to true when you have finished this plan.",
        "",
    ]
    for portal_id in sorted(annotations):
        entry = annotations[portal_id]
        lines.append(f"[plan.{portal_id}]")
        lines.append(f"file = {_quote(entry.file)}")
        lines.append(f"address = {_quote(entry.address)}")
        if entry.north_clock is None:
            lines.append("# north_clock = 0")
        else:
            lines.append(f"north_clock = {entry.north_clock}")
        listed = ", ".join(str(bearing) for bearing in entry.windows_clock)
        lines.append(f"windows_clock = [{listed}]")
        lines.append(f"compass_style = {_quote(entry.compass_style)}")
        lines.append(f"notes = {_quote(entry.notes)}")
        lines.append(f"done = {str(entry.done).lower()}")
        lines.append("")
    return "\n".join(lines)


def plans_in(db_path: Path) -> list[tuple[str, str, str | None, str | None]]:
    """(portal_id, address, floorplan_url, floorplan_path) for every plan.

    A row with only a path is included: `check` caches the plan, and a Listing
    whose media URL has since died is exactly the case the cache exists for.
    """
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        "SELECT portal_id, address, floorplan_url, floorplan_path FROM listings "
        "WHERE floorplan_url IS NOT NULL OR floorplan_path IS NOT NULL ORDER BY id"
    ).fetchall()
    return [
        (row["portal_id"], row["address"] or "", row["floorplan_url"], row["floorplan_path"])
        for row in rows
    ]


SUFFIXES = {"image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif"}


def download_plan(
    portal_id: str,
    url: str | None,
    client: httpx.Client,
    user_agent: str,
    into: Path,
    local: str | None = None,
) -> str | None:
    """Put one floorplan in the corpus. Returns its filename, or None.

    `local` is where `check` cached the plan. Copied rather than downloaded
    whenever it is there and the corpus can hold that kind of file, so building
    the corpus costs the media host nothing and works on a Listing whose URL
    has since died. A `.webp` or a `.pdf` is not a corpus file - the sheet and
    the editor serve only the three types below - so those fall back to the URL.
    """
    existing = [path for path in into.glob(f"{portal_id}.*")]
    if existing:
        return existing[0].name
    cached = Path(local) if local else None
    if cached is not None and cached.is_file() and cached.suffix.lower() in set(SUFFIXES.values()):
        into.mkdir(parents=True, exist_ok=True)
        name = f"{portal_id}{cached.suffix.lower()}"
        (into / name).write_bytes(cached.read_bytes())
        return name
    if not url:
        return None
    try:
        response = client.get(url, headers={"User-Agent": user_agent}, follow_redirects=True)
        response.raise_for_status()
    except Exception:  # noqa: BLE001 - a corpus is allowed to be incomplete
        return None
    media_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    suffix = SUFFIXES.get(media_type)
    if suffix is None:
        return None
    into.mkdir(parents=True, exist_ok=True)
    name = f"{portal_id}{suffix}"
    (into / name).write_bytes(response.content)
    return name


SHEET_STYLE = """
:root { --ink: #1a1a1a; --paper: #fff; --edge: #d8d8d8; --dim: #666; }
:root:not([data-theme="light"]) { }
body { font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 0 auto; padding: 2rem 1.5rem 6rem; max-width: 68rem;
       color: var(--ink); background: var(--paper); }
h1 { font-size: 1.6rem; margin: 0 0 .3rem; }
.lede { color: var(--dim); max-width: 46rem; }
.plan { border-top: 1px solid var(--edge); padding: 2rem 0; }
.plan h2 { font-size: 1.05rem; margin: 0 0 .2rem; font-family: ui-monospace, monospace; }
.plan .where { color: var(--dim); margin: 0 0 1rem; }
.frame { position: relative; display: inline-block; max-width: 100%; }
.frame img { display: block; max-width: 100%; height: auto; }
.frame svg { position: absolute; inset: 0; width: 100%; height: 100%;
             pointer-events: none; opacity: .55; }
pre { background: #f6f6f6; border: 1px solid var(--edge); border-radius: 6px;
      padding: .8rem 1rem; overflow-x: auto; font-size: .85rem; }
"""

# A ring of ticks over the plan, so a bearing can be read off rather than
# estimated. Drawn in the image's own terms, which is what is being annotated.
RING = """<svg viewBox="0 0 100 100" preserveAspectRatio="none">
  <g stroke="#d00" stroke-width=".25" fill="none">
    <line x1="50" y1="0" x2="50" y2="100"/><line x1="0" y1="50" x2="100" y2="50"/>
    <line x1="0" y1="0" x2="100" y2="100"/><line x1="100" y1="0" x2="0" y2="100"/>
  </g>
  <g fill="#d00" font-size="4" font-family="monospace" text-anchor="middle">
    <text x="50" y="5">0</text><text x="95" y="51">90</text>
    <text x="50" y="98">180</text><text x="6" y="51">270</text>
    <text x="82" y="20">45</text><text x="82" y="84">135</text>
    <text x="18" y="84">225</text><text x="18" y="20">315</text>
  </g>
</svg>"""


def render_sheet(entries: list[tuple[Annotation, bytes, str]]) -> str:
    """A self-contained page: every plan, a bearing ring, and its TOML block."""
    parts = [
        "<title>Floorplan ground truth</title>",
        f"<style>{SHEET_STYLE}</style>",
        "<h1>Floorplan ground truth</h1>",
        f"<p class='lede'>Bearings are on the image: <strong>{COMPASS_HINT}</strong>. "
        "The red ring is drawn over each plan as a protractor - read the angle off "
        "it rather than estimating. For each plan, write which way the north "
        "indicator points and which way the main living space's windows face, "
        "into <code>tests/fixtures/floorplans/annotations.toml</code>. A "
        "dual-aspect flat gets two window bearings. If no compass is drawn, "
        "delete the <code>north_clock</code> line.</p>",
    ]
    for entry, image, media_type in entries:
        encoded = base64.b64encode(image).decode()
        block = render_annotations({entry.portal_id: entry}).split("\n\n", 1)[1]
        parts.append(
            f"<section class='plan'><h2>{entry.portal_id}</h2>"
            f"<p class='where'>{_escape(entry.address)}</p>"
            f"<div class='frame'><img src='data:{media_type};base64,{encoded}' "
            f"alt='floorplan {entry.portal_id}'>{RING}</div>"
            f"<pre>{block.strip()}</pre></section>"
        )
    return "\n".join(parts)


# --- editing the answers in a browser, which writes them straight to the file ---


def apply_update(
    annotations: dict[str, Annotation], portal_id: str, payload: dict
) -> dict[str, Annotation]:
    """One plan's answers, replaced. Everything else is left exactly as it was.

    Deliberately not a merge of the payload into the existing entry: the editor
    sends the whole of one plan every time, so a field the annotator cleared has
    to be able to become empty again. Other plans are untouched, which is what
    makes autosaving safe with a whole evening's work already in the file.
    """
    if portal_id not in annotations:
        raise KeyError(portal_id)
    north = payload.get("north_clock")
    if north is not None:
        north = int(north)
        if not 0 <= north < 360:
            raise ValueError(f"north_clock out of range: {north}")
    windows = tuple(int(bearing) % 360 for bearing in payload.get("windows_clock") or ())
    existing = annotations[portal_id]
    updated = dict(annotations)
    updated[portal_id] = Annotation(
        portal_id=portal_id,
        file=existing.file,
        address=existing.address,
        north_clock=north,
        windows_clock=windows,
        compass_style=str(payload.get("compass_style", "")),
        notes=str(payload.get("notes", "")),
        done=bool(payload.get("done", False)),
    )
    return updated


EDITOR_SCRIPT = """
const bearing = (from, to) => {
  const angle = Math.atan2(to.x - from.x, from.y - to.y) * 180 / Math.PI;
  return Math.round((angle + 360) % 360);
};
const state = {};
function say(panel, text, bad) {
  const flag = panel.querySelector('.state');
  flag.textContent = text;
  flag.className = 'state' + (bad ? ' bad' : '');
  if (!bad) setTimeout(() => { if (flag.textContent === text) flag.textContent = ''; }, 1600);
}
function collect(panel) {
  const north = panel.querySelector('[name=north]').value.trim();
  return {
    north_clock: north === '' ? null : Number(north),
    windows_clock: panel.querySelector('[name=windows]').value
      .split(',').map(s => s.trim()).filter(Boolean).map(Number),
    compass_style: panel.querySelector('[name=style]').value,
    notes: panel.querySelector('[name=notes]').value,
    done: panel.querySelector('[name=done]').checked,
  };
}
async function save(panel) {
  const body = collect(panel);
  if (body.north_clock !== null && (isNaN(body.north_clock) || body.north_clock < 0 || body.north_clock > 359)) {
    return say(panel, 'north must be 0-359', true);
  }
  if (body.windows_clock.some(isNaN)) return say(panel, 'windows must be numbers', true);
  const reply = await fetch('/annotations/' + panel.dataset.plan, {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
  });
  say(panel, reply.ok ? 'saved' : 'NOT saved: ' + reply.status, !reply.ok);
  if (reply.ok) progress();
}
function progress() {
  const all = document.querySelectorAll('.plan');
  const done = document.querySelectorAll('[name=done]:checked');
  document.getElementById('progress').textContent = done.length + ' of ' + all.length + ' done';
}
function arm(panel, mode) {
  state[panel.dataset.plan] = {mode, first: null};
  panel.querySelectorAll('.pick').forEach(b => b.classList.remove('armed'));
  panel.querySelector('.pick[data-mode=' + mode + ']').classList.add('armed');
  say(panel, mode === 'north' ? 'click the middle of the compass' : 'click inside the room');
}
document.querySelectorAll('.plan').forEach(panel => {
  panel.querySelectorAll('input, select, textarea').forEach(field => {
    field.addEventListener('change', () => save(panel));
  });
  panel.querySelectorAll('.pick').forEach(button => {
    button.addEventListener('click', () => arm(panel, button.dataset.mode));
  });
  panel.querySelector('.frame').addEventListener('click', event => {
    const picking = state[panel.dataset.plan];
    if (!picking) return;
    const box = event.currentTarget.getBoundingClientRect();
    const at = {x: event.clientX - box.left, y: event.clientY - box.top};
    if (!picking.first) {
      picking.first = at;
      say(panel, picking.mode === 'north' ? 'now click where north points'
                                         : 'now click outside the window wall');
      return;
    }
    const angle = bearing(picking.first, at);
    if (picking.mode === 'north') {
      panel.querySelector('[name=north]').value = angle;
    } else {
      const field = panel.querySelector('[name=windows]');
      field.value = field.value.trim() ? field.value.trim() + ', ' + angle : String(angle);
    }
    state[panel.dataset.plan] = null;
    panel.querySelectorAll('.pick').forEach(b => b.classList.remove('armed'));
    save(panel);
  });
});
progress();
"""

EDITOR_STYLE = SHEET_STYLE + """
.bar { position: sticky; top: 0; background: var(--paper); border-bottom: 1px solid var(--edge);
       padding: .7rem 0; margin-bottom: 1rem; z-index: 5; font-variant-numeric: tabular-nums; }
.fields { display: flex; flex-wrap: wrap; gap: .8rem; align-items: flex-end; margin-top: 1rem; }
.fields label { display: flex; flex-direction: column; font-size: .78rem; color: var(--dim);
                text-transform: uppercase; letter-spacing: .04em; gap: .25rem; }
.fields input[type=text], .fields select { font: inherit; padding: .35rem .5rem;
       border: 1px solid var(--edge); border-radius: 5px; background: var(--paper); color: var(--ink); }
.fields input.narrow { width: 6.5rem; } .fields input.wide { width: 20rem; }
.pick { font: inherit; font-size: .85rem; padding: .35rem .7rem; border: 1px solid var(--edge);
        border-radius: 5px; background: var(--paper); color: var(--ink); cursor: pointer; }
.pick.armed { background: #d00; color: #fff; border-color: #d00; }
.frame { cursor: crosshair; }
.state { font-size: .8rem; color: #0a0; margin-left: .5rem; }
.state.bad { color: #c00; font-weight: 600; }
.done-row { display: flex; align-items: center; gap: .4rem; font-size: .85rem; color: var(--dim); }
"""


def _escape(text: str) -> str:
    """Attribute-safe HTML.

    Escaping the apostrophe alone is not enough. A note holding "&copy;" is
    decoded by the browser into a copyright sign, and the next autosave writes
    that back, so the annotator's own words are rewritten by the act of being
    displayed. Same family as the newline that broke the TOML: text a person
    typed has to survive every layer it passes through. The ampersand goes first
    or it re-escapes the replacements.
    """
    for raw, safe in (
        ("&", "&amp;"),
        ("<", "&lt;"),
        (">", "&gt;"),
        ("'", "&#39;"),
        ('"', "&quot;"),
    ):
        text = text.replace(raw, safe)
    return text


def render_editor(entries: list[Annotation]) -> str:
    """The page the local server serves. Images come from the server, not inline."""
    parts = [
        "<title>Floorplan ground truth</title>",
        "<style>" + EDITOR_STYLE + "</style>",
        "<div class='bar'><strong>Floorplan ground truth</strong> &mdash; "
        "<span id='progress'></span> &mdash; saved to annotations.toml as you go</div>",
        "<p class='lede'>Bearings are on the image: <strong>" + COMPASS_HINT + "</strong>. "
        "Use the two <em>pick</em> buttons and click twice on the drawing rather than "
        "typing angles: for north, click the middle of the indicator and then where it "
        "points; for a window, click inside the room and then outside the glazed wall. "
        "A dual-aspect flat gets two window bearings, comma separated. Leave north empty "
        "if no compass is drawn and set the style to <code>none</code>.</p>",
    ]
    for entry in entries:
        windows = ", ".join(str(bearing) for bearing in entry.windows_clock)
        options = "".join(
            "<option" + (" selected" if entry.compass_style == value else "") + ">" + value
            + "</option>"
            for value in ("", "arrow", "circle", "rose", "none")
        )
        parts.append(
            "<section class='plan' data-plan='" + entry.portal_id + "'>"
            "<h2>" + entry.portal_id + "<span class='state'></span></h2>"
            "<p class='where'>" + _escape(entry.address) + "</p>"
            "<div class='frame'><img src='/plans/" + entry.file + "' alt='floorplan'>" + RING + "</div>"
            "<div class='fields'>"
            "<button class='pick' data-mode='north'>pick north</button>"
            "<button class='pick' data-mode='window'>add a window</button>"
            "<label>north<input type='text' class='narrow' name='north' value='"
            + ("" if entry.north_clock is None else str(entry.north_clock)) + "'></label>"
            "<label>windows<input type='text' class='narrow' name='windows' value='"
            + windows + "'></label>"
            "<label>style<select name='style'>" + options + "</select></label>"
            "<label>notes<input type='text' class='wide' name='notes' value='"
            + _escape(entry.notes) + "'></label>"
            "<span class='done-row'><input type='checkbox' name='done'"
            + (" checked" if entry.done else "") + "> done</span>"
            "</div></section>"
        )
    parts.append("<script>" + EDITOR_SCRIPT + "</script>")
    return "\n".join(parts)


def serve(
    annotations_path: Path,
    corpus: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
):
    """A local editor that writes straight to the annotations file.

    A page opened from disk cannot write to disk, and a download into ~/Downloads
    that then has to be moved by hand is barely better than copy-and-paste. So
    the page is served by this instead, and every change is saved where it
    belongs the moment it is made.

    Bound to the loopback address, and never to 0.0.0.0: it writes a file in the
    working tree with no authentication of any kind, which is fine for something
    only this machine can reach and would not be otherwise.
    """
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    # Every save reads the whole file, replaces one plan, and writes it all back.
    # Two of those interleaving loses whichever finished first, silently, with
    # both requests answering 204 - and the editor autosaves on every field that
    # changes, so overlapping saves are the normal case rather than the unlucky
    # one. Protecting an evening of somebody's annotation is the entire job here.
    writing = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes = b"", content_type: str = "text/html"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - the stdlib's spelling
            if self.path in ("/", "/index.html"):
                current = load_annotations(annotations_path)
                entries = [
                    entry
                    for _, entry in sorted(current.items())
                    if (corpus / entry.file).exists()
                ]
                return self._send(200, render_editor(entries).encode(), "text/html; charset=utf-8")
            if self.path.startswith("/plans/"):
                # `.name` is what keeps a crafted path from walking out of the
                # corpus directory, and the file has to be one we put there.
                name = Path(self.path[len("/plans/") :]).name
                image = corpus / name
                if not image.exists() or image.suffix not in set(SUFFIXES.values()):
                    return self._send(404, b"no such plan", "text/plain")
                kind = {v: k for k, v in SUFFIXES.items()}[image.suffix]
                return self._send(200, image.read_bytes(), kind)
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802 - the stdlib's spelling
            if not self.path.startswith("/annotations/"):
                return self._send(404, b"not found", "text/plain")
            portal_id = self.path[len("/annotations/") :]
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
                # Re-read every time rather than holding the file in memory: the
                # annotator may well be editing the TOML by hand in another
                # window, and losing that silently would be the worst thing this
                # could do.
                with writing:
                    current = load_annotations(annotations_path)
                    updated = apply_update(current, portal_id, payload)
                    annotations_path.write_text(render_annotations(updated))
            except KeyError:
                return self._send(404, b"no such plan", "text/plain")
            except (ValueError, TypeError) as exc:
                return self._send(400, str(exc).encode(), "text/plain")
            return self._send(204)

        def log_message(self, *args):  # noqa: A003 - silence the per-request log
            pass

    return ThreadingHTTPServer((host, port), Handler)
