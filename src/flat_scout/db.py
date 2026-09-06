from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import sqlite_utils

from flat_scout.adjudicate import Grade
from flat_scout.criteria import Criterion
from flat_scout.models import Evaluation, ImageReading, ListingData


class IllegalTransition(Exception):
    pass


TRANSITIONS: dict[str, set[str]] = {
    # "evaluated" is reachable from "new" for Portals we never download:
    # there is no fetch step to pass through.
    "new": {"fetch_pending", "prefiltered_out", "evaluated", "unavailable"},
    "fetch_pending": {
        "fetch_pending",
        "fetch_failed",
        "prefiltered_out",
        "evaluated",
        "unavailable",
    },
    "fetch_failed": set(),
    "prefiltered_out": set(),
    "unavailable": set(),
    # A Decision is recorded straight off an evaluated Listing with
    # `flat-scout approve` / `flat-scout reject`.
    "evaluated": {"approved", "rejected"},
    "approved": set(),
    "rejected": set(),
}

DECISION_STATUSES = {"approved", "rejected"}

# Once a Decision has been recorded it is settled. A second one - however it
# arrives - must be a silent no-op rather than an error: first Decision wins.
DECIDED_STATUSES = {
    "approved",
    "rejected",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS listings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    portal TEXT NOT NULL,
    portal_id TEXT NOT NULL,
    url TEXT NOT NULL,
    source TEXT NOT NULL,
    first_seen TEXT NOT NULL,
    address TEXT, postcode TEXT, price_pcm INTEGER, beds INTEGER,
    sqft INTEGER, floor TEXT, furnished TEXT, pet_notes TEXT,
    description TEXT, listed_on TEXT, image_url TEXT,
    epc_caption TEXT, epc_image_url TEXT, floorplan_url TEXT,
    image_path TEXT, epc_image_path TEXT, floorplan_path TEXT,
    council_tax_band TEXT,
    nearest_station TEXT, nearest_station_miles REAL,
    latitude REAL, longitude REAL,
    fetched_at TEXT, fetch_status TEXT, fetch_attempts INTEGER NOT NULL DEFAULT 0,
    epc_band TEXT, epc_band_source TEXT,
    epc_floor_area_sqm REAL, epc_property_type TEXT,
    layout_verdict TEXT, layout_notes TEXT,
    reception_is_separate INTEGER, desk_space INTEGER, bedroom_has_window INTEGER,
    floor_text TEXT,
    north_clock INTEGER, windows_clock TEXT, window_aspects TEXT,
    north_source TEXT,
    images_read_at TEXT, vision_model TEXT,
    verdict TEXT, score REAL, coverage REAL, reasons TEXT, red_flags TEXT,
    agent_questions TEXT, evaluated_at TEXT, model TEXT, evaluation_mode TEXT,
    status TEXT NOT NULL DEFAULT 'new',
    decided_at TEXT, notes TEXT,
    UNIQUE (portal, portal_id)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    listing_id INTEGER,
    at TEXT NOT NULL,
    kind TEXT NOT NULL,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_listings_status ON listings(status);
-- One row per Listing and Criterion. The grade vector is stored rather than
-- only the composite, because that is what makes the brief tunable: changing a
-- weight replays over the whole corpus with no model call at all, and changing
-- a rubric re-grades only the Criteria whose `rubric_hash` moved.
--
-- `determined` is 0 where nothing was read off the flat - the Criterion was
-- skipped, or a configured default supplied the value. A default moves the
-- score and never the coverage, and can never fire a veto.
CREATE TABLE IF NOT EXISTS criterion_grades (
    listing_id  INTEGER NOT NULL,
    criterion   TEXT NOT NULL,
    grade       REAL,
    determined  INTEGER NOT NULL,
    evidence    TEXT,
    concern     TEXT,
    question    TEXT,
    graded_by   TEXT,
    rubric_hash TEXT NOT NULL,
    graded_at   TEXT NOT NULL,
    PRIMARY KEY (listing_id, criterion)
);
-- Every replicate `grade --measure` ever took, never overwritten. The noise
-- fit needs the raw spread, not the summary `criterion_grades` keeps one row
-- of - so a batch is appended here instead of replacing what came before.
CREATE TABLE IF NOT EXISTS measure_runs (
    listing_id  INTEGER NOT NULL,
    criterion   TEXT NOT NULL,
    run         INTEGER NOT NULL,
    grade       REAL,
    rubric_hash TEXT NOT NULL,
    measured_at TEXT NOT NULL,
    PRIMARY KEY (listing_id, criterion, run, measured_at)
);
CREATE TABLE IF NOT EXISTS posterior_fit (
    input_hash TEXT PRIMARY KEY,
    fitted     TEXT NOT NULL,
    fitted_at  TEXT NOT NULL
);
"""

MUTABLE_FIELDS = (
    "address",
    "postcode",
    "price_pcm",
    "beds",
    "sqft",
    "floor",
    "furnished",
    "pet_notes",
    "description",
    "listed_on",
    "image_url",
    "epc_caption",
    "epc_image_url",
    "floorplan_url",
    "council_tax_band",
    "nearest_station",
    "nearest_station_miles",
    "latitude",
    "longitude",
)

# The columns `set_image_paths` may write, and the only ones. Deliberately not
# in MUTABLE_FIELDS above: an upsert carries what a Portal said about a flat,
# and where a picture landed on this disk is not that.
IMAGE_PATH_COLUMNS = ("image_path", "epc_image_path", "floorplan_path")

# Columns added after the live database was created, with their types. An
# ALTER per entry brings that database up to SCHEMA.
ADDED_COLUMNS = {
    "epc_caption": "TEXT",
    "epc_image_url": "TEXT",
    "council_tax_band": "TEXT",
    "nearest_station": "TEXT",
    "nearest_station_miles": "REAL",
    "image_url": "TEXT",
    "floorplan_url": "TEXT",
    # Where the local image cache put each picture. Written only by
    # `set_image_paths`, never by `upsert`: a path is a fact about this
    # machine's disk, not something a Portal payload can have an opinion on.
    "image_path": "TEXT",
    "epc_image_path": "TEXT",
    "floorplan_path": "TEXT",
    # What the vision step read. Discrete columns rather than one JSON blob:
    # `epc_band` wants to be groupable in a report, and every other column on
    # this table is already flat. `layout_notes` is a list, so it is JSON text
    # exactly as `reasons` and `red_flags` are.
    "epc_band": "TEXT",
    # Set from the image we sent, so a band printed on the official certificate
    # can be told apart from one judged off a bar chart.
    "epc_band_source": "TEXT",
    # Printed on the certificate only. The floor area matters most: the brief
    # calls square footage the best signal, and Rightmove's sizings is often
    # absent where a certificate is attached.
    "epc_floor_area_sqm": "REAL",
    "epc_property_type": "TEXT",
    "layout_verdict": "TEXT",
    "layout_notes": "TEXT",
    "reception_is_separate": "INTEGER",
    "desk_space": "INTEGER",
    # A window in every bedroom was a dealbreaker under the retired
    # `bedroom-window` Criterion, and the floorplan is the only place it can
    # be read. Null means the plan did not settle it.
    "bedroom_has_window": "INTEGER",
    # The floor as the floorplan states it. Kept beside `floor` rather than
    # only merged into it, so a floor that came from a drawing can be told
    # from one a Portal supplied.
    "floor_text": "TEXT",
    # The two bearings are the working behind the aspect, kept because a
    # wrong aspect can only be diagnosed by seeing which of them was misread.
    "north_clock": "INTEGER",
    # JSON lists: a flat may face more than one way, and most of them do.
    "windows_clock": "TEXT",
    "window_aspects": "TEXT",
    "north_source": "TEXT",
    "images_read_at": "TEXT",
    "vision_model": "TEXT",
    "latitude": "REAL",
    "longitude": "REAL",
    # The share of Criterion weight evidence answered. Added alongside the
    # nullable score it accompanies, for a database that predates the split.
    "coverage": "REAL",
    # Which evaluator produced the stored score/Verdict. Existing NULL rows
    # predate the weighted rollout and are therefore holistic. Grade presence
    # cannot answer this because the rollout backfills Grades onto those rows.
    "evaluation_mode": "TEXT",
    # The posterior, frozen beside the score for the same reason score is
    # frozen: the Decision answered these numbers.
    "score_low": "REAL",
    "score_high": "REAL",
    "p_hopeful": "REAL",
    "posterior_fit_hash": "TEXT",
}

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Kept so callers can put things beside the database on the same volume.
        self.path = Path(path)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        # Two processes can hold this file at once on purpose. WAL lets the
        # reader and the writer coexist instead of colliding, and the busy
        # timeout makes a collision wait rather than raise: a bare "database is
        # locked" would abort a whole run over one row.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.db = sqlite_utils.Database(self.conn)
        self.table = self.db["listings"]
        self.conn.executescript(SCHEMA)
        self._add_missing_columns()
        self.conn.commit()

    def _add_missing_columns(self) -> None:
        """Bring an existing database up to the current schema.

        CREATE TABLE IF NOT EXISTS silently skips a table that already exists,
        so new columns need an explicit ALTER on databases created earlier.
        """
        present = {row["name"] for row in self.conn.execute("PRAGMA table_info(listings)")}
        for column, column_type in ADDED_COLUMNS.items():
            if column not in present:
                self.conn.execute(f"ALTER TABLE listings ADD COLUMN {column} {column_type}")

    def record_event(self, listing_id: int | None, kind: str, detail: str = "") -> None:
        self.conn.execute(
            "INSERT INTO events (listing_id, at, kind, detail) VALUES (?, ?, ?, ?)",
            (listing_id, _now(), kind, detail),
        )
        self.conn.commit()

    def events_for(self, listing_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM events WHERE listing_id = ? ORDER BY id", (listing_id,)
        ).fetchall()

    def upsert(self, listing: ListingData, source: str) -> int:
        """Insert or update by (portal, portal_id). A None never overwrites a value."""
        data = asdict(listing)
        existing = self.conn.execute(
            "SELECT id FROM listings WHERE portal = ? AND portal_id = ?",
            (listing.portal, listing.portal_id),
        ).fetchone()
        if existing:
            updates = {k: data[k] for k in MUTABLE_FIELDS if data.get(k) is not None}
            if updates:
                self.table.update(existing["id"], updates)
            return int(existing["id"])
        record = {**data, "source": source, "first_seen": _now()}
        # sqlite-utils takes a dict, so the column order is the dict's own and
        # there is no placeholder list to keep in step. Adding a field to
        # ListingData needs no change here at all.
        self.table.insert(record, pk="id")
        listing_id = int(self.table.last_pk)
        self.record_event(listing_id, "discovered", f"{source}:{listing.url}")
        return listing_id

    def get(self, listing_id: int) -> sqlite3.Row:
        row = self.conn.execute(
            "SELECT * FROM listings WHERE id = ?", (listing_id,)
        ).fetchone()
        if row is None:
            raise KeyError(listing_id)
        return row

    def by_status(self, status: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM listings WHERE status = ? ORDER BY id", (status,)
        ).fetchall()

    def transition(self, listing_id: int, to_status: str) -> bool:
        """Move a Listing to a new status. False means the Decision was already made."""
        current = self.get(listing_id)["status"]
        if current == to_status:
            return False
        if to_status in DECISION_STATUSES and current in DECIDED_STATUSES:
            self.record_event(
                listing_id, "decision_ignored", f"{to_status}, already {current}"
            )
            return False
        if to_status not in TRANSITIONS.get(current, set()):
            raise IllegalTransition(f"{current} -> {to_status} (listing {listing_id})")
        if to_status in DECISION_STATUSES:
            self.conn.execute(
                "UPDATE listings SET status = ?, decided_at = ? WHERE id = ?",
                (to_status, _now(), listing_id),
            )
        else:
            self.conn.execute(
                "UPDATE listings SET status = ? WHERE id = ?", (to_status, listing_id)
            )
        self.conn.commit()
        self.record_event(listing_id, "transition", f"{current} -> {to_status}")
        return True

    def set_fetch_result(self, listing_id: int, listing: ListingData | None, ok: bool) -> None:
        if listing is not None:
            self.upsert(listing, source=self.get(listing_id)["source"])
        self.conn.execute(
            "UPDATE listings SET fetched_at = ?, fetch_status = ?, "
            "fetch_attempts = fetch_attempts + 1 WHERE id = ?",
            (_now(), "ok" if ok else "failed", listing_id),
        )
        self.conn.commit()

    def set_image_paths(self, listing_id: int, paths: dict[str, str]) -> None:
        """Record where the cache put a Listing's pictures. Only the given keys.

        Written one field at a time on purpose: a run that cached the floorplan
        and could not reach the EPC must leave the EPC's path exactly as it was
        rather than blanking it.
        """
        unknown = sorted(set(paths) - set(IMAGE_PATH_COLUMNS))
        if unknown:
            raise ValueError(
                f"not an image path column: {', '.join(unknown)}. "
                f"Expected any of {', '.join(IMAGE_PATH_COLUMNS)}."
            )
        if not paths:
            return
        assignments = ", ".join(f"{column} = ?" for column in paths)
        self.conn.execute(
            f"UPDATE listings SET {assignments} WHERE id = ?",
            (*paths.values(), listing_id),
        )
        self.conn.commit()

    def set_image_reading(
        self, listing_id: int, reading: ImageReading, model: str
    ) -> None:
        """Record what the vision step read, including that it read nothing.

        `images_read_at` is set whatever the reading says. An all-unknown
        reading is a real result — the graph was illegible — and stamping it
        stops a later retry paying to look at the same two images again.
        """
        self.conn.execute(
            "UPDATE listings SET epc_band = ?, epc_band_source = ?, "
            "epc_floor_area_sqm = ?, epc_property_type = ?, "
            "layout_verdict = ?, layout_notes = ?, "
            "reception_is_separate = ?, desk_space = ?, bedroom_has_window = ?, floor_text = ?, "
            "north_clock = ?, windows_clock = ?, window_aspects = ?, north_source = ?, "
            "images_read_at = ?, "
            "vision_model = ? WHERE id = ?",
            (
                reading.epc_band,
                reading.epc_band_source,
                reading.epc_floor_area_sqm,
                reading.epc_property_type,
                reading.layout_verdict,
                json.dumps(reading.layout_notes),
                reading.reception_is_separate,
                reading.desk_space,
                reading.bedroom_has_window,
                reading.floor_text,
                reading.north_clock,
                json.dumps(reading.windows_clock),
                json.dumps(reading.window_aspects),
                reading.north_source,
                _now(),
                model,
                listing_id,
            ),
        )
        self.conn.commit()

    def set_floor_from_plan(self, listing_id: int, floor: str) -> bool:
        """Fill in the floor from a floorplan, if the Portal gave none.

        Never overwrites a floor the Portal supplied, on the same principle as
        `upsert`: a value already on the row was read from the page, and a
        drawing is not grounds to argue with it. Returns whether it wrote.
        """
        current = self.get(listing_id)["floor"]
        if current:
            return False
        self.conn.execute(
            "UPDATE listings SET floor = ? WHERE id = ?", (floor, listing_id)
        )
        self.conn.commit()
        self.record_event(listing_id, "floor_from_plan", floor)
        return True

    def set_evaluation(
        self,
        listing_id: int,
        evaluation: Evaluation,
        model: str,
        mode: str = "holistic",
    ) -> None:
        self.conn.execute(
            "UPDATE listings SET verdict = ?, score = ?, coverage = ?, score_low = ?, "
            "score_high = ?, p_hopeful = ?, posterior_fit_hash = ?, reasons = ?, red_flags = ?, "
            "agent_questions = ?, evaluated_at = ?, model = ?, evaluation_mode = ? "
            "WHERE id = ?",
            (
                evaluation.verdict,
                evaluation.score,
                evaluation.coverage,
                evaluation.score_low,
                evaluation.score_high,
                evaluation.p_hopeful,
                evaluation.posterior_fit_hash,
                json.dumps(evaluation.reasons),
                json.dumps(evaluation.red_flags),
                json.dumps(evaluation.agent_questions),
                _now(),
                model,
                mode,
                listing_id,
            ),
        )
        self.conn.commit()

    def set_adjudication(self, listing_id: int, evaluation: Evaluation) -> None:
        """Replace the derived weighted result without pretending to re-evaluate.

        A queued Listing is re-adjudicated when weights or gates change.  The
        Grade vector and model run are unchanged, so keep their original model
        and timestamp while making the stored, reader-facing result agree with
        the decision in force now.
        """
        self.conn.execute(
            "UPDATE listings SET verdict = ?, score = ?, coverage = ?, score_low = ?, "
            "score_high = ?, p_hopeful = ?, posterior_fit_hash = ?, reasons = ?, red_flags = ?, "
            "agent_questions = ? WHERE id = ?",
            (
                evaluation.verdict,
                evaluation.score,
                evaluation.coverage,
                evaluation.score_low,
                evaluation.score_high,
                evaluation.p_hopeful,
                evaluation.posterior_fit_hash,
                json.dumps(evaluation.reasons),
                json.dumps(evaluation.red_flags),
                json.dumps(evaluation.agent_questions),
                listing_id,
            ),
        )
        self.conn.commit()

    def set_criterion_grades(
        self, listing_id: int, pairs: list[tuple["Criterion", "Grade"]]
    ) -> None:
        """Write one row for each Criterion, replacing any earlier grade.

        A templated question is deliberately not stored. It is rendered at read
        time from the Criterion's current `ask`, so editing that text costs
        nothing - the same reason `ask` is not in the rubric hash.
        """
        self.conn.executemany(
            "INSERT OR REPLACE INTO criterion_grades "
            "(listing_id, criterion, grade, determined, evidence, concern, "
            "question, graded_by, rubric_hash, graded_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    listing_id,
                    criterion.slug,
                    grade.value,
                    int(grade.determined),
                    grade.evidence,
                    grade.concern,
                    grade.question,
                    grade.graded_by,
                    criterion.rubric_hash,
                    _now(),
                )
                for criterion, grade in pairs
            ],
        )
        self.conn.commit()

    def add_measure_runs(
        self,
        listing_id: int,
        criterion: str,
        rubric_hash: str,
        values: list[float | None],
    ) -> None:
        """One `grade --measure` batch for one (Listing, Criterion).

        A NULL grade is a run that returned unknown - kept, because silence
        that comes and goes is itself a fact about a rubric. `measured_at` is
        shared by the batch: it is the group key the noise fit clusters
        replicates by.
        """
        stamp = _now()
        self.conn.executemany(
            "INSERT OR REPLACE INTO measure_runs "
            "(listing_id, criterion, run, grade, rubric_hash, measured_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (listing_id, criterion, run, value, rubric_hash, stamp)
                for run, value in enumerate(values)
            ],
        )
        self.conn.commit()

    def measure_runs(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM measure_runs "
            "ORDER BY criterion, listing_id, measured_at, run"
        ).fetchall()

    def fit_cache(self, input_hash: str) -> str | None:
        row = self.conn.execute(
            "SELECT fitted FROM posterior_fit WHERE input_hash = ?", (input_hash,)
        ).fetchone()
        return row["fitted"] if row else None

    def latest_fit_cache(self) -> tuple[str, str] | None:
        """The last fit that completed, whatever the inputs are now."""
        row = self.conn.execute(
            "SELECT input_hash, fitted FROM posterior_fit "
            "ORDER BY fitted_at DESC LIMIT 1"
        ).fetchone()
        return (row["input_hash"], row["fitted"]) if row else None

    def set_fit_cache(self, input_hash: str, fitted: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO posterior_fit (input_hash, fitted, fitted_at) "
            "VALUES (?, ?, ?)",
            (input_hash, fitted, _now()),
        )
        self.conn.commit()

    def criterion_grades(self, listing_id: int) -> dict[str, sqlite3.Row]:
        rows = self.conn.execute(
            "SELECT * FROM criterion_grades WHERE listing_id = ?", (listing_id,)
        ).fetchall()
        return {row["criterion"]: row for row in rows}

    def stale_criteria(
        self, listing_id: int, criteria: list["Criterion"]
    ) -> list[str]:
        """The slugs whose stored grade no longer matches the rubric on disk.

        A Criterion never graded for this Listing counts as stale, which is how
        a newly added Criterion reaches the backfill.
        """
        stored = self.criterion_grades(listing_id)
        return [
            criterion.slug
            for criterion in criteria
            if criterion.slug not in stored
            or stored[criterion.slug]["rubric_hash"] != criterion.rubric_hash
        ]

    def graded_pairs(
        self, listing_id: int, criteria: list["Criterion"]
    ) -> list[tuple["Criterion", Grade]]:
        """The stored grade vector, rebuilt against the Criteria on disk.

        A Criterion with no stored row comes back unknown rather than absent,
        so the coverage denominator is always the whole brief.

        The `unknown:` rule is applied here rather than trusted from the row.
        `unknown` is kept out of the rubric hash precisely so an edit to it
        regrades nothing - which only holds if the read path applies it.
        Without this, a numeric default reached a Listing graded after the
        edit and no Listing graded before it, and a never-graded Criterion
        never got one at all.
        """
        # Lazily, because grading.py reaches pydantic_ai and this module has
        # no other reason to.
        from flat_scout.grading import apply_unknown

        stored = self.criterion_grades(listing_id)
        pairs = []
        for criterion in criteria:
            row = stored.get(criterion.slug)
            if row is None:
                grade = Grade(criterion.slug, None, False, "never graded")
            else:
                determined = bool(row["determined"])
                grade = Grade(
                    criterion=criterion.slug,
                    # A stored default is not stored evidence, and only a
                    # determined row carries the latter. `apply_unknown` is the
                    # one thing in the codebase that builds a Grade with
                    # `determined=False` and a value; every grader and the model
                    # path alike set `determined = value is not None`. So a
                    # non-determined row's number is always the default that was
                    # in force when it was written, and handing it back would
                    # make `apply_unknown` below see a value and return it
                    # untouched - the old default outliving the edit.
                    #
                    # `unknown` is kept out of the rubric hash precisely so an
                    # edit to it restages nothing, which means this read is the
                    # only place a new default can arrive.
                    # Reconstructed as absent, the rule in force now decides,
                    # and the knob is reversible: setting it back to `skip`
                    # drops the number instead of leaving it stranded in every
                    # row already graded.
                    value=row["grade"] if determined else None,
                    determined=determined,
                    evidence=row["evidence"] or "",
                    graded_by=row["graded_by"] or "",
                    concern=row["concern"],
                    question=row["question"],
                )
            pairs.append((criterion, apply_unknown(criterion, grade)))
        return pairs
