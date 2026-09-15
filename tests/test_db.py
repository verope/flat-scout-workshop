import dataclasses
import json
import sqlite3

import pytest

from flat_scout.db import Database, IllegalTransition
from flat_scout.models import ImageReading, ListingData


def make_listing(**overrides) -> ListingData:
    base = dict(
        portal="rightmove",
        portal_id="92053296",
        url="https://www.rightmove.co.uk/properties/92053296",
        address="Riverlight Quay, Nine Elms",
        postcode="SW8",
        price_pcm=3100,
        beds=1,
        sqft=441,
        floor="6th",
        furnished="Furnished",
        pet_notes=None,
        description="A flat.",
        listed_on="2026-08-16",
    )
    base.update(overrides)
    return ListingData(**base)


def test_upsert_is_idempotent_on_portal_and_id(tmp_path):
    db = Database(tmp_path / "flats.db")
    first = db.upsert(make_listing(), source="manual")
    second = db.upsert(make_listing(price_pcm=3050), source="manual")
    assert first == second
    assert db.get(first)["price_pcm"] == 3050


def test_a_decision_is_recorded_straight_off_an_evaluated_listing(tmp_path):
    """`flat-scout approve` / `reject` act on an evaluated Listing.

    A second Decision is a silent no-op - a stale decision, not a bug - and a
    Decision on a Listing that was never evaluated is a programming error.
    """
    db = Database(tmp_path / "flats.db")
    for decision in ("approved", "rejected"):
        listing_id = db.upsert(make_listing(portal_id=decision), source="manual")
        db.transition(listing_id, "evaluated")
        assert db.transition(listing_id, decision) is True
        assert db.get(listing_id)["status"] == decision
        assert [row["kind"] for row in db.events_for(listing_id)].count("transition") == 2

    assert db.transition(listing_id, "approved") is False
    assert db.get(listing_id)["status"] == "rejected"

    fresh = db.upsert(make_listing(portal_id="fresh"), source="manual")
    with pytest.raises(IllegalTransition):
        db.transition(fresh, "approved")


def test_upsert_never_overwrites_a_known_value_with_none(tmp_path):
    db = Database(tmp_path / "flats.db")
    listing_id = db.upsert(make_listing(), source="alert")
    db.upsert(make_listing(sqft=None, furnished=None), source="alert")
    row = db.get(listing_id)
    assert row["sqft"] == 441
    assert row["furnished"] == "Furnished"


def evaluated_listing(tmp_path) -> tuple[Database, int]:
    db = Database(tmp_path / "flats.db")
    listing_id = db.upsert(make_listing(), source="alert")
    for status in ("fetch_pending", "evaluated"):
        db.transition(listing_id, status)
    return db, listing_id


def test_a_competing_decision_is_a_noop_not_an_error(tmp_path):
    """One partner approves while the other's stale rejection arrives."""
    db, listing_id = evaluated_listing(tmp_path)
    assert db.transition(listing_id, "approved") is True
    assert db.transition(listing_id, "rejected") is False
    assert db.get(listing_id)["status"] == "approved"


def test_the_epc_tax_and_station_signals_round_trip(tmp_path):
    db = Database(tmp_path / "flats.db")
    listing_id = db.upsert(
        make_listing(
            epc_caption="EE Rating",
            council_tax_band="E",
            nearest_station="Elm Road Underground Station",
            nearest_station_miles=0.26,
        ),
        source="manual",
    )
    row = db.get(listing_id)
    assert row["epc_caption"] == "EE Rating"
    assert row["council_tax_band"] == "E"
    assert row["nearest_station"] == "Elm Road Underground Station"
    assert row["nearest_station_miles"] == 0.26


def test_the_headline_photo_round_trips(tmp_path):
    db = Database(tmp_path / "flats.db")
    listing_id = db.upsert(
        make_listing(image_url="https://media.rightmove.co.uk/a.jpeg"), source="manual"
    )
    assert db.get(listing_id)["image_url"] == "https://media.rightmove.co.uk/a.jpeg"


def test_the_floorplan_url_round_trips(tmp_path):
    db = Database(tmp_path / "flats.db")
    listing_id = db.upsert(
        make_listing(floorplan_url="https://media.rightmove.co.uk/property-floorplan/a.jpeg"),
        source="manual",
    )
    assert db.get(listing_id)["floorplan_url"] == (
        "https://media.rightmove.co.uk/property-floorplan/a.jpeg"
    )


def test_a_database_created_before_the_floorplan_column_gains_it_by_alter(tmp_path):
    """The live database has rows and no floorplan_url. Migration must not recreate it."""
    path = tmp_path / "flats.db"
    older = Database(path)
    older.conn.execute("ALTER TABLE listings DROP COLUMN floorplan_url")
    older.conn.execute(
        "INSERT INTO listings (portal, portal_id, url, source, first_seen) "
        "VALUES ('rightmove', '1', 'u', 'alert', '2026-08-16')"
    )
    older.conn.commit()
    older.conn.close()

    db = Database(path)
    assert db.get(1)["floorplan_url"] is None  # the pre-existing row survives
    db.upsert(
        make_listing(
            portal_id="1",
            floorplan_url="https://media.rightmove.co.uk/property-floorplan/a.jpeg",
        ),
        source="alert",
    )
    assert db.get(1)["floorplan_url"] == (
        "https://media.rightmove.co.uk/property-floorplan/a.jpeg"
    )


def test_an_image_reading_round_trips(tmp_path):
    db = Database(tmp_path / "flats.db")
    listing_id = db.upsert(make_listing(), source="manual")
    db.set_image_reading(
        listing_id,
        ImageReading(
            epc_band="C",
            layout_verdict="good",
            layout_notes=["reception 4.2m x 3.6m"],
            reception_is_separate=True,
            desk_space=False,
            bedroom_has_window=True,
        ),
        model="openrouter:z-ai/glm-5.3-flash",
    )
    row = db.get(listing_id)
    assert row["bedroom_has_window"] == 1
    assert row["epc_band"] == "C"
    assert row["layout_verdict"] == "good"
    assert json.loads(row["layout_notes"]) == ["reception 4.2m x 3.6m"]
    assert row["reception_is_separate"] == 1
    assert row["desk_space"] == 0
    assert row["vision_model"] == "openrouter:z-ai/glm-5.3-flash"
    assert row["images_read_at"]


def test_a_certificate_reading_round_trips(tmp_path):
    db = Database(tmp_path / "flats.db")
    listing_id = db.upsert(make_listing(), source="manual")
    db.set_image_reading(
        listing_id,
        ImageReading(
            epc_band="B",
            epc_band_source="certificate",
            epc_floor_area_sqm=51.0,
            epc_property_type="Mid-floor flat",
        ),
        model="openrouter:z-ai/glm-5.3-flash",
    )
    row = db.get(listing_id)
    assert row["epc_band"] == "B"
    assert row["epc_band_source"] == "certificate"
    assert row["epc_floor_area_sqm"] == 51.0
    assert row["epc_property_type"] == "Mid-floor flat"


def test_a_database_created_before_the_certificate_columns_is_migrated(tmp_path):
    path = tmp_path / "flats.db"
    older = Database(path)
    for column in ("epc_band_source", "epc_floor_area_sqm", "epc_property_type"):
        older.conn.execute(f"ALTER TABLE listings DROP COLUMN {column}")
    older.conn.execute(
        "INSERT INTO listings (portal, portal_id, url, source, first_seen) "
        "VALUES ('rightmove', '1', 'u', 'alert', '2026-08-16')"
    )
    older.conn.commit()
    older.conn.close()

    db = Database(path)
    assert db.get(1)["portal_id"] == "1"  # the pre-existing row survives
    db.set_image_reading(
        1,
        ImageReading(epc_band="B", epc_band_source="certificate", epc_floor_area_sqm=51.0),
        model="m",
    )
    assert db.get(1)["epc_floor_area_sqm"] == 51.0


def test_an_unreadable_reading_is_still_recorded_as_having_been_read(tmp_path):
    """images_read_at is what stops a retry paying for the same images twice."""
    db = Database(tmp_path / "flats.db")
    listing_id = db.upsert(make_listing(), source="manual")
    db.set_image_reading(listing_id, ImageReading(), model="openrouter:z-ai/glm-5.3-flash")
    row = db.get(listing_id)
    assert row["epc_band"] is None
    assert row["reception_is_separate"] is None
    assert row["images_read_at"]


def test_a_database_created_before_the_image_reading_columns_is_migrated(tmp_path):
    path = tmp_path / "flats.db"
    older = Database(path)
    for column in (
        "epc_band",
        "layout_verdict",
        "layout_notes",
        "reception_is_separate",
        "desk_space",
        "images_read_at",
        "vision_model",
    ):
        older.conn.execute(f"ALTER TABLE listings DROP COLUMN {column}")
    older.conn.execute(
        "INSERT INTO listings (portal, portal_id, url, source, first_seen) "
        "VALUES ('rightmove', '1', 'u', 'alert', '2026-08-16')"
    )
    older.conn.commit()
    older.conn.close()

    db = Database(path)
    assert db.get(1)["portal_id"] == "1"  # the pre-existing row survives
    db.set_image_reading(1, ImageReading(epc_band="D"), model="m")
    assert db.get(1)["epc_band"] == "D"


def test_a_download_fills_the_new_signals_in_on_a_row_from_an_alert(tmp_path):
    db = Database(tmp_path / "flats.db")
    listing_id = db.upsert(make_listing(), source="alert")
    db.upsert(make_listing(council_tax_band="E", nearest_station="Vauxhall"), source="alert")
    assert db.get(listing_id)["council_tax_band"] == "E"
    assert db.get(listing_id)["nearest_station"] == "Vauxhall"


def test_a_database_created_before_the_new_columns_is_migrated(tmp_path):
    """The live database predates these columns and must not break on upgrade."""
    path = tmp_path / "flats.db"
    old = sqlite3.connect(path)
    old.executescript(
        """
        CREATE TABLE listings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portal TEXT NOT NULL, portal_id TEXT NOT NULL, url TEXT NOT NULL,
            source TEXT NOT NULL, first_seen TEXT NOT NULL,
            address TEXT, postcode TEXT, price_pcm INTEGER, beds INTEGER,
            sqft INTEGER, floor TEXT, furnished TEXT, pet_notes TEXT,
            description TEXT, listed_on TEXT,
            fetched_at TEXT, fetch_status TEXT,
            fetch_attempts INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'new',
            UNIQUE (portal, portal_id)
        );
        """
    )
    old.execute(
        "INSERT INTO listings (portal, portal_id, url, source, first_seen) "
        "VALUES ('rightmove', '1', 'u', 'alert', '2026-08-16')"
    )
    old.commit()
    old.close()

    db = Database(path)
    listing_id = db.upsert(
        make_listing(
            epc_caption="EPC 1",
            council_tax_band="C",
            nearest_station_miles=0.3,
            image_url="https://media.rightmove.co.uk/a.jpeg",
            latitude=51.4818,
            longitude=-0.13687,
        ),
        source="manual",
    )
    row = db.get(listing_id)
    assert row["epc_caption"] == "EPC 1"
    assert row["council_tax_band"] == "C"
    assert row["nearest_station_miles"] == 0.3
    assert row["image_url"] == "https://media.rightmove.co.uk/a.jpeg"
    assert row["latitude"] == 51.4818
    assert row["longitude"] == -0.13687
    assert db.get(1)["portal_id"] == "1"  # the pre-existing row survives


def test_the_coordinate_columns_round_trip(tmp_path):
    db = Database(tmp_path / "flats.db")
    listing_id = db.upsert(
        make_listing(latitude=51.4818, longitude=-0.13687),
        source="manual",
    )
    row = db.get(listing_id)
    assert row["latitude"] == 51.4818
    assert row["longitude"] == -0.13687


# --- The INSERT arity guard -------------------------------------------------
#
# upsert's INSERT is a hand-written column list beside a hand-written string of
# placeholders. Three features have now appended to both, and a bad merge there
# would not crash: it would shift every later value one column left and write,
# say, a description into `listed_on`, silently and permanently. These two tests
# are the guard. The first catches a forgotten `?`; the second catches the
# nastier case where the counts still agree but the order does not.


def test_every_listing_field_round_trips_through_upsert_and_get(tmp_path):
    """A distinct value in every field, so a shifted column cannot pass unnoticed.

    Two adjacent TEXT columns swapped by a merge would survive any test that
    only checks a handful of fields, or that leaves fields None. Nothing here is
    None and no two values are equal.
    """
    fields = [f for f in dataclasses.fields(ListingData)]
    values: dict[str, object] = {}
    for index, field in enumerate(fields, start=1):
        if field.type == "int | None":
            values[field.name] = 1000 + index
        elif field.type == "float | None":
            values[field.name] = round(index + 0.5, 2)
        else:
            values[field.name] = f"{field.name}-{index}"
    listing = ListingData(**values)  # type: ignore[arg-type]

    db = Database(tmp_path / "flats.db")
    row = db.get(db.upsert(listing, source="manual"))

    for field in fields:
        assert row[field.name] == getattr(listing, field.name), (
            f"{field.name} did not survive the round trip - "
            "the INSERT's columns and its values are out of step"
        )


def test_a_database_created_before_the_photo_column_gains_it_by_alter(tmp_path):
    """The live database has rows and no image_url. Migration must not recreate it."""
    path = tmp_path / "flats.db"
    older = Database(path)
    older.conn.execute("ALTER TABLE listings DROP COLUMN image_url")
    older.conn.execute(
        "INSERT INTO listings (portal, portal_id, url, source, first_seen) "
        "VALUES ('rightmove', '1', 'u', 'alert', '2026-08-16')"
    )
    older.conn.commit()
    older.conn.close()

    db = Database(path)
    assert "image_url" in {r["name"] for r in db.conn.execute("PRAGMA table_info(listings)")}
    assert db.get(1)["portal_id"] == "1"  # the pre-existing row survives
    assert db.get(1)["image_url"] is None
    db.upsert(
        make_listing(portal_id="1", image_url="https://media.rightmove.co.uk/a.jpeg"),
        source="alert",
    )
    assert db.get(1)["image_url"] == "https://media.rightmove.co.uk/a.jpeg"


def test_a_database_created_before_evaluation_modes_gains_the_column(tmp_path):
    path = tmp_path / "flats.db"
    older = Database(path)
    older.conn.execute("ALTER TABLE listings DROP COLUMN evaluation_mode")
    older.conn.commit()
    older.conn.close()

    db = Database(path)
    columns = {row["name"] for row in db.conn.execute("PRAGMA table_info(listings)")}
    assert "evaluation_mode" in columns


def test_an_evaluation_with_no_score_stores_null_and_its_coverage(tmp_path):
    from flat_scout.db import Database
    from flat_scout.models import Evaluation, ListingData

    db = Database(tmp_path / "flats.db")
    listing_id = db.upsert(
        ListingData(portal="rightmove", portal_id="1", url="u"), source="manual"
    )
    db.set_evaluation(
        listing_id,
        Evaluation(verdict="borderline", score=None, coverage=0.4, reasons=["thin"]),
        model="test",
    )
    row = db.get(listing_id)
    assert row["score"] is None
    assert row["coverage"] == 0.4
    assert row["verdict"] == "borderline"
    assert row["evaluation_mode"] == "holistic"


def _graded(tmp_path):
    from flat_scout.adjudicate import Grade
    from flat_scout.criteria import Criterion
    from flat_scout.db import Database
    from flat_scout.models import ListingData

    db = Database(tmp_path / "flats.db")
    listing_id = db.upsert(
        ListingData(portal="rightmove", portal_id="1", url="u"), source="manual"
    )
    warmth = Criterion(
        slug="warmth", name="Warmth", description="d", weight=4,
        grade={"field": "epc_band"}, body="b", rubric_hash="aaaa",
    )
    grade = Grade(
        criterion="warmth", value=8.0, determined=True,
        evidence="EPC band C.", graded_by="map",
    )
    return db, listing_id, warmth, grade


def test_a_grade_is_stored_and_read_back_by_slug(tmp_path):
    db, listing_id, warmth, grade = _graded(tmp_path)
    db.set_criterion_grades(listing_id, [(warmth, grade)])
    stored = db.criterion_grades(listing_id)
    assert stored["warmth"]["grade"] == 8.0
    assert stored["warmth"]["determined"] == 1
    assert stored["warmth"]["evidence"] == "EPC band C."
    assert stored["warmth"]["rubric_hash"] == "aaaa"


def test_regrading_replaces_rather_than_duplicates(tmp_path):
    from flat_scout.adjudicate import Grade

    db, listing_id, warmth, grade = _graded(tmp_path)
    db.set_criterion_grades(listing_id, [(warmth, grade)])
    again = Grade(criterion="warmth", value=2.0, determined=True, evidence="EPC band E.")
    db.set_criterion_grades(listing_id, [(warmth, again)])
    stored = db.criterion_grades(listing_id)
    assert len(stored) == 1
    assert stored["warmth"]["grade"] == 2.0


def test_an_unknown_grade_stores_null_and_not_zero(tmp_path):
    from flat_scout.adjudicate import Grade

    db, listing_id, warmth, _ = _graded(tmp_path)
    unknown = Grade(criterion="warmth", value=None, determined=False, evidence="not read")
    db.set_criterion_grades(listing_id, [(warmth, unknown)])
    stored = db.criterion_grades(listing_id)
    assert stored["warmth"]["grade"] is None
    assert stored["warmth"]["determined"] == 0


def test_a_weight_edit_makes_nothing_stale(tmp_path):
    """The whole of the tunability."""
    import dataclasses

    db, listing_id, warmth, grade = _graded(tmp_path)
    db.set_criterion_grades(listing_id, [(warmth, grade)])
    reweighted = dataclasses.replace(warmth, weight=9)
    assert db.stale_criteria(listing_id, [reweighted]) == []


def test_a_rubric_edit_makes_that_criterion_stale(tmp_path):
    import dataclasses

    db, listing_id, warmth, grade = _graded(tmp_path)
    db.set_criterion_grades(listing_id, [(warmth, grade)])
    rewritten = dataclasses.replace(warmth, rubric_hash="bbbb")
    assert db.stale_criteria(listing_id, [rewritten]) == ["warmth"]


def test_a_criterion_never_graded_is_stale(tmp_path):
    import dataclasses

    db, listing_id, warmth, _ = _graded(tmp_path)
    fresh = dataclasses.replace(warmth, slug="quiet", name="Quiet")
    assert db.stale_criteria(listing_id, [warmth, fresh]) == ["warmth", "quiet"]


def test_the_unknown_default_is_applied_on_read_for_a_never_graded_criterion(tmp_path):
    """`unknown` is read-time by design.

    It is deliberately outside the rubric hash, so editing it invalidates no
    stored row - which only holds if the read path applies it. Rebuilding a
    never-graded Criterion as a bare unknown honoured `unknown: skip` by
    accident and dropped a numeric default on the floor.
    """
    import dataclasses

    db, listing_id, warmth, _ = _graded(tmp_path)
    defaulted = dataclasses.replace(warmth, unknown=4)
    [(criterion, grade)] = db.graded_pairs(listing_id, [defaulted])
    assert criterion is defaulted
    assert grade.value == 4
    # Still not evidence: the score weighs it, the coverage does not count it,
    # and no veto can fire on it.
    assert grade.determined is False


def test_the_unknown_default_is_applied_on_read_to_a_stored_null(tmp_path):
    """The row was written under `unknown: skip` and read under `unknown: 4`.

    That edit moves no rubric hash, so nothing is regraded and the read path
    is the only place the new default can arrive.
    """
    import dataclasses

    from flat_scout.adjudicate import Grade

    db, listing_id, warmth, _ = _graded(tmp_path)
    unread = Grade(
        criterion="warmth", value=None, determined=False,
        evidence="The model could not be reached.", graded_by="model:x (failed)",
    )
    db.set_criterion_grades(listing_id, [(warmth, unread)])
    defaulted = dataclasses.replace(warmth, unknown=4)
    [(_, grade)] = db.graded_pairs(listing_id, [defaulted])
    assert grade.value == 4
    assert grade.determined is False


def test_skip_still_reads_back_as_absent(tmp_path):
    """The discrimination for the two above: `unknown: skip` is the shipped
    setting on all five Criteria, and applying a default there would put
    a number on every unread Criterion in the corpus."""
    db, listing_id, warmth, _ = _graded(tmp_path)
    [(_, grade)] = db.graded_pairs(listing_id, [warmth])
    assert warmth.unknown == "skip"
    assert grade.value is None
    assert grade.determined is False


def _defaulted_row(tmp_path):
    """A stored row as `apply_unknown` writes one: not determined, and
    carrying the numeric default in force when it was graded.

    That shape is unambiguous. `apply_unknown` is the only thing in the
    codebase that builds a Grade with `determined=False` and a value - every
    grader and the model path alike set `determined = value is not None` - so
    a stored row like this is a default, never something read off a Listing.
    """
    from flat_scout.adjudicate import Grade

    db, listing_id, warmth, _ = _graded(tmp_path)
    stale_default = Grade(
        criterion="warmth", value=5.0, determined=False,
        evidence="Not determined, so the brief's default of 5 applies.",
        graded_by="model:x",
    )
    db.set_criterion_grades(listing_id, [(warmth, stale_default)])
    return db, listing_id, warmth


def test_a_stored_default_is_dropped_when_the_criterion_goes_back_to_skip(tmp_path):
    """The knob has to be reversible, and `unknown` is outside the rubric hash
    precisely so an edit to it restages nothing.

    Reconstructing the row with its old default already in `grade` meant
    `apply_unknown` saw a value and returned it untouched, so changing the
    default - or changing it back to `skip` - had no effect on anything already
    graded. The value survived every rescore and every report, and no command
    existed to shift it.
    """
    db, listing_id, warmth = _defaulted_row(tmp_path)
    assert warmth.unknown == "skip"
    [(_, grade)] = db.graded_pairs(listing_id, [warmth])
    assert grade.value is None
    assert grade.determined is False


def test_an_edited_default_replaces_the_one_stored_under_the_old_rule(tmp_path):
    """5 was written to the row; the brief now says 3. The read path decides."""
    import dataclasses

    db, listing_id, warmth = _defaulted_row(tmp_path)
    edited = dataclasses.replace(warmth, unknown=3)
    [(_, grade)] = db.graded_pairs(listing_id, [edited])
    assert grade.value == 3
    assert grade.determined is False


def test_a_determined_grade_is_never_mistaken_for_a_default(tmp_path):
    """The discrimination. Evidence read off a Listing is what the row is for,
    and no `unknown` setting may touch it - reading `determined` as the signal
    would be worthless if it also discarded real grades."""
    import dataclasses

    db, listing_id, warmth, grade = _graded(tmp_path)
    db.set_criterion_grades(listing_id, [(warmth, grade)])
    edited = dataclasses.replace(warmth, unknown=3)
    [(_, read_back)] = db.graded_pairs(listing_id, [edited])
    assert read_back.value == 8.0
    assert read_back.determined is True
    assert read_back.evidence == "EPC band C."


def test_a_stored_default_keeps_the_evidence_that_says_why_it_was_not_read(tmp_path):
    """Under `skip` the row reads back as a bare unknown, and the sentence
    naming what could not be read is the whole of what a replay has to show."""
    db, listing_id, warmth = _defaulted_row(tmp_path)
    [(_, grade)] = db.graded_pairs(listing_id, [warmth])
    assert "Not determined" in grade.evidence
    assert grade.graded_by == "model:x"


def test_measure_runs_round_trip(tmp_path):
    db = Database(tmp_path / "flats.db")
    db.add_measure_runs(1, "quiet", "abc123", [4.0, 7.0, None])
    rows = db.measure_runs()
    assert [(r["listing_id"], r["criterion"], r["run"], r["grade"]) for r in rows] == [
        (1, "quiet", 0, 4.0),
        (1, "quiet", 1, 7.0),
        (1, "quiet", 2, None),
    ]
    assert all(r["rubric_hash"] == "abc123" for r in rows)


def test_measure_runs_accumulate_across_calls(tmp_path):
    # A later `grade --measure` must add evidence, not replace it: the fit
    # wants every replicate ever taken under the current rubric.
    db = Database(tmp_path / "flats.db")
    db.add_measure_runs(1, "quiet", "abc123", [4.0])
    db.add_measure_runs(1, "quiet", "abc123", [6.0])
    assert len(db.measure_runs()) == 2


def test_posterior_columns_persist(tmp_path):
    from flat_scout.models import Evaluation

    db = Database(tmp_path / "flats.db")
    listing_id = db.upsert(make_listing(), source="manual")
    db.set_evaluation(
        listing_id,
        Evaluation(
            verdict="hopeful", score=6.8, coverage=0.7,
            score_low=5.9, score_high=7.7, p_hopeful=0.81,
            posterior_fit_hash="fit123",
            reasons=[], red_flags=[], agent_questions=[],
        ),
        model="m", mode="weighted",
    )
    row = db.get(listing_id)
    assert (
        row["score_low"],
        row["score_high"],
        row["p_hopeful"],
        row["posterior_fit_hash"],
    ) == (5.9, 7.7, 0.81, "fit123")


# --- where the image cache put the pictures ---------------------------------


def test_the_image_paths_round_trip_onto_a_listing(tmp_path):
    """`set_image_paths` writes them and `listing_from_row` reads them back:
    that pair is the whole contract between the cache and every reader."""
    from flat_scout.pipeline import listing_from_row

    db = Database(tmp_path / "flats.db")
    listing_id = db.upsert(make_listing(), source="manual")
    db.set_image_paths(
        listing_id,
        {
            "image_path": "data/images/rightmove_92053296/photo.jpg",
            "epc_image_path": "data/images/rightmove_92053296/epc.pdf",
            "floorplan_path": "data/images/rightmove_92053296/floorplan.png",
        },
    )
    listing = listing_from_row(db.get(listing_id))
    assert listing.image_path == "data/images/rightmove_92053296/photo.jpg"
    assert listing.epc_image_path == "data/images/rightmove_92053296/epc.pdf"
    assert listing.floorplan_path == "data/images/rightmove_92053296/floorplan.png"
    # The URLs stay: the path is preferred, and the URL is what refills a cache.
    assert listing.url == "https://www.rightmove.co.uk/properties/92053296"


def test_only_the_cached_pictures_are_written(tmp_path):
    """A run that reached the floorplan and not the EPC must leave the EPC
    exactly as it was, rather than blanking a path that still points at a file."""
    db = Database(tmp_path / "flats.db")
    listing_id = db.upsert(make_listing(), source="manual")
    db.set_image_paths(listing_id, {"epc_image_path": "data/images/r_1/epc.png"})
    db.set_image_paths(listing_id, {"floorplan_path": "data/images/r_1/floorplan.png"})
    row = db.get(listing_id)
    assert row["epc_image_path"] == "data/images/r_1/epc.png"
    assert row["floorplan_path"] == "data/images/r_1/floorplan.png"
    assert row["image_path"] is None


def test_an_unknown_column_is_refused_rather_than_interpolated(tmp_path):
    """The keys reach an UPDATE as column names, so nothing else may pass."""
    db = Database(tmp_path / "flats.db")
    listing_id = db.upsert(make_listing(), source="manual")
    with pytest.raises(ValueError):
        db.set_image_paths(listing_id, {"floorplan_url": "https://media.example.com/p.png"})
    with pytest.raises(ValueError):
        db.set_image_paths(listing_id, {"status = 'approved' -- ": "x"})


def test_nothing_cached_is_a_no_op(tmp_path):
    db = Database(tmp_path / "flats.db")
    listing_id = db.upsert(make_listing(), source="manual")
    db.set_image_paths(listing_id, {})
    assert db.get(listing_id)["image_path"] is None


def test_upsert_never_writes_over_a_stored_image_path(tmp_path):
    """A path is a fact about this machine's disk, and a Portal payload has no
    opinion on it. A re-ingest carrying stale paths must not touch the row."""
    db = Database(tmp_path / "flats.db")
    listing_id = db.upsert(make_listing(), source="alert")
    db.set_image_paths(listing_id, {"image_path": "data/images/rightmove_92053296/photo.jpg"})
    db.upsert(
        make_listing(
            image_path="somewhere/else/photo.jpg",
            epc_image_path="somewhere/else/epc.png",
            floorplan_path="somewhere/else/floorplan.png",
        ),
        source="alert",
    )
    row = db.get(listing_id)
    assert row["image_path"] == "data/images/rightmove_92053296/photo.jpg"
    assert row["epc_image_path"] is None
    assert row["floorplan_path"] is None


def test_a_database_created_before_the_image_path_columns_gains_them_by_alter(tmp_path):
    """The live database has rows and no path columns. Migration, not recreation."""
    path = tmp_path / "flats.db"
    older = Database(path)
    for column in ("image_path", "epc_image_path", "floorplan_path"):
        older.conn.execute(f"ALTER TABLE listings DROP COLUMN {column}")
    older.conn.execute(
        "INSERT INTO listings (portal, portal_id, url, source, first_seen) "
        "VALUES ('rightmove', '1', 'u', 'alert', '2026-08-16')"
    )
    older.conn.commit()
    older.conn.close()

    db = Database(path)
    columns = {row["name"] for row in db.conn.execute("PRAGMA table_info(listings)")}
    assert {"image_path", "epc_image_path", "floorplan_path"} <= columns
    assert db.get(1)["portal_id"] == "1"  # the pre-existing row survives
    assert db.get(1)["image_path"] is None
    db.set_image_paths(1, {"image_path": "data/images/rightmove_1/photo.jpg"})
    assert db.get(1)["image_path"] == "data/images/rightmove_1/photo.jpg"
