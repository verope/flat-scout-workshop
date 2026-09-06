from pathlib import Path

from flat_scout.criteria import Criterion
from flat_scout.grading import apply_unknown, deterministic, field_value
from flat_scout.models import ImageReading, ListingData

# None of the Criteria here is graded by a grader that reads a file beside the
# Criterion files, so the directory is only the argument the signature
# now requires.
CRITERIA = Path("criteria")

BAND_MAP = {"A": 10, "B": 10, "C": 8, "D": 5, "E": 2, "F": 0, "G": 0}


def a_listing(**kwargs) -> ListingData:
    return ListingData(portal="rightmove", portal_id="1", url="u", **kwargs)


def a_criterion(grade: dict, **kwargs) -> Criterion:
    return Criterion(
        slug="warmth", name="Warmth and bills", description="d", weight=4,
        grade=grade, body="rubric", **kwargs,
    )


def test_a_map_grades_off_the_image_reading():
    criterion = a_criterion(
        {"field": "epc_band", "map": BAND_MAP, "evidence": "EPC band {value}."}
    )
    grade = deterministic(criterion, a_listing(), ImageReading(epc_band="C"), CRITERIA)
    assert grade.value == 8
    assert grade.determined is True
    assert grade.evidence == "EPC band C."
    assert grade.graded_by == "map"


def test_a_map_over_an_absent_field_is_unknown_and_not_zero():
    criterion = a_criterion({"field": "epc_band", "map": BAND_MAP})
    grade = deterministic(criterion, a_listing(), ImageReading(), CRITERIA)
    assert grade.value is None
    assert grade.determined is False


def test_a_map_grades_a_false_boolean_without_treating_it_as_absent():
    """bedroom_has_window=False is the whole point, and False is falsy."""
    criterion = a_criterion({"field": "bedroom_has_window", "map": {True: 10, False: 0}})
    grade = deterministic(criterion, a_listing(), ImageReading(bedroom_has_window=False), CRITERIA)
    assert grade.value == 0
    assert grade.determined is True


def test_bands_are_read_top_to_bottom_with_a_catch_all():
    criterion = a_criterion(
        {
            "field": "nearest_station_miles",
            "bands": [{"at_most": 1.5, "grade": 10}, {"at_most": 3.0, "grade": 7}, {"grade": 3}],
            "evidence": "{value} miles to the station in a straight line.",
        }
    )
    assert deterministic(
        criterion, a_listing(nearest_station_miles=1.0), None, CRITERIA
    ).value == 10
    assert deterministic(
        criterion, a_listing(nearest_station_miles=2.2), None, CRITERIA
    ).value == 7
    assert deterministic(
        criterion, a_listing(nearest_station_miles=9.0), None, CRITERIA
    ).value == 3
    assert deterministic(criterion, a_listing(), None, CRITERIA).determined is False


def test_a_maximum_caps_every_band():
    """The capping mechanism itself, exercised on a synthetic Criterion.

    No shipped Criterion uses `max`; the mechanism stays available."""
    criterion = a_criterion(
        {
            "field": "nearest_station_miles",
            "max": 7,
            "bands": [{"at_most": 0.3, "grade": 10}, {"grade": 2}],
        }
    )
    assert deterministic(criterion, a_listing(nearest_station_miles=0.1), None, CRITERIA).value == 7
    assert deterministic(criterion, a_listing(nearest_station_miles=1.4), None, CRITERIA).value == 2


def test_a_value_outside_all_bands_with_no_catch_all_is_unknown():
    """Bands with no catch-all: a gap in the file is an error in the file, not a fact about the flat."""
    criterion = a_criterion(
        {
            "field": "nearest_station_miles",
            "bands": [{"at_most": 0.5, "grade": 10}, {"at_most": 1.5, "grade": 7}],
        }
    )
    # Value 5.0 exceeds both thresholds and no catch-all exists
    grade = deterministic(criterion, a_listing(nearest_station_miles=5.0), None, CRITERIA)
    assert grade.value is None
    assert grade.determined is False


def test_the_image_reading_outranks_the_listing_for_a_shared_field_name():
    listing = a_listing(sqft=400)
    reading = ImageReading(epc_floor_area_sqm=60.0)
    assert field_value(listing, reading, "epc_floor_area_sqm") == 60.0
    assert field_value(listing, reading, "sqft") == 400


def test_an_empty_list_on_the_reading_counts_as_absent():
    assert field_value(a_listing(), ImageReading(), "window_aspects") is None


def test_a_model_criterion_is_not_determined_here():
    grade = deterministic(a_criterion({"model": True}), a_listing(), None, CRITERIA)
    assert grade.determined is False
    assert grade.graded_by == ""


def test_an_unknown_default_supplies_a_value_and_no_coverage():
    criterion = a_criterion({"field": "epc_band", "map": BAND_MAP}, unknown=4)
    grade = apply_unknown(criterion, deterministic(criterion, a_listing(), None, CRITERIA))
    assert grade.value == 4
    assert grade.determined is False
    assert "4" in grade.evidence


def test_skip_leaves_the_grade_absent():
    criterion = a_criterion({"field": "epc_band", "map": BAND_MAP}, unknown="skip")
    grade = apply_unknown(criterion, deterministic(criterion, a_listing(), None, CRITERIA))
    assert grade.value is None
    assert grade.determined is False


def test_an_unknown_default_never_overwrites_a_determined_grade():
    criterion = a_criterion({"field": "epc_band", "map": BAND_MAP}, unknown=4)
    determined_grade = deterministic(criterion, a_listing(), ImageReading(epc_band="A"), CRITERIA)
    grade = apply_unknown(criterion, determined_grade)
    assert grade.value == 10
    assert grade.determined is True


def test_the_unknown_default_keeps_the_graded_by_of_the_attempt():
    """`graded_by` is the record of what was spent, and the default is not it.

    `backfill_grades` counts model spend with `graded_by.startswith("model:")`.
    Overwriting it with "default" makes the first Criterion pairing
    `fallback: model` with a numeric `unknown` undercount its own bill -
    silently, and in the number somebody reads before running the backfill
    over the whole corpus.
    """
    from flat_scout.adjudicate import Grade

    criterion = a_criterion({"model": True}, unknown=4)
    attempted = Grade(
        criterion="warmth", value=None, determined=False,
        evidence="nothing stated", graded_by="model:openai:gpt-4o",
    )
    grade = apply_unknown(criterion, attempted)
    assert grade.value == 4
    assert grade.graded_by == "model:openai:gpt-4o"


def test_a_default_over_nothing_attempted_is_still_graded_by_default():
    """The discrimination: a Criterion that never reached a model must not
    now be counted as spend either."""
    criterion = a_criterion({"field": "epc_band", "map": BAND_MAP}, unknown=4)
    grade = apply_unknown(criterion, deterministic(criterion, a_listing(), None, CRITERIA))
    assert grade.value == 4
    assert grade.graded_by == "default"
