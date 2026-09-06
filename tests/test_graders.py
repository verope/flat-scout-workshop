from pathlib import Path

from flat_scout.criteria import Criterion
from flat_scout.graders import GRADERS
from flat_scout.models import ImageReading, ListingData

CRITERIA = Path("criteria")


def a_listing(**kwargs) -> ListingData:
    return ListingData(portal="rightmove", portal_id="1", url="u", **kwargs)


def grade(slug: str, grader: str, listing, reading=None, directory=CRITERIA):
    criterion = Criterion(
        slug=slug, name=slug, description="d", weight=1,
        grade={"grader": grader}, body="rubric",
    )
    return GRADERS[grader](criterion, listing, reading, directory)


def test_floor_area_grades_the_certificate_in_square_metres():
    """The bands are metric and read off the certificate as it stands."""
    for sqm, expected in ((60.0, 10.0), (55.0, 8.0), (45.0, 0.0)):
        result = grade("floor-area", "floor_area", a_listing(), ImageReading(epc_floor_area_sqm=sqm))
        assert result.determined is True, sqm
        assert result.value == expected, sqm


def test_floor_area_converts_the_listings_square_footage():
    """646 sq ft is 60.0 m² and 538 is 49.98, either side of a band edge.

    The grade alone therefore says whether the conversion happened at all: a
    figure left in square feet would clear every band.
    """
    for sqft, expected in ((646, 10.0), (538, 0.0), (539, 8.0)):
        result = grade("floor-area", "floor_area", a_listing(sqft=sqft))
        assert result.determined is True, sqft
        assert result.value == expected, sqft


def test_floor_area_prefers_the_certificate_where_the_two_measurements_agree():
    """criteria.md: where both exist and agree, the certificate is the firmer.

    60 m² against the listing's 600 sq ft (55.7 m²) is 7.1% apart, which is
    agreement - and on the other side of the 60 band, so the grade alone says
    which figure was read.
    """
    reading = ImageReading(epc_floor_area_sqm=60.0)
    result = grade("floor-area", "floor_area", a_listing(sqft=600), reading)
    assert result.determined is True
    assert result.value == 10.0
    assert "certificate" in result.evidence


def test_floor_area_tolerates_a_gross_versus_net_gap_as_agreement():
    """61 m² against 600 sq ft (55.7 m²): 8.6% apart, which is not a wrong unit.

    The threshold has to leave room for rounding and for a gross figure
    measured against a net one, or a flat with two honest measurements would
    lose the heaviest Criterion in the brief.
    """
    reading = ImageReading(epc_floor_area_sqm=61.0)
    result = grade("floor-area", "floor_area", a_listing(sqft=600), reading)
    assert result.determined is True
    assert result.value == 10.0
    assert "certificate" in result.evidence


def test_floor_area_reads_the_listing_where_there_is_no_certificate():
    result = grade("floor-area", "floor_area", a_listing(sqft=646))
    assert result.determined is True and result.value == 10.0
    assert "listing" in result.evidence


def test_floor_area_is_unknown_with_neither():
    result = grade("floor-area", "floor_area", a_listing())
    assert result.determined is False
    assert result.value is None


def test_floor_area_leaves_a_materially_disputed_size_unresolved():
    """criteria.md's own example: 529 sq ft advertised, a 74 m² certificate.

    74 against 49 is a different flat, not a rounding. Grading it either way
    would be confident and wrong, so nothing is graded and the couple ask.
    """
    reading = ImageReading(epc_floor_area_sqm=74.0)
    result = grade("floor-area", "floor_area", a_listing(sqft=529), reading)
    assert result.value is None
    assert result.determined is False
    assert result.graded_by == "floor_area"
    assert "529" in result.evidence and "74" in result.evidence
    assert result.concern is not None and "529" in result.concern
    assert result.question is not None
    assert "EPC" in result.question


def test_best_aspect_grades_the_best_of_a_dual_aspect_flat():
    """A flat facing north and west is not a west-facing flat."""
    result = grade(
        "aspect", "best_aspect", a_listing(), ImageReading(window_aspects=["N", "W"])
    )
    assert result.value == 10
    assert "N" in result.evidence and "W" in result.evidence


def test_best_aspect_gives_full_credit_from_north_round_to_south():
    """Everything from north through east to south is worth the full 10."""
    for aspect in ("NE", "S"):
        result = grade(
            "aspect", "best_aspect", a_listing(), ImageReading(window_aspects=[aspect])
        )
        assert result.value == 10, aspect
        assert result.determined is True, aspect


def test_best_aspect_docks_the_two_corners_either_side_of_west():
    for aspect in ("SW", "NW"):
        result = grade(
            "aspect", "best_aspect", a_listing(), ImageReading(window_aspects=[aspect])
        )
        assert result.value == 7, aspect
        assert result.determined is True, aspect


def test_best_aspect_marks_west_down_mildly_rather_than_at_nothing():
    """West alone is the worst answer there is, and still not a problem."""
    result = grade(
        "aspect", "best_aspect", a_listing(), ImageReading(window_aspects=["W"])
    )
    assert result.value == 4
    assert result.determined is True


def test_best_aspect_is_unknown_with_no_compass():
    assert grade(
        "aspect", "best_aspect", a_listing(), ImageReading()
    ).determined is False


def test_not_ground_floor_vetoes_on_the_certificate_and_passes_otherwise():
    ground = ImageReading(epc_property_type="Ground-floor flat")
    assert grade("not-ground-floor", "not_ground_floor", a_listing(), ground).value == 0
    mid = ImageReading(epc_property_type="Mid-floor flat")
    assert grade("not-ground-floor", "not_ground_floor", a_listing(), mid).value == 10
    assert grade("not-ground-floor", "not_ground_floor", a_listing()).determined is False


def test_the_floorplan_names_the_floor_where_the_certificate_does_not():
    """The widening this was written for. The floorplan names a floor far more
    often than the certificate names a property type, and ground-floor flats had
    been passing the hard filter."""
    reading = ImageReading(floor_text="14th Floor")
    assert grade("not-ground-floor", "not_ground_floor", a_listing(), reading).value == 10


def test_a_ground_floor_read_off_a_drawing_never_reaches_the_veto():
    """`veto_at_or_below: 0`, so a 0 here would reject the flat outright and
    invisibly. One plan in the corpus was once read as "Ground Floor" when
    the drawing contains neither word."""
    reading = ImageReading(floor_text="Ground Floor")
    answer = grade("not-ground-floor", "not_ground_floor", a_listing(), reading)
    assert answer.value > 0
    assert answer.concern is not None


def test_the_certificate_still_vetoes_because_it_is_printed():
    reading = ImageReading(epc_property_type="Ground-floor flat")
    assert grade("not-ground-floor", "not_ground_floor", a_listing(), reading).value == 0


def test_the_certificate_outranks_the_drawing_where_both_speak():
    reading = ImageReading(epc_property_type="Mid-floor flat", floor_text="Ground Floor")
    assert grade("not-ground-floor", "not_ground_floor", a_listing(), reading).value == 10


def test_a_raised_ground_floor_is_a_ground_floor():
    """A few steps up is not a storey, and it is the phrase an advert reaches
    for when it would rather not say ground floor."""
    reading = ImageReading(floor_text="Raised Ground Floor")
    assert grade("not-ground-floor", "not_ground_floor", a_listing(), reading).value == 2


def test_a_stored_plan_floor_does_not_slip_past_the_image_signals_gate():
    """Found in review. `set_floor_from_plan` merges the plan's floor into
    `listings.floor`, so that field holds either the advert's word or a
    drawing's and nothing on it says which. With `image_signals` off the
    reading is withheld - and grading off `floor` anyway would let a flag that
    promises to hide vision data decide a flat on exactly that."""
    listing = a_listing(floor="Ground Floor")
    assert grade("not-ground-floor", "not_ground_floor", listing, None).determined is False


def test_the_listing_floor_is_read_once_the_plan_has_been_seen_and_named_none():
    """The other half: a reading that named no floor proves the plan is not
    where `listing.floor` came from, so it is the advert's and may be used."""
    listing = a_listing(floor="Fourth Floor")
    reading = ImageReading(layout_verdict="good")
    assert grade("not-ground-floor", "not_ground_floor", listing, reading).value == 10


def test_a_denied_furnishing_is_never_read_as_the_thing_denied():
    """Found in review. Every phrase contains the word "furnished", so "not
    furnished" is a substring match on "furnished" and would grade 10 - the
    exact opposite of the advert, with four weight behind it."""
    for denied in ("Not furnished", "non-furnished", "un-furnished", "un furnished"):
        answer = grade("furnished", "furnishing", a_listing(furnished=denied))
        assert answer.determined is False, denied
    # And the four the Portals really send still grade, including the one that
    # begins with the prefix the check above rejects when it stands apart.
    assert grade("furnished", "furnishing", a_listing(furnished="Unfurnished")).value == 0


def test_furnishing_reads_the_portals_own_words():
    for stated, expected in [
        ("Furnished", 10),
        ("Fully furnished", 10),
        ("Furnished or unfurnished, landlord is flexible", 8),
        ("Part furnished", 5),
        ("Unfurnished", 0),
    ]:
        answer = grade("furnished", "furnishing", a_listing(furnished=stated))
        assert answer.value == expected, stated
        assert answer.determined is True


def test_unfurnished_is_matched_before_furnished():
    """Every one of these phrases contains the word "furnished", so a plain
    substring search in the wrong order grades an unfurnished flat a 10."""
    assert grade("furnished", "furnishing", a_listing(furnished="Unfurnished")).value == 0


def test_an_unstated_furnishing_is_unknown_rather_than_unfurnished():
    assert grade("furnished", "furnishing", a_listing()).determined is False


def test_an_unfamiliar_furnishing_phrase_is_unknown_rather_than_a_guess():
    answer = grade("furnished", "furnishing", a_listing(furnished="Optional furnishing"))
    assert answer.determined is False


def test_the_registry_holds_exactly_the_graders_the_criteria_name():
    assert set(GRADERS) == {
        "floor_area", "best_aspect", "not_ground_floor", "furnishing",
    }
