from pathlib import Path

import pytest

from flat_scout.criteria import CriteriaError, load_criteria, load_criterion, rubric_hash

WARMTH = """---
name: Warmth and bills
description: Low bills and no draughts.
weight: 4
grade:
  field: epc_band
  map: {A: 10, B: 10, C: 8, D: 5, E: 2, F: 0, G: 0}
unknown: skip
ask: "What is the EPC rating?"
---

## Rubric
**10** — band A or B.
"""


def write(directory: Path, name: str, text: str) -> None:
    (directory / name).write_text(text)


def test_loads_a_criterion_and_takes_the_slug_from_the_filename(tmp_path: Path):
    write(tmp_path, "warmth.md", WARMTH)
    [criterion] = load_criteria(tmp_path)
    assert criterion.slug == "warmth"
    assert criterion.name == "Warmth and bills"
    assert criterion.weight == 4
    assert criterion.grade["field"] == "epc_band"
    assert criterion.unknown == "skip"
    assert criterion.ask == "What is the EPC rating?"
    assert "**10**" in criterion.body


def test_a_file_without_frontmatter_is_not_a_criterion(tmp_path: Path):
    write(tmp_path, "warmth.md", WARMTH)
    write(tmp_path, "notes.md", "Some notes about the search, with no frontmatter.\n")
    assert [c.slug for c in load_criteria(tmp_path)] == ["warmth"]


def test_an_absent_weight_stops_the_run(tmp_path: Path):
    write(tmp_path, "broken.md", WARMTH.replace("weight: 4\n", ""))
    with pytest.raises(CriteriaError, match="weight"):
        load_criteria(tmp_path)


def test_a_weight_of_zero_stops_the_run(tmp_path: Path):
    write(tmp_path, "broken.md", WARMTH.replace("weight: 4", "weight: 0"))
    with pytest.raises(CriteriaError, match="above zero"):
        load_criteria(tmp_path)


def test_an_unknown_grader_name_stops_the_run(tmp_path: Path):
    write(
        tmp_path,
        "broken.md",
        "---\nname: X\ndescription: d\nweight: 1\ngrade:\n  grader: no_such\n---\n\nbody\n",
    )
    with pytest.raises(CriteriaError, match="no_such"):
        load_criteria(tmp_path, known_graders={"floor_area"})


def test_two_grade_forms_at_once_stops_the_run(tmp_path: Path):
    write(
        tmp_path,
        "broken.md",
        "---\nname: X\ndescription: d\nweight: 1\n"
        "grade:\n  grader: floor_area\n  model: true\n---\n\nbody\n",
    )
    with pytest.raises(CriteriaError, match="exactly one"):
        load_criteria(tmp_path, known_graders={"floor_area"})


def test_a_map_that_omits_a_value_of_a_closed_set_stops_the_run(tmp_path: Path):
    write(tmp_path, "broken.md", WARMTH.replace(", G: 0", ""))
    with pytest.raises(CriteriaError, match="G"):
        load_criteria(tmp_path)


def test_two_criteria_may_not_share_a_name(tmp_path: Path):
    write(tmp_path, "a.md", WARMTH)
    write(tmp_path, "b.md", WARMTH)
    with pytest.raises(CriteriaError, match="Warmth and bills"):
        load_criteria(tmp_path)


def test_an_empty_body_stops_the_run(tmp_path: Path):
    head = WARMTH.split("---\n\n")[0]
    write(tmp_path, "broken.md", head + "---\n\n   \n")
    with pytest.raises(CriteriaError, match="rubric"):
        load_criteria(tmp_path)


def test_the_hash_ignores_the_keys_applied_at_read_time(tmp_path: Path):
    """A weight edit must not invalidate a stored grade."""
    write(tmp_path, "warmth.md", WARMTH)
    before = load_criteria(tmp_path)[0].rubric_hash
    write(tmp_path, "warmth.md", WARMTH.replace("weight: 4", "weight: 9"))
    assert load_criteria(tmp_path)[0].rubric_hash == before


def test_the_hash_ignores_the_ask_and_the_unknown_rule(tmp_path: Path):
    write(tmp_path, "warmth.md", WARMTH)
    before = load_criteria(tmp_path)[0].rubric_hash
    edited = WARMTH.replace("unknown: skip", "unknown: 5").replace(
        'ask: "What is the EPC rating?"', 'ask: "Which band is it?"'
    )
    write(tmp_path, "warmth.md", edited)
    assert load_criteria(tmp_path)[0].rubric_hash == before


def test_the_hash_moves_when_the_rubric_moves(tmp_path: Path):
    write(tmp_path, "warmth.md", WARMTH)
    before = load_criteria(tmp_path)[0].rubric_hash
    write(tmp_path, "warmth.md", WARMTH.replace("band A or B.", "band A only."))
    assert load_criteria(tmp_path)[0].rubric_hash != before


def test_the_hash_moves_when_the_map_moves(tmp_path: Path):
    write(tmp_path, "warmth.md", WARMTH)
    before = load_criteria(tmp_path)[0].rubric_hash
    write(tmp_path, "warmth.md", WARMTH.replace("C: 8", "C: 9"))
    assert load_criteria(tmp_path)[0].rubric_hash != before


def test_the_hash_is_stable_across_key_order():
    a = rubric_hash({"field": "epc_band", "map": {"A": 10}}, None, "body")
    b = rubric_hash({"map": {"A": 10}, "field": "epc_band"}, None, "body")
    assert a == b


def test_an_unterminated_frontmatter_block_stops_the_run(tmp_path: Path):
    """A file that opens with --- but never closes it is malformed, not absent."""
    write(tmp_path, "broken.md", "---\nname: X\ndescription: d\nweight: 1\n")
    with pytest.raises(CriteriaError, match="never closed"):
        load_criteria(tmp_path)


def write_silence(tmp_path, front, body="## Rubric\nwords\n"):
    path = tmp_path / "pets.md"
    path.write_text(f"---\n{front}\n---\n\n{body}")
    return path


BASE = 'name: Pets\ndescription: d\nweight: 3\ngrade: {model: true}'


def test_silence_prior_parses(tmp_path):
    got = load_criterion(write_silence(tmp_path, BASE + "\nsilence_prior: {0: 0.7, 10: 0.3}"))
    assert got.silence_prior == {0.0: 0.7, 10.0: 0.3}


def test_silence_prior_must_sum_to_one(tmp_path):
    with pytest.raises(CriteriaError):
        load_criterion(write_silence(tmp_path, BASE + "\nsilence_prior: {0: 0.7, 10: 0.4}"))


def test_silence_prior_keys_stay_on_the_grade_scale(tmp_path):
    with pytest.raises(CriteriaError):
        load_criterion(write_silence(tmp_path, BASE + "\nsilence_prior: {0: 0.5, 11: 0.5}"))


def test_silence_prior_needs_a_model_reachable_criterion(tmp_path):
    # Only advert silence is informative, and only a model can certify it.
    front = (
        'name: Lift\ndescription: d\nweight: 2\n'
        'grade: {bands: [[2, 10]], field: nearest_station_miles}\n'
        'silence_prior: {0: 1.0}'
    )
    with pytest.raises(CriteriaError):
        load_criterion(write_silence(tmp_path, front))


def test_silence_prior_stays_out_of_the_rubric_hash(tmp_path):
    with_prior = load_criterion(write_silence(tmp_path, BASE + "\nsilence_prior: {0: 0.7, 10: 0.3}"))
    without = load_criterion(write_silence(tmp_path, BASE))
    assert with_prior.rubric_hash == without.rubric_hash
