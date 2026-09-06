"""The Criteria the evaluator judges a Listing against, read off disk.

One file for each Criterion, so a weight and a rubric sit beside the words that
explain them, and a person can change one without reading the others.

The loader is deliberately strict. A Criterion that failed to load would not
announce itself: it would stop contributing, and every score in the corpus
would move with nothing on screen to say why. So every fault raises.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Fields whose values are a closed set in `models.py`. A `map` over one of
# these must cover every value, because a value the map omits could otherwise
# only be discovered by a Listing arriving with it - months from now, in
# production, as a KeyError inside a grader.
CLOSED_SETS: dict[str, set[Any]] = {
    "epc_band": {"A", "B", "C", "D", "E", "F", "G"},
    "layout_verdict": {"good", "adequate", "poor"},
    "bedroom_has_window": {True, False},
    "reception_is_separate": {True, False},
    "desk_space": {True, False},
}

FRONTMATTER = "---"


class CriteriaError(Exception):
    """A Criterion file is malformed. The run stops."""


@dataclass(frozen=True)
class Criterion:
    slug: str
    name: str
    description: str
    weight: float
    grade: dict[str, Any]
    body: str
    unknown: str | float = "skip"
    fallback: str | None = None
    veto_at_or_below: float | None = None
    ask: str | None = None
    silence_prior: dict[float, float] | None = None
    rubric_hash: str = ""

    @property
    def is_model(self) -> bool:
        return bool(self.grade.get("model"))


def rubric_hash(grade: dict[str, Any], fallback: str | None, body: str) -> str:
    """A fingerprint of everything that can change a grade, and nothing else.

    `weight`, `unknown`, `veto_at_or_below` and `ask` are applied at read time,
    so a change to any of them leaves every stored grade valid. That is the
    whole of the tunability: a weight edit replays over the corpus for free.
    Adding one of those four here would quietly destroy it, and the loss would
    show only as an unexplained model bill.

    Two things outside this file can also change a grade and are deliberately
    NOT fingerprinted: the Python in `graders.py`, and the shared preamble
    (`_brief.md`, `_fields.md`). Edit either and `stale_criteria` still
    reports everything current. That is a real limitation and it has been
    raised more than once, so the reasoning is recorded here rather than in a
    review thread.

    A fingerprint over a shared input covering every Criterion would
    invalidate the ones that never read it, forcing a model re-grade of
    `lift`, `pets` and the rest on an edit that could not touch them - the
    unexplained model bill this hash exists to prevent, arriving by a
    different door. Scoping it to the Criteria that do read the input means
    this loader has to know which grader each Criterion uses and which files
    that grader touches, coupling a module that reads markdown to the
    internals of one that grades flats.

    The escape hatch is explicit instead: `grade --backfill --force` re-grades
    regardless of stored hashes, and names these inputs in its help. An
    operator editing one of them reads it there.
    """
    payload = json.dumps(
        {"grade": grade, "fallback": fallback, "body": body.strip()},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def split_frontmatter(text: str) -> tuple[str, str] | None:
    """The frontmatter and the body, or None if the file opens with neither."""
    if not text.startswith(FRONTMATTER):
        return None
    rest = text[len(FRONTMATTER) :].lstrip("\n")
    end = rest.find(f"\n{FRONTMATTER}")
    if end == -1:
        return None
    return rest[:end], rest[end + len(FRONTMATTER) + 1 :]


def _number(value: Any, path: Path, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CriteriaError(f"{path.name}: `{key}` must be a number, got {value!r}")
    return float(value)


def _check_silence_prior(front: dict, path: Path) -> dict[float, float] | None:
    """Elicited odds for what advert silence suggests.

    Strict for the same reason the rest of the loader is: a malformed prior
    would move every silent Listing's score with nothing on screen to say
    why. Model-reachable only - extraction absence must never take it.
    """
    raw = front.get("silence_prior")
    if raw is None:
        return None
    grade = front.get("grade") or {}
    reachable = bool(grade.get("model")) or front.get("fallback") == "model"
    if not reachable:
        raise CriteriaError(
            f"{path.name}: `silence_prior` needs a Criterion a model can reach"
        )
    if not isinstance(raw, dict) or not raw:
        raise CriteriaError(f"{path.name}: `silence_prior` must be a mapping of grade to probability")
    prior: dict[float, float] = {}
    for key, value in raw.items():
        grade_at = _number(key, path, "silence_prior key")
        share = _number(value, path, "silence_prior value")
        if not 0 <= grade_at <= 10:
            raise CriteriaError(f"{path.name}: `silence_prior` grades must be 0 to 10")
        if share <= 0:
            raise CriteriaError(f"{path.name}: `silence_prior` probabilities must be above zero")
        prior[grade_at] = share
    if abs(sum(prior.values()) - 1.0) > 1e-6:
        raise CriteriaError(f"{path.name}: `silence_prior` probabilities must sum to 1")
    return prior


def _check_grade(grade: Any, path: Path, known_graders: set[str] | None) -> None:
    if not isinstance(grade, dict):
        raise CriteriaError(f"{path.name}: `grade` must be a block")
    forms = (
        bool(grade.get("map")),
        bool(grade.get("bands")),
        bool(grade.get("grader")),
        bool(grade.get("model")),
    )
    if sum(forms) != 1:
        raise CriteriaError(
            f"{path.name}: `grade` needs exactly one of map, bands, grader, model"
        )
    if (grade.get("map") or grade.get("bands")) and not grade.get("field"):
        raise CriteriaError(f"{path.name}: a map or bands needs a `field`")
    if grade.get("map"):
        closed = CLOSED_SETS.get(grade["field"])
        if closed is not None:
            missing = closed - set(grade["map"])
            if missing:
                shown = ", ".join(str(value) for value in sorted(missing, key=str))
                raise CriteriaError(
                    f"{path.name}: the map over `{grade['field']}` omits {shown}"
                )
    name = grade.get("grader")
    if name and known_graders is not None and name not in known_graders:
        raise CriteriaError(f"{path.name}: no grader is registered as `{name}`")


def load_criterion(
    path: Path, known_graders: set[str] | None = None
) -> Criterion | None:
    """One Criterion, or None if the file carries no frontmatter at all.

    None is how `_brief.md` and `_fields.md` stay in `criteria/` without
    becoming Criteria. Once a file opens with frontmatter it is a Criterion,
    and from there every fault raises rather than returning None.
    """
    text = path.read_text()
    split = split_frontmatter(text)
    if split is None:
        if text.startswith(FRONTMATTER):
            raise CriteriaError(f"{path.name}: the frontmatter block is never closed")
        return None
    head, body = split
    try:
        front = yaml.safe_load(head)
    except yaml.YAMLError as exc:
        raise CriteriaError(
            f"{path.name}: the frontmatter is not valid YAML: {exc}"
        ) from exc
    if not isinstance(front, dict):
        raise CriteriaError(f"{path.name}: the frontmatter must be a mapping")

    for key in ("name", "description", "weight", "grade"):
        if key not in front:
            raise CriteriaError(f"{path.name}: `{key}` is required")
    if not body.strip():
        raise CriteriaError(f"{path.name}: the body is the rubric, and it is empty")

    weight = _number(front["weight"], path, "weight")
    if weight <= 0:
        raise CriteriaError(f"{path.name}: `weight` must be above zero")

    unknown = front.get("unknown", "skip")
    if unknown != "skip":
        unknown = _number(unknown, path, "unknown")
        if not 0 <= unknown <= 10:
            raise CriteriaError(f"{path.name}: `unknown` must be 'skip' or 0 to 10")

    fallback = front.get("fallback")
    if fallback not in (None, "model"):
        raise CriteriaError(f"{path.name}: `fallback` must be 'model' if present")

    veto = front.get("veto_at_or_below")
    if veto is not None:
        veto = _number(veto, path, "veto_at_or_below")

    _check_grade(front["grade"], path, known_graders)

    silence = _check_silence_prior(front, path)

    return Criterion(
        slug=path.stem,
        name=str(front["name"]),
        description=str(front["description"]),
        weight=weight,
        grade=front["grade"],
        body=body.strip(),
        unknown=unknown,
        fallback=fallback,
        veto_at_or_below=veto,
        ask=front.get("ask"),
        silence_prior=silence,
        rubric_hash=rubric_hash(front["grade"], fallback, body),
    )


def load_criteria(
    directory: Path, known_graders: set[str] | None = None
) -> list[Criterion]:
    """Every Criterion in `directory`, ordered by slug.

    The order is fixed so a prompt, a report and a stored row list the Criteria
    the same way every time. Nothing downstream depends on it beyond that.
    """
    found: list[Criterion] = []
    for path in sorted(directory.glob("*.md")):
        criterion = load_criterion(path, known_graders)
        if criterion is not None:
            found.append(criterion)
    names: dict[str, str] = {}
    for criterion in found:
        if criterion.name in names:
            raise CriteriaError(
                f"{criterion.slug}.md and {names[criterion.name]}.md "
                f"share the name {criterion.name!r}"
            )
        names[criterion.name] = criterion.slug
    return found


# The files that are shared by every model call about one Listing, in the order
# they are read. They carry no frontmatter, so they are not Criteria: `_brief`
# is who the couple are, and `_fields` is what the data means. Both are
# identical across a Listing's Criterion calls, which is what lets the provider
# cache them as a prefix.
PREAMBLE_FILES = ("_brief.md", "_fields.md")


def load_preamble(directory: Path) -> str:
    """The shared material that opens every Criterion prompt."""
    return "\n\n".join(
        (directory / name).read_text().strip() for name in PREAMBLE_FILES
    )
