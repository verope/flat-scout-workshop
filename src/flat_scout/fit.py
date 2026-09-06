"""The noise fit and the empirical priors, cached in the database.

PyMC runs here and nowhere else. `posterior.py` scores Listings off the
`Fit` this module produces, with numpy alone: the fit is slow-and-rare, the
scoring is fast-and-constant, and the cache keyed on the fit's inputs is
what keeps a restart from paying for sampling it already did.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from flat_scout.criteria import Criterion
from flat_scout.db import Database

GRADE_BINS: tuple[float, ...] = tuple(float(b) for b in range(11))

# One Dirichlet pseudocount per bin. A single real grade moves the prior
# without owning it; an empty corpus is uniform, which is the honest answer.
PSEUDOCOUNT = 1.0

# What an unmeasured instrument is assumed to wobble by, in grade points,
# until measure runs exist at all. Roughly the spread `grade --measure`
# showed on steady rubrics.
DEFAULT_POOLED_SIGMA = 1.5

# Bump when the PyMC structure or inference meaning changes. The numeric
# hyperparameters below are also fingerprinted directly, so editing one cannot
# accidentally reuse a Fit produced under its old value.
FIT_MODEL_VERSION = 1


@dataclass(frozen=True)
class CriterionFit:
    sigma: float
    prior_weights: tuple[float, ...]
    n_grades: int
    n_runs: int


@dataclass(frozen=True)
class Fit:
    criteria: dict[str, CriterionFit]
    pooled_sigma: float
    input_hash: str
    fitted_at: str


def empirical_priors(
    grades_by_slug: dict[str, list[float]],
) -> dict[str, tuple[float, ...]]:
    """π_c: what each Criterion usually is, over the corpus.

    A categorical over integer bins rather than a fitted density, so `pets`
    stays two-lobed. Grades land in the nearest bin.
    """
    priors: dict[str, tuple[float, ...]] = {}
    for slug, grades in grades_by_slug.items():
        counts = [PSEUDOCOUNT] * len(GRADE_BINS)
        for grade in grades:
            bin_at = min(len(GRADE_BINS) - 1, max(0, round(grade)))
            counts[bin_at] += 1.0
        total = sum(counts)
        priors[slug] = tuple(count / total for count in counts)
    return priors


def determined_grades(db: Database, criteria: list[Criterion]) -> dict[str, list[float]]:
    """Every determined grade in the evaluated corpus, keyed by slug.

    Every Criterion gets a key, so a never-graded Criterion still gets a
    (uniform) prior instead of a KeyError at scoring time. `evaluated_at` is
    the durable proof that a Listing passed the hard filters and reached the
    evaluator. Backfill also grades Listings that never got that far, which are
    not the population the prior describes.
    """
    by_slug: dict[str, list[float]] = {criterion.slug: [] for criterion in criteria}
    live_hash = {criterion.slug: criterion.rubric_hash for criterion in criteria}
    rows = db.conn.execute(
        "SELECT g.criterion, g.grade, g.rubric_hash FROM criterion_grades g "
        "JOIN listings l ON l.id = g.listing_id "
        "WHERE l.evaluated_at IS NOT NULL "
        "AND g.determined = 1 AND g.grade IS NOT NULL"
    ).fetchall()
    for row in rows:
        if live_hash.get(row["criterion"]) == row["rubric_hash"]:
            by_slug[row["criterion"]].append(float(row["grade"]))
    return by_slug


def current_runs(db: Database, criteria: list[Criterion]) -> dict[str, list[list[float]]]:
    """Replicate groups per slug, current-rubric only, answered values only.

    A group is one (listing, measured_at) batch. Groups below two answered
    values carry no information about spread and are dropped here rather
    than inside the model.
    """
    live_hash = {criterion.slug: criterion.rubric_hash for criterion in criteria}
    groups: dict[tuple[int, str, str], list[float]] = {}
    for row in db.measure_runs():
        slug = row["criterion"]
        if live_hash.get(slug) != row["rubric_hash"] or row["grade"] is None:
            continue
        groups.setdefault((row["listing_id"], slug, row["measured_at"]), []).append(
            float(row["grade"])
        )
    by_slug: dict[str, list[list[float]]] = {}
    for (_, slug, _), values in sorted(groups.items()):
        if len(values) >= 2:
            by_slug.setdefault(slug, []).append(values)
    return by_slug


def _input_hash(
    criteria: list[Criterion],
    grades: dict[str, list[float]],
    runs: dict[str, list[list[float]]],
) -> str:
    """A fingerprint of one captured set of fit inputs."""
    payload = json.dumps(
        {
            "fit_model": {
                "version": FIT_MODEL_VERSION,
                "grade_bins": GRADE_BINS,
                "pseudocount": PSEUDOCOUNT,
                "default_pooled_sigma": DEFAULT_POOLED_SIGMA,
            },
            "criteria": sorted((c.slug, c.rubric_hash) for c in criteria),
            "grades": {
                slug: sorted(values) for slug, values in grades.items()
            },
            "runs": {
                slug: sorted(groups) for slug, groups in runs.items()
            },
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def fit_input_hash(criteria: list[Criterion], db: Database) -> str:
    """A fingerprint of everything a new fit would read, and nothing else.

    The same contract as `rubric_hash`: a changed input refits, an
    unchanged one reuses. Grades and runs enter raw, so one new graded
    Listing moves the hash.
    """
    return _input_hash(
        criteria,
        determined_grades(db, criteria),
        current_runs(db, criteria),
    )


def fit_sigmas(
    runs_by_slug: dict[str, list[list[float]]],
) -> tuple[dict[str, float], float]:
    """σ_c for each measured Criterion, partially pooled.

    Ten Listings by three runs is thin, so a lone Criterion's spread borrows
    strength from the others through the shared log-normal. The pooled σ is
    the answer for a Criterion nobody measured.
    """
    if not runs_by_slug:
        return {}, DEFAULT_POOLED_SIGMA
    import numpy as np
    import pymc as pm

    slugs = sorted(runs_by_slug)
    slug_index = {slug: i for i, slug in enumerate(slugs)}
    values, group_idx, crit_idx = [], [], []
    group = 0
    for slug in slugs:
        for run_values in runs_by_slug[slug]:
            for value in run_values:
                values.append(value)
                group_idx.append(group)
                crit_idx.append(slug_index[slug])
            group += 1

    with pm.Model():
        mu = pm.Normal("mu", 0.0, 1.0)
        tau = pm.HalfNormal("tau", 1.0)
        log_sigma = pm.Normal("log_sigma", mu, tau, shape=len(slugs))
        latent = pm.Normal("latent", 5.0, 3.0, shape=group)
        pm.Normal(
            "obs",
            mu=latent[np.array(group_idx)],
            sigma=pm.math.exp(log_sigma)[np.array(crit_idx)],
            observed=np.array(values),
        )
        idata = pm.sample(
            1000, tune=1000, chains=2, progressbar=False,
            # The CLI process is multi-threaded. Forking it can deadlock; two chains
            # run sequentially because this fit is small and off the event loop.
            cores=1, random_seed=0, target_accept=0.9,
        )
    posterior = idata.posterior
    sigmas = {
        slug: float(np.exp(posterior["log_sigma"].values[..., i]).mean())
        for i, slug in enumerate(slugs)
    }
    pooled = float(np.exp(posterior["mu"].values).mean())
    return sigmas, pooled


def run_fit(db: Database, criteria: list[Criterion]) -> Fit:
    """Sample the fit off the database as it stands. Always pays."""
    from flat_scout.db import _now

    grades = determined_grades(db, criteria)
    runs = current_runs(db, criteria)
    input_hash = _input_hash(criteria, grades, runs)
    priors = empirical_priors(grades)
    sigmas, pooled = fit_sigmas(runs)
    return Fit(
        criteria={
            criterion.slug: CriterionFit(
                sigma=sigmas.get(criterion.slug, pooled),
                prior_weights=priors[criterion.slug],
                n_grades=len(grades[criterion.slug]),
                n_runs=sum(len(g) for g in runs.get(criterion.slug, [])),
            )
            for criterion in criteria
        },
        pooled_sigma=pooled,
        # Hash the captured inputs above. A grade can land while PyMC samples;
        # hashing the database again would label this older fit as the newer one.
        input_hash=input_hash,
        fitted_at=_now(),
    )


def serialize(fit: Fit) -> str:
    """The JSON the database cache stores a `Fit` as. Shared by `load_fit` and the CLI."""
    return json.dumps(
        {
            "criteria": {
                slug: {
                    "sigma": entry.sigma,
                    "prior_weights": list(entry.prior_weights),
                    "n_grades": entry.n_grades,
                    "n_runs": entry.n_runs,
                }
                for slug, entry in fit.criteria.items()
            },
            "pooled_sigma": fit.pooled_sigma,
            "fitted_at": fit.fitted_at,
        }
    )


def _deserialize(input_hash: str, cached: str) -> Fit:
    raw = json.loads(cached)
    return Fit(
        criteria={
            slug: CriterionFit(
                sigma=entry["sigma"],
                prior_weights=tuple(entry["prior_weights"]),
                n_grades=entry["n_grades"],
                n_runs=entry["n_runs"],
            )
            for slug, entry in raw["criteria"].items()
        },
        pooled_sigma=raw["pooled_sigma"],
        input_hash=input_hash,
        fitted_at=raw["fitted_at"],
    )


def load_cached_fit(db: Database, input_hash: str) -> Fit:
    """The exact historical Fit named by a frozen result."""
    cached = db.fit_cache(input_hash)
    if cached is None:
        raise RuntimeError(f"posterior fit {input_hash} is missing from the cache")
    return _deserialize(input_hash, cached)


def load_fit(
    db: Database, criteria: list[Criterion], *, refit: bool = False
) -> Fit:
    """The cached production fit, or a current fit for an explicit refresh.

    Live scoring never starts PyMC because one new Listing moved the corpus.
    It keeps the last completed fit until `flat-scout fit` refreshes it.
    `refit=True` asks for the current input hash, but an unchanged corpus is
    still a cache hit. A fresh database with no measure runs can build its
    default fit cheaply. Once runs exist, only `flat-scout fit` may start PyMC.
    """
    input_hash = fit_input_hash(criteria, db)
    cached = db.fit_cache(input_hash)
    if cached is not None:
        return _deserialize(input_hash, cached)
    if not refit and (latest := db.latest_fit_cache()) is not None:
        return _deserialize(*latest)
    if not refit and current_runs(db, criteria):
        raise RuntimeError("no cached posterior fit; run `flat-scout fit`")
    fitted = run_fit(db, criteria)
    db.set_fit_cache(fitted.input_hash, serialize(fitted))
    return fitted
