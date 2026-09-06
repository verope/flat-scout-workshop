"""Grade vector and Fit in, score posterior and Verdict out.

The Bayesian sibling of `adjudicate.py`, and pure in the same way: no
model, no database, no file, no PyMC. Everything here is numpy over the
Fit that `fit.py` cached, which is why a Listing scores in under a
millisecond and why every rule is table-testable.

Sampling is seeded by the caller (the Listing id), so a replay reproduces
the stored numbers exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import truncnorm

from flat_scout.adjudicate import Pair, _ranked, vetoes_fired
from flat_scout.criteria import Criterion
from flat_scout.fit import GRADE_BINS, Fit
from flat_scout.models import Evaluation

DRAWS = 10_000
# A borderline whose 90% interval spans this much is "too unknown to call";
# below it, the flat was read and found middling.
WIDE = 3.0
SHOWN = 3

_BINS = np.array(GRADE_BINS)


@dataclass(frozen=True)
class Posterior:
    mean: float
    low: float                      # 5th percentile
    high: float                     # 95th percentile
    p_hopeful: float                # P(S >= threshold)
    width_shares: dict[str, float]  # slug -> share of score variance
    voi: dict[str, float]           # slug -> expected swing in P(S >= t); 0 for determined


def _is_advert_silence(grade) -> bool:
    # A model read the advert and found nothing. A failed call proves
    # nothing about the advert, so it stays extraction absence.
    return grade.graded_by.startswith("model:") and not grade.graded_by.endswith("(failed)")


def _criterion_draws(criterion: Criterion, grade, fit: Fit, rng, draws: int) -> np.ndarray:
    entry = fit.criteria.get(criterion.slug)
    sigma = entry.sigma if entry else fit.pooled_sigma
    if grade.value is not None:
        if grade.determined and grade.graded_by.startswith("model:"):
            a, b = (0.0 - grade.value) / sigma, (10.0 - grade.value) / sigma
            return truncnorm.rvs(a, b, loc=grade.value, scale=sigma, size=draws, random_state=rng)
        # A deterministic fact, or a configured default: a configured opinion
        # is not widened.
        return np.full(draws, float(grade.value))
    if criterion.silence_prior and _is_advert_silence(grade):
        items = sorted(criterion.silence_prior.items())
        values = np.array([value for value, _ in items])
        probs = np.array([share for _, share in items])
        # The loader tolerates a sum within 1e-6; rng.choice wants ~1.5e-8.
        # Normalized here rather than tightening the loader's tolerance.
        probs = probs / probs.sum()
        return rng.choice(values, size=draws, p=probs)
    weights = np.array(entry.prior_weights) if entry else np.full(len(_BINS), 1 / len(_BINS))
    return rng.choice(_BINS, size=draws, p=weights)


def score_posterior(
    pairs: list[Pair], fit: Fit, *, threshold: float, seed: int, draws: int = DRAWS
) -> Posterior:
    rng = np.random.default_rng(seed)
    weights = np.array([criterion.weight for criterion, _ in pairs])
    samples = np.stack(
        [_criterion_draws(criterion, grade, fit, rng, draws) for criterion, grade in pairs]
    )
    scores = weights @ samples / weights.sum()

    variances = samples.var(axis=1) * (weights / weights.sum()) ** 2
    total_var = float(variances.sum())
    width_shares = {
        criterion.slug: (float(v) / total_var if total_var else 0.0)
        for (criterion, _), v in zip(pairs, variances)
    }

    p_hopeful = float((scores >= threshold).mean())
    voi: dict[str, float] = {}
    for index, (criterion, grade) in enumerate(pairs):
        if grade.value is not None:
            voi[criterion.slug] = 0.0
            continue
        swings = 0.0
        drawn = samples[index]
        for value in np.unique(drawn):
            mask = drawn == value
            swings += mask.mean() * abs(float((scores[mask] >= threshold).mean()) - p_hopeful)
        voi[criterion.slug] = swings

    return Posterior(
        mean=float(scores.mean()),
        low=float(np.percentile(scores, 5)),
        high=float(np.percentile(scores, 95)),
        p_hopeful=p_hopeful,
        width_shares=width_shares,
        voi=voi,
    )


def _unknown_flag(pairs: list[Pair], posterior: Posterior) -> str:
    # Rank every real source of width. A partially read Listing can have one
    # tiny unknown and one very noisy model Grade; naming only the unknown
    # would diagnose the opposite of what made the interval wide. Configured
    # defaults and deterministic Grades have zero variance, so they disappear
    # at this filter even though a default is undetermined.
    sources = sorted(
        (
            (criterion, grade)
            for criterion, grade in pairs
            if posterior.width_shares.get(criterion.slug, 0.0) > 0
        ),
        key=lambda pair: posterior.width_shares.get(pair[0].slug, 0.0),
        reverse=True,
    )
    if not sources:
        return "Too uncertain to call."

    unknown_share = sum(
        posterior.width_shares.get(criterion.slug, 0.0)
        for criterion, grade in sources
        if grade.value is None
    )
    label = "Too unknown to call" if unknown_share >= 0.5 else "Too uncertain to call"
    named = [criterion for criterion, _ in sources[:SHOWN]]
    # Share is summed over only what is named, so the sentence never
    # overstates what the named criteria carry.
    share = sum(posterior.width_shares.get(criterion.slug, 0.0) for criterion in named)
    names = ", ".join(criterion.name for criterion in named)
    return f"{label} - {names} carry {share:.0%} of the uncertainty."


def adjudicate_bayesian(
    pairs: list[Pair],
    fit: Fit,
    *,
    hopeful_threshold: float,
    hopeful_confidence: float,
    seed: int,
    draws: int = DRAWS,
) -> Evaluation:
    """The same record `adjudicate` builds, off the posterior.

    Vetoes are untouched arithmetic on determined Grades: silence drags the
    score through its prior and never rejects.
    """
    posterior = score_posterior(
        pairs, fit, threshold=hopeful_threshold, seed=seed, draws=draws
    )
    vetoes = vetoes_fired(pairs)
    if vetoes:
        verdict = "reject"
    elif posterior.p_hopeful >= hopeful_confidence:
        verdict = "hopeful"
    else:
        verdict = "borderline"

    red_flags = [grade.evidence for _, grade in vetoes]
    if verdict == "borderline":
        if (posterior.high - posterior.low) >= WIDE:
            red_flags.append(_unknown_flag(pairs, posterior))
        else:
            red_flags.append(
                "Confidently mediocre - the posterior is narrow but does not "
                "clear the hopeful gate."
            )
    red_flags += [grade.concern for _, grade in pairs if grade.concern]
    red_flags += [
        evidence for evidence in _ranked(pairs, above=False) if evidence not in red_flags
    ]

    asked = []
    for criterion, grade in pairs:
        if grade.question:
            asked.append((posterior.voi.get(criterion.slug, 0.0), criterion.slug, grade.question))
        elif not grade.determined and criterion.ask:
            asked.append((posterior.voi.get(criterion.slug, 0.0), criterion.slug, criterion.ask))
    asked.sort(key=lambda entry: (-entry[0], entry[1]))

    return Evaluation(
        verdict=verdict,
        score=posterior.mean,
        # `pairs` is never empty: the loader guarantees at least one Criterion.
        coverage=sum(c.weight for c, g in pairs if g.determined)
        / sum(c.weight for c, _ in pairs),
        score_low=posterior.low,
        score_high=posterior.high,
        p_hopeful=posterior.p_hopeful,
        posterior_fit_hash=fit.input_hash,
        reasons=_ranked(pairs, above=True),
        red_flags=red_flags,
        agent_questions=[question for _, _, question in asked],
    )
