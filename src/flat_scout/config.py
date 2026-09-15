from __future__ import annotations

import os
import tomllib
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator


class Filters(BaseModel):
    max_price_pcm: int = 6000
    allowed_beds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    exclude_ground_floor: bool = False
    postcodes: list[str] = Field(default_factory=list)


# Every model call goes through OpenRouter, and this is the prefix pydantic-ai
# resolves to its OpenRouter model. The slug after it is OpenRouter's own.
OPENROUTER_PREFIX = "openrouter:"
# Cheap, accepts images, so one slug serves the graders and the vision reads
# alike until there is a reason for two.
DEFAULT_MODEL_SLUG = "z-ai/glm-5.3-flash"


# Handed to every Agent the app builds. The OpenAI client underneath the
# OpenRouter model waits ten minutes for a response by default, and a hung
# connection held two Listings of a corpus run for exactly that. A grader
# call on the default model takes under a minute with its reasoning on, so
# three minutes is generous, and a call that is not back by then is retried
# by `backoff.with_backoff` rather than waited for.
MODEL_SETTINGS: dict = {"timeout": 180}


def openrouter_model(slug: str) -> str:
    """The pydantic-ai name for an OpenRouter slug such as `z-ai/glm-5.3-flash`."""
    return slug if slug.startswith(OPENROUTER_PREFIX) else f"{OPENROUTER_PREFIX}{slug}"


class EvaluationConfig(BaseModel):
    # Set from OPENROUTER_TEXT_MODEL in `.env`, not from `config.toml`: the
    # model is chosen alongside the key that pays for it, and one place for
    # the choice is one fewer to get out of step.
    model: str = openrouter_model(DEFAULT_MODEL_SLUG)
    # Reading a band off a graph is a different job from weighing a brief, and
    # may deserve a different model. Set from OPENROUTER_IMAGE_MODEL; left
    # empty it follows `model`, so there is one knob to turn until there is a
    # reason for two.
    vision_model: str = ""
    hopeful_threshold: float = 6.5
    criteria_path: str = "criteria/criteria.md"
    # Where the Criterion files live. `criteria_path` above is the holistic
    # path's single brief, and both are read while the flag below is false.
    criteria_dir: str = "criteria"
    # Below this share of the total Criterion weight a Listing cannot be
    # hopeful, whatever it scores. A thin advert that answers three Criteria
    # well would otherwise outrank a documented flat with one real weakness.
    #
    # A STARTING POINT, not a considered value: set it from the coverage
    # distribution `grade --backfill` prints, the same way `hopeful_threshold`
    # is set from the score distribution. The first number here was 0.5,
    # invented with nothing behind it, and it gated ordinary flats: adverts
    # rarely mention pets or flooring, and the aspect is absent whenever a
    # plan carries no compass, so an ordinary Listing reads well under half.
    coverage_floor: float = 0.35
    # What an unread Criterion scores. Silence is priced into the Score rather
    # than dropped from it, so an advert cannot win by saying less - see
    # `adjudicate`. Raise it toward 5 to treat silence as neutral, lower it to
    # punish a thin advert harder. Changing it costs no model call: every Grade
    # is stored, so `report --weights` replays the whole corpus on the new
    # value in a second.
    unknown_prior: float = 4.0
    # The p in `P(score >= hopeful_threshold) >= p`. Only read while
    # features.bayesian_verdict is on. Set it from `report --posterior` once you
    # have decisions of your own, not from a guess, and it moves with
    # `hopeful_threshold` in `config.toml`: the two are one gate, and
    # `test_shipped_config_runs_the_point_score_gate` holds this default and the
    # shipped file together.
    hopeful_confidence: float = Field(default=0.50, ge=0, le=1)

    @model_validator(mode="after")
    def _vision_model_defaults_to_the_evaluator(self) -> EvaluationConfig:
        if not self.vision_model:
            self.vision_model = self.model
        return self


class FetchConfig(BaseModel):
    delay_seconds: float = 4.0
    max_attempts: int = 4
    timeout_seconds: float = 30.0
    user_agent: str = "Mozilla/5.0"


class Features(BaseModel):
    # Reading the EPC graph and the floorplan with a vision model. It costs
    # real money per Listing, so it must be switchable off without a rebuild.
    # Off, the evaluator sees the text alone, exactly as it did before.
    image_signals: bool = True
    # Report every wall a flat is glazed on, rather than the best-confirmed one.
    # Off by default because it measures worse: on 24 annotated floorplans the
    # single-aspect output was right on 9 of 17, and reporting them all on 6 of
    # 17 - more true statements, and more false ones with them. The machinery is
    # the same either way, and 13 of those 24 flats really are dual aspect, so
    # this is a threshold to revisit against a bigger corpus rather than a
    # feature to delete.
    multi_aspect: bool = False
    # The weighted Criteria in criteria/*.md, in place of the single holistic
    # score. On by default: the Criteria are what the couple now edit, and a
    # brief nobody can tune was the whole reason for replacing the old score.
    #
    # Turning it off falls back to the holistic evaluator reading
    # `criteria_path`, and is a TWO-value revert: the flag alone leaves the
    # threshold meaning something else than it did. The two paths score on
    # different scales, so set `hopeful_threshold` from a fresh `report` after
    # switching. Both are in config.toml, so either direction is an edit.
    weighted_criteria: bool = True
    # The posterior Verdict of the Bayesian scoring path. Off, the point score
    # and coverage_floor rule exactly as before. On, the gate is
    # P(score >= hopeful_threshold) >= hopeful_confidence.
    bayesian_verdict: bool = False


class Secrets(BaseModel):
    # pydantic-ai's OpenRouter provider reads it from the environment itself;
    # it is carried here so every secret the app knows about is in one place.
    openrouter_api_key: str = ""
    # The switch for logfire, and the only one. Absent means nothing is
    # exported and nothing is configured - see `observe.py`.
    logfire_token: str = ""


class Settings(BaseModel):
    filters: Filters = Filters()
    evaluation: EvaluationConfig = EvaluationConfig()
    fetch: FetchConfig = FetchConfig()
    features: Features = Features()
    secrets: Secrets = Secrets()


def load_settings(config_path: Path = Path("config.toml")) -> Settings:
    raw = tomllib.loads(config_path.read_text())
    load_dotenv()
    secrets = Secrets(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        logfire_token=os.getenv("LOGFIRE_TOKEN", ""),
    )
    evaluation = dict(raw.pop("evaluation", {}))
    # Empty counts as unset: the test suite pins these blank so a developer's
    # own `.env` cannot leak into an assertion about the default.
    if text_slug := os.getenv("OPENROUTER_TEXT_MODEL", ""):
        evaluation["model"] = openrouter_model(text_slug)
    if image_slug := os.getenv("OPENROUTER_IMAGE_MODEL", ""):
        evaluation["vision_model"] = openrouter_model(image_slug)
    return Settings(**raw, evaluation=evaluation, secrets=secrets)
