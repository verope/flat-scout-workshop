from pathlib import Path

import pytest
from pydantic import ValidationError

from flat_scout.config import EvaluationConfig, Features, Settings, load_settings


def test_loads_hard_filters_from_config(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        "[filters]\n"
        "max_price_pcm = 3500\n"
        "allowed_beds = [0, 1]\n"
        "exclude_ground_floor = true\n"
        'postcodes = ["SW8"]\n'
    )
    settings = load_settings(tmp_path / "config.toml")
    assert settings.filters.max_price_pcm == 3500
    assert settings.filters.allowed_beds == [0, 1]
    assert settings.filters.postcodes == ["SW8"]


def test_defaults_apply_for_absent_sections(tmp_path: Path):
    (tmp_path / "config.toml").write_text("[filters]\nmax_price_pcm = 3000\n")
    settings = load_settings(tmp_path / "config.toml")
    assert settings.fetch.max_attempts == 4


def _model_resolution_error(name: str) -> str | None:
    """The message pydantic-ai raises for `name`, or None if it resolves."""
    from pydantic_ai.models import infer_model

    try:
        infer_model(name)
    except Exception as exc:  # noqa: BLE001 - we only care about the message
        return str(exc)
    return None


def test_default_model_is_known_to_pydantic_ai():
    """A bare model name raises UserError: Unknown model, so it must be qualified."""
    error = _model_resolution_error(EvaluationConfig().model)
    assert error is None or "Unknown model" not in error


def test_shipped_config_model_is_known_to_pydantic_ai():
    settings = load_settings(Path("config.toml"))
    error = _model_resolution_error(settings.evaluation.model)
    assert error is None or "Unknown model" not in error


def test_shipped_config_vision_model_is_known_to_pydantic_ai():
    settings = load_settings(Path("config.toml"))
    error = _model_resolution_error(settings.evaluation.vision_model)
    assert error is None or "Unknown model" not in error


def test_the_default_model_is_glm_flash_on_openrouter():
    """Cheap, takes images, and one slug serves both jobs."""
    assert EvaluationConfig().model == "openrouter:z-ai/glm-5.3-flash"
    assert EvaluationConfig().vision_model == "openrouter:z-ai/glm-5.3-flash"


def test_the_text_model_comes_from_the_environment(tmp_path: Path, monkeypatch):
    """`.env` holds the bare OpenRouter slug; the loader qualifies it."""
    (tmp_path / "config.toml").write_text("[filters]\nmax_price_pcm = 3000\n")
    monkeypatch.setenv("OPENROUTER_TEXT_MODEL", "anthropic/claude-sonnet-5")
    settings = load_settings(tmp_path / "config.toml")
    assert settings.evaluation.model == "openrouter:anthropic/claude-sonnet-5"


def test_the_vision_model_follows_the_text_model_unless_it_is_set(
    tmp_path: Path, monkeypatch
):
    """One knob until there is a reason for two."""
    (tmp_path / "config.toml").write_text("[filters]\nmax_price_pcm = 3000\n")
    monkeypatch.setenv("OPENROUTER_TEXT_MODEL", "anthropic/claude-opus-5")
    settings = load_settings(tmp_path / "config.toml")
    assert settings.evaluation.vision_model == "openrouter:anthropic/claude-opus-5"

    monkeypatch.setenv("OPENROUTER_IMAGE_MODEL", "google/gemini-3-flash")
    settings = load_settings(tmp_path / "config.toml")
    assert settings.evaluation.vision_model == "openrouter:google/gemini-3-flash"
    assert settings.evaluation.model == "openrouter:anthropic/claude-opus-5"


def test_the_openrouter_key_is_read_into_secrets(tmp_path: Path, monkeypatch):
    (tmp_path / "config.toml").write_text("[filters]\nmax_price_pcm = 3000\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    settings = load_settings(tmp_path / "config.toml")
    assert settings.secrets.openrouter_api_key == "sk-or-v1-test"


def test_image_signals_is_on_by_default_and_can_be_switched_off(tmp_path: Path):
    """Vision costs money per Listing, so the couple must be able to stop it."""
    assert load_settings(Path("config.toml")).features.image_signals is True
    (tmp_path / "config.toml").write_text("[features]\nimage_signals = false\n")
    assert load_settings(tmp_path / "config.toml").features.image_signals is False


# Four real Listings whose Coverage was measured with no vision signal at all -
# no compass on the plan, or no EPC published, or a Portal whose pages cannot be
# downloaded. That is the ordinary case, not a broken one: `pets` and
# `hard-floors` (4 of the 11 weight) are unknown by design - adverts rarely
# mention either - and the 2 weight that needs vision is absent whenever a plan
# carries no compass.
COVERAGE_WITHOUT_VISION = (0.51, 0.38, 0.54, 0.38)


def test_the_coverage_floor_admits_a_listing_with_no_vision_signal():
    """The floor must not disqualify the ordinary Listing on principle.

    The shipped 0.5 was invented with nothing behind it, and it gated two of
    these four flats before any Grade was read. A floor is a real instrument
    and it should be set from `grade --backfill`'s coverage distribution -
    but until it is, it has to sit under what a normal advert can answer.
    """
    floor = load_settings(Path("config.toml")).evaluation.coverage_floor
    assert all(coverage >= floor for coverage in COVERAGE_WITHOUT_VISION)
    # The shipped file and the default must not drift: `config.toml` is what
    # the CLI reads, and `EvaluationConfig()` is what every test reads.
    assert EvaluationConfig().coverage_floor == floor


def test_bayesian_settings_have_defaults():
    settings = Settings()
    assert settings.evaluation.hopeful_confidence == 0.50
    assert settings.features.bayesian_verdict is False


def test_shipped_config_runs_the_point_score_gate():
    # The shipped file leaves the posterior Verdict off, so the point score and
    # coverage_floor rule. The Pydantic fallback is off too, so an absent
    # [features] section never opts in by accident.
    shipped = load_settings(Path("config.toml"))
    assert shipped.evaluation.hopeful_confidence == EvaluationConfig().hopeful_confidence
    assert shipped.features.bayesian_verdict is False
    assert Features().bayesian_verdict is False


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_hopeful_confidence_must_be_a_probability(confidence):
    with pytest.raises(ValidationError):
        EvaluationConfig(hopeful_confidence=confidence)
