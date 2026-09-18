"""Pure-data contract shared by GPU optimization and local reporting."""

from __future__ import annotations

from dataclasses import dataclass

PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "base_time_per_distance": (0.35, 1.00),
    "difficulty_time_scale": (0.10, 1.20),
    "load_time_scale": (0.05, 0.40),
    "rain_time_scale": (0.05, 1.00),
    "base_stamina_per_distance": (0.05, 0.25),
    "difficulty_stamina_scale": (0.15, 0.80),
    "rain_stamina_scale": (0.02, 0.30),
    "freshness_decay_per_time": (0.008, 0.035),
    "rain_freshness_scale": (0.002, 0.040),
    "strong_hard_advantage": (0.05, 0.65),
    "light_easy_speed_advantage": (0.05, 0.45),
}

LATENT_BOUNDS: dict[str, tuple[float, float]] = {
    "difficulty_time_scale": PARAMETER_BOUNDS["difficulty_time_scale"],
    "strong_hard_advantage": PARAMETER_BOUNDS["strong_hard_advantage"],
    "light_easy_speed_advantage": PARAMETER_BOUNDS["light_easy_speed_advantage"],
    "rain_severity": (0.30, 1.80),
}

INITIAL_LATENTS = {
    "difficulty_time_scale": 0.60,
    "strong_hard_advantage": 0.35,
    "light_easy_speed_advantage": 0.12,
    "rain_severity": 1.0,
}

PROFILE_EPSILONS = (0.0, 0.25, 0.5, 0.75, 1.0)
TARGET_WIN_PROFILE = (0.5, 0.38, 0.27, 0.18, 0.10)

TRAIN_SEED_START = 0
TRAIN_MAPS = 64
VALIDATION_SEED_START = 10_000
VALIDATION_MAPS = 16
TEST_SEED_START = 20_000


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    balance: bool
    specialization: bool
    skill_profile: bool
    weather: bool


ABLATIONS = (
    ObjectiveSpec("viability", False, False, False, False),
    ObjectiveSpec("+balance", True, False, False, False),
    ObjectiveSpec("+specialization", True, True, False, False),
    ObjectiveSpec("+skill_weather", True, True, True, True),
)
