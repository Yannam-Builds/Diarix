from backend.services.model_lifecycle import (
    DEFAULT_IDLE_TIMEOUT_SECONDS,
    NEVER_UNLOAD,
    normalize_idle_timeout,
)


def test_dictation_idle_timeout_accepts_supported_values() -> None:
    for value in (NEVER_UNLOAD, 0, 15, 60, 300, 600, 900, 3600):
        assert normalize_idle_timeout(value) == value


def test_dictation_idle_timeout_falls_back_safely() -> None:
    assert normalize_idle_timeout(None) == DEFAULT_IDLE_TIMEOUT_SECONDS
    assert normalize_idle_timeout(17) == DEFAULT_IDLE_TIMEOUT_SECONDS
