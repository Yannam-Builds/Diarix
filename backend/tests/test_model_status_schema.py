"""Model-free tests for the public model-status contract."""

from backend.backends import AudioInputSpec as RegistryAudioInputSpec
from backend.models import ModelStatus


def test_model_status_accepts_registry_audio_input_dataclass() -> None:
    status = ModelStatus(
        model_name="whisper-base",
        display_name="Whisper Base",
        downloaded=True,
        audio_input=RegistryAudioInputSpec(),
    )

    assert status.audio_input is not None
    assert status.audio_input.sample_rate_hz == 16_000
    assert status.audio_input.channels == 1
    assert status.audio_input.codec == "pcm_s16le"
    assert status.audio_input.container == "wav"

