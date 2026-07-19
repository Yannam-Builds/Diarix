"""
Backend abstraction layer for TTS and STT.

Provides a unified interface for MLX and PyTorch backends,
and a model config registry that eliminates per-engine dispatch maps.
"""

# Install HF compatibility patches before any backend imports transformers /
# huggingface_hub. The module runs ``patch_transformers_mistral_regex`` at
# import time, which wraps transformers' tokenizer load against the
# unconditional HuggingFace metadata call that otherwise raises on
# HF_HUB_OFFLINE=1 and on network failures.
from ..utils import hf_offline_patch  # noqa: F401

import threading
from dataclasses import dataclass, field
from typing import Callable, Protocol, Optional, Tuple, List
from typing_extensions import runtime_checkable
import numpy as np

DEFAULT_LLM_MAX_TOKENS = 512
DEFAULT_LLM_TEMPERATURE = 0.7

ProgressCallback = Callable[[float], None]

from ..utils.platform_detect import get_backend_type

LANGUAGE_CODE_TO_NAME = {
    "zh": "chinese",
    "en": "english",
    "ja": "japanese",
    "ko": "korean",
    "de": "german",
    "fr": "french",
    "ru": "russian",
    "pt": "portuguese",
    "es": "spanish",
    "it": "italian",
}

WHISPER_HF_REPOS = {
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v3",
    "turbo": "openai/whisper-large-v3-turbo",
    "distil-large-v3.5": "distil-whisper/distil-large-v3.5",
}

WHISPER_LANGUAGE_CODES = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca",
    "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms",
    "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la",
    "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be",
    "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
    "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha",
    "ba", "jw", "su", "yue",
]

QWEN3_ASR_LANGUAGE_CODES = [
    "zh", "en", "yue", "ar", "de", "fr", "es", "pt", "id", "it", "ko", "ru",
    "th", "vi", "ja", "tr", "hi", "ms", "nl", "sv", "da", "fi", "pl", "cs",
    "fil", "fa", "el", "hu", "mk", "ro",
]


@dataclass(frozen=True)
class AudioInputSpec:
    """Canonical media format an STT model receives after ingestion."""

    sample_rate_hz: int = 16_000
    channels: int = 1
    sample_format: str = "s16"
    codec: str = "pcm_s16le"
    container: str = "wav"


DEFAULT_STT_AUDIO_INPUT = AudioInputSpec()


@dataclass
class ModelConfig:
    """Declarative config for a downloadable model variant."""

    model_name: str  # e.g. "luxtts", "chatterbox-tts"
    display_name: str  # e.g. "LuxTTS (Fast, CPU-friendly)"
    engine: str  # e.g. "luxtts", "chatterbox"
    hf_repo_id: str  # e.g. "YatharthS/LuxTTS"
    model_size: str = "default"
    size_mb: int = 0
    needs_trim: bool = False
    supports_instruct: bool = False
    languages: list[str] = field(default_factory=lambda: ["en"])
    modality: str = "tts"
    runtime_group: str = "core"
    capabilities: list[str] = field(default_factory=list)
    description: str = ""
    precision_options: list[str] = field(default_factory=list)
    default_precision: Optional[str] = None
    recommended: bool = False
    min_vram_gb: Optional[float] = None
    audio_input: Optional[AudioInputSpec] = None
    artifact_filename: Optional[str] = None
    base_model_id: Optional[str] = None
    source_license: Optional[str] = None


def _stt_model_config(**kwargs) -> ModelConfig:
    """Build an STT config with the shared normalized-audio contract."""
    return ModelConfig(audio_input=DEFAULT_STT_AUDIO_INPUT, **kwargs)


@runtime_checkable
class TTSBackend(Protocol):
    """Protocol for TTS backend implementations."""

    # Each backend class should define MODEL_CONFIGS as a class variable:
    # MODEL_CONFIGS: list[ModelConfig]

    async def load_model(self, model_size: str) -> None:
        """Load TTS model."""
        ...

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """
        Create voice prompt from reference audio.

        Returns:
            Tuple of (voice_prompt_dict, was_cached)
        """
        ...

    async def combine_voice_prompts(
        self,
        audio_paths: List[str],
        reference_texts: List[str],
    ) -> Tuple[np.ndarray, str]:
        """
        Combine multiple voice prompts.

        Returns:
            Tuple of (combined_audio_array, combined_text)
        """
        ...

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio from text.

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        ...

    def unload_model(self) -> None:
        """Unload model to free memory."""
        ...

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        ...

    def _get_model_path(self, model_size: str) -> str:
        """
        Get model path for a given size.

        Returns:
            Model path or HuggingFace Hub ID
        """
        ...


@runtime_checkable
class STTBackend(Protocol):
    """Protocol for STT (Speech-to-Text) backend implementations."""

    async def load_model(self, model_size: str) -> None:
        """Load STT model."""
        ...

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        model_size: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        partial_callback: Optional[Callable[[str], None]] = None,
        segments_callback: Optional[Callable[[list], None]] = None,
    ) -> str:
        """
        Transcribe audio to text.

        ``should_stop``, if given, is polled between audio chunks (where the
        adapter supports chunking) so a cancelled job can stop after the
        in-flight chunk instead of running to completion. ``partial_callback``,
        if given, receives the accumulated transcript text after each chunk
        completes, for live progress display. ``segments_callback``, if given,
        receives the final ``{start, end, text}`` segment list from engines
        with real timestamps (Faster-Whisper, WhisperX); engines without
        honest segment boundaries never invoke it.

        Returns:
            Transcribed text
        """
        ...

    def unload_model(self) -> None:
        """Unload model to free memory."""
        ...

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        ...


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol for local LLM (chat/completion) backend implementations."""

    async def load_model(self, model_size: str) -> None:
        """Load LLM weights and tokenizer."""
        ...

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        model_size: Optional[str] = None,
        examples: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        """Run a single-turn chat completion and return the assistant reply.

        ``examples`` is an optional list of ``(user, assistant)`` pairs
        prepended to the conversation as proper chat turns — small models
        pattern-match on inline system-prompt examples (echoing them
        verbatim for unrelated inputs), but treat structured turns as
        data and generalize instead. Used by the refinement service.
        """
        ...

    def unload_model(self) -> None:
        ...

    def is_loaded(self) -> bool:
        ...


# Global backend instances
_tts_backend: Optional[TTSBackend] = None
_tts_backends: dict[str, TTSBackend] = {}
_tts_backends_lock = threading.Lock()
_stt_backend: Optional[STTBackend] = None
_stt_backends: dict[str, STTBackend] = {}
_stt_backends_lock = threading.Lock()
_llm_backends: dict[str, LLMBackend] = {}
_llm_backends_lock = threading.Lock()

# Supported TTS engines — keyed by engine name, value is the backend class import path.
# The factory function uses this for the if/elif chain; the model configs live on the backend classes.
TTS_ENGINES = {
    "qwen": "Qwen TTS",
    "qwen_custom_voice": "Qwen CustomVoice",
    "luxtts": "LuxTTS",
    "chatterbox": "Chatterbox TTS",
    "chatterbox_turbo": "Chatterbox Turbo",
    "tada": "TADA",
    "kokoro": "Kokoro",
}

LLM_ENGINES = {
    "qwen_llm": "Qwen3 LLM",
}


def _get_qwen_model_configs() -> list[ModelConfig]:
    """Return Qwen model configs with backend-aware HF repo IDs."""
    backend_type = get_backend_type()
    if backend_type == "mlx":
        repo_1_7b = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
        repo_0_6b = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
    else:
        repo_1_7b = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        repo_0_6b = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

    return [
        ModelConfig(
            model_name="qwen-tts-1.7B",
            display_name="Qwen TTS 1.7B",
            engine="qwen",
            hf_repo_id=repo_1_7b,
            model_size="1.7B",
            size_mb=3500,
            supports_instruct=False,  # Base model drops instruct silently
            languages=["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
        ),
        ModelConfig(
            model_name="qwen-tts-0.6B",
            display_name="Qwen TTS 0.6B",
            engine="qwen",
            hf_repo_id=repo_0_6b,
            model_size="0.6B",
            size_mb=1200,
            supports_instruct=False,
            languages=["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
        ),
    ]


def _get_qwen_custom_voice_configs() -> list[ModelConfig]:
    """Return Qwen CustomVoice model configs."""
    return [
        ModelConfig(
            model_name="qwen-custom-voice-1.7B",
            display_name="Qwen CustomVoice 1.7B",
            engine="qwen_custom_voice",
            hf_repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            model_size="1.7B",
            size_mb=3500,
            supports_instruct=True,
            languages=["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
        ),
        ModelConfig(
            model_name="qwen-custom-voice-0.6B",
            display_name="Qwen CustomVoice 0.6B",
            engine="qwen_custom_voice",
            hf_repo_id="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            model_size="0.6B",
            size_mb=1200,
            supports_instruct=True,
            languages=["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
        ),
    ]


def _get_non_qwen_tts_configs() -> list[ModelConfig]:
    """Return model configs for non-Qwen TTS engines.

    These are static — no backend-type branching needed.
    """
    return [
        ModelConfig(
            model_name="luxtts",
            display_name="LuxTTS (Fast, CPU-friendly)",
            engine="luxtts",
            hf_repo_id="YatharthS/LuxTTS",
            size_mb=300,
            languages=["en"],
        ),
        ModelConfig(
            model_name="chatterbox-tts",
            display_name="Chatterbox TTS (Multilingual)",
            engine="chatterbox",
            hf_repo_id="ResembleAI/chatterbox",
            size_mb=3200,
            needs_trim=True,
            languages=[
                "zh",
                "en",
                "ja",
                "ko",
                "de",
                "fr",
                "ru",
                "pt",
                "es",
                "it",
                "he",
                "ar",
                "da",
                "el",
                "fi",
                "hi",
                "ms",
                "nl",
                "no",
                "pl",
                "sv",
                "sw",
                "tr",
            ],
        ),
        ModelConfig(
            model_name="chatterbox-turbo",
            display_name="Chatterbox Turbo (English, Tags)",
            engine="chatterbox_turbo",
            hf_repo_id="ResembleAI/chatterbox-turbo",
            size_mb=1500,
            needs_trim=True,
            languages=["en"],
        ),
        ModelConfig(
            model_name="tada-1b",
            display_name="TADA 1B (English)",
            engine="tada",
            hf_repo_id="HumeAI/tada-1b",
            model_size="1B",
            size_mb=4000,
            languages=["en"],
        ),
        ModelConfig(
            model_name="tada-3b-ml",
            display_name="TADA 3B Multilingual",
            engine="tada",
            hf_repo_id="HumeAI/tada-3b-ml",
            model_size="3B",
            size_mb=8000,
            languages=["en", "ar", "zh", "de", "es", "fr", "it", "ja", "pl", "pt"],
        ),
        ModelConfig(
            model_name="kokoro",
            display_name="Kokoro 82M",
            engine="kokoro",
            hf_repo_id="hexgrad/Kokoro-82M",
            size_mb=350,
            languages=["en", "es", "fr", "hi", "it", "pt", "ja", "zh"],
        ),
    ]


def _get_whisper_configs() -> list[ModelConfig]:
    """Return Whisper STT model configs."""
    return [
        _stt_model_config(
            model_name="whisper-base",
            display_name="Whisper Base",
            engine="whisper",
            hf_repo_id="openai/whisper-base",
            model_size="base",
            modality="stt",
            capabilities=["language_detection", "long_audio", "multilingual"],
            languages=WHISPER_LANGUAGE_CODES,
            description="Compact general-purpose transcription.",
        ),
        _stt_model_config(
            model_name="whisper-small",
            display_name="Whisper Small",
            engine="whisper",
            hf_repo_id="openai/whisper-small",
            model_size="small",
            modality="stt",
            capabilities=["language_detection", "long_audio", "multilingual"],
            languages=WHISPER_LANGUAGE_CODES,
            description="Balanced local transcription.",
        ),
        _stt_model_config(
            model_name="whisper-medium",
            display_name="Whisper Medium",
            engine="whisper",
            hf_repo_id="openai/whisper-medium",
            model_size="medium",
            modality="stt",
            capabilities=["language_detection", "long_audio", "multilingual"],
            languages=WHISPER_LANGUAGE_CODES,
            description="Higher-accuracy multilingual transcription.",
        ),
        _stt_model_config(
            model_name="whisper-large",
            display_name="Whisper Large",
            engine="whisper",
            hf_repo_id="openai/whisper-large-v3",
            model_size="large",
            modality="stt",
            capabilities=["language_detection", "long_audio", "multilingual"],
            languages=WHISPER_LANGUAGE_CODES,
            description="Whisper's highest-quality multilingual model.",
            min_vram_gb=10,
        ),
        _stt_model_config(
            model_name="whisper-turbo",
            display_name="Whisper Turbo",
            engine="whisper",
            hf_repo_id="openai/whisper-large-v3-turbo",
            model_size="turbo",
            modality="stt",
            capabilities=["language_detection", "long_audio", "multilingual"],
            languages=WHISPER_LANGUAGE_CODES,
            description="Near-large quality with faster inference.",
            recommended=True,
            min_vram_gb=6,
        ),
        _stt_model_config(
            model_name="whisper-distil-large-v3.5",
            display_name="Distil-Whisper Large v3.5",
            engine="whisper",
            hf_repo_id="distil-whisper/distil-large-v3.5",
            model_size="distil-large-v3.5",
            size_mb=1600,
            languages=["en"],
            modality="stt",
            capabilities=["long_audio"],
            description="Fast, accurate English transcription distilled from Whisper Large v3.",
            precision_options=["float16", "float32"],
            default_precision="float16",
            min_vram_gb=5,
        ),
    ]


def _get_advanced_stt_configs() -> list[ModelConfig]:
    """Optional STT engines supplied by the advanced speech runtime."""
    european_languages = [
        "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de",
        "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk",
        "sl", "es", "sv", "ru", "uk",
    ]
    return [
        _stt_model_config(
            model_name="faster-whisper-tiny",
            display_name="Faster-Whisper Tiny",
            engine="faster_whisper",
            hf_repo_id="Systran/faster-whisper-tiny",
            model_size="faster-whisper-tiny",
            size_mb=75,
            languages=WHISPER_LANGUAGE_CODES,
            modality="stt",
            runtime_group="advanced-asr",
            capabilities=["language_detection", "segment_timestamps", "vad", "long_audio", "multilingual"],
            description="Compact multilingual CTranslate2 transcription with VAD and native long-audio segmentation.",
            precision_options=["float16", "int8_float16", "int8"],
            default_precision="float16",
            min_vram_gb=1,
        ),
        _stt_model_config(
            model_name="faster-distil-whisper-large-v3",
            display_name="Faster Distil-Whisper Large v3",
            engine="faster_whisper",
            hf_repo_id="Systran/faster-distil-whisper-large-v3",
            model_size="faster-distil-whisper-large-v3",
            size_mb=1500,
            languages=["en"],
            modality="stt",
            runtime_group="advanced-asr",
            capabilities=["segment_timestamps", "vad", "long_audio"],
            description="Fast English-only CTranslate2 transcription optimized for long recordings.",
            precision_options=["float16", "int8_float16", "int8"],
            default_precision="float16",
            min_vram_gb=4,
        ),
        _stt_model_config(
            model_name="whisperx-large-v3",
            display_name="WhisperX Large v3",
            engine="whisperx",
            hf_repo_id="Systran/faster-whisper-large-v3",
            model_size="large-v3",
            size_mb=3100,
            languages=WHISPER_LANGUAGE_CODES,
            modality="stt",
            runtime_group="advanced-asr",
            capabilities=[
                "language_detection", "word_timestamps", "alignment", "vad", "long_audio", "multilingual"
            ],
            description="Meeting transcription with aligned word timestamps and voice activity detection.",
            precision_options=["float16", "int8_float16", "int8"],
            default_precision="float16",
            recommended=True,
            min_vram_gb=8,
        ),
        _stt_model_config(
            model_name="nvidia-parakeet-tdt-0.6b-v3",
            display_name="NVIDIA Parakeet TDT 0.6B v3",
            engine="nemo_asr",
            hf_repo_id="nvidia/parakeet-tdt-0.6b-v3",
            model_size="nvidia-parakeet-tdt-0.6b-v3",
            size_mb=2500,
            languages=european_languages,
            modality="stt",
            runtime_group="advanced-asr",
            capabilities=["language_detection", "word_timestamps", "long_audio", "multilingual"],
            description="High-throughput multilingual transcription for 25 European languages.",
            precision_options=["bfloat16", "float16", "float32"],
            default_precision="bfloat16",
            min_vram_gb=4,
        ),
        _stt_model_config(
            model_name="nvidia-canary-180m-flash",
            display_name="NVIDIA Canary 180M Flash",
            engine="nemo_asr",
            hf_repo_id="nvidia/canary-180m-flash",
            model_size="nvidia-canary-180m-flash",
            size_mb=703,
            languages=["en", "de", "fr", "es"],
            modality="stt",
            runtime_group="advanced-asr",
            capabilities=["translation", "word_timestamps", "long_audio", "multilingual"],
            description="Low-latency NeMo transcription for English, German, French, and Spanish.",
            precision_options=["bfloat16", "float16", "float32"],
            default_precision="bfloat16",
            min_vram_gb=2,
        ),
        _stt_model_config(
            model_name="nvidia-canary-1b-v2",
            display_name="NVIDIA Canary 1B v2",
            engine="nemo_asr",
            hf_repo_id="nvidia/canary-1b-v2",
            model_size="nvidia-canary-1b-v2",
            size_mb=4000,
            languages=european_languages,
            modality="stt",
            runtime_group="advanced-asr",
            capabilities=["translation", "word_timestamps", "long_audio", "multilingual"],
            description="Multilingual transcription and speech translation through NVIDIA NeMo.",
            precision_options=["bfloat16", "float16", "float32"],
            default_precision="bfloat16",
            min_vram_gb=6,
        ),
        _stt_model_config(
            model_name="nvidia-canary-qwen-2.5b",
            display_name="NVIDIA Canary-Qwen 2.5B",
            engine="nemo_asr",
            hf_repo_id="nvidia/canary-qwen-2.5b",
            model_size="nvidia-canary-qwen-2.5b",
            size_mb=4882,
            languages=["en"],
            modality="stt",
            runtime_group="advanced-asr",
            capabilities=["punctuation", "capitalization", "long_audio"],
            description="English speech recognition with punctuation and capitalization through NVIDIA NeMo SALM.",
            precision_options=["bfloat16", "float16"],
            default_precision="bfloat16",
            min_vram_gb=8,
        ),
        _stt_model_config(
            model_name="qwen3-asr-0.6b",
            display_name="Qwen3-ASR 0.6B",
            engine="qwen_asr",
            hf_repo_id="Qwen/Qwen3-ASR-0.6B",
            model_size="qwen3-asr-0.6b",
            size_mb=1800,
            languages=QWEN3_ASR_LANGUAGE_CODES,
            modality="stt",
            runtime_group="advanced-asr",
            capabilities=["language_detection", "streaming", "long_audio", "multilingual"],
            description="Efficient multilingual and streaming speech recognition.",
            precision_options=["bfloat16", "float16"],
            default_precision="bfloat16",
            min_vram_gb=4,
        ),
        _stt_model_config(
            model_name="qwen3-asr-1.7b",
            display_name="Qwen3-ASR 1.7B",
            engine="qwen_asr",
            hf_repo_id="Qwen/Qwen3-ASR-1.7B",
            model_size="qwen3-asr-1.7b",
            size_mb=4200,
            languages=QWEN3_ASR_LANGUAGE_CODES,
            modality="stt",
            runtime_group="advanced-asr",
            capabilities=["language_detection", "streaming", "long_audio", "multilingual"],
            description="High-accuracy multilingual recognition for speech, songs, and difficult audio.",
            precision_options=["bfloat16", "float16"],
            default_precision="bfloat16",
            min_vram_gb=8,
        ),
        _stt_model_config(
            model_name="ibm-granite-speech-3.3-8b",
            display_name="IBM Granite Speech 3.3 8B",
            engine="transformers_asr",
            hf_repo_id="ibm-granite/granite-speech-3.3-8b",
            model_size="ibm-granite-speech-3.3-8b",
            size_mb=18000,
            languages=["en", "fr", "de", "es", "pt"],
            modality="stt",
            runtime_group="advanced-asr",
            capabilities=["translation", "speech_reasoning", "multilingual"],
            description="Heavy speech-language model for transcription, translation, and text follow-up.",
            precision_options=["bfloat16", "float16"],
            default_precision="bfloat16",
            min_vram_gb=18,
        ),
        _stt_model_config(
            model_name="faster-whisper-base",
            display_name="Faster-Whisper Base",
            engine="faster_whisper",
            hf_repo_id="Systran/faster-whisper-base",
            model_size="faster-whisper-base",
            size_mb=140,
            languages=WHISPER_LANGUAGE_CODES,
            modality="stt",
            runtime_group="advanced-asr",
            capabilities=["language_detection", "segment_timestamps", "vad", "long_audio", "multilingual"],
            description="CTranslate2 Base Whisper model for faster multilingual transcription.",
            precision_options=["float16", "int8_float16", "int8"],
            default_precision="float16",
            min_vram_gb=1,
        ),
        _stt_model_config(
            model_name="faster-whisper-small",
            display_name="Faster-Whisper Small",
            engine="faster_whisper",
            hf_repo_id="Systran/faster-whisper-small",
            model_size="faster-whisper-small",
            size_mb=460,
            languages=WHISPER_LANGUAGE_CODES,
            modality="stt",
            runtime_group="advanced-asr",
            capabilities=["language_detection", "segment_timestamps", "vad", "long_audio", "multilingual"],
            description="CTranslate2 Small Whisper model for balanced multilingual transcription.",
            precision_options=["float16", "int8_float16", "int8"],
            default_precision="float16",
            min_vram_gb=2,
        ),
        _stt_model_config(
            model_name="faster-whisper-medium",
            display_name="Faster-Whisper Medium",
            engine="faster_whisper",
            hf_repo_id="Systran/faster-whisper-medium",
            model_size="faster-whisper-medium",
            size_mb=1500,
            languages=WHISPER_LANGUAGE_CODES,
            modality="stt",
            runtime_group="advanced-asr",
            capabilities=["language_detection", "segment_timestamps", "vad", "long_audio", "multilingual"],
            description="CTranslate2 Medium Whisper model for high-accuracy multilingual transcription.",
            precision_options=["float16", "int8_float16", "int8"],
            default_precision="float16",
            min_vram_gb=5,
        ),
        _stt_model_config(
            model_name="faster-whisper-large-v3",
            display_name="Faster-Whisper Large v3",
            engine="faster_whisper",
            hf_repo_id="Systran/faster-whisper-large-v3",
            model_size="faster-whisper-large-v3",
            size_mb=3100,
            languages=WHISPER_LANGUAGE_CODES,
            modality="stt",
            runtime_group="advanced-asr",
            capabilities=["language_detection", "segment_timestamps", "vad", "long_audio", "multilingual"],
            description="CTranslate2 Large v3 Whisper model for maximum multilingual accuracy.",
            precision_options=["float16", "int8_float16", "int8"],
            default_precision="float16",
            min_vram_gb=8,
        ),
        _stt_model_config(
            model_name="faster-whisper-large-v3-turbo",
            display_name="Faster-Whisper Large v3 Turbo",
            engine="faster_whisper",
            hf_repo_id="dropbox-dash/faster-whisper-large-v3-turbo",
            model_size="faster-whisper-large-v3-turbo",
            size_mb=1600,
            languages=WHISPER_LANGUAGE_CODES,
            modality="stt",
            runtime_group="advanced-asr",
            capabilities=["language_detection", "segment_timestamps", "vad", "long_audio", "multilingual"],
            description="CTranslate2 Turbo Whisper model for fast high-accuracy multilingual transcription.",
            precision_options=["float16", "int8_float16", "int8"],
            default_precision="float16",
            min_vram_gb=6,
        ),
    ]


def _get_native_gguf_stt_configs() -> list[ModelConfig]:
    """Handy-compatible GGUF models backed by the in-process transcribe.cpp runtime.

    This catalog deliberately omits GGUF conversions of checkpoints that are
    already represented by Diarix's Whisper, Qwen3-ASR, Canary, Parakeet TDT,
    and Granite entries. The remaining models add genuinely different
    streaming, language, or size choices without presenting duplicate names
    that download the same base model twice.
    """

    return [
        _stt_model_config(
            model_name="parakeet-unified-en-0.6b-gguf",
            display_name="Parakeet Unified EN 0.6B",
            engine="transcribe_cpp",
            hf_repo_id="handy-computer/parakeet-unified-en-0.6b-gguf",
            model_size="parakeet-unified-en-0.6b-gguf",
            size_mb=698,
            languages=["en"],
            modality="stt",
            runtime_group="native-asr",
            capabilities=["streaming", "token_timestamps", "long_audio"],
            description="Fast, accurate live English transcription through the native GGUF runtime.",
            precision_options=["Q8_0"],
            default_precision="Q8_0",
            recommended=True,
            min_vram_gb=2,
            artifact_filename="parakeet-unified-en-0.6b-Q8_0.gguf",
            base_model_id="nvidia/parakeet-unified-en-0.6b",
            source_license="cc-by-4.0",
        ),
        _stt_model_config(
            model_name="nemotron-3.5-asr-streaming-0.6b-gguf",
            display_name="Nemotron Streaming 3.5",
            engine="transcribe_cpp",
            hf_repo_id="handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf",
            model_size="nemotron-3.5-asr-streaming-0.6b-gguf",
            size_mb=716,
            languages=[
                "en", "es", "fr", "it", "pt", "nl", "de", "tr", "ru", "ar",
                "hi", "ja", "ko", "vi", "uk", "pl", "sv", "cs", "nb", "da",
                "bg", "fi", "hr", "sk", "zh", "hu", "ro", "et",
            ],
            modality="stt",
            runtime_group="native-asr",
            capabilities=[
                "language_detection", "streaming", "token_timestamps",
                "long_audio", "multilingual",
            ],
            description="Live multilingual transcription across 28 languages.",
            precision_options=["Q8_0"],
            default_precision="Q8_0",
            recommended=True,
            min_vram_gb=2,
            artifact_filename="nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf",
            base_model_id="nvidia/nemotron-3.5-asr-streaming-0.6b",
            source_license="other",
        ),
        _stt_model_config(
            model_name="moonshine-streaming-tiny-gguf",
            display_name="Moonshine Streaming Tiny",
            engine="transcribe_cpp",
            hf_repo_id="handy-computer/moonshine-streaming-tiny-gguf",
            model_size="moonshine-streaming-tiny-gguf",
            size_mb=49,
            languages=["en"],
            modality="stt",
            runtime_group="native-asr",
            capabilities=["streaming", "long_audio"],
            description="Ultra-light English live dictation model with true incremental text.",
            precision_options=["Q8_0"],
            default_precision="Q8_0",
            min_vram_gb=1,
            artifact_filename="moonshine-streaming-tiny-Q8_0.gguf",
            base_model_id="UsefulSensors/moonshine-streaming-tiny",
            source_license="mit",
        ),
        _stt_model_config(
            model_name="moonshine-streaming-small-gguf",
            display_name="Moonshine Streaming Small",
            engine="transcribe_cpp",
            hf_repo_id="handy-computer/moonshine-streaming-small-gguf",
            model_size="moonshine-streaming-small-gguf",
            size_mb=190,
            languages=["en"],
            modality="stt",
            runtime_group="native-asr",
            capabilities=["streaming", "long_audio"],
            description="Balanced English live dictation model with true incremental text.",
            precision_options=["Q8_0"],
            default_precision="Q8_0",
            min_vram_gb=1,
            artifact_filename="moonshine-streaming-small-Q8_0.gguf",
            base_model_id="UsefulSensors/moonshine-streaming-small",
            source_license="mit",
        ),
        _stt_model_config(
            model_name="moonshine-streaming-medium-gguf",
            display_name="Moonshine Streaming Medium",
            engine="transcribe_cpp",
            hf_repo_id="handy-computer/moonshine-streaming-medium-gguf",
            model_size="moonshine-streaming-medium-gguf",
            size_mb=283,
            languages=["en"],
            modality="stt",
            runtime_group="native-asr",
            capabilities=["streaming", "long_audio"],
            description="Higher-quality English live dictation with true incremental text.",
            precision_options=["Q8_0"],
            default_precision="Q8_0",
            min_vram_gb=1,
            artifact_filename="moonshine-streaming-medium-Q8_0.gguf",
            base_model_id="UsefulSensors/moonshine-streaming-medium",
            source_license="mit",
        ),
        _stt_model_config(
            model_name="sensevoice-small-gguf",
            display_name="SenseVoice Small",
            engine="transcribe_cpp",
            hf_repo_id="handy-computer/SenseVoiceSmall-gguf",
            model_size="sensevoice-small-gguf",
            size_mb=241,
            languages=["zh", "yue", "en", "ja", "ko"],
            modality="stt",
            runtime_group="native-asr",
            capabilities=["language_detection", "long_audio", "multilingual"],
            description="Compact five-language recognition with automatic language detection.",
            precision_options=["Q8_0"],
            default_precision="Q8_0",
            min_vram_gb=1,
            artifact_filename="SenseVoiceSmall-Q8_0.gguf",
            base_model_id="FunAudioLLM/SenseVoiceSmall",
            source_license="other",
        ),
        _stt_model_config(
            model_name="funasr-nano-multilingual-gguf",
            display_name="Fun-ASR Nano Multilingual",
            engine="transcribe_cpp",
            hf_repo_id="handy-computer/Fun-ASR-MLT-Nano-2512-gguf",
            model_size="funasr-nano-multilingual-gguf",
            size_mb=850,
            languages=[
                "zh", "en", "yue", "ja", "ko", "vi", "id", "th", "ms", "tl",
                "ar", "hi", "bg", "hr", "cs", "da", "nl", "et", "fi", "el",
                "hu", "ga", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl",
                "sv",
            ],
            modality="stt",
            runtime_group="native-asr",
            capabilities=["long_audio", "multilingual"],
            description="Compact multilingual recognition across 31 languages.",
            precision_options=["Q8_0"],
            default_precision="Q8_0",
            min_vram_gb=3,
            artifact_filename="Fun-ASR-MLT-Nano-2512-Q8_0.gguf",
            base_model_id="FunAudioLLM/Fun-ASR-MLT-Nano-2512",
            source_license="other",
        ),
        _stt_model_config(
            model_name="gigaam-v3-rnnt-gguf",
            display_name="GigaAM v3 RNN-T",
            engine="transcribe_cpp",
            hf_repo_id="handy-computer/gigaam-v3-rnnt-gguf",
            model_size="gigaam-v3-rnnt-gguf",
            size_mb=261,
            languages=["ru"],
            modality="stt",
            runtime_group="native-asr",
            capabilities=["token_timestamps", "long_audio"],
            description="Compact Russian speech recognition with token timestamps.",
            precision_options=["Q8_0"],
            default_precision="Q8_0",
            min_vram_gb=1,
            artifact_filename="gigaam-v3-rnnt-Q8_0.gguf",
            base_model_id="ai-sage/GigaAM-v3",
            source_license="mit",
        ),
        _stt_model_config(
            model_name="cohere-transcribe-gguf",
            display_name="Cohere Transcribe",
            engine="transcribe_cpp",
            hf_repo_id="handy-computer/cohere-transcribe-03-2026-gguf",
            model_size="cohere-transcribe-gguf",
            size_mb=1689,
            languages=["en", "fr", "de", "es", "it", "pt", "nl", "pl", "el", "ar", "ja", "zh", "vi", "ko"],
            modality="stt",
            runtime_group="native-asr",
            capabilities=["long_audio", "multilingual"],
            description="High-accuracy local transcription across 14 languages.",
            precision_options=["Q5_K_M"],
            default_precision="Q5_K_M",
            recommended=True,
            min_vram_gb=5,
            artifact_filename="cohere-transcribe-03-2026-Q5_K_M.gguf",
            base_model_id="CohereLabs/cohere-transcribe-03-2026",
            source_license="apache-2.0",
        ),
        _stt_model_config(
            model_name="voxtral-mini-4b-realtime-gguf",
            display_name="Voxtral Mini 4B Realtime",
            engine="transcribe_cpp",
            hf_repo_id="handy-computer/Voxtral-Mini-4B-Realtime-2602-gguf",
            model_size="voxtral-mini-4b-realtime-gguf",
            size_mb=3129,
            languages=[
                "en", "fr", "es", "de", "ru", "zh", "ja",
                "it", "pt", "nl", "ar", "hi", "ko",
            ],
            modality="stt",
            runtime_group="native-asr",
            capabilities=[
                "language_detection", "streaming", "long_audio", "multilingual",
            ],
            description=(
                "Low-latency multilingual live transcription across 13 languages "
                "through the native GGUF runtime."
            ),
            precision_options=["Q5_K_M"],
            default_precision="Q5_K_M",
            min_vram_gb=6,
            artifact_filename="Voxtral-Mini-4B-Realtime-2602-Q5_K_M.gguf",
            base_model_id="mistralai/Voxtral-Mini-4B-Realtime-2602",
            source_license="apache-2.0",
        ),
        _stt_model_config(
            model_name="breeze-asr-25-gguf",
            display_name="Breeze ASR 25",
            engine="transcribe_cpp",
            hf_repo_id="handy-computer/Breeze-ASR-25-gguf",
            model_size="breeze-asr-25-gguf",
            size_mb=1107,
            languages=["zh", "en"],
            modality="stt",
            runtime_group="native-asr",
            capabilities=[
                "language_detection", "translation", "segment_timestamps",
                "long_audio", "multilingual",
            ],
            description=(
                "Taiwanese Mandarin and English transcription tuned for "
                "code-switching and subtitle timing."
            ),
            precision_options=["Q5_K_M"],
            default_precision="Q5_K_M",
            min_vram_gb=3,
            artifact_filename="Breeze-ASR-25-Q5_K_M.gguf",
            base_model_id="MediaTek-Research/Breeze-ASR-25",
            source_license="apache-2.0",
        ),
        _stt_model_config(
            model_name="medasr-gguf",
            display_name="MedASR",
            engine="transcribe_cpp",
            hf_repo_id="handy-computer/medasr-gguf",
            model_size="medasr-gguf",
            size_mb=122,
            languages=["en"],
            modality="stt",
            runtime_group="native-asr",
            capabilities=["token_timestamps", "long_audio"],
            description=(
                "Compact English medical-dictation recognition. Separate "
                "Health AI Developer Foundations terms apply; review them before use."
            ),
            precision_options=["Q8_0"],
            default_precision="Q8_0",
            min_vram_gb=1,
            artifact_filename="medasr-Q8_0.gguf",
            base_model_id="google/medasr",
            source_license="health-ai-developer-foundations",
        ),
    ]


def _get_qwen_llm_configs() -> list[ModelConfig]:
    """Return Qwen3 LLM configs with backend-aware HF repo IDs.

    MLX path uses 4-bit community quantizations for Apple Silicon; PyTorch path
    uses the upstream instruct weights.
    """
    backend_type = get_backend_type()
    if backend_type == "mlx":
        repo_0_6 = "mlx-community/Qwen3-0.6B-4bit"
        repo_1_7 = "mlx-community/Qwen3-1.7B-4bit"
        repo_4 = "mlx-community/Qwen3-4B-4bit"
    else:
        repo_0_6 = "Qwen/Qwen3-0.6B"
        repo_1_7 = "Qwen/Qwen3-1.7B"
        repo_4 = "Qwen/Qwen3-4B"

    common_languages = [
        "en", "zh", "ja", "ko", "de", "fr", "ru", "pt", "es", "it",
    ]

    return [
        ModelConfig(
            model_name="qwen3-0.6b",
            display_name="Qwen3 0.6B",
            engine="qwen_llm",
            hf_repo_id=repo_0_6,
            model_size="0.6B",
            size_mb=400 if backend_type == "mlx" else 1400,
            languages=common_languages,
            modality="llm",
        ),
        ModelConfig(
            model_name="qwen3-1.7b",
            display_name="Qwen3 1.7B",
            engine="qwen_llm",
            hf_repo_id=repo_1_7,
            model_size="1.7B",
            size_mb=1100 if backend_type == "mlx" else 3500,
            languages=common_languages,
            modality="llm",
        ),
        ModelConfig(
            model_name="qwen3-4b",
            display_name="Qwen3 4B",
            engine="qwen_llm",
            hf_repo_id=repo_4,
            model_size="4B",
            size_mb=2500 if backend_type == "mlx" else 8000,
            languages=common_languages,
            modality="llm",
        ),
    ]


def get_all_model_configs() -> list[ModelConfig]:
    """Return the full list of model configs (TTS + STT + LLM)."""
    return (
        _get_qwen_model_configs()
        + _get_qwen_custom_voice_configs()
        + _get_non_qwen_tts_configs()
        + _get_whisper_configs()
        + _get_advanced_stt_configs()
        + _get_native_gguf_stt_configs()
        + _get_qwen_llm_configs()
    )


def get_tts_model_configs() -> list[ModelConfig]:
    """Return only TTS model configs."""
    return _get_qwen_model_configs() + _get_qwen_custom_voice_configs() + _get_non_qwen_tts_configs()


def get_llm_model_configs() -> list[ModelConfig]:
    """Return only LLM model configs."""
    return _get_qwen_llm_configs()


def get_stt_model_configs() -> list[ModelConfig]:
    """Return every native and optional speech-to-text model."""
    return _get_whisper_configs() + _get_advanced_stt_configs() + _get_native_gguf_stt_configs()


# Lookup helpers — these replace the if/elif chains in main.py


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """Look up a model config by model_name."""
    for cfg in get_all_model_configs():
        if cfg.model_name == model_name:
            return cfg
    return None


def is_model_config_cached(config: ModelConfig) -> bool:
    """Check the exact cache artifact required by a catalog entry."""
    from .base import is_model_cached

    required_files = [config.artifact_filename] if config.artifact_filename else None
    weight_extensions = (".gguf",) if config.artifact_filename else (".safetensors", ".bin", ".nemo")
    try:
        return is_model_cached(
            config.hf_repo_id,
            weight_extensions=weight_extensions,
            required_files=required_files,
        )
    except TypeError:
        # Compatibility with older adapters and lightweight test doubles that
        # expose the original one-argument cache predicate.
        return is_model_cached(config.hf_repo_id)


def engine_needs_trim(engine: str) -> bool:
    """Whether this engine's output should be run through trim_tts_output."""
    for cfg in get_tts_model_configs():
        if cfg.engine == engine:
            return cfg.needs_trim
    return False


def engine_has_model_sizes(engine: str) -> bool:
    """Whether this engine supports multiple model sizes (only Qwen currently)."""
    configs = [c for c in get_tts_model_configs() if c.engine == engine]
    return len(configs) > 1


async def load_engine_model(engine: str, model_size: str = "default") -> None:
    """Load a model for the given engine, handling engines with multiple model sizes."""
    backend = get_tts_backend_for_engine(engine)
    if engine in ("qwen", "qwen_custom_voice"):
        await backend.load_model_async(model_size)
    elif engine == "tada":
        await backend.load_model(model_size)
    else:
        await backend.load_model()


async def ensure_model_cached_or_raise(engine: str, model_size: str = "default") -> None:
    """Check if a model is cached, raise HTTPException if not. Used by streaming endpoint."""
    from fastapi import HTTPException

    backend = get_tts_backend_for_engine(engine)
    cfg = None
    for c in get_tts_model_configs():
        if c.engine == engine and c.model_size == model_size:
            cfg = c
            break

    if engine in ("qwen", "qwen_custom_voice", "tada"):
        if not backend._is_model_cached(model_size):
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_size} is not downloaded yet. Use /generate to trigger a download.",
            )
    else:
        if not backend._is_model_cached():
            display = cfg.display_name if cfg else engine
            raise HTTPException(
                status_code=400,
                detail=f"{display} model is not downloaded yet. Use /generate to trigger a download.",
            )


def unload_model_by_config(config: ModelConfig) -> bool:
    """Unload a model given its config. Returns True if it was loaded, False otherwise."""
    from . import get_tts_backend_for_engine
    from ..services import tts, llm as llm_service

    if config.modality == "stt":
        stt_model = get_stt_backend_for_engine(config.engine)
        if stt_model.is_loaded() and stt_model.model_size == config.model_size:
            stt_model.unload_model()
            return True
        return False

    if config.engine == "qwen_llm":
        backend = llm_service.get_llm_model()
        loaded_size = getattr(backend, "_current_model_size", None) or getattr(backend, "model_size", None)
        if backend.is_loaded() and loaded_size == config.model_size:
            backend.unload_model()
            return True
        return False

    if config.engine == "qwen":
        tts_model = tts.get_tts_model()
        loaded_size = getattr(tts_model, "_current_model_size", None) or getattr(tts_model, "model_size", None)
        if tts_model.is_loaded() and loaded_size == config.model_size:
            tts.unload_tts_model()
            return True
        return False

    if config.engine == "qwen_custom_voice":
        backend = get_tts_backend_for_engine(config.engine)
        loaded_size = getattr(backend, "_current_model_size", None) or getattr(backend, "model_size", None)
        if backend.is_loaded() and loaded_size == config.model_size:
            backend.unload_model()
            return True
        return False

    # All other TTS engines
    backend = get_tts_backend_for_engine(config.engine)
    if backend.is_loaded():
        backend.unload_model()
        return True
    return False


def check_model_loaded(config: ModelConfig) -> bool:
    """Check if a model is currently loaded."""
    from . import get_tts_backend_for_engine
    from ..services import tts, transcribe, llm as llm_service

    try:
        if config.modality == "stt":
            backend = get_stt_backend_for_engine(config.engine)
            return backend.is_loaded() and getattr(backend, "model_size", None) == config.model_size

        if config.engine == "qwen_llm":
            backend = llm_service.get_llm_model()
            loaded_size = getattr(backend, "_current_model_size", None) or getattr(backend, "model_size", None)
            return backend.is_loaded() and loaded_size == config.model_size

        if config.engine == "qwen":
            tts_model = tts.get_tts_model()
            loaded_size = getattr(tts_model, "_current_model_size", None) or getattr(tts_model, "model_size", None)
            return tts_model.is_loaded() and loaded_size == config.model_size

        if config.engine == "qwen_custom_voice":
            backend = get_tts_backend_for_engine(config.engine)
            loaded_size = getattr(backend, "_current_model_size", None) or getattr(backend, "model_size", None)
            return backend.is_loaded() and loaded_size == config.model_size

        backend = get_tts_backend_for_engine(config.engine)
        return backend.is_loaded()
    except Exception:
        return False


def get_model_load_func(config: ModelConfig):
    """Return a weight-only download operation for the model manager.

    Downloading and loading are deliberately separate operations.  A model's
    optional inference package (for example ``qwen_tts`` or NeMo) must not be
    imported just to place its HuggingFace files in the shared cache.  Loading
    still happens through the normal backend path when the model is first used.
    """

    async def download_model_weights() -> None:
        import asyncio

        import huggingface_hub

        from .base import (
            materialize_windows_snapshot_links,
            model_load_progress,
        )

        def download() -> None:
            cached = is_model_config_cached(config)
            if config.artifact_filename:
                with model_load_progress(config.model_name, cached):
                    from pathlib import Path

                    artifact_path = Path(
                        huggingface_hub.hf_hub_download(
                            repo_id=config.hf_repo_id,
                            filename=config.artifact_filename,
                            local_files_only=False,
                        )
                    )
                    materialize_windows_snapshot_links(artifact_path.parent)
                return

            download_options = {"repo_id": config.hf_repo_id}
            if config.model_name == "nvidia-parakeet-tdt-0.6b-v3":
                # Diarix runs Parakeet through NVIDIA's supported NeMo path.
                # Avoid the duplicate Transformers checkpoint stored beside
                # the .nemo archive.
                download_options["allow_patterns"] = [
                    "*.nemo",
                    "*.json",
                    "*.model",
                    "*.txt",
                ]

            with model_load_progress(
                config.model_name,
                cached,
            ):
                snapshot_path = huggingface_hub.snapshot_download(**download_options)
                if config.engine == "faster_whisper":
                    materialize_windows_snapshot_links(snapshot_path)

        await asyncio.to_thread(download)

    return download_model_weights


def get_tts_backend() -> TTSBackend:
    """
    Get or create the default (Qwen) TTS backend instance based on platform.

    Returns:
        TTS backend instance (MLX or PyTorch)
    """
    return get_tts_backend_for_engine("qwen")


def get_tts_backend_for_engine(engine: str) -> TTSBackend:
    """
    Get or create a TTS backend for the given engine.

    Args:
        engine: Engine name (e.g. "qwen", "luxtts", "chatterbox", "chatterbox_turbo")

    Returns:
        TTS backend instance
    """
    global _tts_backends

    # Fast path: check without lock
    if engine in _tts_backends:
        return _tts_backends[engine]

    # Slow path: create with lock to avoid duplicate instantiation
    with _tts_backends_lock:
        # Double-check after acquiring lock
        if engine in _tts_backends:
            return _tts_backends[engine]

        if engine == "qwen":
            backend_type = get_backend_type()
            if backend_type == "mlx":
                from .mlx_backend import MLXTTSBackend

                backend = MLXTTSBackend()
            else:
                from .pytorch_backend import PyTorchTTSBackend

                backend = PyTorchTTSBackend()
        elif engine == "luxtts":
            from .luxtts_backend import LuxTTSBackend

            backend = LuxTTSBackend()
        elif engine == "chatterbox":
            from .chatterbox_backend import ChatterboxTTSBackend

            backend = ChatterboxTTSBackend()
        elif engine == "chatterbox_turbo":
            from .chatterbox_turbo_backend import ChatterboxTurboTTSBackend

            backend = ChatterboxTurboTTSBackend()
        elif engine == "tada":
            from .hume_backend import HumeTadaBackend

            backend = HumeTadaBackend()
        elif engine == "kokoro":
            from .kokoro_backend import KokoroTTSBackend

            backend = KokoroTTSBackend()
        elif engine == "qwen_custom_voice":
            from .qwen_custom_voice_backend import QwenCustomVoiceBackend

            backend = QwenCustomVoiceBackend()
        else:
            raise ValueError(f"Unknown TTS engine: {engine}. Supported: {list(TTS_ENGINES.keys())}")

        _tts_backends[engine] = backend
        return backend


def get_stt_backend() -> STTBackend:
    """
    Get or create STT backend instance based on platform.

    Returns:
        STT backend instance (MLX or PyTorch)
    """
    return get_stt_backend_for_engine("whisper")


def get_stt_backend_for_engine(engine: str) -> STTBackend:
    """Return a lazily-created STT adapter for an engine family."""
    global _stt_backend

    if engine in _stt_backends:
        return _stt_backends[engine]

    with _stt_backends_lock:
        if engine in _stt_backends:
            return _stt_backends[engine]

        if engine == "whisper":
            if get_backend_type() == "mlx":
                from .mlx_backend import MLXSTTBackend
                backend = MLXSTTBackend()
            else:
                from .pytorch_backend import PyTorchSTTBackend
                backend = PyTorchSTTBackend()
            _stt_backend = backend
        elif engine == "whisperx":
            from .stt.whisperx_backend import WhisperXSTTBackend
            backend = WhisperXSTTBackend()
        elif engine == "faster_whisper":
            from .stt.faster_whisper_backend import FasterWhisperSTTBackend
            backend = FasterWhisperSTTBackend()
        elif engine == "transformers_asr":
            from .stt.transformers_backend import TransformersASRBackend
            backend = TransformersASRBackend()
        elif engine == "nemo_asr":
            from .stt.nemo_backend import NeMoASRBackend
            backend = NeMoASRBackend()
        elif engine == "qwen_asr":
            from .stt.qwen_backend import QwenASRBackend
            backend = QwenASRBackend()
        elif engine == "transcribe_cpp":
            from .stt.transcribe_cpp_backend import TranscribeCppSTTBackend
            backend = TranscribeCppSTTBackend()
        else:
            raise ValueError(f"Unknown STT engine: {engine}")

        _stt_backends[engine] = backend
        return backend


def unload_all_stt_backends() -> None:
    """Unload every instantiated STT adapter, including optional runtimes."""
    for backend in list(_stt_backends.values()):
        backend.unload_model()


def unload_other_stt_backends(selected_engine: str) -> None:
    """Keep at most one STT engine family resident in memory."""
    for engine, backend in list(_stt_backends.items()):
        if engine != selected_engine and backend.is_loaded():
            backend.unload_model()


def resolve_stt_config(model: Optional[str]) -> ModelConfig:
    """Resolve a global model id or a legacy Whisper size to an STT config."""
    requested = model or "whisper-turbo"
    for config in get_stt_model_configs():
        if requested in (config.model_name, config.model_size):
            return config
    raise ValueError(f"Unknown transcription model: {requested}")


def get_llm_backend() -> LLMBackend:
    """Get or create the default Qwen3 LLM backend based on platform."""
    return get_llm_backend_for_engine("qwen_llm")


def get_llm_backend_for_engine(engine: str) -> LLMBackend:
    """Get or create an LLM backend for the given engine."""
    global _llm_backends

    if engine in _llm_backends:
        return _llm_backends[engine]

    with _llm_backends_lock:
        if engine in _llm_backends:
            return _llm_backends[engine]

        if engine == "qwen_llm":
            backend_type = get_backend_type()
            if backend_type == "mlx":
                from .qwen_llm_backend import MLXQwenLLMBackend

                backend = MLXQwenLLMBackend()
            else:
                from .qwen_llm_backend import PyTorchQwenLLMBackend

                backend = PyTorchQwenLLMBackend()
        else:
            raise ValueError(f"Unknown LLM engine: {engine}. Supported: {list(LLM_ENGINES.keys())}")

        _llm_backends[engine] = backend
        return backend


def reset_backends():
    """Reset backend instances (useful for testing)."""
    global _tts_backend, _tts_backends, _stt_backend, _stt_backends, _llm_backends
    _tts_backend = None
    _tts_backends.clear()
    _stt_backend = None
    _stt_backends.clear()
    _llm_backends.clear()
