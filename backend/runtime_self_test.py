"""Import-only release checks for every model engine in the Diarix catalog."""

from __future__ import annotations

import importlib
import traceback
from typing import Any

RUNTIME_IMPORT_CHECKS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "whisper": (("transformers", ("WhisperForConditionalGeneration", "WhisperProcessor")),),
    "whisperx": (("whisperx", ("load_model", "load_audio")),),
    "faster_whisper": (("faster_whisper", ("WhisperModel",)),),
    "nemo_asr": (
        ("nemo.collections.asr", ("models",)),
        ("nemo.collections.speechlm2.models", ("SALM",)),
    ),
    "qwen_asr": (("qwen_asr", ("Qwen3ASRModel",)),),
    "transcribe_cpp": (("transcribe_cpp", ("Model", "Session", "Stream")),),
    "transformers_asr": (("transformers", ("pipeline",)),),
    "qwen": (("qwen_tts", ("Qwen3TTSModel",)),),
    "qwen_custom_voice": (("qwen_tts", ("Qwen3TTSModel",)),),
    "luxtts": (("zipvoice.luxvoice", ("LuxTTS",)),),
    "chatterbox": (("chatterbox.mtl_tts", ("ChatterboxMultilingualTTS",)),),
    "chatterbox_turbo": (("chatterbox.tts_turbo", ("ChatterboxTurboTTS",)),),
    "tada": (
        ("tada.modules.aligner", ("AlignerConfig",)),
        ("tada.modules.encoder", ("Encoder",)),
        ("tada.modules.tada", ("TadaConfig", "TadaForCausalLM")),
    ),
    "kokoro": (("kokoro", ("KModel", "KPipeline")),),
    "qwen_llm": (("transformers", ("AutoModelForCausalLM", "AutoTokenizer")),),
}


def _prepare_engine(engine: str) -> None:
    if engine == "tada":
        from backend.utils.dac_shim import install_dac_shim

        install_dac_shim()


def run_runtime_self_test() -> dict[str, Any]:
    """Import each catalog engine and report missing modules or attributes."""
    results: dict[str, dict[str, Any]] = {}
    for engine, checks in RUNTIME_IMPORT_CHECKS.items():
        try:
            _prepare_engine(engine)
            imported: list[str] = []
            for module_name, attributes in checks:
                module = importlib.import_module(module_name)
                missing = [name for name in attributes if not hasattr(module, name)]
                if missing:
                    raise AttributeError(f"{module_name} is missing required attributes: {', '.join(missing)}")
                imported.append(module_name)
            results[engine] = {"ok": True, "modules": imported}
        except Exception as exc:
            results[engine] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }

    failed = [engine for engine, result in results.items() if not result["ok"]]
    return {"ok": not failed, "failed": failed, "engines": results}
