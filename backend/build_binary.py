"""
PyInstaller build script for creating standalone Python server binary.

Usage:
    python build_binary.py           # Build default (CPU) server binary
    python build_binary.py --cuda    # Build CUDA-enabled server binary
    python build_binary.py --require-media-tools  # Release build with FFmpeg/FFprobe
"""

import argparse
import logging
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

logger = logging.getLogger(__name__)

NAGISA_PYINSTALLER_ARGS = [
    "--collect-all",
    "nagisa",
    "--copy-metadata",
    "nagisa",
]


class MediaToolPackagingError(RuntimeError):
    """Raised when FFmpeg/FFprobe cannot be packaged safely."""


@dataclass(frozen=True)
class MediaToolPaths:
    """Resolved paths for the two media executables shipped with the server."""

    ffmpeg: Path
    ffprobe: Path


def _canonical_media_tool_name(tool: str, system_name: str | None = None) -> str:
    """Return the runtime filename expected under ``sys._MEIPASS/tools``."""
    system_name = system_name or platform.system()
    suffix = ".exe" if system_name == "Windows" else ""
    return f"{tool}{suffix}"


def _tool_names_to_try(tool: str, system_name: str) -> tuple[str, ...]:
    canonical = _canonical_media_tool_name(tool, system_name)
    if canonical == tool:
        return (tool,)
    return (canonical, tool)


def _resolve_explicit_media_binary(
    value: str,
    *,
    env_var: str,
    search_path: str,
    which: Callable[..., str | None],
) -> Path:
    """Resolve an explicit binary path or command without silently falling back."""
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if expanded.is_file():
        return expanded.resolve()

    found = which(value, path=search_path)
    if found:
        return Path(found).resolve()

    raise MediaToolPackagingError(
        f"{env_var} is set to {value!r}, but it is neither an existing file "
        "nor a command available on PATH."
    )


def resolve_media_tools(
    *,
    env: Mapping[str, str] | None = None,
    require: bool = False,
    system_name: str | None = None,
    which: Callable[..., str | None] = shutil.which,
) -> MediaToolPaths | None:
    """Resolve FFmpeg and FFprobe for a server build.

    Resolution precedence for each tool is its explicit binary environment
    variable, ``DIARIX_FFMPEG_DIR``, then ``PATH``. Explicit settings are
    authoritative: a bad path is a packaging error instead of silently using
    a different installation.
    """
    env = os.environ if env is None else env
    system_name = system_name or platform.system()
    search_path = env.get("PATH", "")

    explicit_vars = {
        "ffmpeg": "FFMPEG_BINARY",
        "ffprobe": "FFPROBE_BINARY",
    }
    tool_dir = None
    tool_dir_value = env.get("DIARIX_FFMPEG_DIR")
    needs_tool_dir = tool_dir_value and any(not env.get(env_var) for env_var in explicit_vars.values())
    if needs_tool_dir:
        tool_dir = Path(os.path.expandvars(os.path.expanduser(tool_dir_value))).resolve()
        if not tool_dir.is_dir():
            raise MediaToolPackagingError(
                f"DIARIX_FFMPEG_DIR is set to {tool_dir_value!r}, but that directory does not exist."
            )

    resolved: dict[str, Path] = {}
    missing: list[str] = []

    for tool, env_var in explicit_vars.items():
        explicit_value = env.get(env_var)
        if explicit_value:
            resolved[tool] = _resolve_explicit_media_binary(
                explicit_value,
                env_var=env_var,
                search_path=search_path,
                which=which,
            )
            continue

        if tool_dir is not None:
            candidate = next(
                (tool_dir / name for name in _tool_names_to_try(tool, system_name) if (tool_dir / name).is_file()),
                None,
            )
            if candidate is None:
                expected = ", ".join(_tool_names_to_try(tool, system_name))
                raise MediaToolPackagingError(
                    f"DIARIX_FFMPEG_DIR={str(tool_dir)!r} does not contain {expected}."
                )
            resolved[tool] = candidate.resolve()
            continue

        found = None
        for command in _tool_names_to_try(tool, system_name):
            found = which(command, path=search_path)
            if found:
                break
        if found:
            resolved[tool] = Path(found).resolve()
        else:
            missing.append(tool)

    if missing:
        message = (
            "FFmpeg media tools are unavailable for packaging: missing "
            f"{', '.join(missing)}. Set DIARIX_FFMPEG_DIR to a directory containing both tools, "
            "set FFMPEG_BINARY and FFPROBE_BINARY explicitly, or make both commands available on PATH."
        )
        if require:
            raise MediaToolPackagingError(f"{message} Release builds require both tools.")
        logger.warning("%s The server will be built without bundled media tools.", message)
        return None

    return MediaToolPaths(ffmpeg=resolved["ffmpeg"], ffprobe=resolved["ffprobe"])


def stage_media_tools(
    media_tools: MediaToolPaths,
    staging_dir: Path,
    *,
    system_name: str | None = None,
) -> MediaToolPaths:
    """Copy tools to canonical filenames before handing them to PyInstaller."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged = {}
    for tool, source in (("ffmpeg", media_tools.ffmpeg), ("ffprobe", media_tools.ffprobe)):
        destination = staging_dir / _canonical_media_tool_name(tool, system_name)
        shutil.copy2(source, destination)
        staged[tool] = destination
    return MediaToolPaths(ffmpeg=staged["ffmpeg"], ffprobe=staged["ffprobe"])


def media_tool_pyinstaller_args(media_tools: MediaToolPaths) -> list[str]:
    """Build deterministic PyInstaller args for ``sys._MEIPASS/tools``."""
    return [
        "--add-binary",
        f"{media_tools.ffmpeg}{os.pathsep}tools",
        "--add-binary",
        f"{media_tools.ffprobe}{os.pathsep}tools",
    ]


def _run_pyinstaller(args: list[str]) -> None:
    """Import PyInstaller only for an actual build so resolver tests stay pure."""
    import PyInstaller.__main__

    PyInstaller.__main__.run(args)


def is_apple_silicon():
    """Check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def build_server(cuda=False, rocm=False, require_media_tools=False):
    """Build Python server as standalone binary.

    Args:
        cuda: If True, build with CUDA support and name the binary
              diarix-server-cuda instead of diarix-server.
        rocm: If True, build with ROCm support and name the binary
              diarix-server-rocm instead of diarix-server.
        require_media_tools: If True, fail unless both FFmpeg and FFprobe
              can be bundled into the server.
    """
    if cuda and rocm:
        raise ValueError("Cannot build with both CUDA and ROCm support")

    backend_dir = Path(__file__).parent
    media_tools = resolve_media_tools(require=require_media_tools)

    if rocm:
        binary_name = "diarix-server-rocm"
    elif cuda:
        binary_name = "diarix-server-cuda"
    else:
        binary_name = "diarix-server"

    # PyInstaller arguments
    # CUDA and ROCm builds use --onedir so we can split the output into two archives:
    #   1. Server core (~200-400MB) — versioned with the app
    #   2. GPU libs (~2GB) — versioned independently (only redownloaded on
    #      GPU toolkit / torch major version changes)
    # CPU builds remain --onefile for simplicity.
    pack_mode = "--onedir" if (cuda or rocm) else "--onefile"
    args = [
        "server.py",  # Use server.py as entry point instead of main.py
        pack_mode,
        "--name",
        binary_name,
    ]

    # Hide console window on Windows only. On macOS/Linux the sidecar needs
    # stdout/stderr for Tauri to capture logs.
    if platform.system() == "Windows":
        args.append("--noconsole")

    # numpy 2.x / torch ABI mismatch fix: install memmove fallback for
    # torch.from_numpy() before the app starts. Runtime hooks run after
    # FrozenImporter is registered so frozen torch/numpy are importable.
    # Paths are passed relative to backend_dir because os.chdir(backend_dir)
    # runs before PyInstaller. Absolute paths would get baked into the
    # generated .spec, breaking reproducible builds on other machines / CI.
    args.extend(
        [
            "--runtime-hook",
            "pyi_rth_numpy_compat.py",
            # Stub torch.compiler.disable before transformers imports
            # flex_attention, which otherwise triggers torch._dynamo →
            # torch._numpy._ufuncs and crashes at module load under
            # PyInstaller. See pyi_rth_torch_compiler_disable.py.
            "--runtime-hook",
            "pyi_rth_torch_compiler_disable.py",
            # Per-module collection overrides (e.g. forcing scipy.stats._distn_infrastructure
            # to bundle .py source alongside .pyc so the runtime hook can source-patch it).
            "--additional-hooks-dir",
            "pyi_hooks",
        ]
    )

    # Add local qwen_tts path if specified (for editable installs)
    qwen_tts_path = os.getenv("QWEN_TTS_PATH")
    if qwen_tts_path and Path(qwen_tts_path).exists():
        args.extend(["--paths", str(qwen_tts_path)])
        logger.info("Using local qwen_tts source from: %s", qwen_tts_path)

    # Add common hidden imports
    args.extend(
        [
            "--hidden-import",
            "backend",
            "--hidden-import",
            "backend.main",
            "--hidden-import",
            "backend.config",
            "--hidden-import",
            "backend.database",
            "--hidden-import",
            "backend.models",
            "--hidden-import",
            "backend.services.profiles",
            "--hidden-import",
            "backend.services.history",
            "--hidden-import",
            "backend.services.tts",
            "--hidden-import",
            "backend.services.transcribe",
            "--hidden-import",
            "backend.utils.platform_detect",
            "--hidden-import",
            "backend.backends",
            "--hidden-import",
            "backend.backends.pytorch_backend",
            "--hidden-import",
            "backend.backends.stt",
            "--hidden-import",
            "backend.backends.stt.common",
            "--hidden-import",
            "backend.backends.stt.whisperx_backend",
            "--hidden-import",
            "backend.backends.stt.transformers_backend",
            "--hidden-import",
            "backend.backends.stt.nemo_backend",
            "--hidden-import",
            "backend.backends.stt.qwen_backend",
            "--hidden-import",
            "backend.backends.qwen_custom_voice_backend",
            "--hidden-import",
            "backend.utils.audio",
            "--hidden-import",
            "backend.utils.cache",
            "--hidden-import",
            "backend.utils.progress",
            "--hidden-import",
            "backend.utils.hf_progress",
            "--hidden-import",
            "backend.services.cuda",
            "--hidden-import",
            "backend.services.effects",
            "--hidden-import",
            "backend.utils.effects",
            "--hidden-import",
            "backend.services.versions",
            "--hidden-import",
            "backend.services.resource_limits",
            "--hidden-import",
            "backend.runtime_self_test",
            "--hidden-import",
            "pedalboard",
            "--hidden-import",
            "chatterbox",
            "--hidden-import",
            "chatterbox.tts_turbo",
            "--hidden-import",
            "chatterbox.mtl_tts",
            "--hidden-import",
            "backend.backends.chatterbox_backend",
            "--hidden-import",
            "backend.backends.chatterbox_turbo_backend",
            # chatterbox multilingual uses spacy_pkuseg for Chinese word
            # segmentation, which ships pickled dict files (dicts/default.pkl)
            # and native .so extensions that --hidden-import alone won't bundle.
            "--collect-all",
            "spacy_pkuseg",
            "--hidden-import",
            "backend.backends.luxtts_backend",
            "--hidden-import",
            "zipvoice",
            "--hidden-import",
            "zipvoice.luxvoice",
            "--collect-all",
            "zipvoice",
            "--collect-all",
            "linacodec",
            "--hidden-import",
            "torch",
            "--hidden-import",
            "transformers",
            "--hidden-import",
            "fastapi",
            "--hidden-import",
            "uvicorn",
            "--hidden-import",
            "sqlalchemy",
            # librosa uses lazy_loader which generates .pyi stub files at
            # install time and reads them at runtime to discover submodules.
            # --hidden-import alone doesn't bundle the stubs, causing
            # "Cannot load imports from non-existent stub" at runtime.
            "--collect-all",
            "lazy_loader",
            "--collect-all",
            "librosa",
            "--hidden-import",
            "soundfile",
            "--hidden-import",
            "qwen_tts",
            "--hidden-import",
            "qwen_tts.inference",
            "--hidden-import",
            "qwen_tts.inference.qwen3_tts_model",
            "--hidden-import",
            "qwen_tts.inference.qwen3_tts_tokenizer",
            "--hidden-import",
            "qwen_tts.core",
            "--hidden-import",
            "qwen_tts.cli",
            "--copy-metadata",
            "qwen-tts",
            "--copy-metadata",
            "requests",
            "--copy-metadata",
            "transformers",
            # Transformers 4.57 checks torchcodec's installed version while
            # importing Whisper, even though Diarix loads normalized PCM
            # directly and never invokes TorchCodec for transcription.
            "--copy-metadata",
            "torchcodec",
            "--copy-metadata",
            "huggingface-hub",
            "--copy-metadata",
            "tokenizers",
            "--copy-metadata",
            "safetensors",
            "--copy-metadata",
            "tqdm",
            "--hidden-import",
            "requests",
            # qwen_tts uses inspect.getsource() at runtime to locate
            # modeling_qwen3_tts.py — needs physical .py source files bundled
            "--collect-all",
            "qwen_tts",
            # Fix for pkg_resources and jaraco namespace packages
            "--hidden-import",
            "pkg_resources.extern",
            "--collect-submodules",
            "jaraco",
            # inflect uses typeguard @typechecked which calls inspect.getsource()
            # at import time — needs .py source files, not just .pyc bytecode
            "--collect-all",
            "inflect",
            # perth ships pretrained watermark model files (hparams.yaml, .pth.tar)
            # in perth/perth_net/pretrained/ — needed by chatterbox at runtime
            "--collect-all",
            "perth",
            # piper_phonemize ships espeak-ng-data/ (phoneme tables, language dicts)
            # needed by LuxTTS for text-to-phoneme conversion
            "--collect-all",
            "piper_phonemize",
            # HumeAI TADA — speech-language model using Llama + flow matching
            "--hidden-import",
            "backend.backends.hume_backend",
            "--hidden-import",
            "tada",
            "--hidden-import",
            "tada.modules",
            "--hidden-import",
            "tada.modules.tada",
            "--hidden-import",
            "tada.modules.encoder",
            "--hidden-import",
            "tada.modules.decoder",
            "--hidden-import",
            "tada.modules.aligner",
            "--hidden-import",
            "tada.modules.acoustic_spkr_verf",
            "--hidden-import",
            "tada.nn",
            "--hidden-import",
            "tada.nn.vibevoice",
            "--hidden-import",
            "tada.utils",
            "--hidden-import",
            "tada.utils.gray_code",
            "--hidden-import",
            "tada.utils.text",
            # DAC shim — provides dac.nn.layers.Snake1d without the real
            # descript-audio-codec package (which pulls onnx/tensorboard via
            # descript-audiotools). The shim is in backend/utils/dac_shim.py.
            "--hidden-import",
            "backend.utils.dac_shim",
            "--hidden-import",
            "torchaudio",
            "--collect-submodules",
            "tada",
            # Kokoro 82M — lightweight TTS engine using misaki G2P
            # collect-all is required because transformers introspects .py source
            # files at runtime (e.g. _can_set_attn_implementation opens the class
            # file); hidden-import alone only bundles bytecode.
            "--hidden-import",
            "backend.backends.kokoro_backend",
            "--collect-all",
            "kokoro",
            # misaki ships G2P data files (dictionaries, phoneme tables)
            # that must be bundled for espeak/en/ja/zh G2P to work
            "--collect-all",
            "misaki",
            # language_tags ships JSON data files (index.json etc.) loaded at
            # runtime via: misaki → phonemizer → segments → csvw → language_tags
            "--collect-all",
            "language_tags",
            # espeakng_loader ships the entire espeak-ng-data directory (369 files)
            # loaded at import time by misaki.espeak via get_data_path()
            "--collect-all",
            "espeakng_loader",
            # spacy en_core_web_sm model — misaki.en tries to spacy.cli.download()
            # at runtime if not found, which calls pip as a subprocess and crashes
            # the frozen binary. Bundle the model so spacy.util.is_package() passes.
            "--collect-all",
            "en_core_web_sm",
            "--copy-metadata",
            "en_core_web_sm",
            "--hidden-import",
            "en_core_web_sm",
            # unidic-lite ships the MeCab dictionary used by fugashi (pulled in
            # by misaki[ja]). The dict lives in unidic_lite/dicdir/ and is
            # discovered via the package's DICDIR constant, so the data files
            # must be collected or Japanese Kokoro voices crash at runtime.
            "--collect-all",
            "unidic_lite",
            "--hidden-import",
            "loguru",
            # MCP server — Streamable-HTTP endpoint and the 4 voicebox.* tools.
            # FastMCP pulls in a chain of deps (mcp, cyclopts, openapi-pydantic,
            # etc.) that don't auto-discover cleanly under PyInstaller, so we
            # collect them whole. Small compared to torch.
            "--hidden-import",
            "backend.mcp_server",
            "--hidden-import",
            "backend.mcp_server.server",
            "--hidden-import",
            "backend.mcp_server.tools",
            "--hidden-import",
            "backend.mcp_server.context",
            "--hidden-import",
            "backend.mcp_server.resolve",
            "--hidden-import",
            "backend.mcp_server.events",
            "--collect-all",
            "fastmcp",
            "--collect-all",
            "mcp",
            "--hidden-import",
            "sse_starlette",
        ]
    )

    # Add CUDA/ROCm-specific hidden imports
    if cuda or rocm:
        variant = "ROCm" if rocm else "CUDA"
        logger.info("Building with %s support", variant)
        gpu_hidden = [
            "--hidden-import",
            "torch.cuda",
        ]
        # cudnn is NVIDIA-specific; ROCm uses MIOpen under the abstraction layer
        if cuda:
            gpu_hidden.extend(
                [
                    "--hidden-import",
                    "torch.backends.cudnn",
                    "--collect-all",
                    "whisperx",
                    "--collect-all",
                    "faster_whisper",
                    "--collect-all",
                    "nemo",
                    # pytorch-lightning imports lightning_fabric through NeMo;
                    # its __version__ module reads version.info from disk.
                    "--collect-all",
                    "lightning_fabric",
                    # NeMo imports CUDA Python's compiled helpers dynamically.
                    # Collect the package so cydriver/cyruntime survive freezing.
                    "--collect-all",
                    "cuda.bindings",
                    "--hidden-import",
                    "cuda.bindings.cydriver",
                    "--hidden-import",
                    "cuda.bindings.cyruntime",
                    "--collect-all",
                    "qwen_asr",
                    # qwen-asr imports Nagisa's forced aligner at module load.
                    # Nagisa 0.2.x relies on sibling modules being available as
                    # physical source files; hook-nagisa.py selects that mode.
                    *NAGISA_PYINSTALLER_ARGS,
                    "--copy-metadata",
                    "whisperx",
                    "--copy-metadata",
                    "faster-whisper",
                    "--copy-metadata",
                    "nemo_toolkit",
                    "--copy-metadata",
                    "qwen-asr",
                ]
            )
        args.extend(gpu_hidden)

    if rocm:
        # rocm_sdk imports its backend packages dynamically via
        # importlib.import_module(py_package_name), which PyInstaller's
        # static analyzer cannot see. We must collect them explicitly —
        # otherwise only the pure-python rocm_sdk wrapper ships and
        # rocm_sdk.find_libraries crashes with UnboundLocalError at boot.
        #
        # The backend packages also contain the HIP/MIOpen/hipBLAS DLLs
        # under bin/ (plus ~750 MB of tensile kernel files under
        # bin/rocblas/library and bin/hipblaslt/library) — collect-all
        # walks the tree recursively so both DLLs and kernel data are
        # bundled. See rocm_sdk/_dist_info.py for the package mapping.
        args.extend(
            [
                "--collect-all",
                "rocm_sdk",
                "--collect-all",
                "_rocm_sdk_core",
                "--collect-all",
                "_rocm_sdk_libraries_custom",
                "--collect-all",
                "rocm_sdk_core",
                "--collect-all",
                "rocm_sdk_libraries_custom",
                "--hidden-import",
                "_rocm_sdk_core",
                "--hidden-import",
                "_rocm_sdk_libraries_custom",
                "--hidden-import",
                "rocm_sdk_core",
                "--hidden-import",
                "rocm_sdk_libraries_custom",
                "--copy-metadata",
                "rocm",
                "--copy-metadata",
                "rocm-sdk-core",
                "--copy-metadata",
                "rocm-sdk-libraries-custom",
                # Repair rocm_sdk.find_libraries (masks UnboundLocalError
                # with a readable ModuleNotFoundError on missing backends).
                "--runtime-hook",
                "pyi_rth_rocm_sdk.py",
            ]
        )

    # Exclude NVIDIA CUDA packages from non-CUDA builds to keep binary small.
    # When building from a venv with CUDA torch installed, PyInstaller would
    # bundle ~3GB of NVIDIA shared libraries. We exclude both the Python
    # modules and the binary DLLs. This applies to CPU and ROCm builds.
    if not cuda:
        nvidia_packages = [
            "nvidia",
            "nvidia.cublas",
            "nvidia.cuda_cupti",
            "nvidia.cuda_nvrtc",
            "nvidia.cuda_runtime",
            "nvidia.cudnn",
            "nvidia.cufft",
            "nvidia.curand",
            "nvidia.cusolver",
            "nvidia.cusparse",
            "nvidia.nccl",
            "nvidia.nvjitlink",
            "nvidia.nvtx",
        ]
        for pkg in nvidia_packages:
            args.extend(["--exclude-module", pkg])

    # Add MLX-specific imports if building on Apple Silicon (never for GPU builds)
    if is_apple_silicon() and not cuda and not rocm:
        logger.info("Building for Apple Silicon - including MLX dependencies")
        args.extend(
            [
                "--hidden-import",
                "backend.backends.mlx_backend",
                "--hidden-import",
                "mlx",
                "--hidden-import",
                "mlx.core",
                "--hidden-import",
                "mlx.nn",
                "--hidden-import",
                "mlx_audio",
                "--hidden-import",
                "mlx_audio.tts",
                "--hidden-import",
                "mlx_audio.stt",
                "--hidden-import",
                "mlx_lm",
                "--hidden-import",
                "backend.backends.qwen_llm_backend",
                "--collect-submodules",
                "mlx",
                "--collect-submodules",
                "mlx_audio",
                "--collect-submodules",
                "mlx_lm",
                # Use --collect-all so PyInstaller bundles both data files AND
                # native shared libraries (.dylib, .metallib) for MLX.
                # Previously only --collect-data was used, which caused MLX to
                # raise OSError at runtime inside the bundled binary because
                # the Metal shader libraries were missing.
                "--collect-all",
                "mlx",
                "--collect-all",
                "mlx_audio",
                # mlx_lm ships chat_templates/ JSON files and loads tool_parsers
                # submodules dynamically via importlib at tokenizer load time,
                # which --hidden-import alone can't resolve.
                "--collect-all",
                "mlx_lm",
            ]
        )
    elif not cuda and not rocm:
        logger.info("Building for non-Apple Silicon platform - PyTorch only")

    dist_dir = str(backend_dir / "dist")
    build_dir = str(backend_dir / "build")

    args.extend(
        [
            "--distpath",
            dist_dir,
            "--workpath",
            build_dir,
            "--noconfirm",
            "--clean",
        ]
    )

    media_tools_staging = None
    if media_tools is not None:
        media_tools_staging = tempfile.TemporaryDirectory(prefix="diarix-media-tools-")
        staged_media_tools = stage_media_tools(media_tools, Path(media_tools_staging.name))
        args.extend(media_tool_pyinstaller_args(staged_media_tools))
        logger.info(
            "Bundling FFmpeg media tools under sys._MEIPASS/tools: ffmpeg=%s, ffprobe=%s",
            media_tools.ffmpeg,
            media_tools.ffprobe,
        )

    # Change to backend directory
    os.chdir(backend_dir)

    # For CPU builds on Windows, ensure we're using CPU-only torch.
    # If CUDA or ROCm torch is installed (local dev), swap to CPU torch before
    # building, then restore afterwards. This prevents PyInstaller from bundling
    # GPU libraries into the CPU binary.
    restore_torch = None
    try:
        if not cuda and not rocm and platform.system() == "Windows":
            import subprocess

            cuda_result = subprocess.run(
                [sys.executable, "-c", "import torch; print(torch.version.cuda or '')"], capture_output=True, text=True
            )
            rocm_result = subprocess.run(
                [sys.executable, "-c", "import torch; print(torch.version.hip or '')"], capture_output=True, text=True
            )

            if cuda_result.stdout.strip():
                restore_torch = "cuda"
                logger.info("CUDA torch detected — installing CPU torch for CPU build...")
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "torch",
                        "torchvision",
                        "torchaudio",
                        "--index-url",
                        "https://download.pytorch.org/whl/cpu",
                        "--force-reinstall",
                        "--no-deps",
                        "-q",
                    ],
                    check=True,
                )
            elif rocm_result.stdout.strip():
                restore_torch = "rocm"
                logger.info("ROCm torch detected — installing CPU torch for CPU build...")
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "torch",
                        "torchvision",
                        "torchaudio",
                        "--index-url",
                        "https://download.pytorch.org/whl/cpu",
                        "--force-reinstall",
                        "--no-deps",
                        "-q",
                    ],
                    check=True,
                )

        # For ROCm builds on Windows, ensure ROCm torch is installed.
        if rocm and platform.system() == "Windows":
            import subprocess

            if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12):
                raise RuntimeError(
                    "ROCm wheels are cp312-cp312-specific; "
                    f"got {sys.implementation.name} {sys.version.split()[0]}. "
                    "Use CPython 3.12 to build the ROCm binary."
                )

            result = subprocess.run(
                [sys.executable, "-c", "import torch; print(torch.version.hip or '')"], capture_output=True, text=True
            )
            has_rocm_torch = bool(result.stdout.strip())
            if not has_rocm_torch:
                logger.info("ROCm torch not detected — installing ROCm torch for ROCm build...")

                # Determine what to restore BEFORE overwriting the environment
                cuda_result = subprocess.run(
                    [sys.executable, "-c", "import torch; print(torch.version.cuda or '')"],
                    capture_output=True,
                    text=True,
                )
                if cuda_result.stdout.strip():
                    restore_torch = "cuda"
                else:
                    restore_torch = "cpu"

                # Now overwrite the environment safely
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
                        "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
                        "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
                        "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz",
                        "--no-deps",
                        "-q",
                    ],
                    check=True,
                )
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
                        "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
                        "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
                        "--force-reinstall",
                        "--no-deps",
                        "-q",
                    ],
                    check=True,
                )

        # Run PyInstaller
        _run_pyinstaller(args)
    finally:
        if media_tools_staging is not None:
            try:
                media_tools_staging.cleanup()
            except OSError as exc:
                logger.warning("Failed to clean temporary media-tool staging directory: %s", exc)

        # Restore torch if we swapped it out (even on build failure)
        if restore_torch == "cuda":
            logger.info("Restoring CUDA torch...")
            import subprocess

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "torch",
                    "torchvision",
                    "torchaudio",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu128",
                    "--force-reinstall",
                    "--no-deps",
                    "-q",
                ],
                check=True,
            )
        elif restore_torch == "rocm":
            logger.info("Restoring ROCm torch...")
            import subprocess

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
                    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
                    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
                    "--force-reinstall",
                    "--no-deps",
                    "-q",
                ],
                check=True,
            )
        elif restore_torch == "cpu":
            logger.info("Restoring CPU torch...")
            import subprocess

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "torch",
                    "torchvision",
                    "torchaudio",
                    "--index-url",
                    "https://download.pytorch.org/whl/cpu",
                    "--force-reinstall",
                    "--no-deps",
                    "-q",
                ],
                check=True,
            )


    logger.info("Binary built in %s", backend_dir / "dist" / binary_name)


def build_shim():
    """Build the voicebox-mcp stdio shim as a tiny standalone binary.

    This is the bridge for MCP clients that only speak stdio — it proxies
    JSON-RPC to the main diarix-server's /mcp endpoint. Keep it small: no
    torch, no ML deps, just httpx + asyncio.
    """
    backend_dir = Path(__file__).parent

    args = [
        "mcp_shim/__main__.py",
        "--onefile",
        "--name",
        "voicebox-mcp",
        # Stdio-only — no console hiding needed on Windows since the parent
        # MCP client is spawning this as a child process and wants stdio.
        "--hidden-import",
        "backend.mcp_shim",
        "--hidden-import",
        "backend.mcp_shim.__main__",
        "--hidden-import",
        "httpx",
        "--hidden-import",
        "httpx._transports.default",
        "--hidden-import",
        "anyio",
        # Exclude everything heavy that httpx/asyncio don't actually need so
        # the binary stays tiny (~15 MB instead of ~400 MB).
        "--exclude-module",
        "torch",
        "--exclude-module",
        "transformers",
        "--exclude-module",
        "mlx",
        "--exclude-module",
        "mlx_audio",
        "--exclude-module",
        "mlx_lm",
        "--exclude-module",
        "qwen_tts",
        "--exclude-module",
        "chatterbox",
        "--exclude-module",
        "zipvoice",
        "--exclude-module",
        "tada",
        "--exclude-module",
        "kokoro",
        "--exclude-module",
        "misaki",
        "--exclude-module",
        "spacy",
        "--exclude-module",
        "librosa",
        "--exclude-module",
        "numba",
        "--exclude-module",
        "numpy",
        "--exclude-module",
        "pedalboard",
        "--exclude-module",
        "fastapi",
        "--exclude-module",
        "uvicorn",
        "--exclude-module",
        "sqlalchemy",
        "--exclude-module",
        "fastmcp",
        "--exclude-module",
        "mcp",
    ]

    dist_dir = str(backend_dir / "dist")
    build_dir = str(backend_dir / "build")
    args.extend(
        [
            "--distpath",
            dist_dir,
            "--workpath",
            build_dir,
            "--noconfirm",
            "--clean",
        ]
    )

    os.chdir(backend_dir)
    _run_pyinstaller(args)
    logger.info("Shim built: %s", backend_dir / "dist" / "voicebox-mcp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build voicebox binaries")
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Build CUDA-enabled binary (diarix-server-cuda)",
    )
    parser.add_argument(
        "--rocm",
        action="store_true",
        help="Build ROCm-enabled binary (diarix-server-rocm) for AMD GPUs",
    )
    parser.add_argument(
        "--shim",
        action="store_true",
        help="Build the voicebox-mcp stdio shim binary instead of the server",
    )
    parser.add_argument(
        "--require-media-tools",
        action="store_true",
        help="Fail unless FFmpeg and FFprobe can be bundled under the server's tools directory",
    )
    cli_args = parser.parse_args()
    try:
        if cli_args.shim:
            build_shim()
        else:
            build_server(
                cuda=cli_args.cuda,
                rocm=cli_args.rocm,
                require_media_tools=cli_args.require_media_tools,
            )
    except MediaToolPackagingError as exc:
        parser.exit(2, f"Media tool packaging error: {exc}\n")

