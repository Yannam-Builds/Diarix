import os
import runpy
import tempfile
import unittest
from pathlib import Path

from backend.build_binary import (
    MediaToolPackagingError,
    MediaToolPaths,
    NAGISA_PYINSTALLER_ARGS,
    media_tool_pyinstaller_args,
    resolve_media_tools,
    stage_media_tools,
)


class MediaToolPackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_tool(self, relative_path: str, content: bytes = b"tool") -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_explicit_binary_paths_override_ffmpeg_directory(self):
        ffmpeg = self.write_tool("custom-encoder")
        ffprobe = self.write_tool("custom-prober")

        result = resolve_media_tools(
            env={
                "DIARIX_FFMPEG_DIR": str(self.root / "missing-directory"),
                "FFMPEG_BINARY": str(ffmpeg),
                "FFPROBE_BINARY": str(ffprobe),
                "PATH": "",
            },
            require=True,
            system_name="Linux",
        )

        self.assertEqual(result, MediaToolPaths(ffmpeg=ffmpeg.resolve(), ffprobe=ffprobe.resolve()))

    def test_ffmpeg_directory_resolves_canonical_windows_names(self):
        ffmpeg = self.write_tool("media-tools/ffmpeg.exe")
        ffprobe = self.write_tool("media-tools/ffprobe.exe")
        tool_dir = ffmpeg.parent

        result = resolve_media_tools(
            env={"DIARIX_FFMPEG_DIR": str(tool_dir), "PATH": ""},
            require=True,
            system_name="Windows",
        )

        self.assertEqual(result, MediaToolPaths(ffmpeg=ffmpeg.resolve(), ffprobe=ffprobe.resolve()))

    def test_path_resolution_uses_the_supplied_path(self):
        ffmpeg = self.write_tool("ffmpeg")
        ffprobe = self.write_tool("ffprobe")
        calls = []

        def fake_which(command: str, *, path: str):
            calls.append((command, path))
            return {"ffmpeg": str(ffmpeg), "ffprobe": str(ffprobe)}.get(command)

        result = resolve_media_tools(
            env={"PATH": "isolated-search-path"},
            require=True,
            system_name="Linux",
            which=fake_which,
        )

        self.assertEqual(result, MediaToolPaths(ffmpeg=ffmpeg.resolve(), ffprobe=ffprobe.resolve()))
        self.assertEqual(
            calls,
            [
                ("ffmpeg", "isolated-search-path"),
                ("ffprobe", "isolated-search-path"),
            ],
        )

    def test_required_media_tools_fail_with_actionable_error(self):
        with self.assertRaisesRegex(MediaToolPackagingError, "Release builds require both tools"):
            resolve_media_tools(
                env={"PATH": ""},
                require=True,
                system_name="Linux",
                which=lambda _command, *, path: None,
            )

    def test_optional_build_can_continue_without_media_tools(self):
        with self.assertLogs("backend.build_binary", level="WARNING") as logs:
            result = resolve_media_tools(
                env={"PATH": ""},
                system_name="Linux",
                which=lambda _command, *, path: None,
            )

        self.assertIsNone(result)
        self.assertIn("built without bundled media tools", "\n".join(logs.output))

    def test_bad_explicit_binary_does_not_silently_fall_back(self):
        with self.assertRaisesRegex(MediaToolPackagingError, "FFMPEG_BINARY is set"):
            resolve_media_tools(
                env={
                    "FFMPEG_BINARY": "missing-custom-ffmpeg",
                    "FFPROBE_BINARY": "missing-custom-ffprobe",
                    "PATH": "",
                },
                system_name="Linux",
                which=lambda _command, *, path: None,
            )

    def test_staging_canonicalizes_names_and_targets_tools_directory(self):
        ffmpeg = self.write_tool("sources/encoder-custom", b"ffmpeg")
        ffprobe = self.write_tool("sources/probe-custom", b"ffprobe")

        staged = stage_media_tools(
            MediaToolPaths(ffmpeg=ffmpeg, ffprobe=ffprobe),
            self.root / "staging",
            system_name="Windows",
        )

        self.assertEqual(staged.ffmpeg.name, "ffmpeg.exe")
        self.assertEqual(staged.ffprobe.name, "ffprobe.exe")
        self.assertEqual(staged.ffmpeg.read_bytes(), b"ffmpeg")
        self.assertEqual(staged.ffprobe.read_bytes(), b"ffprobe")
        self.assertEqual(
            media_tool_pyinstaller_args(staged),
            [
                "--add-binary",
                f"{staged.ffmpeg}{os.pathsep}tools",
                "--add-binary",
                f"{staged.ffprobe}{os.pathsep}tools",
            ],
        )

    def test_qwen_asr_packages_nagisa_as_physical_source(self):
        self.assertEqual(
            NAGISA_PYINSTALLER_ARGS,
            ["--collect-all", "nagisa", "--copy-metadata", "nagisa"],
        )
        hook = runpy.run_path(
            str(
                Path(__file__).parents[1]
                / "pyi_hooks"
                / "hook-nagisa.py"
            )
        )
        self.assertEqual(hook["module_collection_mode"], "py")

    def test_nemo_packages_dynamic_cuda_python_extensions(self):
        source = Path(__file__).parents[1].joinpath("build_binary.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"cuda.bindings"', source)
        self.assertIn('"cuda.bindings.cydriver"', source)
        self.assertIn('"cuda.bindings.cyruntime"', source)
        self.assertIn('"lightning_fabric"', source)


if __name__ == "__main__":
    unittest.main()
