import pytest

from backend.services import cuda, rocm


def test_downloadable_backends_use_diarix_releases():
    expected = "https://github.com/Yannam-Builds/Diarix/releases/download"

    assert cuda.GITHUB_RELEASES_URL == expected
    assert rocm.GITHUB_RELEASES_URL == expected


def test_downloadable_backend_executables_use_diarix_names():
    assert cuda.get_cuda_exe_name().startswith("diarix-server-cuda")
    assert rocm.get_rocm_exe_name().startswith("diarix-server-rocm")


def test_cuda_manifest_supports_split_release_assets():
    manifest = {
        "version": cuda.CUDA_LIBS_VERSION,
        "archives": [
            {"archive": f"cuda-libs-{cuda.CUDA_LIBS_VERSION}-part1.tar.gz"},
            {"archive": f"cuda-libs-{cuda.CUDA_LIBS_VERSION}-part2.tar.gz"},
        ],
    }

    assert cuda._cuda_lib_archive_names(manifest) == [
        f"cuda-libs-{cuda.CUDA_LIBS_VERSION}-part1.tar.gz",
        f"cuda-libs-{cuda.CUDA_LIBS_VERSION}-part2.tar.gz",
    ]


@pytest.mark.parametrize(
    "archive",
    ["../cuda-libs.tar.gz", "folder/cuda-libs.tar.gz", "other-runtime.tar.gz"],
)
def test_cuda_manifest_rejects_unsafe_or_unexpected_assets(archive):
    manifest = {
        "version": cuda.CUDA_LIBS_VERSION,
        "archives": [{"archive": archive}],
    }

    with pytest.raises(ValueError):
        cuda._cuda_lib_archive_names(manifest)
