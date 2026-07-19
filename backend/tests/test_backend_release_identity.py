from backend.services import cuda, rocm


def test_downloadable_backends_use_diarix_releases():
    expected = "https://github.com/Yannam-Builds/Diarix/releases/download"

    assert cuda.GITHUB_RELEASES_URL == expected
    assert rocm.GITHUB_RELEASES_URL == expected


def test_downloadable_backend_executables_use_diarix_names():
    assert cuda.get_cuda_exe_name().startswith("diarix-server-cuda")
    assert rocm.get_rocm_exe_name().startswith("diarix-server-rocm")
