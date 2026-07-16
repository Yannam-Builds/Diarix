"""Windows cache compatibility for native model runtimes."""

from pathlib import Path

from backend.backends import base


def test_native_windows_path_strips_extended_drive_prefix(monkeypatch):
    monkeypatch.setattr(base.platform, "system", lambda: "Windows")

    assert base.native_windows_path(r"\\?\Z:\models\snapshot") == r"Z:\models\snapshot"


def test_native_windows_path_strips_extended_unc_prefix(monkeypatch):
    monkeypatch.setattr(base.platform, "system", lambda: "Windows")

    assert (
        base.native_windows_path(r"\\?\UNC\server\share\models")
        == r"\\server\share\models"
    )


def test_materialize_windows_snapshot_links_replaces_symlink(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(base.platform, "system", lambda: "Windows")
    blob = tmp_path / "blobs" / "weights"
    blob.parent.mkdir()
    blob.write_bytes(b"model weights")
    snapshot = tmp_path / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    link = snapshot / "model.bin"
    link.symlink_to(blob)

    result = base.materialize_windows_snapshot_links(snapshot)

    assert result == snapshot
    assert link.read_bytes() == b"model weights"
    assert not link.is_symlink()


def test_materialize_windows_snapshot_links_is_noop_off_windows(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(base.platform, "system", lambda: "Linux")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    assert base.materialize_windows_snapshot_links(snapshot) == snapshot
