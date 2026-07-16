"""Release-contract tests for catalog runtime import coverage."""

from types import SimpleNamespace

import backend.runtime_self_test as runtime_self_test
from backend.backends import get_all_model_configs
from backend.runtime_self_test import RUNTIME_IMPORT_CHECKS, run_runtime_self_test


def test_runtime_self_test_covers_every_catalog_engine() -> None:
    catalog_engines = {config.engine for config in get_all_model_configs()}
    assert catalog_engines == set(RUNTIME_IMPORT_CHECKS)


def test_runtime_self_test_reports_import_failures(monkeypatch) -> None:
    def fake_import(module_name: str):
        if module_name == "qwen_asr":
            raise ModuleNotFoundError("No module named 'prepro'")
        attributes = {
            attribute: object()
            for checks in RUNTIME_IMPORT_CHECKS.values()
            for candidate, names in checks
            if candidate == module_name
            for attribute in names
        }
        return SimpleNamespace(**attributes)

    monkeypatch.setattr(runtime_self_test.importlib, "import_module", fake_import)
    monkeypatch.setattr(runtime_self_test, "_prepare_engine", lambda _engine: None)

    result = run_runtime_self_test()

    assert result["ok"] is False
    assert result["failed"] == ["qwen_asr"]
    assert "prepro" in result["engines"]["qwen_asr"]["error"]
