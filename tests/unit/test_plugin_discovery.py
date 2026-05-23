from __future__ import annotations

from core.plugin import discover_plugins


class _EP:
    def __init__(self, name: str, should_fail: bool = False) -> None:
        self.name = name
        self._should_fail = should_fail

    def load(self):
        if self._should_fail:
            raise RuntimeError("boom")
        return object()


class _SelectableEPs:
    def __init__(self, selected):
        self._selected = selected

    def select(self, *, group: str):
        return list(self._selected.get(group, []))


def test_discover_plugins_with_select_api(monkeypatch) -> None:
    selected = {
        "genlib.plugins": [_EP("a"), _EP("b", should_fail=True), _EP("c")],
    }
    monkeypatch.setattr("core.plugin.metadata.entry_points", lambda: _SelectableEPs(selected))
    assert discover_plugins() == ["a", "c"]


def test_discover_plugins_with_legacy_mapping_api(monkeypatch) -> None:
    mapping = {
        "genlib.plugins": [_EP("x"), _EP("y")],
        "other.group": [_EP("z")],
    }
    monkeypatch.setattr("core.plugin.metadata.entry_points", lambda: mapping)
    assert discover_plugins("genlib.plugins") == ["x", "y"]
    assert discover_plugins("other.group") == ["z"]

