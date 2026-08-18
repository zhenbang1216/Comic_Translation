from types import SimpleNamespace

from comic_codex.resources.runtime import capture_runtime_snapshot


def test_runtime_snapshot_tolerates_cuda_driver_error(monkeypatch) -> None:
    broken_cuda = SimpleNamespace(
        is_available=lambda: (_ for _ in ()).throw(RuntimeError("driver mismatch"))
    )

    def fake_import(name: str):
        if name == "torch":
            return SimpleNamespace(cuda=broken_cuda)
        raise ImportError(name)

    monkeypatch.setattr("comic_codex.resources.runtime.import_module", fake_import)

    snapshot = capture_runtime_snapshot()

    assert snapshot.cuda_allocated_mb is None
    assert snapshot.cuda_reserved_mb is None
