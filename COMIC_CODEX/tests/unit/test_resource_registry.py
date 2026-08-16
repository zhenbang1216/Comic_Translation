from collections.abc import Callable

import pytest

from comic_codex.resources.registry import ResourceLimitError, ResourceRegistry


class ClosableFake:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def raise_if_called() -> ClosableFake:
    raise AssertionError("loader must not be called for an acquired resource")


def test_acquire_loads_once_and_release_calls_close() -> None:
    resource = ClosableFake()
    cleanup_calls = 0

    def cleanup() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1

    registry = ResourceRegistry(max_resources=1, cleanup=cleanup)

    first = registry.acquire("ocr", lambda: resource)
    second = registry.acquire("ocr", raise_if_called)
    registry.release("ocr")

    assert first is second
    assert resource.closed is True
    assert cleanup_calls == 1


def test_limit_blocks_second_heavy_resource_before_loading() -> None:
    registry = ResourceRegistry(max_resources=1)
    second_loader_called = False

    def load_second() -> object:
        nonlocal second_loader_called
        second_loader_called = True
        return object()

    registry.acquire("detector", object)

    with pytest.raises(ResourceLimitError):
        registry.acquire("translator", load_second)

    assert second_loader_called is False


def test_release_all_uses_reverse_acquisition_order() -> None:
    released: list[str] = []

    class OrderedResource:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            released.append(self.name)

    registry = ResourceRegistry(max_resources=2)
    registry.acquire("first", lambda: OrderedResource("first"))
    registry.acquire("second", lambda: OrderedResource("second"))

    registry.release_all()

    assert released == ["second", "first"]


def test_release_all_aggregates_close_errors() -> None:
    def broken(name: str) -> Callable[[], None]:
        def close() -> None:
            raise RuntimeError(name)

        return close

    class BrokenResource:
        def __init__(self, close: Callable[[], None]) -> None:
            self.close = close

    registry = ResourceRegistry(max_resources=2)
    registry.acquire("one", lambda: BrokenResource(broken("one")))
    registry.acquire("two", lambda: BrokenResource(broken("two")))

    with pytest.raises(ExceptionGroup) as captured:
        registry.release_all()

    assert [str(error) for error in captured.value.exceptions] == ["two", "one"]
    assert registry.loaded_resource_ids == ()
