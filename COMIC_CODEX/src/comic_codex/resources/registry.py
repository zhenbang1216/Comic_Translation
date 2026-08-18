from collections.abc import Callable
from typing import TypeVar, cast

T = TypeVar("T")


class ResourceLimitError(RuntimeError):
    pass


class ResourceRegistry:
    def __init__(
        self, max_resources: int, cleanup: Callable[[], None] | None = None
    ) -> None:
        if max_resources < 1:
            raise ValueError("max_resources must be at least 1")
        self._max_resources = max_resources
        self._cleanup = cleanup
        self._resources: dict[str, object] = {}

    @property
    def loaded_resource_ids(self) -> tuple[str, ...]:
        return tuple(self._resources)

    def acquire(self, resource_id: str, loader: Callable[[], T]) -> T:
        if resource_id in self._resources:
            return cast(T, self._resources[resource_id])
        if len(self._resources) >= self._max_resources:
            raise ResourceLimitError(
                f"resource limit {self._max_resources} reached before loading {resource_id!r}"
            )
        resource = loader()
        self._resources[resource_id] = resource
        return resource

    def release(self, resource_id: str) -> None:
        resource = self._resources.get(resource_id)
        if resource is None:
            return
        try:
            close = getattr(resource, "close", None)
            if callable(close):
                close()
        finally:
            del self._resources[resource_id]
            if self._cleanup is not None:
                self._cleanup()

    def release_all(self) -> None:
        errors: list[Exception] = []
        for resource_id in reversed(self.loaded_resource_ids):
            try:
                self.release(resource_id)
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("one or more resources failed to close", errors)

