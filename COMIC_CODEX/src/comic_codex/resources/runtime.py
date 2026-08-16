from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    system_total_mb: float | None
    system_available_mb: float | None
    cuda_allocated_mb: float | None
    cuda_reserved_mb: float | None


def capture_runtime_snapshot() -> RuntimeSnapshot:
    system_total_mb: float | None = None
    system_available_mb: float | None = None
    cuda_allocated_mb: float | None = None
    cuda_reserved_mb: float | None = None

    try:
        psutil = import_module("psutil")
        memory = psutil.virtual_memory()
        system_total_mb = float(memory.total) / (1024 * 1024)
        system_available_mb = float(memory.available) / (1024 * 1024)
    except (ImportError, AttributeError):
        pass

    try:
        torch = import_module("torch")
        if bool(torch.cuda.is_available()):
            cuda_allocated_mb = float(torch.cuda.memory_allocated()) / (1024 * 1024)
            cuda_reserved_mb = float(torch.cuda.memory_reserved()) / (1024 * 1024)
    except (ImportError, AttributeError):
        pass

    return RuntimeSnapshot(
        system_total_mb=system_total_mb,
        system_available_mb=system_available_mb,
        cuda_allocated_mb=cuda_allocated_mb,
        cuda_reserved_mb=cuda_reserved_mb,
    )

