import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock

_write_locks_guard = Lock()
_write_locks: dict[Path, Lock] = {}


def _lock_for(path: Path) -> Lock:
    resolved = path.resolve()
    with _write_locks_guard:
        return _write_locks.setdefault(resolved, Lock())


def write_text_atomically(path: Path, content: str) -> None:
    with _lock_for(path):
        _write_text_atomically_unlocked(path, content)


def _write_text_atomically_unlocked(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

