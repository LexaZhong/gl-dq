"""File storage for config and knowledge: local directory or Unity Catalog Volume."""
from __future__ import annotations

import io
import os
from abc import ABC, abstractmethod
from pathlib import Path


class Storage(ABC):
    """Paths are '/'-separated and relative to the storage root."""

    @abstractmethod
    def read_text(self, path: str) -> str | None: ...

    @abstractmethod
    def write_text(self, path: str, text: str) -> None: ...

    @abstractmethod
    def list(self, prefix: str, suffix: str = "") -> list[str]:
        """Relative paths of files directly under `prefix`."""

    @abstractmethod
    def version(self, path: str) -> str | None:
        """Opaque change token (modification time); None if the file does not exist."""


class LocalStorage(Storage):
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _p(self, path: str) -> Path:
        return self.root / path

    def read_text(self, path):
        p = self._p(path)
        return p.read_text() if p.exists() else None

    def write_text(self, path, text):
        p = self._p(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, p)

    def list(self, prefix, suffix=""):
        d = self._p(prefix)
        if not d.is_dir():
            return []
        return sorted(f"{prefix}/{f.name}" for f in d.iterdir() if f.is_file() and f.name.endswith(suffix))

    def version(self, path):
        p = self._p(path)
        return str(p.stat().st_mtime_ns) if p.exists() else None

    def __repr__(self):
        return f"LocalStorage({self.root})"


class VolumeStorage(Storage):
    """Unity Catalog Volume via the Databricks Files API (works inside Databricks Apps)."""

    def __init__(self, root: str):
        from databricks.sdk import WorkspaceClient

        self.root = root.rstrip("/")
        self.w = WorkspaceClient()

    def _p(self, path: str) -> str:
        return f"{self.root}/{path}"

    def read_text(self, path):
        from databricks.sdk.errors import NotFound

        try:
            return self.w.files.download(self._p(path)).contents.read().decode()
        except NotFound:
            return None

    def write_text(self, path, text):
        self.w.files.upload(self._p(path), io.BytesIO(text.encode()), overwrite=True)

    def list(self, prefix, suffix=""):
        from databricks.sdk.errors import NotFound

        try:
            entries = self.w.files.list_directory_contents(self._p(prefix))
            return sorted(f"{prefix}/{e.name}" for e in entries if not e.is_directory and e.name.endswith(suffix))
        except NotFound:
            return []

    def version(self, path):
        from databricks.sdk.errors import NotFound

        try:
            return str(self.w.files.get_metadata(self._p(path)).last_modified)
        except NotFound:
            return None

    def __repr__(self):
        return f"VolumeStorage({self.root})"


def make_storage(location: str, base: Path) -> Storage:
    if location.startswith("/Volumes/"):
        return VolumeStorage(location)
    p = Path(location)
    return LocalStorage(p if p.is_absolute() else base / p)
