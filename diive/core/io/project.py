"""
CORE.IO.PROJECT: DIIVE PROJECT FOLDERS
======================================

A *diive project* is a self-contained, portable folder that bundles a dataset
with everything needed to reopen it in the same state: the data as parquet, the
full per-variable metadata (tags, notes, origin/parent/provenance), and a slot
for caller-owned extras (site coordinates, active date range, ...).

Layout::

    MyProject.diive/
    ├── __diive__        marker -> {"diive_project": true, "format_version": 1}
    ├── project.json     manifest (name, timestamps, metadata, extras, data ref)
    └── data.parquet     the dataset (diive parquet)

The marker file identifies the folder cheaply (no manifest parsing needed). This
module owns the format only; ``extras`` is written/read verbatim, so callers (the
GUI) can stash their own values without this module knowing about them.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pandas import DataFrame

from diive.core.io.files import load_parquet, save_parquet
from diive.core.metadata import MetadataStore

#: Suffix for a project folder and the files inside it.
PROJECT_SUFFIX = ".diive"
MARKER_FILE = "__diive__"
MANIFEST_FILE = "project.json"
DATA_FILE = "data.parquet"
FORMAT_VERSION = 1


@dataclass
class DiiveProject:
    """An in-memory diive project: a dataset plus its metadata and extras."""

    name: str
    data: DataFrame
    metadata: MetadataStore = field(default_factory=MetadataStore)
    extras: dict = field(default_factory=dict)


def is_project(folder) -> bool:
    """True if ``folder`` is a diive project (has the marker file)."""
    try:
        return (Path(folder) / MARKER_FILE).is_file()
    except OSError:
        return False


def project_name_to_dirname(name: str) -> str:
    """Folder name for a project: ``<name>.diive`` (suffix added once)."""
    name = name.strip()
    return name if name.endswith(PROJECT_SUFFIX) else f"{name}{PROJECT_SUFFIX}"


def save_project(folder, project: DiiveProject) -> Path:
    """Write ``project`` into ``folder`` (created if needed). Returns the folder.

    The marker, manifest, and ``data.parquet`` are (re)written; an existing diive
    project at ``folder`` is updated in place.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    (folder / MARKER_FILE).write_text(
        json.dumps({"diive_project": True, "format_version": FORMAT_VERSION}),
        encoding="utf-8")

    save_parquet(filename=Path(DATA_FILE).stem, data=project.data, outpath=folder)

    try:
        from diive import __version__ as diive_version
    except Exception:
        diive_version = ""
    manifest = {
        "name": project.name,
        "format_version": FORMAT_VERSION,
        "diive_version": diive_version,
        "data_file": DATA_FILE,
        "metadata": project.metadata.to_dict(),
        "extras": project.extras,
    }
    (folder / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return folder


def load_project(folder) -> DiiveProject:
    """Read a diive project from ``folder``. Raises if it is not a project."""
    folder = Path(folder)
    if not is_project(folder):
        raise ValueError(f"Not a diive project (no {MARKER_FILE}): {folder}")
    manifest = json.loads((folder / MANIFEST_FILE).read_text(encoding="utf-8"))
    data = load_parquet(filepath=str(folder / manifest.get("data_file", DATA_FILE)))
    store = MetadataStore()
    store.load_dict(manifest.get("metadata"))
    return DiiveProject(
        name=manifest.get("name", folder.stem),
        data=data, metadata=store, extras=dict(manifest.get("extras") or {}))
