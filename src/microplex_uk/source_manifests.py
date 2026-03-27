"""Typed source-manifest accessors for microplex-uk."""

from __future__ import annotations

from functools import cache
from pathlib import Path

from microplex.core import SourceManifest, load_source_manifest


def _manifest_dir() -> Path:
    return Path(__file__).resolve().parent / "manifests"


@cache
def load_uk_source_manifest(name: str) -> SourceManifest:
    """Load one UK source manifest by name."""
    return load_source_manifest(_manifest_dir() / f"{name}.json")
