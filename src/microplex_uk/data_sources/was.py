"""UK Wealth and Assets Survey (WAS) source provider."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import pandas as pd
from microplex.core import (
    EntityObservation,
    EntityType,
    ObservationFrame,
    Shareability,
    SourceColumnValueType,
    SourceDescriptor,
    SourceQuery,
    TimeStructure,
    apply_source_query,
)

from microplex_uk.source_manifests import load_uk_source_manifest


def _read_was_households(file_path: Path) -> pd.DataFrame:
    manifest = load_uk_source_manifest("was")
    observation = manifest.observation_for(EntityType.HOUSEHOLD)
    raw = pd.read_csv(
        file_path,
        sep="\t",
        usecols=[column_spec.raw_column for column_spec in observation.columns],
        low_memory=False,
    )
    normalized = pd.DataFrame(index=raw.index)
    for column_spec in observation.columns:
        if column_spec.raw_column not in raw.columns:
            continue
        series = raw[column_spec.raw_column]
        if column_spec.value_type is SourceColumnValueType.CATEGORICAL:
            normalized[column_spec.canonical_name] = series
        else:
            normalized[column_spec.canonical_name] = pd.to_numeric(
                series,
                errors="coerce",
            )
    return normalized


@dataclass(frozen=True)
class UKWASSourceProvider:
    """Load a UK WAS household extract into an observation frame."""

    file_path: str | Path
    source_name: str = "uk_was"
    shareability: Shareability = Shareability.RESTRICTED
    time_structure: TimeStructure = TimeStructure.REPEATED_CROSS_SECTION
    population: str = "UK households"

    def __post_init__(self) -> None:
        object.__setattr__(self, "file_path", Path(self.file_path))
        if not self.file_path.exists():
            raise FileNotFoundError(self.file_path)

    @cached_property
    def manifest(self):
        return load_uk_source_manifest("was")

    @cached_property
    def descriptor(self) -> SourceDescriptor:
        households = _read_was_households(self.file_path)
        observation = self.manifest.observation_for(EntityType.HOUSEHOLD)
        return SourceDescriptor(
            name=self.source_name,
            shareability=self.shareability,
            time_structure=self.time_structure,
            archetype=self.manifest.archetype,
            population=self.population,
            description=self.manifest.description,
            observations=(
                EntityObservation(
                    entity=EntityType.HOUSEHOLD,
                    key_column=observation.key_column,
                    variable_names=observation.observed_variable_names(households.columns),
                    weight_column=observation.weight_column,
                    period_column=observation.period_column,
                ),
            ),
        )

    def load_frame(self, query: SourceQuery | None = None) -> ObservationFrame:
        households = _read_was_households(self.file_path)
        frame = ObservationFrame(
            source=self.descriptor,
            tables={EntityType.HOUSEHOLD: households},
        )
        frame.validate()
        return apply_source_query(frame, query)
