"""UK Survey of Personal Incomes (SPI) source provider."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
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


def _infer_survey_year(file_path: Path) -> int:
    stem_match = re.search(r"(20\d{2})", file_path.stem)
    if stem_match is not None:
        return int(stem_match.group(1))
    parent_match = re.search(r"(20\d{2})", file_path.parent.name)
    if parent_match is not None:
        return int(parent_match.group(1))
    raise ValueError(f"Could not infer survey year from '{file_path}'")


def _read_spi_table(file_path: Path) -> pd.DataFrame:
    manifest = load_uk_source_manifest("spi")
    observation = manifest.observation_for(EntityType.TAX_UNIT)
    raw = pd.read_csv(
        file_path,
        sep="\t",
        usecols=[column_spec.raw_column for column_spec in observation.columns],
    )
    normalized = pd.DataFrame(index=raw.index)
    normalized[observation.key_column] = np.arange(1, len(raw) + 1, dtype=np.int64)
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
            ).fillna(0.0)
    return normalized


@dataclass(frozen=True)
class UKSPISourceProvider:
    """Load a UK SPI tax-unit extract into a multientity observation frame."""

    file_path: str | Path
    source_name: str = "uk_spi"
    shareability: Shareability = Shareability.RESTRICTED
    time_structure: TimeStructure = TimeStructure.REPEATED_CROSS_SECTION
    survey_year: int | None = None
    population: str = "UK tax units"

    def __post_init__(self) -> None:
        object.__setattr__(self, "file_path", Path(self.file_path))
        if not self.file_path.exists():
            raise FileNotFoundError(self.file_path)
        if self.survey_year is None:
            object.__setattr__(self, "survey_year", _infer_survey_year(self.file_path))

    @cached_property
    def manifest(self):
        return load_uk_source_manifest("spi")

    @cached_property
    def descriptor(self) -> SourceDescriptor:
        tax_units = _read_spi_table(self.file_path)
        tax_units["year"] = int(self.survey_year)
        observation = self.manifest.observation_for(EntityType.TAX_UNIT)
        return SourceDescriptor(
            name=self.source_name,
            shareability=self.shareability,
            time_structure=self.time_structure,
            archetype=self.manifest.archetype,
            population=self.population,
            description=self.manifest.description,
            observations=(
                EntityObservation(
                    entity=EntityType.TAX_UNIT,
                    key_column=observation.key_column,
                    variable_names=observation.observed_variable_names(tax_units.columns),
                    weight_column=observation.weight_column,
                    period_column=observation.period_column,
                ),
            ),
        )

    def load_frame(self, query: SourceQuery | None = None) -> ObservationFrame:
        tax_units = _read_spi_table(self.file_path)
        tax_units["year"] = int(self.survey_year)
        frame = ObservationFrame(
            source=self.descriptor,
            tables={EntityType.TAX_UNIT: tax_units},
        )
        frame.validate()
        return apply_source_query(frame, query)
