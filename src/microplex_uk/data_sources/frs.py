"""UK FRS source provider built from PolicyEngine-style H5 datasets."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import h5py
import pandas as pd
from microplex.core import (
    EntityObservation,
    EntityRelationship,
    EntityType,
    ObservationFrame,
    RelationshipCardinality,
    Shareability,
    SourceDescriptor,
    SourceQuery,
    TimeStructure,
    apply_source_query,
)

from microplex_uk.source_manifests import load_uk_source_manifest


def _read_h5_table(file_path: Path, key: str) -> pd.DataFrame:
    with h5py.File(file_path, "r") as handle:
        dataset = handle[key]["table"][:]
    frame = pd.DataFrame.from_records(dataset)
    if "index" in frame.columns:
        frame = frame.drop(columns=["index"])
    return _decode_bytes(frame)


def _decode_bytes(frame: pd.DataFrame) -> pd.DataFrame:
    decoded = frame.copy()
    for column in decoded.columns:
        sample = decoded[column].dropna()
        if sample.empty:
            continue
        if isinstance(sample.iloc[0], (bytes, bytearray)):
            decoded[column] = decoded[column].map(
                lambda value: value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value
            )
    return decoded


def _extract_period(file_path: Path) -> int:
    time_period = _read_h5_table(file_path, "time_period")
    if "values" not in time_period.columns or time_period.empty:
        raise ValueError(f"Dataset '{file_path}' is missing time_period values")
    value = time_period.iloc[0]["values"]
    return int(value)


def _derive_benefit_unit_households(persons: pd.DataFrame) -> pd.DataFrame:
    mapping = persons.loc[:, ["benefit_unit_id", "household_id"]].drop_duplicates()
    duplicate_units = mapping["benefit_unit_id"].duplicated(keep=False)
    if duplicate_units.any():
        duplicates = sorted(mapping.loc[duplicate_units, "benefit_unit_id"].unique().tolist())
        raise ValueError(
            "Benefit units must map to exactly one household; found duplicates "
            f"for: {duplicates}"
        )
    return mapping


def _materialize_tables(file_path: Path) -> tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    period = _extract_period(file_path)
    manifest = load_uk_source_manifest("frs")
    person_spec = manifest.observation_for(EntityType.PERSON)
    benefit_unit_spec = manifest.observation_for(EntityType.BENEFIT_UNIT)
    household_spec = manifest.observation_for(EntityType.HOUSEHOLD)

    persons = _read_h5_table(file_path, person_spec.table_name or "person").rename(
        columns=person_spec.aliases
    )
    persons["year"] = period

    benefit_units = _read_h5_table(
        file_path,
        benefit_unit_spec.table_name or "benunit",
    ).rename(columns=benefit_unit_spec.aliases)
    benefit_units["year"] = period

    households = _read_h5_table(
        file_path,
        household_spec.table_name or "household",
    ).rename(columns=household_spec.aliases)
    households["year"] = period

    benefit_unit_households = _derive_benefit_unit_households(persons)
    benefit_units = benefit_units.merge(
        benefit_unit_households,
        on="benefit_unit_id",
        how="left",
        validate="one_to_one",
    )
    if benefit_units["household_id"].isna().any():
        missing = sorted(
            benefit_units.loc[
                benefit_units["household_id"].isna(),
                "benefit_unit_id",
            ].unique().tolist()
        )
        raise ValueError(
            "Benefit units must map to a household via person rows; missing mappings "
            f"for: {missing}"
        )

    return period, persons, benefit_units, households

@dataclass(frozen=True)
class UKFRSSourceProvider:
    """Load a UK FRS-style H5 dataset into a multientity observation frame."""

    file_path: str | Path
    source_name: str = "uk_frs"
    shareability: Shareability = Shareability.PUBLIC
    time_structure: TimeStructure = TimeStructure.REPEATED_CROSS_SECTION
    population: str = "UK resident households"

    def __post_init__(self) -> None:
        object.__setattr__(self, "file_path", Path(self.file_path))
        if not self.file_path.exists():
            raise FileNotFoundError(self.file_path)

    @cached_property
    def manifest(self):
        return load_uk_source_manifest("frs")

    @cached_property
    def descriptor(self) -> SourceDescriptor:
        _, persons, benefit_units, households = _materialize_tables(self.file_path)
        person_spec = self.manifest.observation_for(EntityType.PERSON)
        benefit_unit_spec = self.manifest.observation_for(EntityType.BENEFIT_UNIT)
        household_spec = self.manifest.observation_for(EntityType.HOUSEHOLD)
        return SourceDescriptor(
            name=self.source_name,
            shareability=self.shareability,
            time_structure=self.time_structure,
            archetype=self.manifest.archetype,
            population=self.population,
            description=self.manifest.description,
            observations=(
                EntityObservation(
                    entity=EntityType.PERSON,
                    key_column=person_spec.key_column,
                    variable_names=person_spec.observed_variable_names(persons.columns),
                    period_column=person_spec.period_column,
                ),
                EntityObservation(
                    entity=EntityType.BENEFIT_UNIT,
                    key_column=benefit_unit_spec.key_column,
                    variable_names=benefit_unit_spec.observed_variable_names(
                        benefit_units.columns
                    ),
                    period_column=benefit_unit_spec.period_column,
                ),
                EntityObservation(
                    entity=EntityType.HOUSEHOLD,
                    key_column=household_spec.key_column,
                    variable_names=household_spec.observed_variable_names(
                        households.columns
                    ),
                    weight_column=household_spec.weight_column,
                    period_column=household_spec.period_column,
                ),
            ),
        )

    def load_frame(self, query: SourceQuery | None = None) -> ObservationFrame:
        _, persons, benefit_units, households = _materialize_tables(self.file_path)
        frame = ObservationFrame(
            source=self.descriptor,
            tables={
                EntityType.PERSON: persons,
                EntityType.BENEFIT_UNIT: benefit_units,
                EntityType.HOUSEHOLD: households,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
                EntityRelationship(
                    parent_entity=EntityType.BENEFIT_UNIT,
                    child_entity=EntityType.PERSON,
                    parent_key="benefit_unit_id",
                    child_key="benefit_unit_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.BENEFIT_UNIT,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        frame.validate()
        return apply_source_query(frame, query)
