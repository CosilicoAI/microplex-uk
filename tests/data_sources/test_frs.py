from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from microplex.core import EntityType, SourceArchetype, SourceQuery

from microplex_uk.data_sources import UKFRSSourceProvider


def test_frs_source_provider_loads_multientity_frame(tmp_path: Path):
    dataset_path = tmp_path / "mini_frs.h5"
    _write_mini_frs_dataset(dataset_path)

    provider = UKFRSSourceProvider(dataset_path, source_name="mini_frs")
    frame = provider.load_frame()

    assert set(provider.descriptor.observed_entities) == {
        EntityType.PERSON,
        EntityType.BENEFIT_UNIT,
        EntityType.HOUSEHOLD,
    }
    assert set(frame.tables) == {
        EntityType.PERSON,
        EntityType.BENEFIT_UNIT,
        EntityType.HOUSEHOLD,
    }
    assert frame.tables[EntityType.PERSON]["benefit_unit_id"].tolist() == [101, 101, 202]
    assert frame.tables[EntityType.BENEFIT_UNIT]["household_id"].tolist() == [1, 2]
    assert provider.descriptor.observation_for(EntityType.HOUSEHOLD).weight_column == "household_weight"
    assert provider.descriptor.archetype is SourceArchetype.HOUSEHOLD_INCOME


def test_frs_source_provider_filters_on_period(tmp_path: Path):
    dataset_path = tmp_path / "mini_frs.h5"
    _write_mini_frs_dataset(dataset_path)

    provider = UKFRSSourceProvider(dataset_path)
    filtered = provider.load_frame(SourceQuery(period=2023))

    assert len(filtered.tables[EntityType.PERSON]) == 3
    empty = provider.load_frame(SourceQuery(period=2024))
    assert empty.tables[EntityType.PERSON].empty


def test_frs_source_provider_rejects_orphan_benefit_units(tmp_path: Path):
    dataset_path = tmp_path / "mini_frs_orphan.h5"
    _write_mini_frs_dataset(dataset_path, include_orphan_benefit_unit=True)

    provider = UKFRSSourceProvider(dataset_path)

    try:
        provider.load_frame()
    except ValueError as exc:
        assert "missing mappings" in str(exc)
    else:
        raise AssertionError("Expected orphan benefit unit mapping failure")


def _write_mini_frs_dataset(
    path: Path,
    *,
    include_orphan_benefit_unit: bool = False,
) -> None:
    person_dtype = np.dtype(
        [
            ("index", "<i8"),
            ("person_id", "<i8"),
            ("person_benunit_id", "<i8"),
            ("person_household_id", "<i8"),
            ("age", "<f8"),
            ("gender", "S6"),
            ("employment_income", "<f8"),
        ]
    )
    benunit_dtype = np.dtype(
        [
            ("index", "<i8"),
            ("benunit_id", "<i8"),
            ("is_married", "u1"),
        ]
    )
    household_dtype = np.dtype(
        [
            ("index", "<i8"),
            ("household_id", "<i8"),
            ("household_weight", "<f8"),
            ("region", "S8"),
        ]
    )
    time_dtype = np.dtype([("index", "<i8"), ("values", "<i8")])

    with h5py.File(path, "w") as handle:
        person_group = handle.create_group("person")
        person_group.create_dataset(
            "table",
            data=np.array(
                [
                    (0, 1001, 101, 1, 45.0, b"FEMALE", 12000.0),
                    (1, 1002, 101, 1, 43.0, b"MALE", 15000.0),
                    (2, 2001, 202, 2, 31.0, b"FEMALE", 9000.0),
                ],
                dtype=person_dtype,
            ),
        )
        benunit_group = handle.create_group("benunit")
        benunit_rows = [
            (0, 101, 1),
            (1, 202, 0),
        ]
        if include_orphan_benefit_unit:
            benunit_rows.append((2, 303, 0))
        benunit_group.create_dataset(
            "table",
            data=np.array(benunit_rows, dtype=benunit_dtype),
        )
        household_group = handle.create_group("household")
        household_group.create_dataset(
            "table",
            data=np.array(
                [
                    (0, 1, 1.5, b"north"),
                    (1, 2, 2.5, b"south"),
                ],
                dtype=household_dtype,
            ),
        )
        time_group = handle.create_group("time_period")
        time_group.create_dataset("table", data=np.array([(0, 2023)], dtype=time_dtype))
