from __future__ import annotations

from pathlib import Path

import pandas as pd
from microplex.core import EntityType, SourceArchetype, SourceQuery

from microplex_uk.data_sources import UKSPISourceProvider


def test_spi_source_provider_loads_tax_unit_frame(tmp_path: Path):
    spi_path = tmp_path / "put2021uk.tab"
    _write_mini_spi_dataset(spi_path)

    provider = UKSPISourceProvider(spi_path, source_name="mini_spi")
    frame = provider.load_frame(SourceQuery(period=2021))

    assert set(frame.tables) == {EntityType.TAX_UNIT}
    assert provider.descriptor.archetype is SourceArchetype.TAX_MICRODATA
    assert frame.tables[EntityType.TAX_UNIT]["tax_unit_id"].tolist() == [1, 2]
    assert frame.tables[EntityType.TAX_UNIT]["employment_income"].tolist() == [35000, 12000]
    assert frame.tables[EntityType.TAX_UNIT]["year"].tolist() == [2021, 2021]


def test_spi_source_provider_filters_out_other_periods(tmp_path: Path):
    spi_path = tmp_path / "put2021uk.tab"
    _write_mini_spi_dataset(spi_path)

    provider = UKSPISourceProvider(spi_path)
    empty = provider.load_frame(SourceQuery(period=2022))

    assert empty.tables[EntityType.TAX_UNIT].empty


def _write_mini_spi_dataset(path: Path) -> None:
    pd.DataFrame(
        {
            "FACT": [10.0, 20.0],
            "PAY": [35_000, 12_000],
            "INCBBS": [120, 80],
            "INCPROP": [0, 250],
            "PROFITS": [1_000, 0],
            "DIVIDENDS": [200, 50],
            "PENSION": [0, 3_000],
            "OTHERINV": [25, 10],
            "TAXINC": [36_000, 15_000],
            "TOTTAX_DEVO_TXP": [5_000, 1_200],
            "TAX_CRED": [100, 0],
            "GORCODE": [7, 3],
            "SEX": [1, 2],
            "MAR": [2, 1],
            "AGERANGE": [5, 3],
            "TAXPAYER": [1, 1],
            "SEINC_NUM": [1, 0],
            "MAINSRCE": [3, 1],
        }
    ).to_csv(path, sep="\t", index=False)
