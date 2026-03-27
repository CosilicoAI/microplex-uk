from __future__ import annotations

from pathlib import Path

import pandas as pd
from microplex.core import EntityType, SourceArchetype, SourceQuery

from microplex_uk.data_sources import UKWASSourceProvider


def test_was_source_provider_loads_household_frame(tmp_path: Path):
    was_path = tmp_path / "was_round_7_hhold_eul_march_2022.tab"
    _write_mini_was_dataset(was_path)

    provider = UKWASSourceProvider(was_path, source_name="mini_was")
    frame = provider.load_frame(SourceQuery(period=2018))

    assert set(frame.tables) == {EntityType.HOUSEHOLD}
    assert provider.descriptor.archetype is SourceArchetype.WEALTH
    assert frame.tables[EntityType.HOUSEHOLD]["household_id"].tolist() == [101, 102]
    assert frame.tables[EntityType.HOUSEHOLD]["net_wealth"].tolist() == [250000, 400000]


def test_was_source_provider_filters_out_other_periods(tmp_path: Path):
    was_path = tmp_path / "was_round_7_hhold_eul_march_2022.tab"
    _write_mini_was_dataset(was_path)

    provider = UKWASSourceProvider(was_path)
    empty = provider.load_frame(SourceQuery(period=2020))

    assert empty.tables[EntityType.HOUSEHOLD].empty


def _write_mini_was_dataset(path: Path) -> None:
    pd.DataFrame(
        {
            "CASER7": [101, 102],
            "yearr7": [2018, 2018],
            "R7xshhwgt": [10.0, 20.0],
            "gorr7": [1, 7],
            "hholdtyper7": [2, 3],
            "hrpsexr7": [1, 2],
            "HRPDVAge8r7": [5, 6],
            "numadultr7": [2, 1],
            "numch18r7": [1, 0],
            "dvprirntr7": [1, 2],
            "TotWlthR7": [250000, 400000],
            "TotmortR7": [50000, 0],
            "DVTotinc_bhcR7": [42000, 38000],
            "DVTotinc_ahcR7": [39000, 36000],
            "DVPropertyR7": [150000, 250000],
            "HFINWR7_SUM": [60000, 90000],
            "HFINWNTR7_Sum": [30000, 85000],
            "DVHValueR7": [180000, 300000],
            "DVHseValR7_sum": [20000, 10000],
            "DVBLDValR7_sum": [0, 5000],
            "DVSaValR7_aggr": [12000, 24000],
            "vcarnr7": [1, 2],
            "TOTPENR7_aggr": [30000, 50000],
            "DVValDBTR7_aggr": [10000, 20000],
            "DVFESHARESR7_aggr": [2000, 3000],
            "DVFShUKVR7_aggr": [4000, 6000],
            "DVIISAVR7_aggR": [8000, 12000],
            "DVFCollVR7_aggr": [1000, 2000],
        }
    ).to_csv(path, sep="\t", index=False)
