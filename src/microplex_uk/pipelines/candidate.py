"""Build and benchmark fused UK candidate datasets."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from microplex.core import EntityType
from microplex.geography import GeographyAssignmentPlan, GeographyProvider
from microplex.targets import TargetQuery

from microplex_uk.data_sources import UKSPISourceProvider, UKWASSourceProvider
from microplex_uk.data_sources.frs import _extract_period, _read_h5_table
from microplex_uk.geography import apply_uk_candidate_geography
from microplex_uk.policyengine import (
    PolicyEngineUKBenchmarkComparison,
    PolicyEngineUKBenchmarkResult,
    PolicyEngineUKDirectBenchmarkComparison,
    PolicyEngineUKDirectBenchmarkResult,
    compare_policyengine_uk_benchmark,
    compare_policyengine_uk_direct_benchmark,
)
from microplex_uk.targets import PolicyEngineUKTargetProvider

UK_REGION_CODE_TO_NAME = {
    1: "NORTH_EAST",
    2: "NORTH_WEST",
    3: "YORKSHIRE",
    4: "EAST_MIDLANDS",
    5: "WEST_MIDLANDS",
    6: "EAST_OF_ENGLAND",
    7: "LONDON",
    8: "SOUTH_EAST",
    9: "SOUTH_WEST",
    10: "WALES",
    11: "SCOTLAND",
    12: "NORTHERN_IRELAND",
    13: "UNKNOWN",
    14: "UNKNOWN",
    -1: "UNKNOWN",
}
WAS_REGION_CODE_TO_NAME = {
    1: "NORTH_EAST",
    2: "NORTH_WEST",
    4: "YORKSHIRE",
    5: "EAST_MIDLANDS",
    6: "WEST_MIDLANDS",
    7: "EAST_OF_ENGLAND",
    8: "LONDON",
    9: "SOUTH_EAST",
    10: "SOUTH_WEST",
    11: "WALES",
    12: "SCOTLAND",
}
SPI_AGE_BAND_BOUNDS: tuple[tuple[int, int, int], ...] = (
    (1, 16, 25),
    (2, 25, 35),
    (3, 35, 45),
    (4, 45, 55),
    (5, 55, 65),
    (6, 65, 74),
    (7, 74, 200),
)
PERSON_INCOME_COLUMNS: tuple[str, ...] = (
    "employment_income",
    "private_pension_income",
    "self_employment_income",
    "tax_free_savings_income",
    "savings_interest_income",
    "dividend_income",
    "property_income",
    "maintenance_income",
    "miscellaneous_income",
    "private_transfer_income",
    "lump_sum_income",
)


class UKDonorCombineStrategy(str, Enum):
    REPLACE = "replace"
    MAX = "max"


@dataclass(frozen=True)
class UKDonorOutputDefaultSpec:
    column: str
    value: Any


@dataclass(frozen=True)
class UKDonorVariableSpec:
    recipient_column: str
    donor_column: str | None = None
    score_column: str | None = None
    combine_strategy: UKDonorCombineStrategy = UKDonorCombineStrategy.REPLACE
    fill_value: float = 0.0

    @property
    def resolved_donor_column(self) -> str:
        return self.donor_column or self.recipient_column


@dataclass(frozen=True)
class UKDonorBlockSpec:
    name: str
    recipient_table: str
    match_columns: tuple[str, ...]
    donor_weight_column: str
    variables: tuple[UKDonorVariableSpec, ...]
    recipient_mask_column: str | None = None
    output_defaults: tuple[UKDonorOutputDefaultSpec, ...] = ()


class UKBenchmarkMode(str, Enum):
    STANDARD = "standard"
    DIRECT = "direct"


@dataclass
class UKCandidateDataset:
    person: pd.DataFrame
    benunit: pd.DataFrame
    household: pd.DataFrame
    time_period: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(
        self,
        path: str | Path,
        *,
        python_executable: str | Path = "python3",
    ) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer_python = _resolve_hdf_writer_python(python_executable)
        with tempfile.TemporaryDirectory(prefix="microplex_uk_candidate_") as temp_dir:
            temp_root = Path(temp_dir)
            person_path, person_schema_path = _write_transfer_frame(
                temp_root, "person", self.person
            )
            benunit_path, benunit_schema_path = _write_transfer_frame(
                temp_root, "benunit", self.benunit
            )
            household_path, household_schema_path = _write_transfer_frame(
                temp_root, "household", self.household
            )
            script = """
import json
import sys
import pandas as pd

def load_frame(csv_path, schema_path):
    schema = json.loads(open(schema_path).read())
    frame = pd.read_csv(csv_path, low_memory=False)
    frame = frame[schema["columns"]]
    for column, dtype in schema["dtypes"].items():
        if dtype.startswith(("int", "uint", "float")) or dtype in {"bool"}:
            frame[column] = frame[column].astype(dtype)
    return frame

person = load_frame(sys.argv[1], sys.argv[2])
benunit = load_frame(sys.argv[3], sys.argv[4])
household = load_frame(sys.argv[5], sys.argv[6])
time_period = int(sys.argv[7])
output_path = sys.argv[8]

with pd.HDFStore(output_path, mode="w") as store:
    store.put("person", person, format="table", data_columns=True)
    store.put("benunit", benunit, format="table", data_columns=True)
    store.put("household", household, format="table", data_columns=True)
    store.put("time_period", pd.Series([time_period]), format="table")
"""
            completed = subprocess.run(
                [
                    writer_python,
                    "-c",
                    script,
                    str(person_path),
                    str(person_schema_path),
                    str(benunit_path),
                    str(benunit_schema_path),
                    str(household_path),
                    str(household_schema_path),
                    str(int(self.time_period)),
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Failed to write UK candidate HDF dataset: "
                    + (completed.stderr or completed.stdout).strip()
                )
        return output_path


@dataclass
class UKCandidateBenchmarkArtifacts:
    label: str
    candidate_dataset_path: Path
    comparison_path: Path
    comparison: PolicyEngineUKBenchmarkComparison | PolicyEngineUKDirectBenchmarkComparison


def _write_transfer_frame(
    temp_root: Path,
    name: str,
    frame: pd.DataFrame,
) -> tuple[Path, Path]:
    csv_path = temp_root / f"{name}.csv"
    schema_path = temp_root / f"{name}.schema.json"
    frame.to_csv(csv_path, index=False, float_format="%.17g")
    schema_path.write_text(
        json.dumps(
            {
                "columns": list(frame.columns),
                "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
            }
        )
    )
    return csv_path, schema_path


def default_uk_candidate_donor_block_specs() -> tuple[UKDonorBlockSpec, ...]:
    return (
        UKDonorBlockSpec(
            name="spi_capital_income",
            recipient_table="person",
            match_columns=("region", "gender", "age_band"),
            donor_weight_column="weight",
            recipient_mask_column="is_spi_eligible",
            variables=(
                UKDonorVariableSpec(
                    recipient_column="dividend_income",
                    combine_strategy=UKDonorCombineStrategy.MAX,
                ),
                UKDonorVariableSpec(
                    recipient_column="savings_interest_income",
                    combine_strategy=UKDonorCombineStrategy.MAX,
                ),
                UKDonorVariableSpec(
                    recipient_column="property_income",
                    combine_strategy=UKDonorCombineStrategy.MAX,
                ),
            ),
        ),
        UKDonorBlockSpec(
            name="was_household_wealth",
            recipient_table="household",
            match_columns=(
                "region",
                "is_renting",
                "adult_count",
                "child_count",
                "household_income_band",
            ),
            donor_weight_column="household_weight",
            variables=(
                UKDonorVariableSpec("property_wealth", score_column="household_income_bhc"),
                UKDonorVariableSpec(
                    "corporate_wealth",
                    score_column="household_income_bhc",
                ),
                UKDonorVariableSpec(
                    "gross_financial_wealth",
                    score_column="household_income_bhc",
                ),
                UKDonorVariableSpec(
                    "net_financial_wealth",
                    score_column="household_income_bhc",
                ),
                UKDonorVariableSpec(
                    "main_residence_value",
                    score_column="household_income_bhc",
                ),
                UKDonorVariableSpec(
                    "other_residential_property_value",
                    score_column="household_income_bhc",
                ),
                UKDonorVariableSpec(
                    "non_residential_property_value",
                    score_column="household_income_bhc",
                ),
                UKDonorVariableSpec("savings", score_column="household_income_bhc"),
                UKDonorVariableSpec("num_vehicles", score_column="household_income_bhc"),
                UKDonorVariableSpec("net_wealth", score_column="household_income_bhc"),
            ),
            output_defaults=(
                UKDonorOutputDefaultSpec("property_purchased", False),
            ),
        ),
    )


def build_fused_uk_candidate_from_tables(
    *,
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    time_period: int,
    spi_tax_units: pd.DataFrame | None = None,
    was_households: pd.DataFrame | None = None,
    donor_block_specs: tuple[UKDonorBlockSpec, ...] | None = None,
    geography_provider: GeographyProvider | None = None,
    geography_assignment_plan: GeographyAssignmentPlan | None = None,
    geography_random_state: int | None = None,
    seed: int = 0,
) -> UKCandidateDataset:
    block_specs = donor_block_specs or default_uk_candidate_donor_block_specs()
    rng = np.random.default_rng(seed)

    candidate_person = person.copy()
    candidate_benunit = benunit.copy()
    candidate_household = household.copy()

    spi_donors = (
        _prepare_spi_donors(spi_tax_units) if spi_tax_units is not None else None
    )
    was_donors = (
        _prepare_was_donors(was_households) if was_households is not None else None
    )

    person_features = _build_person_features(candidate_person, candidate_household)
    for block_spec in block_specs:
        if block_spec.recipient_table != "person" or spi_donors is None:
            continue
        candidate_person = _apply_donor_block(
            candidate_person,
            recipient_features=person_features,
            donor_frame=spi_donors,
            block_spec=block_spec,
            rng=rng,
        )
        person_features = _build_person_features(candidate_person, candidate_household)

    household_features = _build_household_features(candidate_person, candidate_household)
    for block_spec in block_specs:
        if block_spec.recipient_table != "household" or was_donors is None:
            continue
        candidate_household = _apply_donor_block(
            candidate_household,
            recipient_features=household_features,
            donor_frame=was_donors,
            block_spec=block_spec,
            rng=rng,
        )
        candidate_household = _apply_block_output_defaults(
            candidate_household,
            block_spec=block_spec,
        )
        household_features = _build_household_features(candidate_person, candidate_household)

    if geography_provider is not None:
        candidate_person, candidate_benunit, candidate_household = apply_uk_candidate_geography(
            person=candidate_person,
            benunit=candidate_benunit,
            household=candidate_household,
            geography_provider=geography_provider,
            assignment_plan=geography_assignment_plan,
            random_state=seed if geography_random_state is None else geography_random_state,
        )

    return UKCandidateDataset(
        person=candidate_person,
        benunit=candidate_benunit,
        household=candidate_household,
        time_period=int(time_period),
        metadata={
            "donor_blocks": [block.name for block in block_specs],
            "used_spi": spi_tax_units is not None,
            "used_was": was_households is not None,
            "used_geography_provider": geography_provider is not None,
        },
    )


def build_fused_uk_candidate_dataset(
    *,
    frs_dataset_path: str | Path,
    spi_source_path: str | Path | None = None,
    was_source_path: str | Path | None = None,
    donor_block_specs: tuple[UKDonorBlockSpec, ...] | None = None,
    geography_provider: GeographyProvider | None = None,
    geography_assignment_plan: GeographyAssignmentPlan | None = None,
    geography_random_state: int | None = None,
    policy_period: int | None = None,
    seed: int = 0,
) -> UKCandidateDataset:
    frs_path = Path(frs_dataset_path)
    time_period = (
        int(policy_period) if policy_period is not None else _extract_period(frs_path)
    )
    person = _read_h5_table(frs_path, "person")
    benunit = _read_h5_table(frs_path, "benunit")
    household = _read_h5_table(frs_path, "household")

    spi_tax_units = None
    if spi_source_path is not None:
        spi_tax_units = UKSPISourceProvider(spi_source_path).load_frame().tables[
            EntityType.TAX_UNIT
        ]

    was_households = None
    if was_source_path is not None:
        was_households = UKWASSourceProvider(was_source_path).load_frame().tables[
            EntityType.HOUSEHOLD
        ]

    return build_fused_uk_candidate_from_tables(
        person=person,
        benunit=benunit,
        household=household,
        time_period=time_period,
        spi_tax_units=spi_tax_units,
        was_households=was_households,
        donor_block_specs=donor_block_specs,
        geography_provider=geography_provider,
        geography_assignment_plan=geography_assignment_plan,
        geography_random_state=geography_random_state,
        seed=seed,
    )


def build_and_benchmark_fused_uk_candidate(
    *,
    label: str,
    artifacts_dir: str | Path,
    frs_dataset_path: str | Path,
    baseline_dataset_path: str | Path,
    python_executable: str | Path,
    policyengine_uk_repo_dir: str | Path,
    policyengine_uk_data_repo_dir: str | Path,
    spi_source_path: str | Path | None = None,
    was_source_path: str | Path | None = None,
    donor_block_specs: tuple[UKDonorBlockSpec, ...] | None = None,
    geography_provider: GeographyProvider | None = None,
    geography_assignment_plan: GeographyAssignmentPlan | None = None,
    geography_random_state: int | None = None,
    policy_period: int | None = None,
    benchmark_mode: UKBenchmarkMode = UKBenchmarkMode.STANDARD,
    target_provider: PolicyEngineUKTargetProvider | None = None,
    target_query: TargetQuery | None = None,
    baseline_benchmark_result: (
        PolicyEngineUKBenchmarkResult | PolicyEngineUKDirectBenchmarkResult | None
    ) = None,
    seed: int = 0,
) -> UKCandidateBenchmarkArtifacts:
    effective_target_provider = target_provider or PolicyEngineUKTargetProvider(
        policyengine_uk_data_repo_dir,
        policyengine_uk_repo_dir=policyengine_uk_repo_dir,
    )
    dataset = build_fused_uk_candidate_dataset(
        frs_dataset_path=frs_dataset_path,
        spi_source_path=spi_source_path,
        was_source_path=was_source_path,
        donor_block_specs=donor_block_specs,
        geography_provider=geography_provider,
        geography_assignment_plan=geography_assignment_plan,
        geography_random_state=geography_random_state,
        policy_period=policy_period,
        seed=seed,
    )

    artifact_dir = Path(artifacts_dir) / label
    artifact_dir.mkdir(parents=True, exist_ok=True)
    candidate_dataset_path = dataset.save(
        artifact_dir / "candidate.h5",
        python_executable=python_executable,
    )
    benchmark_metadata = {
        "candidate_label": label,
        "sources": {
            "frs": str(Path(frs_dataset_path)),
            "spi": str(Path(spi_source_path)) if spi_source_path is not None else None,
            "was": str(Path(was_source_path)) if was_source_path is not None else None,
        },
        "donor_blocks": dataset.metadata.get("donor_blocks", []),
        "benchmark_mode": benchmark_mode.value,
    }
    if benchmark_mode is UKBenchmarkMode.DIRECT:
        comparison = compare_policyengine_uk_direct_benchmark(
            candidate_dataset_path=candidate_dataset_path,
            baseline_dataset_path=baseline_dataset_path,
            time_period=dataset.time_period,
            python_executable=python_executable,
            policyengine_uk_repo_dir=policyengine_uk_repo_dir,
            policyengine_uk_data_repo_dir=policyengine_uk_data_repo_dir,
            target_provider=effective_target_provider,
            target_query=target_query,
            baseline_result=baseline_benchmark_result,
            metadata=benchmark_metadata,
        )
    else:
        comparison = compare_policyengine_uk_benchmark(
            candidate_dataset_path=candidate_dataset_path,
            baseline_dataset_path=baseline_dataset_path,
            time_period=dataset.time_period,
            python_executable=python_executable,
            policyengine_uk_repo_dir=policyengine_uk_repo_dir,
            policyengine_uk_data_repo_dir=policyengine_uk_data_repo_dir,
            target_provider=effective_target_provider,
            baseline_result=baseline_benchmark_result,
            metadata=benchmark_metadata,
        )
    comparison_path = comparison.save(artifact_dir / "comparison.json")
    return UKCandidateBenchmarkArtifacts(
        label=label,
        candidate_dataset_path=candidate_dataset_path,
        comparison_path=comparison_path,
        comparison=comparison,
    )


def _prepare_spi_donors(spi_tax_units: pd.DataFrame) -> pd.DataFrame:
    donors = spi_tax_units.copy()
    donors["region"] = _normalize_region(
        pd.to_numeric(donors.get("region_code"), errors="coerce").map(
            UK_REGION_CODE_TO_NAME
        )
    )
    donors["gender"] = _normalize_gender(
        pd.to_numeric(donors.get("sex"), errors="coerce").map({1: "MALE", 2: "FEMALE"})
    )
    donors["age_band"] = (
        pd.to_numeric(donors.get("age_range_code"), errors="coerce")
        .fillna(0)
        .astype(int)
    )
    for column in (
        "weight",
        "dividend_income",
        "savings_interest_income",
        "property_income",
        "private_pension_income",
        "self_employment_income",
    ):
        if column in donors.columns:
            donors[column] = _numeric(donors[column])
    return donors


def _prepare_was_donors(was_households: pd.DataFrame) -> pd.DataFrame:
    donors = was_households.copy()
    donors["region"] = _normalize_region(
        pd.to_numeric(donors.get("region_code"), errors="coerce").map(
            WAS_REGION_CODE_TO_NAME
        )
    )
    donors["is_renting"] = _as_bool_flag(donors.get("is_renter_code"), positive_values={1})
    donors["adult_count"] = _household_size_band(donors.get("num_adults"))
    donors["child_count"] = _child_count_band(donors.get("num_children"))
    donors["household_income_bhc"] = _numeric(donors.get("household_income_bhc"))
    donors["household_income_band"] = _income_band(donors["household_income_bhc"])
    donors["corporate_wealth"] = (
        _numeric(donors.get("pensions"), index=donors.index)
        - _numeric(donors.get("db_pensions"), index=donors.index)
        + _numeric(donors.get("emp_shares_options"), index=donors.index)
        + _numeric(donors.get("uk_shares"), index=donors.index)
        + _numeric(donors.get("investment_isas"), index=donors.index)
        + _numeric(donors.get("unit_investment_trusts"), index=donors.index)
    )
    for column in (
        "household_weight",
        "property_wealth",
        "gross_financial_wealth",
        "net_financial_wealth",
        "main_residence_value",
        "other_residential_property_value",
        "non_residential_property_value",
        "savings",
        "num_vehicles",
        "net_wealth",
        "mortgage_liabilities",
        "corporate_wealth",
    ):
        if column in donors.columns:
            donors[column] = _numeric(donors[column])
    return donors


def _build_person_features(
    person: pd.DataFrame,
    household: pd.DataFrame,
) -> pd.DataFrame:
    features = pd.DataFrame(index=person.index)
    household_region = household.set_index("household_id")["region"]
    features["region"] = _normalize_region(
        person["person_household_id"].map(household_region)
    )
    features["gender"] = _normalize_gender(person["gender"])
    features["age_band"] = _age_to_spi_age_band(person["age"])
    features["is_spi_eligible"] = _numeric(person["age"]) >= 16
    return features


def _build_household_features(
    person: pd.DataFrame,
    household: pd.DataFrame,
) -> pd.DataFrame:
    person_working = person.copy()
    person_working["age"] = _numeric(person_working["age"])
    for column in PERSON_INCOME_COLUMNS:
        if column not in person_working.columns:
            person_working[column] = 0.0
        else:
            person_working[column] = _numeric(person_working[column])
    person_working["is_adult"] = person_working["age"] >= 18
    person_working["is_child"] = person_working["age"] < 18
    person_working["person_income_total"] = person_working[list(PERSON_INCOME_COLUMNS)].sum(
        axis=1
    )
    aggregates = (
        person_working.groupby("person_household_id", dropna=False)
        .agg(
            num_adults=("is_adult", "sum"),
            num_children=("is_child", "sum"),
            household_income_bhc=("person_income_total", "sum"),
        )
        .rename_axis("household_id")
        .reset_index()
    )
    features = household.copy().merge(
        aggregates,
        on="household_id",
        how="left",
    )
    features["num_adults"] = _numeric(features.get("num_adults")).fillna(0.0)
    features["num_children"] = _numeric(features.get("num_children")).fillna(0.0)
    features["household_income_bhc"] = _numeric(
        features.get("household_income_bhc")
    ).fillna(0.0)
    features["region"] = _normalize_region(features["region"]).replace(
        {"NORTHERN_IRELAND": "WALES"}
    )
    features["is_renting"] = features["tenure_type"].astype(str).str.startswith("RENT")
    features["adult_count"] = _household_size_band(features["num_adults"])
    features["child_count"] = _child_count_band(features["num_children"])
    features["household_income_band"] = _income_band(features["household_income_bhc"])
    return features.set_index(household.index)


def _apply_donor_block(
    recipient_table: pd.DataFrame,
    *,
    recipient_features: pd.DataFrame,
    donor_frame: pd.DataFrame,
    block_spec: UKDonorBlockSpec,
    rng: np.random.Generator,
) -> pd.DataFrame:
    updated = recipient_table.copy()
    mask = pd.Series(True, index=recipient_features.index)
    if block_spec.recipient_mask_column is not None:
        mask = recipient_features[block_spec.recipient_mask_column].astype(bool)
    if not mask.any():
        return updated

    eligible_features = recipient_features.loc[mask].copy()
    match_levels = [block_spec.match_columns[:count] for count in range(len(block_spec.match_columns), 0, -1)]
    match_levels.append(tuple())
    recipient_key_cache = {
        level: _group_keys(eligible_features, level)
        for level in match_levels
    }
    donor_key_cache = {
        level: _group_keys(donor_frame, level)
        for level in match_levels
    }
    donor_group_cache = {
        level: donor_frame.groupby(donor_key_cache[level], dropna=False).groups
        for level in match_levels
    }
    for variable_spec in block_spec.variables:
        donor_column = variable_spec.resolved_donor_column
        if donor_column not in donor_frame.columns:
            continue
        score_column = variable_spec.score_column or variable_spec.recipient_column
        if score_column in updated.columns:
            scores = updated.loc[mask, score_column]
        else:
            scores = pd.Series(0.0, index=eligible_features.index, dtype=float)
        matched = _groupwise_rank_match(
            recipient_features=eligible_features,
            donor_frame=donor_frame,
            match_columns=block_spec.match_columns,
            donor_value_column=donor_column,
            donor_weight_column=block_spec.donor_weight_column,
            score_series=_numeric(scores).reindex(eligible_features.index).fillna(0.0),
            rng=rng,
            fill_value=variable_spec.fill_value,
            match_levels=match_levels,
            recipient_key_cache=recipient_key_cache,
            donor_group_cache=donor_group_cache,
        )

        if variable_spec.recipient_column in updated.columns:
            current_values = _numeric(updated.loc[mask, variable_spec.recipient_column]).fillna(
                variable_spec.fill_value
            )
        else:
            current_values = pd.Series(
                variable_spec.fill_value,
                index=eligible_features.index,
                dtype=float,
            )

        if variable_spec.combine_strategy is UKDonorCombineStrategy.MAX:
            combined = np.maximum(current_values.to_numpy(dtype=float), matched.to_numpy(dtype=float))
            updated.loc[mask, variable_spec.recipient_column] = combined
        else:
            updated.loc[mask, variable_spec.recipient_column] = matched.to_numpy(
                dtype=float
            )
    return updated


def _apply_block_output_defaults(
    recipient_table: pd.DataFrame,
    *,
    block_spec: UKDonorBlockSpec,
) -> pd.DataFrame:
    if not block_spec.output_defaults:
        return recipient_table
    updated = recipient_table.copy()
    for output_default in block_spec.output_defaults:
        column = output_default.column
        value = output_default.value
        if column not in updated.columns:
            updated[column] = pd.Series(value, index=updated.index)
            continue
        updated[column] = updated[column].where(updated[column].notna(), value)
    return updated


def _groupwise_rank_match(
    *,
    recipient_features: pd.DataFrame,
    donor_frame: pd.DataFrame,
    match_columns: tuple[str, ...],
    donor_value_column: str,
    donor_weight_column: str,
    score_series: pd.Series,
    rng: np.random.Generator,
    fill_value: float,
    match_levels: list[tuple[str, ...]],
    recipient_key_cache: dict[tuple[str, ...], pd.Series],
    donor_group_cache: dict[tuple[str, ...], dict[Any, Any]],
) -> pd.Series:
    donor_values = _numeric(donor_frame[donor_value_column]).replace([np.inf, -np.inf], np.nan)
    donor_weights = _numeric(donor_frame.get(donor_weight_column)).fillna(0.0)
    valid = donor_values.notna()
    donor_frame = donor_frame.loc[valid].copy()
    donor_values = donor_values.loc[valid]
    donor_weights = donor_weights.loc[valid]
    if donor_frame.empty:
        return pd.Series(fill_value, index=recipient_features.index, dtype=float)

    result = pd.Series(np.nan, index=recipient_features.index, dtype=float)
    for level in match_levels:
        unresolved = result.isna()
        if not unresolved.any():
            break
        recipient_subset = recipient_features.loc[unresolved]
        recipient_keys = recipient_key_cache[level].loc[recipient_subset.index]
        donor_groups = donor_group_cache[level]
        for group_key, recipient_index in recipient_keys.groupby(recipient_keys).groups.items():
            donor_index = donor_groups.get(group_key)
            if donor_index is None:
                continue
            donor_group_values = donor_values.loc[donor_index]
            donor_group_weights = donor_weights.loc[donor_index]
            result.loc[recipient_index] = _rank_match_sample(
                size=len(recipient_index),
                donor_values=donor_group_values.to_numpy(dtype=float),
                donor_weights=donor_group_weights.to_numpy(dtype=float),
                recipient_scores=score_series.loc[recipient_index].to_numpy(dtype=float),
                rng=rng,
            )

    return result.fillna(fill_value)


def _group_keys(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    if not columns:
        return pd.Series([("__all__",)] * len(frame), index=frame.index, dtype=object)
    normalized = pd.DataFrame(index=frame.index)
    for column in columns:
        series = frame[column]
        if pd.api.types.is_bool_dtype(series):
            normalized[column] = series.fillna(False).astype(bool).astype(str)
        elif pd.api.types.is_numeric_dtype(series):
            normalized[column] = pd.to_numeric(series, errors="coerce").fillna(-1).astype(int).astype(str)
        else:
            normalized[column] = series.astype(str).fillna("UNKNOWN")
    key_parts = normalized[columns[0]].astype(str)
    for column in columns[1:]:
        key_parts = key_parts + "|" + normalized[column].astype(str)
    return key_parts


def _rank_match_sample(
    *,
    size: int,
    donor_values: np.ndarray,
    donor_weights: np.ndarray,
    recipient_scores: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if size == 0 or len(donor_values) == 0:
        return np.zeros(size, dtype=float)
    probabilities = None
    clipped_weights = np.clip(donor_weights.astype(float), a_min=0.0, a_max=None)
    weight_sum = float(clipped_weights.sum())
    if weight_sum > 0.0 and len(clipped_weights) == len(donor_values):
        probabilities = clipped_weights / weight_sum
    sampled = rng.choice(
        donor_values.astype(float),
        size=size,
        replace=True,
        p=probabilities,
    )
    sampled.sort()
    clean_scores = np.asarray(recipient_scores, dtype=float)
    if np.isnan(clean_scores).all():
        clean_scores = np.zeros_like(clean_scores)
    else:
        clean_scores = np.nan_to_num(clean_scores, nan=float(np.nanmedian(clean_scores)))
    order = np.argsort(clean_scores, kind="mergesort")
    matched = np.empty(size, dtype=float)
    matched[order] = sampled
    return matched


def _normalize_region(series: pd.Series | Any) -> pd.Series:
    values = pd.Series(series) if series is not None else pd.Series(dtype=object)
    return (
        values.astype(str)
        .str.strip()
        .str.upper()
        .replace({"NAN": "UNKNOWN", "NONE": "UNKNOWN"})
        .fillna("UNKNOWN")
    )


def _normalize_gender(series: pd.Series | Any) -> pd.Series:
    values = pd.Series(series) if series is not None else pd.Series(dtype=object)
    return (
        values.astype(str)
        .str.strip()
        .str.upper()
        .replace({"NAN": "UNKNOWN", "NONE": "UNKNOWN", "0": "UNKNOWN"})
        .fillna("UNKNOWN")
    )


def _numeric(series: pd.Series | Any, *, index: pd.Index | None = None) -> pd.Series:
    if series is None:
        return pd.Series(0.0, index=index, dtype=float)
    return pd.to_numeric(pd.Series(series), errors="coerce").fillna(0.0)


def _age_to_spi_age_band(series: pd.Series | Any) -> pd.Series:
    ages = _numeric(series)
    bands = np.zeros(len(ages), dtype=int)
    for band_code, lower, upper in SPI_AGE_BAND_BOUNDS:
        mask = (ages >= lower) & (ages < upper)
        bands[mask.to_numpy()] = band_code
    return pd.Series(bands, index=ages.index, dtype=int)


def _income_band(series: pd.Series | Any) -> pd.Series:
    values = _numeric(series)
    bins = [-np.inf, 20_000, 40_000, 60_000, 100_000, 150_000, np.inf]
    labels = ["lt20k", "20k_40k", "40k_60k", "60k_100k", "100k_150k", "150k_plus"]
    return pd.cut(values, bins=bins, labels=labels, include_lowest=True).astype(str)


def _household_size_band(series: pd.Series | Any) -> pd.Series:
    values = _numeric(series).round().astype(int)
    return values.clip(lower=0, upper=3).map(
        {0: "0", 1: "1", 2: "2", 3: "3plus"}
    )


def _child_count_band(series: pd.Series | Any) -> pd.Series:
    values = _numeric(series).round().astype(int)
    return values.clip(lower=0, upper=2).map(
        {0: "0", 1: "1", 2: "2plus"}
    )


def _as_bool_flag(series: pd.Series | Any, *, positive_values: set[int]) -> pd.Series:
    values = pd.to_numeric(pd.Series(series), errors="coerce").fillna(0).astype(int)
    return values.isin(positive_values)


def _resolve_hdf_writer_python(preferred: str | Path) -> str:
    candidates = [
        str(preferred),
        sys.executable,
        "/opt/homebrew/bin/python3",
        "/usr/bin/python3",
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        resolved = shutil.which(candidate) if "/" not in candidate else candidate
        if resolved is None or ("/" in candidate and not Path(candidate).exists()):
            continue
        try:
            completed = subprocess.run(
                [
                    resolved,
                    "-c",
                    "import pandas; import tables",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            continue
        if completed.returncode == 0:
            return resolved
    raise RuntimeError(
        "Could not find a Python executable with pandas and tables installed for UK candidate HDF export"
    )
