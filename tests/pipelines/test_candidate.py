from __future__ import annotations

import json
from pathlib import Path

import h5py
import pandas as pd

from microplex_uk.data_sources.frs import _read_h5_table
from microplex_uk.geography import build_static_uk_geography_provider
from microplex_uk.pipelines import (
    UKBenchmarkMode,
    UKCandidateDataset,
    build_and_benchmark_fused_uk_candidate,
    build_fused_uk_candidate_dataset,
    build_fused_uk_candidate_from_tables,
    default_uk_candidate_donor_block_specs,
)
from microplex_uk.policyengine import PolicyEngineUKBenchmarkResult


def test_build_fused_uk_candidate_from_tables_enriches_person_and_household_values():
    person = pd.DataFrame(
        {
            "person_id": [1, 2],
            "person_benunit_id": [10, 20],
            "person_household_id": [100, 200],
            "age": [58, 40],
            "gender": ["MALE", "FEMALE"],
            "dividend_income": [10.0, 5.0],
            "savings_interest_income": [1.0, 0.0],
            "property_income": [0.0, 0.0],
            "private_pension_income": [50.0, 10.0],
            "self_employment_income": [0.0, 0.0],
            "employment_income": [40_000.0, 25_000.0],
            "miscellaneous_income": [0.0, 0.0],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [10, 20], "is_married": [1, 0]})
    household = pd.DataFrame(
        {
            "household_id": [100, 200],
            "household_weight": [1.5, 2.0],
            "region": ["LONDON", "SCOTLAND"],
            "tenure_type": ["OWNED_OUTRIGHT", "RENT_FROM_COUNCIL"],
            "council_tax": [1500.0, 900.0],
        }
    )
    spi_tax_units = pd.DataFrame(
        {
            "weight": [10.0, 20.0],
            "dividend_income": [100.0, 80.0],
            "savings_interest_income": [20.0, 15.0],
            "property_income": [10.0, 5.0],
            "private_pension_income": [200.0, 150.0],
            "self_employment_income": [0.0, 0.0],
            "region_code": [7, 11],
            "sex": [1, 2],
            "age_range_code": [5, 3],
        }
    )
    was_households = pd.DataFrame(
        {
            "household_weight": [15.0, 25.0],
            "region_code": [8, 12],
            "is_renter_code": [2, 1],
            "num_adults": [2, 1],
            "num_children": [0, 0],
            "household_income_bhc": [40_000.0, 25_000.0],
            "property_wealth": [250_000.0, 120_000.0],
            "gross_financial_wealth": [40_000.0, 20_000.0],
            "net_financial_wealth": [35_000.0, 18_000.0],
            "main_residence_value": [300_000.0, 150_000.0],
            "other_residential_property_value": [20_000.0, 0.0],
            "non_residential_property_value": [0.0, 0.0],
            "savings": [15_000.0, 9_000.0],
            "num_vehicles": [1.0, 1.0],
            "net_wealth": [320_000.0, 140_000.0],
            "pensions": [30_000.0, 15_000.0],
            "db_pensions": [10_000.0, 5_000.0],
            "emp_shares_options": [2_000.0, 500.0],
            "uk_shares": [4_000.0, 1_000.0],
            "investment_isas": [8_000.0, 2_000.0],
            "unit_investment_trusts": [1_000.0, 500.0],
        }
    )

    candidate = build_fused_uk_candidate_from_tables(
        person=person,
        benunit=benunit,
        household=household,
        time_period=2024,
        spi_tax_units=spi_tax_units,
        was_households=was_households,
        seed=7,
    )

    assert candidate.person["dividend_income"].tolist() == [100.0, 80.0]
    assert candidate.person["private_pension_income"].tolist() == [50.0, 10.0]
    assert candidate.person["self_employment_income"].tolist() == [0.0, 0.0]
    assert "property_wealth" in candidate.household.columns
    assert sorted(candidate.household["property_wealth"].tolist()) == [120_000.0, 250_000.0]
    assert sorted(candidate.household["corporate_wealth"].tolist()) == [14_000.0, 35_000.0]
    assert candidate.household["property_purchased"].tolist() == [False, False]


def test_build_fused_uk_candidate_from_tables_preserves_observed_purchase_flow():
    person = pd.DataFrame(
        {
            "person_id": [1],
            "person_benunit_id": [10],
            "person_household_id": [100],
            "age": [58],
            "gender": ["MALE"],
            "dividend_income": [10.0],
            "savings_interest_income": [1.0],
            "property_income": [0.0],
            "private_pension_income": [50.0],
            "self_employment_income": [0.0],
            "employment_income": [40_000.0],
            "miscellaneous_income": [0.0],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [10], "is_married": [1]})
    household = pd.DataFrame(
        {
            "household_id": [100],
            "household_weight": [1.5],
            "region": ["LONDON"],
            "tenure_type": ["OWNED_OUTRIGHT"],
            "council_tax": [1500.0],
            "property_purchased": [True],
        }
    )
    was_households = pd.DataFrame(
        {
            "household_weight": [15.0],
            "region_code": [8],
            "is_renter_code": [2],
            "num_adults": [2],
            "num_children": [0],
            "household_income_bhc": [40_000.0],
            "property_wealth": [250_000.0],
            "gross_financial_wealth": [40_000.0],
            "net_financial_wealth": [35_000.0],
            "main_residence_value": [300_000.0],
            "other_residential_property_value": [20_000.0],
            "non_residential_property_value": [0.0],
            "savings": [15_000.0],
            "num_vehicles": [1.0],
            "net_wealth": [320_000.0],
            "pensions": [30_000.0],
            "db_pensions": [10_000.0],
            "emp_shares_options": [2_000.0],
            "uk_shares": [4_000.0],
            "investment_isas": [8_000.0],
            "unit_investment_trusts": [1_000.0],
        }
    )

    candidate = build_fused_uk_candidate_from_tables(
        person=person,
        benunit=benunit,
        household=household,
        time_period=2024,
        was_households=was_households,
        seed=7,
    )

    assert candidate.household["property_purchased"].tolist() == [True]


def test_default_uk_candidate_donor_block_specs_keep_spi_to_capital_income():
    block_specs = default_uk_candidate_donor_block_specs()
    spi_block = next(block for block in block_specs if block.name == "spi_capital_income")

    assert {variable.recipient_column for variable in spi_block.variables} == {
        "dividend_income",
        "savings_interest_income",
        "property_income",
    }


def test_build_fused_uk_candidate_from_tables_assigns_atomic_uk_geography():
    person = pd.DataFrame(
        {
            "person_id": [1, 2],
            "person_benunit_id": [10, 20],
            "person_household_id": [100, 200],
            "age": [30, 40],
            "gender": ["MALE", "FEMALE"],
            "dividend_income": [0.0, 0.0],
            "savings_interest_income": [0.0, 0.0],
            "property_income": [0.0, 0.0],
            "private_pension_income": [0.0, 0.0],
            "self_employment_income": [0.0, 0.0],
            "employment_income": [30_000.0, 25_000.0],
            "miscellaneous_income": [0.0, 0.0],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [10, 20], "is_married": [0, 0]})
    household = pd.DataFrame(
        {
            "household_id": [100, 200],
            "household_weight": [1.0, 1.0],
            "region": ["LONDON", "SCOTLAND"],
            "tenure_type": ["OWNED_OUTRIGHT", "RENT_FROM_COUNCIL"],
            "council_tax": [1500.0, 900.0],
        }
    )
    geography_provider = build_static_uk_geography_provider(
        pd.DataFrame(
            {
                "oa_code": ["OA100", "OA200"],
                "region": ["LONDON", "SCOTLAND"],
                "country": ["ENGLAND", "SCOTLAND"],
                "lsoa": ["LSOA100", "LSOA200"],
                "local_authority": ["E09000001", "S12000033"],
                "constituency": ["E14000530", "S14000024"],
                "assignment_probability": [1.0, 1.0],
            }
        )
    )

    candidate = build_fused_uk_candidate_from_tables(
        person=person,
        benunit=benunit,
        household=household,
        time_period=2024,
        geography_provider=geography_provider,
        seed=7,
    )

    assert candidate.household["oa_code"].tolist() == ["OA100", "OA200"]
    assert candidate.household["local_authority"].tolist() == [
        "E09000001",
        "S12000033",
    ]
    assert candidate.person["oa_code"].tolist() == ["OA100", "OA200"]
    assert candidate.person["country"].tolist() == ["ENGLAND", "SCOTLAND"]
    assert candidate.benunit["constituency"].tolist() == ["E14000530", "S14000024"]
    assert candidate.metadata["used_geography_provider"] is True


def test_uk_candidate_dataset_save_writes_policyengine_readable_h5(tmp_path: Path):
    dataset = UKCandidateDataset(
        person=pd.DataFrame({"person_id": [1], "person_household_id": [10]}),
        benunit=pd.DataFrame({"benunit_id": [10]}),
        household=pd.DataFrame({"household_id": [10], "household_weight": [1.0]}),
        time_period=2024,
    )

    output_path = dataset.save(tmp_path / "candidate.h5")

    with h5py.File(output_path, mode="r") as handle:
        assert set(handle.keys()) == {"person", "benunit", "household", "time_period"}


def test_uk_candidate_dataset_save_preserves_table_values(tmp_path: Path):
    dataset = UKCandidateDataset(
        person=pd.DataFrame(
            {
                "person_id": pd.Series([1, 2], dtype="int64"),
                "person_household_id": pd.Series([10, 10], dtype="int64"),
                "employment_income": pd.Series([1234.56789, 0.0], dtype="float64"),
                "is_household_head": pd.Series([1, 0], dtype="uint8"),
            }
        ),
        benunit=pd.DataFrame({"benunit_id": pd.Series([10], dtype="int64")}),
        household=pd.DataFrame(
            {
                "household_id": pd.Series([10], dtype="int64"),
                "household_weight": pd.Series([1.25], dtype="float64"),
                "household_owns_tv": pd.Series([1], dtype="uint8"),
            }
        ),
        time_period=2024,
    )

    output_path = dataset.save(tmp_path / "candidate.h5")

    assert _read_h5_table(output_path, "person").equals(dataset.person)
    assert _read_h5_table(output_path, "household").equals(dataset.household)


def test_build_and_benchmark_fused_uk_candidate_saves_artifacts(
    monkeypatch,
    tmp_path: Path,
):
    dataset = UKCandidateDataset(
        person=pd.DataFrame({"person_id": [1], "person_household_id": [10]}),
        benunit=pd.DataFrame({"benunit_id": [10]}),
        household=pd.DataFrame({"household_id": [10], "household_weight": [1.0]}),
        time_period=2024,
    )

    monkeypatch.setattr(
        "microplex_uk.pipelines.candidate.build_fused_uk_candidate_dataset",
        lambda **kwargs: dataset,
    )

    class _FakeComparison:
        def __init__(self):
            self.mean_abs_relative_error_delta = -0.1

        def save(self, path: Path) -> Path:
            path.write_text(json.dumps({"status": "ok"}))
            return path

    class _FakeTargetProvider:
        pass

    monkeypatch.setattr(
        "microplex_uk.pipelines.candidate.compare_policyengine_uk_benchmark",
        lambda **kwargs: _FakeComparison(),
    )

    artifacts = build_and_benchmark_fused_uk_candidate(
        label="mini_candidate",
        artifacts_dir=tmp_path,
        frs_dataset_path="/tmp/frs.h5",
        baseline_dataset_path="/tmp/baseline.h5",
        python_executable="/tmp/python",
        policyengine_uk_repo_dir="/tmp/policyengine-uk",
        policyengine_uk_data_repo_dir="/tmp/policyengine-uk-data",
        target_provider=_FakeTargetProvider(),
    )

    assert artifacts.candidate_dataset_path.exists()
    assert json.loads(artifacts.comparison_path.read_text())["status"] == "ok"


def test_build_and_benchmark_fused_uk_candidate_forwards_baseline_result(
    monkeypatch,
    tmp_path: Path,
):
    class _FakeTargetProvider:
        pass

    dataset = UKCandidateDataset(
        person=pd.DataFrame({"person_id": [1], "person_household_id": [10]}),
        benunit=pd.DataFrame({"benunit_id": [10]}),
        household=pd.DataFrame({"household_id": [10], "household_weight": [1.0]}),
        time_period=2024,
    )

    monkeypatch.setattr(
        "microplex_uk.pipelines.candidate.build_fused_uk_candidate_dataset",
        lambda **kwargs: dataset,
    )

    baseline_result = PolicyEngineUKBenchmarkResult(
        dataset_path="/tmp/baseline.h5",
        time_period=2024,
        target_count=0,
        mean_abs_relative_error=0.2,
        max_abs_relative_error=0.2,
        metrics=[],
    )

    def fake_compare(**kwargs):
        assert kwargs["baseline_result"] is baseline_result

        class _FakeComparison:
            def __init__(self):
                self.mean_abs_relative_error_delta = -0.1

            def save(self, path: Path) -> Path:
                path.write_text(json.dumps({"status": "ok"}))
                return path

        return _FakeComparison()

    monkeypatch.setattr(
        "microplex_uk.pipelines.candidate.compare_policyengine_uk_benchmark",
        fake_compare,
    )

    artifacts = build_and_benchmark_fused_uk_candidate(
        label="mini_candidate",
        artifacts_dir=tmp_path,
        frs_dataset_path="/tmp/frs.h5",
        baseline_dataset_path="/tmp/baseline.h5",
        python_executable="/tmp/python",
        policyengine_uk_repo_dir="/tmp/policyengine-uk",
        policyengine_uk_data_repo_dir="/tmp/policyengine-uk-data",
        target_provider=_FakeTargetProvider(),
        baseline_benchmark_result=baseline_result,
    )

    assert artifacts.candidate_dataset_path.exists()


def test_build_and_benchmark_fused_uk_candidate_direct_mode_uses_direct_harness(
    monkeypatch,
    tmp_path: Path,
):
    class _FakeTargetProvider:
        pass

    dataset = UKCandidateDataset(
        person=pd.DataFrame({"person_id": [1], "person_household_id": [10]}),
        benunit=pd.DataFrame({"benunit_id": [10]}),
        household=pd.DataFrame({"household_id": [10], "household_weight": [1.0]}),
        time_period=2024,
    )

    monkeypatch.setattr(
        "microplex_uk.pipelines.candidate.build_fused_uk_candidate_dataset",
        lambda **kwargs: dataset,
    )

    target_query = object()
    baseline_result = object()

    def fake_compare(**kwargs):
        assert kwargs["target_query"] is target_query
        assert kwargs["baseline_result"] is baseline_result

        class _FakeComparison:
            def __init__(self):
                self.mean_abs_relative_error_delta = -0.1

            def save(self, path: Path) -> Path:
                path.write_text(json.dumps({"status": "ok"}))
                return path

        return _FakeComparison()

    monkeypatch.setattr(
        "microplex_uk.pipelines.candidate.compare_policyengine_uk_direct_benchmark",
        fake_compare,
    )

    artifacts = build_and_benchmark_fused_uk_candidate(
        label="mini_candidate",
        artifacts_dir=tmp_path,
        frs_dataset_path="/tmp/frs.h5",
        baseline_dataset_path="/tmp/baseline.h5",
        python_executable="/tmp/python",
        policyengine_uk_repo_dir="/tmp/policyengine-uk",
        policyengine_uk_data_repo_dir="/tmp/policyengine-uk-data",
        target_provider=_FakeTargetProvider(),
        benchmark_mode=UKBenchmarkMode.DIRECT,
        target_query=target_query,
        baseline_benchmark_result=baseline_result,
    )

    assert artifacts.candidate_dataset_path.exists()
    assert json.loads(artifacts.comparison_path.read_text())["status"] == "ok"


def test_build_fused_uk_candidate_dataset_allows_policy_period_override(
    monkeypatch,
):
    monkeypatch.setattr(
        "microplex_uk.pipelines.candidate._extract_period",
        lambda _: 2023,
    )
    monkeypatch.setattr(
        "microplex_uk.pipelines.candidate._read_h5_table",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "microplex_uk.pipelines.candidate.build_fused_uk_candidate_from_tables",
        lambda **kwargs: UKCandidateDataset(
            person=pd.DataFrame(),
            benunit=pd.DataFrame(),
            household=pd.DataFrame(),
            time_period=kwargs["time_period"],
        ),
    )

    dataset = build_fused_uk_candidate_dataset(
        frs_dataset_path="/tmp/frs.h5",
        policy_period=2024,
    )

    assert dataset.time_period == 2024
