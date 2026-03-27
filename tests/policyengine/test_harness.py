from __future__ import annotations

import json
from pathlib import Path

import pytest
from microplex.core import EntityType
from microplex.targets import TargetAggregation, TargetQuery, TargetSet, TargetSpec

from microplex_uk.policyengine import (
    PolicyEngineUKBenchmarkComparison,
    PolicyEngineUKBenchmarkResult,
    PolicyEngineUKDirectBenchmarkComparison,
    PolicyEngineUKDirectBenchmarkResult,
    compare_policyengine_uk_benchmark,
    compare_policyengine_uk_direct_benchmark,
    run_policyengine_uk_direct_loss,
    run_policyengine_uk_loss,
)


class _FakeTargetProvider:
    def load_target_set(self, query: TargetQuery | None = None) -> TargetSet:
        assert query is None or query.period == 2024
        return TargetSet(
            [
                TargetSpec(
                    name="income_tax",
                    entity=EntityType.HOUSEHOLD,
                    value=100.0,
                    period=2024,
                    measure="income_tax",
                    aggregation=TargetAggregation.SUM,
                    source="hmrc",
                    metadata={
                        "variable": "income_tax",
                        "geographic_level": "national",
                        "is_count": False,
                    },
                ),
                TargetSpec(
                    name="population",
                    entity=EntityType.HOUSEHOLD,
                    value=200.0,
                    period=2024,
                    aggregation=TargetAggregation.COUNT,
                    source="ons",
                    metadata={
                        "variable": "people",
                        "geographic_level": "national",
                        "is_count": True,
                    },
                ),
            ]
        )


def test_run_policyengine_uk_loss_maps_payload(monkeypatch):
    monkeypatch.setattr(
        "microplex_uk.policyengine.harness._run_policyengine_uk_loss_payload",
        lambda **kwargs: {
            "time_period": 2024,
            "target_count": 2,
            "mean_abs_relative_error": 0.15,
            "max_abs_relative_error": 0.25,
            "metrics": [
                {
                    "name": "income_tax",
                    "estimate": 110.0,
                    "target": 100.0,
                    "error": 10.0,
                    "abs_error": 10.0,
                    "rel_error": 0.1,
                    "abs_rel_error": 0.1,
                },
                {
                    "name": "population",
                    "estimate": 190.0,
                    "target": 200.0,
                    "error": -10.0,
                    "abs_error": 10.0,
                    "rel_error": -0.05,
                    "abs_rel_error": 0.05,
                },
            ],
        },
    )

    result = run_policyengine_uk_loss(
        "/tmp/candidate.h5",
        time_period=2024,
        python_executable="/tmp/python",
        policyengine_uk_repo_dir="/tmp/policyengine-uk",
        policyengine_uk_data_repo_dir="/tmp/policyengine-uk-data",
        target_provider=_FakeTargetProvider(),
    )

    assert result.dataset_path == "/tmp/candidate.h5"
    assert result.target_count == 2
    assert result.mean_abs_relative_error == 0.15
    assert result.metrics[0].name == "income_tax"
    assert result.metrics[0].metadata["source"] == "hmrc"


def test_compare_policyengine_uk_benchmark_and_save(monkeypatch, tmp_path: Path):
    payloads = iter(
        [
            {
                "time_period": 2024,
                "target_count": 2,
                "mean_abs_relative_error": 0.10,
                "max_abs_relative_error": 0.20,
                "metrics": [
                    {
                        "name": "income_tax",
                        "estimate": 102.0,
                        "target": 100.0,
                        "error": 2.0,
                        "abs_error": 2.0,
                        "rel_error": 0.02,
                        "abs_rel_error": 0.02,
                    },
                    {
                        "name": "population",
                        "estimate": 190.0,
                        "target": 200.0,
                        "error": -10.0,
                        "abs_error": 10.0,
                        "rel_error": -0.05,
                        "abs_rel_error": 0.05,
                    },
                ],
            },
            {
                "time_period": 2024,
                "target_count": 2,
                "mean_abs_relative_error": 0.15,
                "max_abs_relative_error": 0.30,
                "metrics": [
                    {
                        "name": "income_tax",
                        "estimate": 110.0,
                        "target": 100.0,
                        "error": 10.0,
                        "abs_error": 10.0,
                        "rel_error": 0.10,
                        "abs_rel_error": 0.10,
                    },
                    {
                        "name": "population",
                        "estimate": 198.0,
                        "target": 200.0,
                        "error": -2.0,
                        "abs_error": 2.0,
                        "rel_error": -0.01,
                        "abs_rel_error": 0.01,
                    },
                ],
            },
        ]
    )
    monkeypatch.setattr(
        "microplex_uk.policyengine.harness._run_policyengine_uk_loss_payload",
        lambda **kwargs: next(payloads),
    )

    comparison = compare_policyengine_uk_benchmark(
        candidate_dataset_path="/tmp/candidate.h5",
        baseline_dataset_path="/tmp/baseline.h5",
        time_period=2024,
        python_executable="/tmp/python",
        policyengine_uk_repo_dir="/tmp/policyengine-uk",
        policyengine_uk_data_repo_dir="/tmp/policyengine-uk-data",
        target_provider=_FakeTargetProvider(),
        metadata={"suite": "uk_spike"},
    )

    assert isinstance(comparison, PolicyEngineUKBenchmarkComparison)
    assert comparison.common_target_count == 2
    assert comparison.target_win_rate == 0.5
    assert comparison.mean_abs_relative_error_delta == pytest.approx(-0.05)
    assert "source" in comparison.grouped_summaries
    assert comparison.grouped_summaries["source"][0].group_field == "source"

    output_path = comparison.save(tmp_path / "comparison.json")
    saved = json.loads(output_path.read_text())
    assert saved["metadata"]["suite"] == "uk_spike"
    assert saved["candidate"]["mean_abs_relative_error"] == 0.10
    assert saved["grouped_summaries"]["source"][0]["group_field"] == "source"


def test_compare_policyengine_uk_benchmark_reuses_provided_baseline(monkeypatch):
    calls: list[str] = []

    def fake_run(dataset_path, **kwargs):
        calls.append(str(dataset_path))
        return PolicyEngineUKBenchmarkResult(
            dataset_path=str(dataset_path),
            time_period=2024,
            target_count=1,
            mean_abs_relative_error=0.1,
            max_abs_relative_error=0.1,
            metrics=[],
        )

    monkeypatch.setattr(
        "microplex_uk.policyengine.harness.run_policyengine_uk_loss",
        fake_run,
    )

    baseline = PolicyEngineUKBenchmarkResult(
        dataset_path="/tmp/baseline.h5",
        time_period=2024,
        target_count=1,
        mean_abs_relative_error=0.2,
        max_abs_relative_error=0.2,
        metrics=[],
    )

    comparison = compare_policyengine_uk_benchmark(
        candidate_dataset_path="/tmp/candidate.h5",
        baseline_dataset_path="/tmp/baseline.h5",
        time_period=2024,
        python_executable="/tmp/python",
        policyengine_uk_repo_dir="/tmp/policyengine-uk",
        policyengine_uk_data_repo_dir="/tmp/policyengine-uk-data",
        target_provider=_FakeTargetProvider(),
        baseline_result=baseline,
    )

    assert comparison.baseline is baseline
    assert calls == ["/tmp/candidate.h5"]


def test_run_policyengine_uk_direct_loss_maps_supported_and_unsupported(monkeypatch):
    monkeypatch.setattr(
        "microplex_uk.policyengine.harness._run_policyengine_uk_direct_loss_payload",
        lambda **kwargs: {
            "time_period": 2024,
            "target_count": 3,
            "supported_target_count": 2,
            "unsupported_target_count": 1,
            "mean_abs_relative_error": 0.2,
            "max_abs_relative_error": 0.4,
            "metrics": [
                {
                    "name": "income_tax",
                    "estimate": 110.0,
                    "target": 100.0,
                    "error": 10.0,
                    "abs_error": 10.0,
                    "rel_error": 0.1,
                    "abs_rel_error": 0.1,
                    "metadata": {"source": "hmrc", "variable": "income_tax"},
                },
                {
                    "name": "population",
                    "estimate": 180.0,
                    "target": 200.0,
                    "error": -20.0,
                    "abs_error": 20.0,
                    "rel_error": -0.1,
                    "abs_rel_error": 0.1,
                    "metadata": {"source": "ons", "variable": "people"},
                },
            ],
            "unsupported_targets": [
                {
                    "name": "constituency_income_tax",
                    "reason": "missing_feature:constituency",
                    "metadata": {"geographic_level": "constituency"},
                }
            ],
        },
    )

    result = run_policyengine_uk_direct_loss(
        "/tmp/candidate.h5",
        time_period=2024,
        python_executable="/tmp/python",
        policyengine_uk_repo_dir="/tmp/policyengine-uk",
        policyengine_uk_data_repo_dir="/tmp/policyengine-uk-data",
        target_provider=_FakeTargetProvider(),
    )

    assert result.supported_target_count == 2
    assert result.unsupported_target_count == 1
    assert result.unsupported_targets[0].reason == "missing_feature:constituency"


def test_compare_policyengine_uk_direct_benchmark_and_save(
    monkeypatch, tmp_path: Path
):
    payloads = iter(
        [
            {
                "time_period": 2024,
                "target_count": 3,
                "supported_target_count": 2,
                "unsupported_target_count": 1,
                "mean_abs_relative_error": 0.10,
                "max_abs_relative_error": 0.20,
                "metrics": [
                    {
                        "name": "income_tax",
                        "estimate": 102.0,
                        "target": 100.0,
                        "error": 2.0,
                        "abs_error": 2.0,
                        "rel_error": 0.02,
                        "abs_rel_error": 0.02,
                        "metadata": {
                            "source": "hmrc",
                            "variable": "income_tax",
                            "geographic_level": "national",
                        },
                    },
                    {
                        "name": "population",
                        "estimate": 190.0,
                        "target": 200.0,
                        "error": -10.0,
                        "abs_error": 10.0,
                        "rel_error": -0.05,
                        "abs_rel_error": 0.05,
                        "metadata": {
                            "source": "ons",
                            "variable": "people",
                            "geographic_level": "national",
                        },
                    },
                ],
                "unsupported_targets": [
                    {
                        "name": "constituency_income_tax",
                        "reason": "missing_feature:constituency",
                        "metadata": {"geographic_level": "constituency"},
                    }
                ],
            },
            {
                "time_period": 2024,
                "target_count": 3,
                "supported_target_count": 2,
                "unsupported_target_count": 1,
                "mean_abs_relative_error": 0.15,
                "max_abs_relative_error": 0.30,
                "metrics": [
                    {
                        "name": "income_tax",
                        "estimate": 110.0,
                        "target": 100.0,
                        "error": 10.0,
                        "abs_error": 10.0,
                        "rel_error": 0.10,
                        "abs_rel_error": 0.10,
                        "metadata": {
                            "source": "hmrc",
                            "variable": "income_tax",
                            "geographic_level": "national",
                        },
                    },
                    {
                        "name": "population",
                        "estimate": 198.0,
                        "target": 200.0,
                        "error": -2.0,
                        "abs_error": 2.0,
                        "rel_error": -0.01,
                        "abs_rel_error": 0.01,
                        "metadata": {
                            "source": "ons",
                            "variable": "people",
                            "geographic_level": "national",
                        },
                    },
                ],
                "unsupported_targets": [
                    {
                        "name": "constituency_income_tax",
                        "reason": "missing_feature:constituency",
                        "metadata": {"geographic_level": "constituency"},
                    }
                ],
            },
        ]
    )
    monkeypatch.setattr(
        "microplex_uk.policyengine.harness._run_policyengine_uk_direct_loss_payload",
        lambda **kwargs: next(payloads),
    )

    comparison = compare_policyengine_uk_direct_benchmark(
        candidate_dataset_path="/tmp/candidate.h5",
        baseline_dataset_path="/tmp/baseline.h5",
        time_period=2024,
        python_executable="/tmp/python",
        policyengine_uk_repo_dir="/tmp/policyengine-uk",
        policyengine_uk_data_repo_dir="/tmp/policyengine-uk-data",
        target_provider=_FakeTargetProvider(),
        metadata={"suite": "uk_direct"},
    )

    assert isinstance(comparison, PolicyEngineUKDirectBenchmarkComparison)
    assert comparison.common_target_count == 2
    assert comparison.target_win_rate == 0.5
    assert comparison.candidate.unsupported_target_count == 1

    output_path = comparison.save(tmp_path / "direct_comparison.json")
    saved = json.loads(output_path.read_text())
    assert saved["metadata"]["suite"] == "uk_direct"
    assert saved["candidate"]["unsupported_target_count"] == 1


def test_compare_policyengine_uk_direct_benchmark_reuses_provided_baseline(
    monkeypatch,
):
    calls: list[str] = []

    def fake_run(dataset_path, **kwargs):
        calls.append(str(dataset_path))
        return PolicyEngineUKDirectBenchmarkResult(
            dataset_path=str(dataset_path),
            time_period=2024,
            target_count=1,
            supported_target_count=1,
            unsupported_target_count=0,
            mean_abs_relative_error=0.1,
            max_abs_relative_error=0.1,
            metrics=[],
            unsupported_targets=[],
        )

    monkeypatch.setattr(
        "microplex_uk.policyengine.harness.run_policyengine_uk_direct_loss",
        fake_run,
    )

    baseline = PolicyEngineUKDirectBenchmarkResult(
        dataset_path="/tmp/baseline.h5",
        time_period=2024,
        target_count=1,
        supported_target_count=1,
        unsupported_target_count=0,
        mean_abs_relative_error=0.2,
        max_abs_relative_error=0.2,
        metrics=[],
        unsupported_targets=[],
    )

    comparison = compare_policyengine_uk_direct_benchmark(
        candidate_dataset_path="/tmp/candidate.h5",
        baseline_dataset_path="/tmp/baseline.h5",
        time_period=2024,
        python_executable="/tmp/python",
        policyengine_uk_repo_dir="/tmp/policyengine-uk",
        policyengine_uk_data_repo_dir="/tmp/policyengine-uk-data",
        target_provider=_FakeTargetProvider(),
        baseline_result=baseline,
    )

    assert comparison.baseline is baseline
    assert calls == ["/tmp/candidate.h5"]
