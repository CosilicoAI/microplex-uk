"""Benchmark candidate UK datasets against PolicyEngine UK targets."""

from __future__ import annotations

import json
import subprocess
import tempfile
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from microplex.targets import TargetQuery

from microplex_uk.targets import PolicyEngineUKTargetProvider


@dataclass(frozen=True)
class PolicyEngineUKTargetMetric:
    name: str
    estimate: float
    target: float
    error: float
    abs_error: float
    rel_error: float
    abs_rel_error: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyEngineUKBenchmarkResult:
    dataset_path: str
    time_period: int
    target_count: int
    mean_abs_relative_error: float
    max_abs_relative_error: float
    metrics: list[PolicyEngineUKTargetMetric] = field(default_factory=list)


@dataclass(frozen=True)
class PolicyEngineUKTargetDelta:
    name: str
    candidate_abs_rel_error: float
    baseline_abs_rel_error: float
    abs_rel_error_delta: float
    candidate_estimate: float
    baseline_estimate: float
    target: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyEngineUKGroupSummary:
    group_field: str
    group_value: str
    target_count: int
    candidate_mean_abs_relative_error: float
    baseline_mean_abs_relative_error: float
    mean_abs_relative_error_delta: float
    target_win_rate: float


@dataclass(frozen=True)
class PolicyEngineUKUnsupportedTarget:
    name: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyEngineUKBenchmarkComparison:
    candidate: PolicyEngineUKBenchmarkResult
    baseline: PolicyEngineUKBenchmarkResult
    mean_abs_relative_error_delta: float
    target_win_rate: float
    common_target_count: int
    deltas: list[PolicyEngineUKTargetDelta] = field(default_factory=list)
    grouped_summaries: dict[str, list[PolicyEngineUKGroupSummary]] = field(
        default_factory=dict
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "candidate": _benchmark_result_to_dict(self.candidate),
                    "baseline": _benchmark_result_to_dict(self.baseline),
                    "mean_abs_relative_error_delta": self.mean_abs_relative_error_delta,
                    "target_win_rate": self.target_win_rate,
                    "common_target_count": self.common_target_count,
                    "deltas": [asdict(delta) for delta in self.deltas],
                    "grouped_summaries": {
                        field_name: [asdict(summary) for summary in summaries]
                        for field_name, summaries in self.grouped_summaries.items()
                    },
                    "metadata": self.metadata,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return output_path


@dataclass
class PolicyEngineUKDirectBenchmarkResult:
    dataset_path: str
    time_period: int
    target_count: int
    supported_target_count: int
    unsupported_target_count: int
    mean_abs_relative_error: float
    max_abs_relative_error: float
    metrics: list[PolicyEngineUKTargetMetric] = field(default_factory=list)
    unsupported_targets: list[PolicyEngineUKUnsupportedTarget] = field(
        default_factory=list
    )


@dataclass
class PolicyEngineUKDirectBenchmarkComparison:
    candidate: PolicyEngineUKDirectBenchmarkResult
    baseline: PolicyEngineUKDirectBenchmarkResult
    mean_abs_relative_error_delta: float
    target_win_rate: float
    common_target_count: int
    deltas: list[PolicyEngineUKTargetDelta] = field(default_factory=list)
    grouped_summaries: dict[str, list[PolicyEngineUKGroupSummary]] = field(
        default_factory=dict
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "candidate": _direct_benchmark_result_to_dict(self.candidate),
                    "baseline": _direct_benchmark_result_to_dict(self.baseline),
                    "mean_abs_relative_error_delta": self.mean_abs_relative_error_delta,
                    "target_win_rate": self.target_win_rate,
                    "common_target_count": self.common_target_count,
                    "deltas": [asdict(delta) for delta in self.deltas],
                    "grouped_summaries": {
                        field_name: [asdict(summary) for summary in summaries]
                        for field_name, summaries in self.grouped_summaries.items()
                    },
                    "metadata": self.metadata,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return output_path


def run_policyengine_uk_loss(
    dataset_path: str | Path,
    *,
    time_period: int,
    python_executable: str | Path,
    policyengine_uk_repo_dir: str | Path,
    policyengine_uk_data_repo_dir: str | Path,
    target_provider: PolicyEngineUKTargetProvider | None = None,
) -> PolicyEngineUKBenchmarkResult:
    payload = _run_policyengine_uk_loss_payload(
        dataset_path=dataset_path,
        time_period=time_period,
        python_executable=python_executable,
        policyengine_uk_repo_dir=policyengine_uk_repo_dir,
        policyengine_uk_data_repo_dir=policyengine_uk_data_repo_dir,
    )
    metrics = [
        PolicyEngineUKTargetMetric(**metric_payload)
        for metric_payload in payload["metrics"]
    ]
    if target_provider is not None:
        metrics = _attach_target_metadata(metrics, target_provider, time_period=time_period)
    return PolicyEngineUKBenchmarkResult(
        dataset_path=str(Path(dataset_path)),
        time_period=int(payload["time_period"]),
        target_count=int(payload["target_count"]),
        mean_abs_relative_error=float(payload["mean_abs_relative_error"]),
        max_abs_relative_error=float(payload["max_abs_relative_error"]),
        metrics=metrics,
    )


def compare_policyengine_uk_benchmark(
    *,
    candidate_dataset_path: str | Path,
    baseline_dataset_path: str | Path,
    time_period: int,
    python_executable: str | Path,
    policyengine_uk_repo_dir: str | Path,
    policyengine_uk_data_repo_dir: str | Path,
    target_provider: PolicyEngineUKTargetProvider | None = None,
    baseline_result: PolicyEngineUKBenchmarkResult | None = None,
    metadata: dict[str, Any] | None = None,
) -> PolicyEngineUKBenchmarkComparison:
    effective_target_provider = target_provider or PolicyEngineUKTargetProvider(
        policyengine_uk_data_repo_dir,
        policyengine_uk_repo_dir=policyengine_uk_repo_dir,
    )
    candidate = run_policyengine_uk_loss(
        candidate_dataset_path,
        time_period=time_period,
        python_executable=python_executable,
        policyengine_uk_repo_dir=policyengine_uk_repo_dir,
        policyengine_uk_data_repo_dir=policyengine_uk_data_repo_dir,
        target_provider=effective_target_provider,
    )
    baseline = baseline_result or run_policyengine_uk_loss(
        baseline_dataset_path,
        time_period=time_period,
        python_executable=python_executable,
        policyengine_uk_repo_dir=policyengine_uk_repo_dir,
        policyengine_uk_data_repo_dir=policyengine_uk_data_repo_dir,
        target_provider=effective_target_provider,
    )

    candidate_metrics = {metric.name: metric for metric in candidate.metrics}
    baseline_metrics = {metric.name: metric for metric in baseline.metrics}
    common_names = sorted(set(candidate_metrics) & set(baseline_metrics))
    deltas = [
        PolicyEngineUKTargetDelta(
            name=name,
            candidate_abs_rel_error=candidate_metrics[name].abs_rel_error,
            baseline_abs_rel_error=baseline_metrics[name].abs_rel_error,
            abs_rel_error_delta=(
                candidate_metrics[name].abs_rel_error
                - baseline_metrics[name].abs_rel_error
            ),
            candidate_estimate=candidate_metrics[name].estimate,
            baseline_estimate=baseline_metrics[name].estimate,
            target=candidate_metrics[name].target,
            metadata=dict(candidate_metrics[name].metadata),
        )
        for name in common_names
    ]
    wins = sum(
        delta.candidate_abs_rel_error < delta.baseline_abs_rel_error
        for delta in deltas
    )

    return PolicyEngineUKBenchmarkComparison(
        candidate=candidate,
        baseline=baseline,
        mean_abs_relative_error_delta=(
            candidate.mean_abs_relative_error - baseline.mean_abs_relative_error
        ),
        target_win_rate=(wins / len(deltas)) if deltas else 0.0,
        common_target_count=len(deltas),
        deltas=deltas,
        grouped_summaries=_build_grouped_summaries(deltas),
        metadata=dict(metadata or {}),
    )


def run_policyengine_uk_direct_loss(
    dataset_path: str | Path,
    *,
    time_period: int,
    python_executable: str | Path,
    policyengine_uk_repo_dir: str | Path,
    policyengine_uk_data_repo_dir: str | Path,
    target_provider: PolicyEngineUKTargetProvider | None = None,
    target_query: TargetQuery | None = None,
) -> PolicyEngineUKDirectBenchmarkResult:
    effective_target_provider = target_provider or PolicyEngineUKTargetProvider(
        policyengine_uk_data_repo_dir,
        policyengine_uk_repo_dir=policyengine_uk_repo_dir,
    )
    effective_query = target_query or TargetQuery(period=time_period)
    target_set = effective_target_provider.load_target_set(effective_query)
    payload = _run_policyengine_uk_direct_loss_payload(
        dataset_path=dataset_path,
        time_period=time_period,
        python_executable=python_executable,
        policyengine_uk_repo_dir=policyengine_uk_repo_dir,
        target_specs=target_set.targets,
    )
    metrics = [
        PolicyEngineUKTargetMetric(**metric_payload)
        for metric_payload in payload["metrics"]
    ]
    unsupported_targets = [
        PolicyEngineUKUnsupportedTarget(**unsupported_payload)
        for unsupported_payload in payload["unsupported_targets"]
    ]
    return PolicyEngineUKDirectBenchmarkResult(
        dataset_path=str(Path(dataset_path)),
        time_period=int(payload["time_period"]),
        target_count=int(payload["target_count"]),
        supported_target_count=int(payload["supported_target_count"]),
        unsupported_target_count=int(payload["unsupported_target_count"]),
        mean_abs_relative_error=float(payload["mean_abs_relative_error"]),
        max_abs_relative_error=float(payload["max_abs_relative_error"]),
        metrics=metrics,
        unsupported_targets=unsupported_targets,
    )


def compare_policyengine_uk_direct_benchmark(
    *,
    candidate_dataset_path: str | Path,
    baseline_dataset_path: str | Path,
    time_period: int,
    python_executable: str | Path,
    policyengine_uk_repo_dir: str | Path,
    policyengine_uk_data_repo_dir: str | Path,
    target_provider: PolicyEngineUKTargetProvider | None = None,
    target_query: TargetQuery | None = None,
    baseline_result: PolicyEngineUKDirectBenchmarkResult | None = None,
    metadata: dict[str, Any] | None = None,
) -> PolicyEngineUKDirectBenchmarkComparison:
    effective_target_provider = target_provider or PolicyEngineUKTargetProvider(
        policyengine_uk_data_repo_dir,
        policyengine_uk_repo_dir=policyengine_uk_repo_dir,
    )
    candidate = run_policyengine_uk_direct_loss(
        candidate_dataset_path,
        time_period=time_period,
        python_executable=python_executable,
        policyengine_uk_repo_dir=policyengine_uk_repo_dir,
        policyengine_uk_data_repo_dir=policyengine_uk_data_repo_dir,
        target_provider=effective_target_provider,
        target_query=target_query,
    )
    baseline = baseline_result or run_policyengine_uk_direct_loss(
        baseline_dataset_path,
        time_period=time_period,
        python_executable=python_executable,
        policyengine_uk_repo_dir=policyengine_uk_repo_dir,
        policyengine_uk_data_repo_dir=policyengine_uk_data_repo_dir,
        target_provider=effective_target_provider,
        target_query=target_query,
    )
    candidate_metrics = {metric.name: metric for metric in candidate.metrics}
    baseline_metrics = {metric.name: metric for metric in baseline.metrics}
    common_names = sorted(set(candidate_metrics) & set(baseline_metrics))
    deltas = [
        PolicyEngineUKTargetDelta(
            name=name,
            candidate_abs_rel_error=candidate_metrics[name].abs_rel_error,
            baseline_abs_rel_error=baseline_metrics[name].abs_rel_error,
            abs_rel_error_delta=(
                candidate_metrics[name].abs_rel_error
                - baseline_metrics[name].abs_rel_error
            ),
            candidate_estimate=candidate_metrics[name].estimate,
            baseline_estimate=baseline_metrics[name].estimate,
            target=candidate_metrics[name].target,
            metadata=dict(candidate_metrics[name].metadata),
        )
        for name in common_names
    ]
    wins = sum(
        delta.candidate_abs_rel_error < delta.baseline_abs_rel_error
        for delta in deltas
    )
    return PolicyEngineUKDirectBenchmarkComparison(
        candidate=candidate,
        baseline=baseline,
        mean_abs_relative_error_delta=(
            candidate.mean_abs_relative_error - baseline.mean_abs_relative_error
        ),
        target_win_rate=(wins / len(deltas)) if deltas else 0.0,
        common_target_count=len(deltas),
        deltas=deltas,
        grouped_summaries=_build_grouped_summaries(deltas),
        metadata=dict(metadata or {}),
    )


def _run_policyengine_uk_loss_payload(
    *,
    dataset_path: str | Path,
    time_period: int,
    python_executable: str | Path,
    policyengine_uk_repo_dir: str | Path,
    policyengine_uk_data_repo_dir: str | Path,
) -> dict[str, Any]:
    script = textwrap.dedent(
        """
import json
import importlib
import sys
import types
from pathlib import Path

import pandas as pd

package = types.ModuleType("policyengine_uk_data")
package.__path__ = [str(Path(sys.argv[4]) / "policyengine_uk_data")]
sys.modules["policyengine_uk_data"] = package

utils_package = types.ModuleType("policyengine_uk_data.utils")
utils_package.__path__ = [str(Path(sys.argv[4]) / "policyengine_uk_data" / "utils")]
sys.modules["policyengine_uk_data.utils"] = utils_package

storage_package = types.ModuleType("policyengine_uk_data.storage")
storage_package.__path__ = [str(Path(sys.argv[4]) / "policyengine_uk_data" / "storage")]
storage_package.STORAGE_FOLDER = Path(sys.argv[4]) / "policyengine_uk_data" / "storage"
sys.modules["policyengine_uk_data.storage"] = storage_package

sys.path.insert(0, sys.argv[3])
sys.path.insert(0, sys.argv[4])

from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk import Microsimulation

build_loss_matrix = importlib.import_module(
    "policyengine_uk_data.targets.build_loss_matrix"
)

dataset = UKSingleYearDataset(file_path=sys.argv[1])
matrix, targets = build_loss_matrix.create_target_matrix(
    dataset,
    str(int(sys.argv[2])),
)
simulation = Microsimulation(dataset=dataset)
weights = simulation.calculate("household_weight", int(sys.argv[2])).values
estimates = weights @ matrix
results = pd.DataFrame(
    {
        "name": estimates.index,
        "estimate": estimates.values,
        "target": targets,
    }
)
results["error"] = results["estimate"] - results["target"]
results["abs_error"] = results["error"].abs()
results["rel_error"] = results["error"] / results["target"]
results["abs_rel_error"] = results["rel_error"].abs()
payload = {
    "time_period": int(sys.argv[2]),
    "target_count": int(len(results)),
    "mean_abs_relative_error": float(results["abs_rel_error"].mean()),
    "max_abs_relative_error": float(results["abs_rel_error"].max()),
    "metrics": results.to_dict(orient="records"),
}
print(json.dumps(payload))
"""
    )
    completed = subprocess.run(
        [
            str(python_executable),
            "-c",
            script,
            str(Path(dataset_path)),
            str(int(time_period)),
            str(Path(policyengine_uk_repo_dir)),
            str(Path(policyengine_uk_data_repo_dir)),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "PolicyEngine UK benchmark failed")
    return json.loads(completed.stdout)


def _run_policyengine_uk_direct_loss_payload(
    *,
    dataset_path: str | Path,
    time_period: int,
    python_executable: str | Path,
    policyengine_uk_repo_dir: str | Path,
    target_specs: list[object],
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="microplex_uk_targets_",
        delete=False,
    ) as handle:
        target_payload_path = Path(handle.name)
        json.dump([_target_spec_to_payload(target) for target in target_specs], handle)
    script = textwrap.dedent(
        """
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[3])

from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk import Microsimulation


def normalize(values):
    if hasattr(values, "decode_to_str"):
        return pd.Series(values.decode_to_str())
    array = np.asarray(values)
    if array.dtype.kind in ("S", "U", "O"):
        return pd.Series(array.astype(str))
    return pd.Series(array)


def numeric(values):
    return pd.to_numeric(pd.Series(values), errors="coerce")


def apply_filter(values, operator, expected):
    if operator == "==":
        return values.astype(str) == str(expected)
    if operator == "!=":
        return values.astype(str) != str(expected)
    if operator == "in":
        return values.astype(str).isin([str(item) for item in expected])
    if operator == "not_in":
        return ~values.astype(str).isin([str(item) for item in expected])
    numeric_values = numeric(values)
    expected_value = float(expected)
    if operator == ">":
        return numeric_values > expected_value
    if operator == ">=":
        return numeric_values >= expected_value
    if operator == "<":
        return numeric_values < expected_value
    if operator == "<=":
        return numeric_values <= expected_value
    raise ValueError(f"Unsupported operator: {operator}")


dataset = UKSingleYearDataset(file_path=sys.argv[1])
period = int(sys.argv[2])
sim = Microsimulation(dataset=dataset)
sim.default_calculation_period = str(period)
targets = json.loads(Path(sys.argv[4]).read_text())

household_frame = dataset.household.copy()
weights = numeric(sim.calculate("household_weight", period).values).fillna(0.0)
feature_cache = {}
unsupported = []
metrics = []


def get_feature(name):
    if name in feature_cache:
        return feature_cache[name]
    if name in household_frame.columns:
        series = normalize(household_frame[name].values)
        feature_cache[name] = series
        return series
    if name in {"constituency", "local_authority"}:
        raise KeyError(name)
    values = sim.calculate(name, period, map_to="household").values
    series = normalize(values)
    feature_cache[name] = series
    return series


for target in targets:
    metadata = target.get("metadata", {})
    if metadata.get("has_custom_compute"):
        unsupported.append(
            {"name": target["name"], "reason": "custom_compute", "metadata": metadata}
        )
        continue
    try:
        mask = pd.Series(True, index=household_frame.index)
        for filter_spec in target.get("filters", []):
            feature = filter_spec["feature"]
            if feature in {"local_authority", "constituency"} and feature not in household_frame.columns:
                raise KeyError(feature)
            feature_values = get_feature(feature)
            mask = mask & apply_filter(
                feature_values,
                filter_spec["operator"],
                filter_spec["value"],
            ).fillna(False)
        if target["aggregation"] == "count":
            estimate = float((weights * mask.astype(float)).sum())
        else:
            measure = target.get("measure")
            if measure is None:
                raise ValueError("sum/mean target missing measure")
            measure_values = numeric(get_feature(measure)).fillna(0.0)
            if target["aggregation"] == "sum":
                estimate = float((weights * mask.astype(float) * measure_values).sum())
            elif target["aggregation"] == "mean":
                denom = float((weights * mask.astype(float)).sum())
                if denom <= 0.0:
                    raise ValueError("mean target has zero support")
                estimate = float((weights * mask.astype(float) * measure_values).sum() / denom)
            else:
                raise ValueError(f"Unsupported aggregation {target['aggregation']}")
        target_value = float(target["value"])
        error = estimate - target_value
        rel_error = error / target_value if target_value != 0 else 0.0
        metrics.append(
            {
                "name": target["name"],
                "estimate": estimate,
                "target": target_value,
                "error": error,
                "abs_error": abs(error),
                "rel_error": rel_error,
                "abs_rel_error": abs(rel_error),
                "metadata": metadata,
            }
        )
    except KeyError as exc:
        unsupported.append(
            {
                "name": target["name"],
                "reason": f"missing_feature:{exc.args[0]}",
                "metadata": metadata,
            }
        )
    except Exception as exc:
        unsupported.append(
            {
                "name": target["name"],
                "reason": f"unsupported:{type(exc).__name__}",
                "metadata": metadata,
            }
        )

supported_count = len(metrics)
abs_rel_errors = [metric["abs_rel_error"] for metric in metrics]
payload = {
    "time_period": period,
    "target_count": len(targets),
    "supported_target_count": supported_count,
    "unsupported_target_count": len(unsupported),
    "mean_abs_relative_error": float(sum(abs_rel_errors) / supported_count) if supported_count else 0.0,
    "max_abs_relative_error": float(max(abs_rel_errors)) if abs_rel_errors else 0.0,
    "metrics": metrics,
    "unsupported_targets": unsupported,
}
print(json.dumps(payload))
"""
    )
    try:
        completed = subprocess.run(
            [
                str(python_executable),
                "-c",
                script,
                str(Path(dataset_path)),
                str(int(time_period)),
                str(Path(policyengine_uk_repo_dir)),
                str(target_payload_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                completed.stderr.strip()
                or completed.stdout.strip()
                or "PolicyEngine UK direct benchmark failed"
            )
        return json.loads(completed.stdout)
    finally:
        target_payload_path.unlink(missing_ok=True)


def _benchmark_result_to_dict(result: PolicyEngineUKBenchmarkResult) -> dict[str, Any]:
    return {
        "dataset_path": result.dataset_path,
        "time_period": result.time_period,
        "target_count": result.target_count,
        "mean_abs_relative_error": result.mean_abs_relative_error,
        "max_abs_relative_error": result.max_abs_relative_error,
        "metrics": [asdict(metric) for metric in result.metrics],
    }


def _direct_benchmark_result_to_dict(
    result: PolicyEngineUKDirectBenchmarkResult,
) -> dict[str, Any]:
    return {
        "dataset_path": result.dataset_path,
        "time_period": result.time_period,
        "target_count": result.target_count,
        "supported_target_count": result.supported_target_count,
        "unsupported_target_count": result.unsupported_target_count,
        "mean_abs_relative_error": result.mean_abs_relative_error,
        "max_abs_relative_error": result.max_abs_relative_error,
        "metrics": [asdict(metric) for metric in result.metrics],
        "unsupported_targets": [
            asdict(unsupported) for unsupported in result.unsupported_targets
        ],
    }


def _target_spec_to_payload(target: object) -> dict[str, Any]:
    return {
        "name": target.name,
        "value": float(target.value),
        "measure": target.measure,
        "aggregation": target.aggregation.value,
        "filters": [
            {
                "feature": target_filter.feature,
                "operator": target_filter.operator.value,
                "value": target_filter.value,
            }
            for target_filter in target.filters
        ],
        "metadata": dict(target.metadata),
    }


def _attach_target_metadata(
    metrics: list[PolicyEngineUKTargetMetric],
    target_provider: PolicyEngineUKTargetProvider,
    *,
    time_period: int,
) -> list[PolicyEngineUKTargetMetric]:
    target_set = target_provider.load_target_set(TargetQuery(period=time_period))
    metadata_by_name = {
        target.name: {
            "source": target.source,
            "entity": target.entity.value,
            "measure": target.measure,
            "aggregation": target.aggregation.value,
            **target.metadata,
        }
        for target in target_set.targets
    }
    return [
        PolicyEngineUKTargetMetric(
            name=metric.name,
            estimate=metric.estimate,
            target=metric.target,
            error=metric.error,
            abs_error=metric.abs_error,
            rel_error=metric.rel_error,
            abs_rel_error=metric.abs_rel_error,
            metadata=dict(metadata_by_name.get(metric.name, {})),
        )
        for metric in metrics
    ]


def _build_grouped_summaries(
    deltas: list[PolicyEngineUKTargetDelta],
) -> dict[str, list[PolicyEngineUKGroupSummary]]:
    grouped: dict[str, list[PolicyEngineUKGroupSummary]] = {}
    for field_name in ("source", "geographic_level", "variable", "is_count"):
        summaries = _summaries_for_field(deltas, field_name)
        if summaries:
            grouped[field_name] = summaries
    return grouped


def _summaries_for_field(
    deltas: list[PolicyEngineUKTargetDelta],
    field_name: str,
) -> list[PolicyEngineUKGroupSummary]:
    buckets: dict[str, list[PolicyEngineUKTargetDelta]] = {}
    for delta in deltas:
        value = delta.metadata.get(field_name)
        if value is None:
            continue
        buckets.setdefault(str(value), []).append(delta)
    summaries: list[PolicyEngineUKGroupSummary] = []
    for group_value, group_deltas in sorted(buckets.items()):
        target_count = len(group_deltas)
        candidate_mean = sum(
            delta.candidate_abs_rel_error for delta in group_deltas
        ) / target_count
        baseline_mean = sum(
            delta.baseline_abs_rel_error for delta in group_deltas
        ) / target_count
        wins = sum(
            delta.candidate_abs_rel_error < delta.baseline_abs_rel_error
            for delta in group_deltas
        )
        summaries.append(
            PolicyEngineUKGroupSummary(
                group_field=field_name,
                group_value=group_value,
                target_count=target_count,
                candidate_mean_abs_relative_error=candidate_mean,
                baseline_mean_abs_relative_error=baseline_mean,
                mean_abs_relative_error_delta=(candidate_mean - baseline_mean),
                target_win_rate=(wins / target_count),
            )
        )
    return summaries
