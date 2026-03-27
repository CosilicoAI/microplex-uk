"""UK-specific source adapters and benchmark harnesses for microplex."""

from microplex_uk.data_sources import (
    UKFRSSourceProvider,
    UKSPISourceProvider,
    UKWASSourceProvider,
)
from microplex_uk.pipelines import (
    UKCandidateBenchmarkArtifacts,
    UKCandidateDataset,
    UKDonorBlockSpec,
    UKDonorCombineStrategy,
    UKDonorVariableSpec,
    build_and_benchmark_fused_uk_candidate,
    build_fused_uk_candidate_dataset,
    build_fused_uk_candidate_from_tables,
    default_uk_candidate_donor_block_specs,
)
from microplex_uk.policyengine import (
    PolicyEngineUKBenchmarkComparison,
    PolicyEngineUKBenchmarkResult,
    PolicyEngineUKDirectBenchmarkComparison,
    PolicyEngineUKDirectBenchmarkResult,
    PolicyEngineUKGroupSummary,
    PolicyEngineUKTargetDelta,
    PolicyEngineUKTargetMetric,
    PolicyEngineUKUnsupportedTarget,
    compare_policyengine_uk_benchmark,
    compare_policyengine_uk_direct_benchmark,
    run_policyengine_uk_direct_loss,
    run_policyengine_uk_loss,
)
from microplex_uk.targets import PolicyEngineUKTargetProvider

__all__ = [
    "UKFRSSourceProvider",
    "UKSPISourceProvider",
    "UKWASSourceProvider",
    "UKDonorCombineStrategy",
    "UKDonorVariableSpec",
    "UKDonorBlockSpec",
    "UKCandidateDataset",
    "UKCandidateBenchmarkArtifacts",
    "build_fused_uk_candidate_from_tables",
    "build_fused_uk_candidate_dataset",
    "build_and_benchmark_fused_uk_candidate",
    "default_uk_candidate_donor_block_specs",
    "PolicyEngineUKTargetProvider",
    "PolicyEngineUKTargetMetric",
    "PolicyEngineUKBenchmarkResult",
    "PolicyEngineUKDirectBenchmarkResult",
    "PolicyEngineUKGroupSummary",
    "PolicyEngineUKTargetDelta",
    "PolicyEngineUKUnsupportedTarget",
    "PolicyEngineUKBenchmarkComparison",
    "PolicyEngineUKDirectBenchmarkComparison",
    "run_policyengine_uk_loss",
    "run_policyengine_uk_direct_loss",
    "compare_policyengine_uk_benchmark",
    "compare_policyengine_uk_direct_benchmark",
]
