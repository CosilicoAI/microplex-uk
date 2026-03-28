"""UK candidate dataset build helpers."""

from microplex_uk.geography import (
    UK_ATOMIC_GEOGRAPHY_ID_COLUMN,
    UK_GEOGRAPHY_PROBABILITY_COLUMN,
    UK_PARENT_GEOGRAPHY_COLUMNS,
    apply_uk_candidate_geography,
    build_static_uk_geography_provider,
    default_uk_atomic_geography_assignment_plan,
)
from microplex_uk.pipelines.candidate import (
    UKBenchmarkMode,
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

__all__ = [
    "UKCandidateBenchmarkArtifacts",
    "UKCandidateDataset",
    "UKBenchmarkMode",
    "UKDonorBlockSpec",
    "UKDonorCombineStrategy",
    "UKDonorVariableSpec",
    "build_fused_uk_candidate_from_tables",
    "build_fused_uk_candidate_dataset",
    "build_and_benchmark_fused_uk_candidate",
    "default_uk_candidate_donor_block_specs",
    "UK_ATOMIC_GEOGRAPHY_ID_COLUMN",
    "UK_PARENT_GEOGRAPHY_COLUMNS",
    "UK_GEOGRAPHY_PROBABILITY_COLUMN",
    "default_uk_atomic_geography_assignment_plan",
    "build_static_uk_geography_provider",
    "apply_uk_candidate_geography",
]
