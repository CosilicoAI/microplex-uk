"""UK atomic geography helpers built on the core microplex geography API."""

from __future__ import annotations

import pandas as pd
from microplex.geography import (
    AtomicGeographyCrosswalk,
    GeographyAssignmentPlan,
    GeographyProvider,
    StaticGeographyProvider,
    materialize_geographies,
)

UK_ATOMIC_GEOGRAPHY_ID_COLUMN = "oa_code"
UK_PARENT_GEOGRAPHY_COLUMNS: tuple[str, ...] = (
    "lsoa",
    "local_authority",
    "constituency",
    "region",
    "country",
)
UK_GEOGRAPHY_PROBABILITY_COLUMN = "assignment_probability"


def default_uk_atomic_geography_assignment_plan(
    *,
    partition_columns: tuple[str, ...] = ("region",),
    atomic_id_column: str = UK_ATOMIC_GEOGRAPHY_ID_COLUMN,
    geography_columns: tuple[str, ...] = UK_PARENT_GEOGRAPHY_COLUMNS,
    probability_column: str = UK_GEOGRAPHY_PROBABILITY_COLUMN,
) -> GeographyAssignmentPlan:
    """Assign UK households to an atomic geography independently of target grain."""
    return GeographyAssignmentPlan(
        partition_columns=partition_columns,
        atomic_id_column=atomic_id_column,
        geography_columns=geography_columns,
        probability_column=probability_column,
        sync_partition_columns=False,
    )


def build_static_uk_geography_provider(
    data: pd.DataFrame,
    *,
    atomic_id_column: str = UK_ATOMIC_GEOGRAPHY_ID_COLUMN,
    geography_columns: tuple[str, ...] = UK_PARENT_GEOGRAPHY_COLUMNS,
    probability_column: str = UK_GEOGRAPHY_PROBABILITY_COLUMN,
    default_partition_columns: tuple[str, ...] = ("region",),
) -> StaticGeographyProvider:
    """Create a UK geography provider from an in-memory crosswalk."""
    return StaticGeographyProvider(
        crosswalk=AtomicGeographyCrosswalk(
            data=data.copy(),
            atomic_id_column=atomic_id_column,
            geography_columns=geography_columns,
            probability_column=probability_column,
        ),
        default_partition_columns=default_partition_columns,
    )


def apply_uk_candidate_geography(
    *,
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    geography_provider: GeographyProvider,
    assignment_plan: GeographyAssignmentPlan | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign atomic UK geographies once and broadcast parents to all entity tables."""
    plan = assignment_plan or default_uk_atomic_geography_assignment_plan()
    query = plan.to_query()
    crosswalk = geography_provider.load_crosswalk(query)

    household_result = household.copy()
    if plan.atomic_id_column not in household_result.columns:
        assigner = geography_provider.load_assigner(query)
        household_result = assigner.assign(
            household_result,
            atomic_id_column=plan.atomic_id_column,
            random_state=random_state,
        )

    household_result = materialize_geographies(
        household_result,
        crosswalk,
        columns=plan.requested_geography_columns(),
        atomic_id_column=plan.atomic_id_column,
        overwrite=False,
    )

    household_geo_columns = tuple(
        dict.fromkeys((plan.atomic_id_column, *plan.requested_geography_columns()))
    )
    household_lookup = household_result[["household_id", *household_geo_columns]].drop_duplicates(
        "household_id"
    )

    person_result = _broadcast_household_geography(
        person,
        household_lookup=household_lookup,
        household_id_column="person_household_id",
        geography_columns=household_geo_columns,
    )
    benunit_result = _broadcast_benunit_geography(
        benunit,
        person=person_result,
        household_lookup=household_lookup,
        geography_columns=household_geo_columns,
    )
    return person_result, benunit_result, household_result


def _broadcast_household_geography(
    frame: pd.DataFrame,
    *,
    household_lookup: pd.DataFrame,
    household_id_column: str,
    geography_columns: tuple[str, ...],
) -> pd.DataFrame:
    if household_id_column not in frame.columns:
        return frame.copy()
    lookup = household_lookup.rename(columns={"household_id": household_id_column})
    return frame.merge(lookup, on=household_id_column, how="left")


def _broadcast_benunit_geography(
    benunit: pd.DataFrame,
    *,
    person: pd.DataFrame,
    household_lookup: pd.DataFrame,
    geography_columns: tuple[str, ...],
) -> pd.DataFrame:
    if "benunit_id" not in benunit.columns or "person_benunit_id" not in person.columns:
        return benunit.copy()
    benunit_household_map = (
        person[["person_benunit_id", "person_household_id"]]
        .drop_duplicates()
        .rename(
            columns={
                "person_benunit_id": "benunit_id",
                "person_household_id": "household_id",
            }
        )
    )
    if benunit_household_map["benunit_id"].duplicated().any():
        raise ValueError("Cannot broadcast geography: benunit maps to multiple households")
    enriched = benunit.merge(benunit_household_map, on="benunit_id", how="left")
    lookup = household_lookup.copy()
    return enriched.merge(lookup, on="household_id", how="left")
