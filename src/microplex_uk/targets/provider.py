"""PolicyEngine UK targets exposed through the canonical microplex target API."""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from microplex.core import EntityType
from microplex.targets import (
    FilterOperator,
    TargetAggregation,
    TargetFilter,
    TargetProvider,
    TargetQuery,
    TargetSet,
    TargetSpec,
    apply_target_query,
)


@contextmanager
def _prepend_sys_paths(paths: tuple[Path, ...]) -> Iterator[None]:
    inserted: list[str] = []
    for path in reversed(paths):
        path_str = str(path)
        sys.path.insert(0, path_str)
        inserted.append(path_str)
    try:
        yield
    finally:
        for path_str in inserted:
            try:
                sys.path.remove(path_str)
            except ValueError:
                pass


@contextmanager
def _namespace_package(
    name: str,
    package_path: Path,
    *,
    attrs: dict[str, object] | None = None,
) -> Iterator[None]:
    existing_package = sys.modules.get(name)
    existing_submodules = {
        module_name: module
        for module_name, module in sys.modules.items()
        if module_name.startswith(f"{name}.")
    }
    if existing_package is None:
        package = types.ModuleType(name)
        package.__path__ = [str(package_path)]  # type: ignore[attr-defined]
        for key, value in (attrs or {}).items():
            setattr(package, key, value)
        sys.modules[name] = package
    try:
        yield
    finally:
        if existing_package is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = existing_package
        for module_name in list(sys.modules):
            if module_name.startswith(f"{name}."):
                sys.modules.pop(module_name, None)
        sys.modules.update(existing_submodules)


def _coerce_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _load_raw_targets(
    data_repo_dir: Path,
    year: int | None,
    *,
    policyengine_uk_repo_dir: Path | None = None,
) -> list[object]:
    paths = (data_repo_dir,) if policyengine_uk_repo_dir is None else (
        policyengine_uk_repo_dir,
        data_repo_dir,
    )
    base_path = data_repo_dir / "policyengine_uk_data"
    with _prepend_sys_paths(paths), _namespace_package(
        "policyengine_uk_data",
        base_path,
    ), _namespace_package(
        "policyengine_uk_data.utils",
        base_path / "utils",
    ), _namespace_package(
        "policyengine_uk_data.storage",
        base_path / "storage",
        attrs={"STORAGE_FOLDER": base_path / "storage"},
    ):
        registry = importlib.import_module("policyengine_uk_data.targets.registry")
        return list(registry.get_all_targets(year=year))


@dataclass(frozen=True)
class PolicyEngineUKTargetProvider(TargetProvider):
    """Load UK targets from a local policyengine-uk-data repository."""

    data_repo_dir: str | Path
    policyengine_uk_repo_dir: str | Path | None = None
    default_geographic_levels: tuple[str, ...] = (
        "national",
        "region",
        "country",
        "constituency",
        "local_authority",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_repo_dir", Path(self.data_repo_dir))
        if not self.data_repo_dir.exists():
            raise FileNotFoundError(self.data_repo_dir)
        if self.policyengine_uk_repo_dir is not None:
            object.__setattr__(
                self,
                "policyengine_uk_repo_dir",
                Path(self.policyengine_uk_repo_dir),
            )
            if not self.policyengine_uk_repo_dir.exists():
                raise FileNotFoundError(self.policyengine_uk_repo_dir)

    def load_target_set(self, query: TargetQuery | None = None) -> TargetSet:
        period = None if query is None else query.period
        year = None if period is None else int(period)
        provider_filters = {} if query is None else query.provider_filters
        geographic_levels = _coerce_tuple(
            provider_filters.get("geographic_levels", self.default_geographic_levels)
        )
        variables = set(_coerce_tuple(provider_filters.get("variables")))
        sources = set(_coerce_tuple(provider_filters.get("sources")))

        raw_targets = _load_raw_targets(
            self.data_repo_dir,
            year,
            policyengine_uk_repo_dir=self.policyengine_uk_repo_dir,
        )

        selected_specs: list[TargetSpec] = []
        for raw_target in raw_targets:
            if geographic_levels and raw_target.geographic_level.value not in geographic_levels:
                continue
            if variables and raw_target.variable not in variables:
                continue
            if sources and raw_target.source not in sources:
                continue
            value = raw_target.values.get(year) if year is not None else None
            if year is None:
                latest_year = max(raw_target.values)
                value = raw_target.values[latest_year]
                target_period = latest_year
            else:
                if value is None:
                    continue
                target_period = year
            selected_specs.append(self._to_target_spec(raw_target, target_period, value))

        return apply_target_query(TargetSet(selected_specs), query)

    def _to_target_spec(self, raw_target: object, period: int, value: float) -> TargetSpec:
        filters = list(_geography_filters(raw_target))
        filters.extend(_breakdown_filters(raw_target))
        aggregation = (
            TargetAggregation.COUNT if raw_target.is_count else TargetAggregation.SUM
        )
        measure = None if raw_target.is_count else raw_target.variable
        return TargetSpec(
            name=raw_target.name,
            entity=EntityType.HOUSEHOLD,
            value=float(value),
            period=period,
            measure=measure,
            aggregation=aggregation,
            filters=tuple(filters),
            source=raw_target.source,
            units=raw_target.unit.value,
            metadata={
                "variable": raw_target.variable,
                "geographic_level": raw_target.geographic_level.value,
                "geo_code": raw_target.geo_code,
                "geo_name": raw_target.geo_name,
                "breakdown_variable": raw_target.breakdown_variable,
                "lower_bound": raw_target.lower_bound,
                "upper_bound": raw_target.upper_bound,
                "is_count": raw_target.is_count,
                "reference_url": raw_target.reference_url,
                "forecast_vintage": raw_target.forecast_vintage,
                "has_custom_compute": raw_target.custom_compute is not None,
            },
        )


def _geography_filters(raw_target: object) -> tuple[TargetFilter, ...]:
    geography_features = {
        "region": "region",
        "country": "country",
        "local_authority": "local_authority",
        "constituency": "constituency",
    }
    feature = geography_features.get(raw_target.geographic_level.value)
    if feature is None or raw_target.geo_code is None:
        return ()
    return (TargetFilter(feature=feature, operator=FilterOperator.EQ, value=raw_target.geo_code),)


def _breakdown_filters(raw_target: object) -> tuple[TargetFilter, ...]:
    if raw_target.breakdown_variable is None:
        return ()
    filters: list[TargetFilter] = []
    if raw_target.lower_bound is not None:
        filters.append(
            TargetFilter(
                feature=raw_target.breakdown_variable,
                operator=FilterOperator.GTE,
                value=raw_target.lower_bound,
            )
        )
    if raw_target.upper_bound is not None:
        filters.append(
            TargetFilter(
                feature=raw_target.breakdown_variable,
                operator=FilterOperator.LT,
                value=raw_target.upper_bound,
            )
        )
    return tuple(filters)
