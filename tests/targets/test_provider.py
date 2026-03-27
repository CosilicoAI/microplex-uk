from __future__ import annotations

from pathlib import Path

from microplex.core import EntityType
from microplex.targets import FilterOperator, TargetAggregation, TargetQuery

from microplex_uk.targets.provider import PolicyEngineUKTargetProvider


class _FakeGeoLevel:
    def __init__(self, value: str):
        self.value = value


class _FakeUnit:
    def __init__(self, value: str):
        self.value = value


class _FakeTarget:
    def __init__(
        self,
        *,
        name: str,
        variable: str,
        source: str,
        unit: str,
        geographic_level: str,
        values: dict[int, float],
        geo_code: str | None = None,
        geo_name: str | None = None,
        breakdown_variable: str | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        is_count: bool = False,
        reference_url: str | None = None,
        forecast_vintage: str | None = None,
        custom_compute=None,
    ):
        self.name = name
        self.variable = variable
        self.source = source
        self.unit = _FakeUnit(unit)
        self.geographic_level = _FakeGeoLevel(geographic_level)
        self.values = values
        self.geo_code = geo_code
        self.geo_name = geo_name
        self.breakdown_variable = breakdown_variable
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.is_count = is_count
        self.reference_url = reference_url
        self.forecast_vintage = forecast_vintage
        self.custom_compute = custom_compute


def test_target_provider_maps_uk_targets_to_canonical_specs(monkeypatch, tmp_path: Path):
    provider = PolicyEngineUKTargetProvider(tmp_path)
    fake_targets = [
        _FakeTarget(
            name="hmrc/income_tax",
            variable="income_tax",
            source="hmrc",
            unit="gbp",
            geographic_level="region",
            geo_code="UKI",
            values={2024: 100.0},
            breakdown_variable="total_income",
            lower_bound=0.0,
            upper_bound=50_000.0,
        ),
        _FakeTarget(
            name="ons/population",
            variable="people",
            source="ons",
            unit="count",
            geographic_level="national",
            values={2024: 200.0},
            is_count=True,
        ),
    ]

    monkeypatch.setattr(
        "microplex_uk.targets.provider._load_raw_targets",
        lambda data_repo_dir, year, **kwargs: fake_targets,
    )

    target_set = provider.load_target_set(TargetQuery(period=2024))

    assert len(target_set.targets) == 2

    amount_target = target_set.targets[0]
    assert amount_target.entity is EntityType.HOUSEHOLD
    assert amount_target.aggregation is TargetAggregation.SUM
    assert amount_target.measure == "income_tax"
    assert amount_target.filters[0].feature == "region"
    assert amount_target.filters[0].operator is FilterOperator.EQ
    assert amount_target.filters[1].feature == "total_income"
    assert amount_target.filters[1].operator is FilterOperator.GTE
    assert amount_target.filters[2].operator is FilterOperator.LT

    count_target = target_set.targets[1]
    assert count_target.aggregation is TargetAggregation.COUNT
    assert count_target.measure is None
    assert count_target.metadata["is_count"] is True


def test_target_provider_applies_provider_filters(monkeypatch, tmp_path: Path):
    provider = PolicyEngineUKTargetProvider(tmp_path)
    fake_targets = [
        _FakeTarget(
            name="hmrc/income_tax",
            variable="income_tax",
            source="hmrc",
            unit="gbp",
            geographic_level="region",
            values={2024: 100.0},
        ),
        _FakeTarget(
            name="ons/population",
            variable="people",
            source="ons",
            unit="count",
            geographic_level="national",
            values={2024: 200.0},
            is_count=True,
        ),
    ]
    monkeypatch.setattr(
        "microplex_uk.targets.provider._load_raw_targets",
        lambda data_repo_dir, year, **kwargs: fake_targets,
    )

    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "geographic_levels": ("region",),
                "variables": ("income_tax",),
                "sources": ("hmrc",),
            },
        )
    )

    assert [target.name for target in target_set.targets] == ["hmrc/income_tax"]
