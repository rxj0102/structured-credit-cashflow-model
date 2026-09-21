import numpy as np
import pytest

from srt_model.historical_loss_analysis import (
    VintageConfig,
    generate_vintage_data,
    compute_vintage_curve,
    derive_base_case_assumptions,
)


def test_generates_all_requested_vintages():
    df = generate_vintage_data(VintageConfig(n_vintages=6, seed=5))
    assert df["vintage_year"].nunique() == 6


def test_reproducible_with_same_seed():
    df1 = generate_vintage_data(VintageConfig(seed=42))
    df2 = generate_vintage_data(VintageConfig(seed=42))
    assert np.allclose(df1["cumulative_default_rate"], df2["cumulative_default_rate"])


def test_cumulative_default_rate_is_monotonic_non_decreasing_per_vintage(vintage_data):
    for _, group in vintage_data.groupby("vintage_year"):
        ordered = group.sort_values("mob")["cumulative_default_rate"].to_numpy()
        assert np.all(np.diff(ordered) >= -1e-9)


def test_vintage_curve_reaches_full_pct_of_ultimate(vintage_curve):
    assert vintage_curve["pct_of_ultimate"].iloc[-1] == pytest.approx(1.0)


def test_base_case_assumptions_are_valid_probabilities(base_case_assumptions):
    assert 0.0 < base_case_assumptions["base_case_cumulative_default_rate"] < 1.0
    assert 0.0 < base_case_assumptions["base_case_lgd"] < 1.0
    assert base_case_assumptions["base_case_cumulative_loss_rate"] == pytest.approx(
        base_case_assumptions["base_case_cumulative_default_rate"]
        * base_case_assumptions["base_case_lgd"],
        rel=1e-9,
    )


def test_incremental_default_rate_sums_to_cumulative(vintage_data):
    for _, group in vintage_data.groupby("vintage_year"):
        ordered = group.sort_values("mob")
        assert ordered["incremental_default_rate"].sum() == pytest.approx(
            ordered["cumulative_default_rate"].iloc[-1], abs=1e-6
        )
