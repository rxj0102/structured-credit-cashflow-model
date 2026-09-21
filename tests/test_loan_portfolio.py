import pytest

from srt_model.loan_portfolio import (
    PortfolioConfig,
    generate_synthetic_portfolio,
    portfolio_summary,
)


def test_generates_requested_number_of_loans():
    df = generate_synthetic_portfolio(PortfolioConfig(n_loans=500, seed=1))
    assert len(df) == 500


def test_generation_is_reproducible_with_same_seed():
    df1 = generate_synthetic_portfolio(PortfolioConfig(n_loans=300, seed=77))
    df2 = generate_synthetic_portfolio(PortfolioConfig(n_loans=300, seed=77))
    pd_testing_cols = ["loan_id", "current_balance", "interest_rate", "pd_1yr", "lgd"]
    for col in pd_testing_cols:
        assert (df1[col] == df2[col]).all()


def test_balances_are_within_configured_bounds(small_portfolio):
    cfg = PortfolioConfig()
    assert (small_portfolio["original_balance"] >= cfg.min_balance).all()
    assert (small_portfolio["original_balance"] <= cfg.max_balance).all()
    assert (small_portfolio["current_balance"] >= 0).all()
    assert (small_portfolio["current_balance"] <= small_portfolio["original_balance"] + 1e-6).all()


def test_pd_and_lgd_are_valid_probabilities(small_portfolio):
    assert (small_portfolio["pd_1yr"] > 0).all()
    assert (small_portfolio["pd_1yr"] < 1).all()
    assert (small_portfolio["lgd"] > 0).all()
    assert (small_portfolio["lgd"] < 1).all()


def test_summary_sector_concentration_sums_to_one(small_portfolio):
    summary = portfolio_summary(small_portfolio)
    total = sum(summary["sector_concentration"].values())
    assert total == pytest.approx(1.0, abs=1e-9)


def test_summary_rating_distribution_sums_to_one(small_portfolio):
    summary = portfolio_summary(small_portfolio)
    total = sum(summary["rating_distribution"].values())
    assert total == pytest.approx(1.0, abs=1e-9)


def test_obligor_concentration_bounds(small_portfolio):
    summary = portfolio_summary(small_portfolio)
    assert 0.0 < summary["largest_obligor_concentration"] <= 1.0
    assert summary["largest_obligor_concentration"] <= summary["top10_obligor_concentration"]
    assert summary["top10_obligor_concentration"] <= 1.0
