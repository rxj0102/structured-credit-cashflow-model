import numpy as np
import pytest

from srt_model.monte_carlo_risk import (
    MonteCarloConfig,
    simulate_portfolio_losses,
    analyze_tranche_risk,
    tranche_loss_distribution,
)
from srt_model.cashflow_waterfall import Tranche


def test_simulation_is_reproducible_with_same_seed(small_portfolio):
    cfg = MonteCarloConfig(n_paths=500, seed=123)
    result1 = simulate_portfolio_losses(small_portfolio, cfg)
    result2 = simulate_portfolio_losses(small_portfolio, cfg)
    assert np.array_equal(result1["portfolio_losses"], result2["portfolio_losses"])
    assert np.array_equal(result1["default_counts"], result2["default_counts"])


def test_losses_are_bounded_by_total_balance(small_portfolio):
    result = simulate_portfolio_losses(small_portfolio, MonteCarloConfig(n_paths=500, seed=7))
    assert (result["portfolio_losses"] >= 0).all()
    assert (result["portfolio_losses"] <= result["total_balance"] + 1e-6).all()


def test_higher_correlation_increases_tail_risk_for_similar_expected_loss(small_portfolio):
    low_rho = simulate_portfolio_losses(small_portfolio, MonteCarloConfig(n_paths=4000, rho=0.01, seed=42))
    high_rho = simulate_portfolio_losses(small_portfolio, MonteCarloConfig(n_paths=4000, rho=0.5, seed=42))
    # Correlation should not move the mean much but should fatten the tail.
    low_var99 = np.percentile(low_rho["portfolio_losses"], 99)
    high_var99 = np.percentile(high_rho["portfolio_losses"], 99)
    assert high_var99 > low_var99


def test_convergence_more_paths_reduces_standard_error(small_portfolio):
    cfg_small = MonteCarloConfig(n_paths=300, seed=5)
    cfg_large = MonteCarloConfig(n_paths=6000, seed=5)
    small_result = simulate_portfolio_losses(small_portfolio, cfg_small)
    large_result = simulate_portfolio_losses(small_portfolio, cfg_large)
    se_small = small_result["portfolio_losses"].std(ddof=1) / np.sqrt(cfg_small.n_paths)
    se_large = large_result["portfolio_losses"].std(ddof=1) / np.sqrt(cfg_large.n_paths)
    assert se_large < se_small


def test_tail_metrics_ordering(small_portfolio):
    results = analyze_tranche_risk(small_portfolio, config=MonteCarloConfig(n_paths=3000, seed=9))
    pm = results["portfolio_loss_metrics"]
    assert pm["var_99"] >= pm["var_95"]
    assert pm["expected_shortfall_99"] >= pm["var_99"] - 1e-6
    assert pm["expected_shortfall_95"] >= pm["var_95"] - 1e-6
    assert pm["expected_loss"] <= pm["var_95"] + 1e-6


def test_tranche_loss_distribution_respects_attachment_detachment():
    tranche = Tranche("Mezzanine", attachment_pct=0.04, detachment_pct=0.12, coupon_annual=0.10)
    pool_balance = 1_000_000_000.0
    portfolio_losses = np.array([0.0, 30_000_000.0, 60_000_000.0, 200_000_000.0])
    tranche_losses = tranche_loss_distribution(portfolio_losses, pool_balance, tranche)
    expected = np.array([0.0, 0.0, 20_000_000.0, 80_000_000.0])
    assert np.allclose(tranche_losses, expected)


def test_junior_tranches_have_higher_expected_loss_than_senior(small_portfolio):
    results = analyze_tranche_risk(small_portfolio, config=MonteCarloConfig(n_paths=3000, seed=13))
    senior_el = results["tranche_loss_metrics"]["Senior"]["expected_loss_pct"]
    mezz_el = results["tranche_loss_metrics"]["Mezzanine"]["expected_loss_pct"]
    first_loss_el = results["tranche_loss_metrics"]["First-Loss"]["expected_loss_pct"]
    assert first_loss_el >= mezz_el >= senior_el
