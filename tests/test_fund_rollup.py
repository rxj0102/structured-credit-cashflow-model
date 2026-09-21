import numpy as np
import pandas as pd
import pytest

from srt_model.fund_rollup import (
    FundConfig,
    apply_leverage_and_hedge,
    run_fee_waterfall,
    fund_rollup,
)


@pytest.fixture
def flat_tranche_cashflows():
    """A simple, hand-computable tranche cash flow series: $100 interest
    per month for 12 months, then a single $1,000 principal repayment."""
    rows = []
    for m in range(1, 13):
        principal = 1000.0 if m == 12 else 0.0
        rows.append({"month": m, "interest_paid": 100.0, "principal_paid": principal})
    return pd.DataFrame(rows)


def test_equity_investment_matches_advance_rate(flat_tranche_cashflows):
    cfg = FundConfig(advance_rate=0.6)
    result = apply_leverage_and_hedge(flat_tranche_cashflows, tranche_notional=1000.0, config=cfg)
    assert result["equity_investment"] == pytest.approx(400.0)
    assert result["fund_cashflows"]["debt_balance"].iloc[0] == pytest.approx(
        600.0 - result["fund_cashflows"]["debt_principal"].iloc[0]
    )


def test_zero_advance_rate_means_no_leverage(flat_tranche_cashflows):
    cfg = FundConfig(advance_rate=0.0, hedge_cost_annual_bps=0.0, management_fee_annual=0.0)
    result = apply_leverage_and_hedge(flat_tranche_cashflows, tranche_notional=1000.0, config=cfg)
    fund_cf = result["fund_cashflows"]
    gross = flat_tranche_cashflows["interest_paid"] + flat_tranche_cashflows["principal_paid"]
    assert np.allclose(fund_cf["distributable_cf"], gross)
    assert result["equity_investment"] == pytest.approx(1000.0)


def test_debt_balance_is_non_increasing(flat_tranche_cashflows):
    cfg = FundConfig(advance_rate=0.5)
    result = apply_leverage_and_hedge(flat_tranche_cashflows, tranche_notional=1000.0, config=cfg)
    debt = result["fund_cashflows"]["debt_balance"].to_numpy()
    assert np.all(np.diff(debt) <= 1e-9)


def test_fee_waterfall_pays_return_of_capital_before_preferred_return():
    contributed_capital = 1000.0
    distributable = np.array([500.0] + [0.0] * 11)
    cfg = FundConfig(hurdle_rate_annual=0.08, carry_rate=0.20)
    result = run_fee_waterfall(distributable, contributed_capital, cfg)
    assert result["lp_cf"].iloc[0] == pytest.approx(500.0)
    assert result.loc[0, "unreturned_capital"] == pytest.approx(500.0)


def test_fee_waterfall_conserves_total_cash():
    contributed_capital = 1000.0
    rng = np.random.default_rng(0)
    distributable = rng.uniform(0, 200, size=36)
    cfg = FundConfig()
    result = run_fee_waterfall(distributable, contributed_capital, cfg)
    assert (result["lp_cf"] + result["gp_cf"]).sum() == pytest.approx(distributable.sum(), rel=1e-6)


def test_gp_catchup_targets_correct_ultimate_split():
    # Enough cash to fully return capital, pay preferred, catch the GP up,
    # then split a large residual 80/20.
    contributed_capital = 1000.0
    distributable = np.array([2000.0] + [0.0] * 11)
    cfg = FundConfig(hurdle_rate_annual=0.08, carry_rate=0.20)
    result = run_fee_waterfall(distributable, contributed_capital, cfg)
    total_profit = distributable.sum() - contributed_capital
    total_gp = result["gp_cf"].sum()
    # GP's overall share of profit should approach the carry rate once
    # there is enough profit for the catch-up to complete.
    assert total_gp / total_profit == pytest.approx(cfg.carry_rate, abs=0.02)


def test_fund_rollup_total_loss_gives_negative_hundred_irr():
    cf = pd.DataFrame({"month": range(1, 13), "interest_paid": 0.0, "principal_paid": 0.0})
    result = fund_rollup(cf, tranche_notional=1000.0, config=FundConfig(advance_rate=0.0))
    assert result["net_lp_irr"] == pytest.approx(-1.0)
    assert result["net_lp_moic"] == pytest.approx(0.0)


def test_fund_rollup_positive_case_has_reasonable_irr(flat_tranche_cashflows):
    result = fund_rollup(flat_tranche_cashflows, tranche_notional=1000.0, config=FundConfig(advance_rate=0.0, hedge_cost_annual_bps=0.0, management_fee_annual=0.0))
    assert result["net_lp_irr"] > 0.0
    assert result["net_lp_moic"] > 1.0
