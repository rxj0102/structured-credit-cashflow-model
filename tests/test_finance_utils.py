import pytest

from srt_model.finance_utils import (
    irr,
    npv,
    moic,
    periodic_rate_from_annual,
    annualize_cumulative_rate,
    level_payment_amortization,
)


def test_irr_matches_known_simple_case():
    # -100 today, +110 in one period => IRR is exactly 10%.
    result = irr([-100.0, 110.0])
    assert result == pytest.approx(0.10, abs=1e-6)


def test_irr_npv_is_zero_at_solved_rate():
    cashflows = [-1000.0, 200.0, 300.0, 400.0, 500.0]
    rate = irr(cashflows)
    assert npv(rate, cashflows) == pytest.approx(0.0, abs=1e-6)


def test_irr_total_loss_returns_negative_one():
    # An outflow followed only by zeros is a complete loss -> -100% IRR.
    assert irr([-500.0, 0.0, 0.0, 0.0]) == -1.0


def test_irr_requires_sign_change():
    with pytest.raises(ValueError):
        irr([100.0, 50.0, 25.0])


def test_moic_basic():
    assert moic([100.0], [130.0]) == pytest.approx(1.30)


def test_annualize_cumulative_rate_matches_periodic_rate_from_annual():
    # Annualizing a cumulative rate observed over N years should equal the
    # periodic-rate conversion evaluated at N periods per "year unit".
    cum_rate = 0.30
    years = 5
    result = annualize_cumulative_rate(cum_rate, years)
    expected = periodic_rate_from_annual(cum_rate, years)
    assert result == pytest.approx(expected)


def test_level_payment_amortization_fully_pays_off_balance():
    balance = 100_000.0
    annual_rate = 0.06
    n_periods = 36
    payment = level_payment_amortization(balance, annual_rate, n_periods)

    bal = balance
    r = annual_rate / 12.0
    for _ in range(n_periods):
        interest = bal * r
        principal = payment - interest
        bal -= principal
    assert bal == pytest.approx(0.0, abs=1e-4)
