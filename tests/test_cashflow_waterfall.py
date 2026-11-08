from dataclasses import replace

import numpy as np
import pytest

from srt_model.cashflow_waterfall import (
    PoolAssumptions,
    Tranche,
    build_pool_cashflows,
    run_waterfall,
    tranche_notional,
    DEFAULT_TRANCHES,
)


def test_pool_cashflows_amortize_to_zero(pool_cashflows):
    assert pool_cashflows["ending_balance"].iloc[-1] == pytest.approx(0.0, abs=1.0)


def test_pool_cashflows_are_reproducible(pool_assumptions):
    cf1 = build_pool_cashflows(pool_assumptions)
    cf2 = build_pool_cashflows(pool_assumptions)
    assert np.allclose(cf1["ending_balance"], cf2["ending_balance"])


def test_zero_default_zero_prepay_recovers_full_principal():
    assumptions = PoolAssumptions(
        beginning_balance=1_000_000.0,
        wac=0.06,
        wam_months=36,
        annual_cdr=0.0,
        lgd=0.4,
        annual_cpr=0.0,
    )
    cf = build_pool_cashflows(assumptions)
    total_principal = cf["scheduled_principal"].sum() + cf["prepayment"].sum()
    assert total_principal == pytest.approx(1_000_000.0, rel=1e-6)
    assert cf["defaulted_balance"].sum() == pytest.approx(0.0, abs=1e-6)


def test_tranche_notionals_sum_to_pool_balance(pool_assumptions):
    total = sum(tranche_notional(t, pool_assumptions.beginning_balance) for t in DEFAULT_TRANCHES)
    assert total == pytest.approx(pool_assumptions.beginning_balance)


def test_no_loss_scenario_all_tranches_fully_intact():
    assumptions = PoolAssumptions(
        beginning_balance=1_000_000_000.0, wac=0.08, wam_months=60,
        annual_cdr=0.0, lgd=0.4, annual_cpr=0.05,
    )
    cf = build_pool_cashflows(assumptions)
    results = run_waterfall(cf, assumptions.beginning_balance)
    for name, t in results["tranches"].items():
        assert t["cumulative_loss_absorbed"] == pytest.approx(0.0, abs=1.0)


def test_high_stress_wipes_out_first_loss_before_mezzanine():
    assumptions = PoolAssumptions(
        beginning_balance=1_000_000_000.0, wac=0.08, wam_months=60,
        annual_cdr=0.08, lgd=0.6, annual_cpr=0.05,
    )
    cf = build_pool_cashflows(assumptions)
    results = run_waterfall(cf, assumptions.beginning_balance)
    first_loss = results["tranches"]["First-Loss"]
    mezz = results["tranches"]["Mezzanine"]
    assert first_loss["pct_of_notional_lost"] == pytest.approx(1.0, abs=1e-6)
    assert mezz["cumulative_loss_absorbed"] > 0.0


def test_senior_untouched_while_subordinate_tranches_absorb_losses():
    assumptions = PoolAssumptions(
        beginning_balance=1_000_000_000.0, wac=0.08, wam_months=60,
        annual_cdr=0.05, lgd=0.5, annual_cpr=0.05,
    )
    cf = build_pool_cashflows(assumptions)
    results = run_waterfall(cf, assumptions.beginning_balance)
    assert results["tranches"]["Senior"]["cumulative_loss_absorbed"] == pytest.approx(0.0, abs=1.0)
    assert results["tranches"]["First-Loss"]["cumulative_loss_absorbed"] > 0.0


def test_losses_never_exceed_tranche_notional():
    assumptions = PoolAssumptions(
        beginning_balance=1_000_000_000.0, wac=0.08, wam_months=60,
        annual_cdr=0.25, lgd=0.9, annual_cpr=0.0,
    )
    cf = build_pool_cashflows(assumptions)
    results = run_waterfall(cf, assumptions.beginning_balance)
    for t in results["tranches"].values():
        assert t["cumulative_loss_absorbed"] <= t["original_notional"] + 1e-6
        assert t["ending_balance"] >= -1e-6


def test_run_waterfall_accepts_unordered_tranche_list():
    shuffled = list(reversed(DEFAULT_TRANCHES))
    assumptions = PoolAssumptions(
        beginning_balance=500_000_000.0, wac=0.08, wam_months=48,
        annual_cdr=0.02, lgd=0.4, annual_cpr=0.05,
    )
    cf = build_pool_cashflows(assumptions)
    results_shuffled = run_waterfall(cf, assumptions.beginning_balance, tranches=shuffled)
    results_ordered = run_waterfall(cf, assumptions.beginning_balance, tranches=DEFAULT_TRANCHES)
    for name in results_ordered["tranche_order"]:
        assert results_shuffled["tranches"][name]["cumulative_loss_absorbed"] == pytest.approx(
            results_ordered["tranches"][name]["cumulative_loss_absorbed"]
        )


def test_waterfall_distributes_all_cash_and_repays_all_balances():
    a = PoolAssumptions(
        beginning_balance=1e9, wac=0.07, wam_months=60,
        annual_cdr=0.05, lgd=0.4, annual_cpr=0.1,
    )
    cf = build_pool_cashflows(a)
    res = run_waterfall(cf, 1e9)
    paid = sum(
        t["cashflows"][["interest_paid", "principal_paid"]].sum().sum()
        for t in res["tranches"].values()
    )
    assert paid == pytest.approx(cf["total_cash_available"].sum(), rel=1e-9)
    for t in res["tranches"].values():
        assert t["ending_balance"] == pytest.approx(0.0, abs=1.0)
