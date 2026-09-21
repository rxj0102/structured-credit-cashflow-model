import pytest

from srt_model.scenario_sensitivity import (
    ScenarioDefinition,
    SCENARIOS,
    apply_scenario,
    run_all_scenarios,
    summarize_scenarios,
)


def test_scenarios_are_ordered_by_increasing_severity():
    cdrs = [s.cdr_multiplier for s in SCENARIOS]
    lgds = [s.lgd_multiplier for s in SCENARIOS]
    cprs = [s.cpr_multiplier for s in SCENARIOS]
    assert cdrs == sorted(cdrs)
    assert lgds == sorted(lgds)
    assert cprs == sorted(cprs, reverse=True)


def test_apply_scenario_scales_assumptions_correctly(pool_assumptions):
    scenario = ScenarioDefinition("Test", cdr_multiplier=2.0, lgd_multiplier=1.5, cpr_multiplier=0.5)
    stressed = apply_scenario(pool_assumptions, scenario)
    assert stressed.annual_cdr == pytest.approx(pool_assumptions.annual_cdr * 2.0)
    assert stressed.lgd == pytest.approx(pool_assumptions.lgd * 1.5)
    assert stressed.annual_cpr == pytest.approx(pool_assumptions.annual_cpr * 0.5)
    # Everything else should be unchanged.
    assert stressed.wac == pool_assumptions.wac
    assert stressed.beginning_balance == pool_assumptions.beginning_balance


def test_base_scenario_is_a_no_op(pool_assumptions):
    base = next(s for s in SCENARIOS if s.name == "Base")
    stressed = apply_scenario(pool_assumptions, base)
    assert stressed == pool_assumptions


def test_stress_scenarios_degrade_mezzanine_tranche_yield(pool_assumptions):
    # Unlike the thin first-loss/equity tranche (whose yield can actually
    # rise under mild stress because slower prepayments let it capture more
    # excess spread before being wiped out), the fixed-coupon mezzanine
    # tranche has no such offsetting benefit: its yield should degrade (or
    # stay flat) monotonically as credit stress increases.
    results = run_all_scenarios(pool_assumptions)
    base_yield = results["Base"]["waterfall"]["tranches"]["Mezzanine"]["annualized_yield"]
    stress_yield = results["Stress"]["waterfall"]["tranches"]["Mezzanine"]["annualized_yield"]
    severe_yield = results["Severe Stress"]["waterfall"]["tranches"]["Mezzanine"]["annualized_yield"]
    assert stress_yield <= base_yield + 1e-9
    assert severe_yield <= stress_yield + 1e-9


def test_summarize_scenarios_base_has_zero_yield_change(pool_assumptions):
    results = run_all_scenarios(pool_assumptions)
    table = summarize_scenarios(results)
    base_rows = table[table["scenario"] == "Base"]
    assert (base_rows["yield_change_vs_base_bps"].abs() < 1e-6).all()


def test_summarize_scenarios_has_one_row_per_scenario_tranche_pair(pool_assumptions):
    results = run_all_scenarios(pool_assumptions)
    table = summarize_scenarios(results)
    n_scenarios = len(results)
    n_tranches = len(results["Base"]["waterfall"]["tranche_order"])
    assert len(table) == n_scenarios * n_tranches
