"""
scenario_sensitivity.py
=========================

Runs base / stress / severe-stress scenario analysis on the pool cash
flow and tranche waterfall model, varying default rate (CDR), loss
severity (LGD) and prepayment speed (CPR) assumptions, and reports the
resulting impact on tranche-level yields.

ALL DATA IS SYNTHETIC. See README.md for the full disclaimer.

Scenario design
----------------
Multipliers are applied to the base-case assumptions derived in
``historical_loss_analysis.py``:

================  =====  ======  =============
Scenario          CDR    LGD     CPR
================  =====  ======  =============
Base                1.0x   1.0x    1.0x
Stress              2.0x   1.20x   0.60x
Severe Stress       3.5x   1.45x   0.30x
================  =====  ======  =============

Rationale: in a downturn, defaults rise, recoveries weaken (LGD rises),
and voluntary prepayments slow as refinancing options dry up -- all three
levers move against the tranche investor simultaneously, which is the
standard way SRT/CLO analysts construct a stress scenario.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional

import pandas as pd

from .cashflow_waterfall import PoolAssumptions, build_pool_cashflows, run_waterfall, DEFAULT_TRANCHES, Tranche
from .finance_utils import annualize_cumulative_rate


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    cdr_multiplier: float
    lgd_multiplier: float
    cpr_multiplier: float


SCENARIOS: List[ScenarioDefinition] = [
    ScenarioDefinition("Base", cdr_multiplier=1.00, lgd_multiplier=1.00, cpr_multiplier=1.00),
    ScenarioDefinition("Stress", cdr_multiplier=2.00, lgd_multiplier=1.20, cpr_multiplier=0.60),
    ScenarioDefinition("Severe Stress", cdr_multiplier=3.50, lgd_multiplier=1.45, cpr_multiplier=0.30),
]


def apply_scenario(base_assumptions: PoolAssumptions, scenario: ScenarioDefinition) -> PoolAssumptions:
    """Return a new PoolAssumptions with the scenario's multipliers applied,
    clipping LGD to a physically valid [0, 1] range."""
    stressed_lgd = min(base_assumptions.lgd * scenario.lgd_multiplier, 0.99)
    return replace(
        base_assumptions,
        annual_cdr=base_assumptions.annual_cdr * scenario.cdr_multiplier,
        lgd=stressed_lgd,
        annual_cpr=base_assumptions.annual_cpr * scenario.cpr_multiplier,
    )


def run_scenario(
    base_assumptions: PoolAssumptions,
    scenario: ScenarioDefinition,
    original_pool_balance: Optional[float] = None,
    tranches: Optional[List[Tranche]] = None,
) -> Dict[str, object]:
    """Run the full pool cash flow + waterfall model under one scenario."""
    stressed = apply_scenario(base_assumptions, scenario)
    pool_cf = build_pool_cashflows(stressed)
    balance = original_pool_balance or base_assumptions.beginning_balance
    waterfall = run_waterfall(pool_cf, balance, tranches)
    return {
        "scenario": scenario,
        "assumptions": stressed,
        "pool_cashflows": pool_cf,
        "waterfall": waterfall,
    }


def run_all_scenarios(
    base_assumptions: PoolAssumptions,
    scenarios: Optional[List[ScenarioDefinition]] = None,
    tranches: Optional[List[Tranche]] = None,
) -> Dict[str, Dict[str, object]]:
    scenarios = scenarios or SCENARIOS
    return {
        s.name: run_scenario(base_assumptions, s, tranches=tranches) for s in scenarios
    }


def summarize_scenarios(scenario_results: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    """Flatten scenario results into a tidy DataFrame: one row per
    (scenario, tranche) with yield, loss absorbed, and assumption inputs."""
    rows = []
    for scenario_name, result in scenario_results.items():
        assumptions = result["assumptions"]
        waterfall = result["waterfall"]
        for tranche_name in waterfall["tranche_order"]:
            t = waterfall["tranches"][tranche_name]
            rows.append(
                {
                    "scenario": scenario_name,
                    "tranche": tranche_name,
                    "annual_cdr": assumptions.annual_cdr,
                    "lgd": assumptions.lgd,
                    "annual_cpr": assumptions.annual_cpr,
                    "annualized_yield": t["annualized_yield"],
                    "cumulative_loss_absorbed": t["cumulative_loss_absorbed"],
                    "pct_of_notional_lost": t["pct_of_notional_lost"],
                    "original_notional": t["original_notional"],
                }
            )
    df = pd.DataFrame(rows)
    base_yields = df[df["scenario"] == "Base"].set_index("tranche")["annualized_yield"]
    df["yield_change_vs_base_bps"] = df.apply(
        lambda r: (r["annualized_yield"] - base_yields.get(r["tranche"], float("nan"))) * 10_000,
        axis=1,
    )
    return df


def main() -> None:
    import os

    from .loan_portfolio import generate_synthetic_portfolio, portfolio_summary
    from .historical_loss_analysis import generate_vintage_data, derive_base_case_assumptions, MATURITY_MOB

    port_df = generate_synthetic_portfolio()
    summary = portfolio_summary(port_df)
    vintage_df = generate_vintage_data()
    base_case = derive_base_case_assumptions(vintage_df)

    base_assumptions = PoolAssumptions(
        beginning_balance=summary["total_current_balance"],
        wac=summary["weighted_average_coupon"],
        wam_months=int(summary["weighted_average_life_months"]),
        annual_cdr=annualize_cumulative_rate(
            base_case["base_case_cumulative_default_rate"], MATURITY_MOB / 12
        ),
        lgd=base_case["base_case_lgd"],
    )

    results = run_all_scenarios(base_assumptions)
    table = summarize_scenarios(results)

    os.makedirs("output", exist_ok=True)
    table.to_csv("output/scenario_sensitivity.csv", index=False)

    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print(table[["scenario", "tranche", "annual_cdr", "lgd", "annualized_yield", "yield_change_vs_base_bps"]])


if __name__ == "__main__":
    main()
