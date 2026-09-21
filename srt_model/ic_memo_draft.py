"""
ic_memo_draft.py
==================

Orchestrates every other module in the package and assembles a structured
draft Investment Committee (IC) memo in Markdown, pulling every figure
directly from the underlying model outputs (no numbers are hard-coded in
this module).

ALL DATA IS SYNTHETIC. See README.md for the full disclaimer. This memo
is a formatting/orchestration exercise, not a real investment
recommendation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

from .loan_portfolio import PortfolioConfig, generate_synthetic_portfolio, portfolio_summary
from .historical_loss_analysis import (
    VintageConfig,
    MATURITY_MOB,
    generate_vintage_data,
    compute_vintage_curve,
    derive_base_case_assumptions,
)
from .cashflow_waterfall import PoolAssumptions, build_pool_cashflows, run_waterfall, DEFAULT_TRANCHES
from .monte_carlo_risk import MonteCarloConfig, analyze_tranche_risk
from .scenario_sensitivity import run_all_scenarios, summarize_scenarios
from .fund_rollup import FundConfig, fund_rollup
from .finance_utils import annualize_cumulative_rate


@dataclass(frozen=True)
class MemoConfig:
    portfolio_seed: int = 42
    vintage_seed: int = 7
    mc_seed: int = 2024
    n_loans: int = 3000
    mc_paths: int = 8000
    fund_config: FundConfig = FundConfig()


def _fmt_pct(x: float, decimals: int = 2) -> str:
    return f"{x * 100:.{decimals}f}%"


def _fmt_usd(x: float) -> str:
    return f"${x:,.0f}"


def _run_full_model(config: Optional[MemoConfig] = None) -> dict:
    cfg = config or MemoConfig()

    portfolio = generate_synthetic_portfolio(PortfolioConfig(n_loans=cfg.n_loans, seed=cfg.portfolio_seed))
    summary = portfolio_summary(portfolio)

    vintage_df = generate_vintage_data(VintageConfig(seed=cfg.vintage_seed))
    vintage_curve = compute_vintage_curve(vintage_df)
    base_case = derive_base_case_assumptions(vintage_df, vintage_curve)

    base_assumptions = PoolAssumptions(
        beginning_balance=summary["total_current_balance"],
        wac=summary["weighted_average_coupon"],
        wam_months=int(summary["weighted_average_life_months"]),
        annual_cdr=annualize_cumulative_rate(
            base_case["base_case_cumulative_default_rate"], MATURITY_MOB / 12
        ),
        lgd=base_case["base_case_lgd"],
    )
    base_pool_cf = build_pool_cashflows(base_assumptions)
    base_waterfall = run_waterfall(base_pool_cf, base_assumptions.beginning_balance)

    mc_results = analyze_tranche_risk(
        portfolio,
        original_pool_balance=summary["total_current_balance"],
        config=MonteCarloConfig(n_paths=cfg.mc_paths, seed=cfg.mc_seed),
    )

    scenario_results = run_all_scenarios(base_assumptions)
    scenario_table = summarize_scenarios(scenario_results)

    fund_rollups = {}
    for scenario_name, result in scenario_results.items():
        mezz = result["waterfall"]["tranches"]["Mezzanine"]
        fund_rollups[scenario_name] = fund_rollup(mezz["cashflows"], mezz["original_notional"], cfg.fund_config)

    return {
        "config": cfg,
        "portfolio": portfolio,
        "summary": summary,
        "vintage_df": vintage_df,
        "vintage_curve": vintage_curve,
        "base_case": base_case,
        "base_assumptions": base_assumptions,
        "base_waterfall": base_waterfall,
        "mc_results": mc_results,
        "scenario_results": scenario_results,
        "scenario_table": scenario_table,
        "fund_rollups": fund_rollups,
    }


def generate_memo(config: Optional[MemoConfig] = None, as_of: Optional[date] = None) -> str:
    """Generate the full IC memo as a Markdown string."""
    model = _run_full_model(config)
    summary = model["summary"]
    base_case = model["base_case"]
    base_waterfall = model["base_waterfall"]
    mc = model["mc_results"]
    scenario_table = model["scenario_table"]
    fund_rollups = model["fund_rollups"]
    as_of = as_of or date.today()

    sections = []

    # --- Header ---
    sections.append(
        f"# Investment Committee Memo (DRAFT)\n\n"
        f"**Transaction:** Synthetic Significant Risk Transfer (SRT) — Commercial Loan Portfolio\n\n"
        f"**Date:** {as_of.isoformat()}\n\n"
        f"> **DISCLAIMER: All obligor, loan, and performance data in this memo is randomly "
        f"generated for illustrative/educational purposes only. This is not a real transaction, "
        f"and none of the figures below should be used for an actual investment decision.**"
    )

    # --- 1. Transaction overview ---
    sections.append(
        "## 1. Transaction Overview\n\n"
        f"The originating bank is seeking significant risk transfer on a reference portfolio of "
        f"**{summary['n_loans']:,} commercial loans** to **{summary['n_obligors']:,} obligors**, "
        f"with a current aggregate balance of **{_fmt_usd(summary['total_current_balance'])}**. "
        f"The transaction is structured with a synthetic/cash tranched note structure across "
        f"{len(DEFAULT_TRANCHES)} tranches (senior, mezzanine, first-loss); the fund is contemplated "
        f"to invest in the **mezzanine** tranche, which is the tranche typically transferred to "
        f"third-party investors in an SRT transaction while the originating bank retains the senior "
        f"and first-loss pieces."
    )

    # --- 2. Collateral / obligor summary ---
    top_sectors = list(summary["sector_concentration"].items())[:5]
    sector_lines = "\n".join(f"  - {s}: {_fmt_pct(w, 1)}" for s, w in top_sectors)
    rating_lines = "\n".join(
        f"  - {r}: {_fmt_pct(w, 1)}" for r, w in summary["rating_distribution"].items() if w > 0
    )
    sections.append(
        "## 2. Collateral / Obligor Summary\n\n"
        f"- **Loan count / obligor count:** {summary['n_loans']:,} / {summary['n_obligors']:,}\n"
        f"- **Total current balance:** {_fmt_usd(summary['total_current_balance'])}\n"
        f"- **Average loan size:** {_fmt_usd(summary['avg_loan_size'])}\n"
        f"- **Weighted average coupon (WAC):** {_fmt_pct(summary['weighted_average_coupon'])}\n"
        f"- **Weighted average life:** {summary['weighted_average_life_months']:.0f} months\n"
        f"- **Weighted average 1-year PD:** {_fmt_pct(summary['weighted_average_pd_1yr'])}\n"
        f"- **Weighted average LGD:** {_fmt_pct(summary['weighted_average_lgd'])}\n"
        f"- **Top-5 sector concentration:** {_fmt_pct(summary['top5_sector_concentration'], 1)}\n"
        f"{sector_lines}\n"
        f"- **Rating distribution (by balance):**\n{rating_lines}\n"
        f"- **Top-10 obligor concentration:** {_fmt_pct(summary['top10_obligor_concentration'], 1)}\n"
        f"- **Largest single obligor:** {_fmt_pct(summary['largest_obligor_concentration'], 2)}"
    )

    # --- 3. Historical loss analysis ---
    sections.append(
        "## 3. Historical Loss Analysis\n\n"
        f"A vintage-based static-pool analysis of {model['vintage_df']['vintage_year'].nunique()} annual "
        f"origination vintages was used to establish base-case loss assumptions, extrapolating each "
        f"vintage's ultimate cumulative default rate off a dollar-weighted vintage development curve.\n\n"
        f"- **Base-case cumulative default rate (to full seasoning):** "
        f"{_fmt_pct(base_case['base_case_cumulative_default_rate'])}\n"
        f"- **Base-case LGD:** {_fmt_pct(base_case['base_case_lgd'])}\n"
        f"- **Base-case cumulative loss rate:** {_fmt_pct(base_case['base_case_cumulative_loss_rate'])}\n"
        f"- **Implied base-case annual default rate (CDR) used in cash flow model:** "
        f"{_fmt_pct(model['base_assumptions'].annual_cdr, 3)}"
    )

    # --- 4. Structure and tranching ---
    tranche_lines = []
    for name in base_waterfall["tranche_order"]:
        t = base_waterfall["tranches"][name]
        tranche_lines.append(
            f"  - **{name}:** notional {_fmt_usd(t['original_notional'])}, "
            f"base-case yield {_fmt_pct(t['annualized_yield'])}, "
            f"cumulative loss absorbed {_fmt_usd(t['cumulative_loss_absorbed'])} "
            f"({_fmt_pct(t['pct_of_notional_lost'], 1)} of notional)"
        )
    sections.append(
        "## 4. Structure and Tranching\n\n"
        "Base-case pool cash flows (interest, scheduled principal, prepayments, net of "
        "defaults/recoveries) are allocated sequentially through the tranche structure below "
        "(senior-first principal and interest pay, reverse-seniority loss allocation):\n\n"
        + "\n".join(tranche_lines)
    )

    # --- 5. Monte Carlo risk summary ---
    pm = mc["portfolio_loss_metrics"]
    mezz_m = mc["tranche_loss_metrics"]["Mezzanine"]
    fl_m = mc["tranche_loss_metrics"]["First-Loss"]
    sections.append(
        "## 5. Monte Carlo Risk Summary\n\n"
        f"A single-factor Gaussian copula simulation was run over **{mc['n_paths']:,} paths** "
        f"to capture correlated obligor default risk.\n\n"
        f"**Portfolio level:**\n"
        f"- Expected loss: {_fmt_pct(pm['expected_loss_pct'])}\n"
        f"- 95th / 99th percentile loss (VaR): {_fmt_pct(pm['var_95_pct'])} / {_fmt_pct(pm['var_99_pct'])}\n"
        f"- 99% Expected Shortfall (CVaR): {_fmt_pct(pm['expected_shortfall_99_pct'])}\n\n"
        f"**Mezzanine tranche (fund's investment):**\n"
        f"- Expected loss: {_fmt_pct(mezz_m['expected_loss_pct'])}\n"
        f"- 95th / 99th percentile loss: {_fmt_pct(mezz_m['var_95_pct'])} / {_fmt_pct(mezz_m['var_99_pct'])}\n"
        f"- 99% Expected Shortfall: {_fmt_pct(mezz_m['expected_shortfall_99_pct'])}\n\n"
        f"**First-loss tranche (retained by bank):**\n"
        f"- Expected loss: {_fmt_pct(fl_m['expected_loss_pct'])}\n"
        f"- Probability of full wipeout: {_fmt_pct(fl_m['prob_full_wipeout'], 1)}"
    )

    # --- 6. Scenario sensitivity ---
    mezz_scn = scenario_table[scenario_table["tranche"] == "Mezzanine"]
    scn_lines = []
    for _, row in mezz_scn.iterrows():
        scn_lines.append(
            f"  - **{row['scenario']}** (CDR x, LGD, CPR -> annual CDR {row['annual_cdr']:.2%}, "
            f"LGD {row['lgd']:.1%}, CPR {row['annual_cpr']:.1%}): "
            f"mezzanine yield {row['annualized_yield']:.2%} "
            f"({row['yield_change_vs_base_bps']:+.0f} bps vs. base)"
        )
    sections.append(
        "## 6. Scenario Sensitivity\n\n"
        "Base / Stress / Severe Stress scenarios were run, jointly stressing default rate, LGD, "
        "and prepayment speed:\n\n" + "\n".join(scn_lines)
    )

    # --- 7. Fund-level economics ---
    fund_lines = []
    for scenario_name, rollup in fund_rollups.items():
        fund_lines.append(
            f"  - **{scenario_name}:** Net LP IRR {_fmt_pct(rollup['net_lp_irr'])}, "
            f"MOIC {rollup['net_lp_moic']:.2f}x "
            f"(equity invested {_fmt_usd(rollup['equity_investment'])}, "
            f"total GP carry {_fmt_usd(rollup['total_gp_carry'])})"
        )
    sections.append(
        "## 7. Fund-Level Economics\n\n"
        f"Applying fund-level leverage (facility advance rate "
        f"{_fmt_pct(model['config'].fund_config.advance_rate, 0)}), an FX hedging cost of "
        f"{model['config'].fund_config.hedge_cost_annual_bps:.0f} bps/annum, and the fee waterfall "
        f"(management fee {_fmt_pct(model['config'].fund_config.management_fee_annual, 2)}, "
        f"{_fmt_pct(model['config'].fund_config.hurdle_rate_annual, 0)} hurdle, "
        f"{_fmt_pct(model['config'].fund_config.carry_rate, 0)} carry) to the mezzanine tranche cash flows:\n\n"
        + "\n".join(fund_lines)
    )

    # --- 8. Recommendation ---
    base_irr = fund_rollups["Base"]["net_lp_irr"]
    severe_irr = fund_rollups.get("Severe Stress", {}).get("net_lp_irr", float("nan"))
    sections.append(
        "## 8. Recommendation (Draft)\n\n"
        f"Under base-case assumptions derived from the historical vintage analysis, the mezzanine "
        f"investment generates a projected net LP IRR of {_fmt_pct(base_irr)}. Under the severe "
        f"stress scenario, net LP IRR moves to {_fmt_pct(severe_irr)}, reflecting the combined "
        f"effect of higher defaults, weaker recoveries, and slower prepayments (extension risk). "
        f"Key risk factors to monitor include sector concentration "
        f"({_fmt_pct(summary['top5_sector_concentration'], 1)} in the top 5 sectors) and single-name "
        f"concentration ({_fmt_pct(summary['top10_obligor_concentration'], 1)} in the top 10 obligors). "
        f"**This is a synthetic-data illustration only and does not constitute an investment "
        f"recommendation.**"
    )

    return "\n\n".join(sections) + "\n"


def main() -> None:
    import os

    memo = generate_memo()
    os.makedirs("output", exist_ok=True)
    with open("output/IC_Memo.md", "w") as f:
        f.write(memo)
    print(memo)
    print("\nSaved memo to output/IC_Memo.md")


if __name__ == "__main__":
    main()
