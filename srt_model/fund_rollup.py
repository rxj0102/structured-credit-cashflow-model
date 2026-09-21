"""
fund_rollup.py
================

Rolls a transaction tranche's cash flows (typically the mezzanine note an
SRT investor/fund purchases) up into a fund-level model: applies
fund-level leverage (a financing/repo facility), an FX hedging cost drag
for a non-USD LP share class, and a fee waterfall (management fee +
carried interest with a hurdle), to produce net LP-level IRR and MOIC.

ALL DATA IS SYNTHETIC. See README.md for the full disclaimer.

Methodology / simplifying assumptions
--------------------------------------
* The fund invests in a single tranche at par. A financing facility
  advances ``advance_rate`` of the investment (e.g. 50%); the LP/GP fund
  equity funds the remainder.
* The facility is a cash-sweep loan: each period, facility interest is
  paid first from the tranche's collections, then any remaining cash
  sweeps facility principal until the facility is fully repaid; only
  after that does cash flow to fund equity. This is a standard structure
  for asset-based "leverage facilities" used by credit funds.
* The LP share class is denominated in a non-USD currency and the fund
  hedges that FX exposure; the hedge cost is modeled as an annualized
  basis-point drag on the outstanding equity NAV (a simplification for a
  cost that in practice depends on the forward curve/interest-rate
  differential).
* Management fee accrues monthly on outstanding invested equity NAV and
  is paid out of distributable cash before any LP/GP split.
* Carried interest uses a standard "European" (whole-fund, deal-by-deal
  accrual) waterfall: (1) return of capital, (2) preferred return
  (compounding on unreturned capital), (3) 100% GP catch-up up to the
  target carry split, (4) residual split (LP get ``1 - carry_rate``, GP
  gets ``carry_rate``).
* The LP is assumed to fund 100% of the fund equity (no separate GP
  co-invest tranche), which keeps the LP-level IRR calculation simple
  without materially changing the mechanics being illustrated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .finance_utils import irr, moic as moic_fn


@dataclass(frozen=True)
class FundConfig:
    advance_rate: float = 0.50            # fraction of tranche notional financed by the facility
    financing_rate_annual: float = 0.075  # facility all-in interest rate
    hedge_cost_annual_bps: float = 150.0  # FX hedge drag, bps per annum on equity NAV
    management_fee_annual: float = 0.015  # % of invested equity NAV per annum
    hurdle_rate_annual: float = 0.08      # LP preferred return
    carry_rate: float = 0.20              # GP carried interest on profit above hurdle
    fund_commitment_pct: float = 1.00     # % of the tranche the fund invests in


def apply_leverage_and_hedge(
    tranche_cashflows: pd.DataFrame,
    tranche_notional: float,
    config: Optional[FundConfig] = None,
) -> Dict[str, object]:
    """Split tranche-level cash flows into facility debt service and
    residual equity cash flow, net of the FX hedge cost drag and the
    management fee.

    Returns a dict with the period-by-period DataFrame and the initial
    equity investment amount (the LP's day-1 capital contribution).
    """
    cfg = config or FundConfig()
    invested_notional = tranche_notional * cfg.fund_commitment_pct
    debt_balance = invested_notional * cfg.advance_rate
    equity_investment = invested_notional * (1.0 - cfg.advance_rate)

    monthly_financing_rate = cfg.financing_rate_annual / 12.0
    monthly_hedge_rate = cfg.hedge_cost_annual_bps / 10_000.0 / 12.0
    monthly_fee_rate = cfg.management_fee_annual / 12.0

    rows = []
    equity_nav = equity_investment
    for _, period in tranche_cashflows.iterrows():
        gross_cf = (period["interest_paid"] + period["principal_paid"]) * cfg.fund_commitment_pct

        debt_interest = debt_balance * monthly_financing_rate
        cash_after_interest = gross_cf - debt_interest
        debt_principal = min(max(cash_after_interest, 0.0), debt_balance)
        debt_balance -= debt_principal
        equity_cf_pre_costs = cash_after_interest - debt_principal

        hedge_cost = equity_nav * monthly_hedge_rate
        management_fee = equity_nav * monthly_fee_rate
        distributable_cf = equity_cf_pre_costs - hedge_cost - management_fee

        # Equity NAV declines by whatever equity cash flow was actually
        # generated this period (a simple NAV roll-forward for fee/hedge
        # base purposes).
        equity_nav = max(equity_nav - max(equity_cf_pre_costs, 0.0), 0.0)

        rows.append(
            {
                "month": period["month"],
                "gross_tranche_cf": gross_cf,
                "debt_interest": debt_interest,
                "debt_principal": debt_principal,
                "debt_balance": debt_balance,
                "hedge_cost": hedge_cost,
                "management_fee": management_fee,
                "distributable_cf": distributable_cf,
            }
        )

    df = pd.DataFrame(rows)
    return {
        "fund_cashflows": df,
        "equity_investment": equity_investment,
        "invested_notional": invested_notional,
    }


def run_fee_waterfall(
    distributable_cf: np.ndarray,
    contributed_capital: float,
    config: Optional[FundConfig] = None,
) -> pd.DataFrame:
    """Allocate a distributable cash flow series between LP and GP using a
    European (whole-fund) return-of-capital / preferred-return / GP
    catch-up / residual-split waterfall, accrued period by period.
    """
    cfg = config or FundConfig()
    monthly_hurdle = (1.0 + cfg.hurdle_rate_annual) ** (1 / 12) - 1.0

    unreturned_capital = contributed_capital
    pref_balance = 0.0
    cumulative_pref_paid = 0.0
    cumulative_catchup_paid = 0.0

    rows = []
    for month_idx, cash in enumerate(distributable_cf, start=1):
        # Preferred return accrues on capital not yet returned.
        pref_balance += unreturned_capital * monthly_hurdle

        remaining = max(cash, 0.0)
        lp_cf = 0.0
        gp_cf = 0.0

        # 1) Return of capital
        roc = min(remaining, unreturned_capital)
        lp_cf += roc
        unreturned_capital -= roc
        remaining -= roc

        # 2) Preferred return
        pref_paid = min(remaining, pref_balance)
        lp_cf += pref_paid
        pref_balance -= pref_paid
        cumulative_pref_paid += pref_paid
        remaining -= pref_paid

        # 3) GP 100% catch-up, until GP carry = carry_rate / (1 - carry_rate) * pref paid
        catchup_target = (cfg.carry_rate / (1.0 - cfg.carry_rate)) * cumulative_pref_paid
        catchup_due = max(catchup_target - cumulative_catchup_paid, 0.0)
        catchup_paid = min(remaining, catchup_due)
        gp_cf += catchup_paid
        cumulative_catchup_paid += catchup_paid
        remaining -= catchup_paid

        # 4) Residual split
        lp_cf += remaining * (1.0 - cfg.carry_rate)
        gp_cf += remaining * cfg.carry_rate

        rows.append(
            {
                "month": month_idx,
                "distributable_cf": cash,
                "lp_cf": lp_cf,
                "gp_cf": gp_cf,
                "unreturned_capital": unreturned_capital,
                "pref_balance": pref_balance,
            }
        )

    return pd.DataFrame(rows)


def fund_rollup(
    tranche_cashflows: pd.DataFrame,
    tranche_notional: float,
    config: Optional[FundConfig] = None,
) -> Dict[str, object]:
    """End-to-end fund roll-up: leverage + hedge + fees + carry -> net LP
    IRR and MOIC."""
    cfg = config or FundConfig()
    leveraged = apply_leverage_and_hedge(tranche_cashflows, tranche_notional, cfg)
    fund_cf = leveraged["fund_cashflows"]
    equity_investment = leveraged["equity_investment"]

    waterfall = run_fee_waterfall(fund_cf["distributable_cf"].to_numpy(), equity_investment, cfg)

    lp_cashflow_series = [-equity_investment] + waterfall["lp_cf"].tolist()
    gp_cashflow_series = [0.0] + waterfall["gp_cf"].tolist()

    try:
        monthly_lp_irr = irr(lp_cashflow_series)
        net_lp_irr = (1 + monthly_lp_irr) ** 12 - 1
    except ValueError:
        net_lp_irr = float("nan")

    net_lp_moic = moic_fn([equity_investment], waterfall["lp_cf"].tolist())

    return {
        "fund_cashflows": fund_cf,
        "waterfall": waterfall,
        "equity_investment": equity_investment,
        "total_lp_distributions": float(waterfall["lp_cf"].sum()),
        "total_gp_carry": float(waterfall["gp_cf"].sum()),
        "net_lp_irr": float(net_lp_irr),
        "net_lp_moic": float(net_lp_moic),
    }


def main() -> None:
    import os

    from .loan_portfolio import generate_synthetic_portfolio, portfolio_summary
    from .historical_loss_analysis import generate_vintage_data, derive_base_case_assumptions, MATURITY_MOB
    from .cashflow_waterfall import PoolAssumptions
    from .scenario_sensitivity import run_all_scenarios
    from .finance_utils import annualize_cumulative_rate

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

    scenario_results = run_all_scenarios(base_assumptions)

    os.makedirs("output", exist_ok=True)
    for scenario_name, result in scenario_results.items():
        mezz = result["waterfall"]["tranches"]["Mezzanine"]
        rollup = fund_rollup(mezz["cashflows"], mezz["original_notional"])
        print(
            f"[{scenario_name}] Mezzanine notional=${mezz['original_notional']:,.0f} "
            f"tranche_yield={mezz['annualized_yield']:.2%} | "
            f"Net LP IRR={rollup['net_lp_irr']:.2%} MOIC={rollup['net_lp_moic']:.2f}x"
        )


if __name__ == "__main__":
    main()
