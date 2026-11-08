"""
cashflow_waterfall.py
======================

Projects aggregate portfolio (pool-level) cash flows net of defaults and
losses, then allocates those cash flows through a tranched capital
structure typical of a synthetic/cash SRT transaction (senior / mezzanine
/ first-loss), computing each tranche's cash flow stream, realized yield,
and loss-absorption sequence.

ALL DATA IS SYNTHETIC. See README.md for the full disclaimer.

Methodology / simplifying assumptions
--------------------------------------
* The loan pool is modeled as a single aggregated amortizing balance
  (standard ABS "pool factor" approach) characterized by its weighted
  average coupon (WAC) and weighted average life (WAL/WAM), rather than
  loan-by-loan cash flows. This is the standard level of granularity for
  a waterfall/tranche cash flow model.
* Annual assumptions (CDR = conditional/constant default rate, CPR =
  conditional prepayment rate) are converted to monthly rates via the
  standard compound formula: monthly rate = 1 - (1 - annual rate)^(1/12).
* Defaulted balances are removed from the performing pool at the month of
  default; recoveries (= defaulted balance * (1 - LGD)) are collected
  after a fixed ``recovery_lag_months``.
* Net period loss = gross defaults in the period - recoveries collected
  in the period. Losses are allocated to tranches in reverse seniority
  (first-loss/equity absorbs first) as permanent write-downs of tranche
  balance -- i.e. no subsequent write-up from later recoveries. This is a
  simplifying (conservative-for-mezz-and-senior) assumption common to a
  first-pass model.
* Principal collections (scheduled amortization + prepayments) are paid
  sequentially, senior tranche first, consistent with a typical SRT note
  structure where the senior piece amortizes fastest.
* Each tranche earns its own coupon on its beginning balance; interest is
  paid sequentially by seniority, with any residual after paying all
  tranche coupons flowing to the first-loss/equity tranche as excess
  spread.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .finance_utils import irr, periodic_rate_from_annual, level_payment_amortization, annualize_cumulative_rate

# ---------------------------------------------------------------------------
# Pool-level assumption / cash flow projection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolAssumptions:
    """Aggregate pool assumptions driving the cash flow projection."""

    beginning_balance: float
    wac: float                       # weighted average coupon, annual
    wam_months: int                  # weighted average remaining life, months
    annual_cdr: float                # annual conditional default rate
    lgd: float                       # loss given default
    annual_cpr: float = 0.05         # annual conditional prepayment rate
    recovery_lag_months: int = 12
    servicing_fee_annual: float = 0.0025


def build_pool_cashflows(assumptions: PoolAssumptions) -> pd.DataFrame:
    """Project monthly pool cash flows (interest, scheduled principal,
    defaults, recoveries, prepayments) to zero balance / maturity.

    Returns a DataFrame indexed by month (1-based) with columns:
    beginning_balance, interest_collected, scheduled_principal,
    defaulted_balance, recoveries_received, prepayment, net_loss,
    ending_balance, total_principal_collected (sched + prepay, excludes
    defaults), total_cash_available (interest + total_principal_collected
    + recoveries_received).
    """
    a = assumptions
    monthly_mdr = periodic_rate_from_annual(a.annual_cdr, 12)
    monthly_smm = periodic_rate_from_annual(a.annual_cpr, 12)
    monthly_wac = a.wac / 12.0
    monthly_fee = a.servicing_fee_annual / 12.0

    payment = level_payment_amortization(a.beginning_balance, a.wac, a.wam_months)

    rows = []
    balance = a.beginning_balance
    # Pending recoveries keyed by the month they will be received.
    pending_recoveries: Dict[int, float] = {}
    month = 0
    max_months = a.wam_months + a.recovery_lag_months + 1

    while (balance > 1e-6 or any(v > 1e-6 for v in pending_recoveries.values())) and month < max_months:
        month += 1
        beginning_balance = balance
        interest_collected = beginning_balance * monthly_wac
        servicing_fee = beginning_balance * monthly_fee

        if beginning_balance <= 1e-6:
            scheduled_principal = 0.0
            defaulted_balance = 0.0
            prepayment = 0.0
        else:
            interest_only = beginning_balance * monthly_wac
            scheduled_principal = max(min(payment - interest_only, beginning_balance), 0.0)
            balance_after_sched = beginning_balance - scheduled_principal
            defaulted_balance = balance_after_sched * monthly_mdr
            balance_after_default = balance_after_sched - defaulted_balance
            prepayment = balance_after_default * monthly_smm

        recoveries_received = pending_recoveries.pop(month, 0.0)
        if defaulted_balance > 0:
            recovery_amount = defaulted_balance * (1.0 - a.lgd)
            recv_month = month + a.recovery_lag_months
            pending_recoveries[recv_month] = pending_recoveries.get(recv_month, 0.0) + recovery_amount

        ending_balance = max(beginning_balance - scheduled_principal - defaulted_balance - prepayment, 0.0)
        net_loss = defaulted_balance - recoveries_received
        total_principal_collected = scheduled_principal + prepayment
        total_cash_available = interest_collected - servicing_fee + total_principal_collected + recoveries_received

        rows.append(
            {
                "month": month,
                "beginning_balance": beginning_balance,
                "interest_collected": interest_collected,
                "servicing_fee": servicing_fee,
                "scheduled_principal": scheduled_principal,
                "defaulted_balance": defaulted_balance,
                "recoveries_received": recoveries_received,
                "prepayment": prepayment,
                "net_loss": net_loss,
                "ending_balance": ending_balance,
                "total_principal_collected": total_principal_collected,
                "total_cash_available": total_cash_available,
            }
        )
        balance = ending_balance

    # Flush any recoveries still pending after the balance has amortized to zero.
    if pending_recoveries:
        for recv_month in sorted(pending_recoveries):
            month = max(month, recv_month)
        for recv_month, amount in sorted(pending_recoveries.items()):
            rows.append(
                {
                    "month": recv_month,
                    "beginning_balance": 0.0,
                    "interest_collected": 0.0,
                    "servicing_fee": 0.0,
                    "scheduled_principal": 0.0,
                    "defaulted_balance": 0.0,
                    "recoveries_received": amount,
                    "prepayment": 0.0,
                    "net_loss": -amount,
                    "ending_balance": 0.0,
                    "total_principal_collected": 0.0,
                    "total_cash_available": amount,
                }
            )

    df = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    # Recoveries flushed after amortization end may create duplicate months
    # with zero-balance rows; combine same-month rows.
    df = df.groupby("month", as_index=False).sum(numeric_only=True)
    return df


# ---------------------------------------------------------------------------
# Tranche structure and waterfall allocation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tranche:
    name: str
    attachment_pct: float   # % of original pool balance, loss begins to hit this tranche above this point
    detachment_pct: float   # % of original pool balance, tranche fully wiped out above this point
    coupon_annual: float


DEFAULT_TRANCHES: List[Tranche] = [
    Tranche("Senior", attachment_pct=0.08, detachment_pct=1.00, coupon_annual=0.055),
    Tranche("Mezzanine", attachment_pct=0.02, detachment_pct=0.08, coupon_annual=0.120),
    Tranche("First-Loss", attachment_pct=0.00, detachment_pct=0.02, coupon_annual=0.00),
]


def tranche_notional(tranche: Tranche, original_pool_balance: float) -> float:
    return (tranche.detachment_pct - tranche.attachment_pct) * original_pool_balance


def run_waterfall(
    pool_cashflows: pd.DataFrame,
    original_pool_balance: float,
    tranches: Optional[List[Tranche]] = None,
) -> Dict[str, object]:
    """Allocate pool cash flows and losses through the tranche structure.

    Tranches must be supplied ordered from most senior to most junior.
    Returns a dict with per-tranche cash flow DataFrames, summary metrics
    (yield, cumulative loss absorbed, % of notional lost), and the pool
    cash flow DataFrame the allocation was run against.
    """
    tranches = tranches or DEFAULT_TRANCHES
    # Ensure senior-first ordering for the sequential-pay logic below.
    ordered = sorted(tranches, key=lambda t: -t.attachment_pct)

    balances = {t.name: tranche_notional(t, original_pool_balance) for t in ordered}
    original_balances = dict(balances)
    tranche_rows: Dict[str, List[dict]] = {t.name: [] for t in ordered}

    for _, period in pool_cashflows.iterrows():
        interest_available = period["interest_collected"] - period["servicing_fee"]
        net_loss = max(period["net_loss"], 0.0)
        # All recovery cash is principal proceeds. The write-down is already
        # net of same-period recoveries (net_loss), so paying the recovery
        # cash out as principal keeps tranche balances and cash consistent:
        # balance reduction = sched + prepay + defaults. Recoveries beyond the
        # period's defaults pay down balances written down earlier.
        principal_available_total = period["total_principal_collected"] + period["recoveries_received"]

        # --- Loss allocation: reverse-seniority (junior-most first) ---
        remaining_loss = net_loss
        loss_by_tranche = {}
        for t in reversed(ordered):
            absorb = min(remaining_loss, balances[t.name])
            loss_by_tranche[t.name] = absorb
            balances[t.name] -= absorb
            remaining_loss -= absorb

        # --- Interest waterfall: senior-first ---
        remaining_interest = max(interest_available, 0.0)
        interest_by_tranche = {}
        for t in ordered:
            due = balances[t.name] * (t.coupon_annual / 12.0)
            paid = min(due, remaining_interest) if t.coupon_annual > 0 else 0.0
            interest_by_tranche[t.name] = paid
            remaining_interest -= paid
        # Residual interest (excess spread) flows to the most junior tranche.
        if ordered:
            junior_name = ordered[-1].name
            interest_by_tranche[junior_name] += max(remaining_interest, 0.0)

        # --- Principal waterfall: senior-first, capped at tranche balance ---
        remaining_principal = max(principal_available_total, 0.0)
        principal_by_tranche = {}
        for t in ordered:
            pay = min(remaining_principal, balances[t.name])
            principal_by_tranche[t.name] = pay
            balances[t.name] -= pay
            remaining_principal -= pay
        # Any leftover principal (e.g. pool fully repaid, tranches all at 0)
        # flows to the most junior tranche as a residual cash sweep.
        if ordered and remaining_principal > 1e-6:
            principal_by_tranche[ordered[-1].name] += remaining_principal

        for t in ordered:
            tranche_rows[t.name].append(
                {
                    "month": period["month"],
                    "beginning_balance": balances[t.name] + principal_by_tranche[t.name] + loss_by_tranche[t.name],
                    "interest_paid": interest_by_tranche[t.name],
                    "principal_paid": principal_by_tranche[t.name],
                    "loss_absorbed": loss_by_tranche[t.name],
                    "ending_balance": balances[t.name],
                }
            )

    results: Dict[str, object] = {"tranches": {}}
    for t in ordered:
        tdf = pd.DataFrame(tranche_rows[t.name])
        notional = original_balances[t.name]
        cash_in = -notional
        cashflow_series = [cash_in] + (tdf["interest_paid"] + tdf["principal_paid"]).tolist()
        try:
            monthly_yield = irr(cashflow_series) if notional > 0 else float("nan")
            annual_yield = (1 + monthly_yield) ** 12 - 1 if notional > 0 else float("nan")
        except ValueError:
            annual_yield = float("nan")

        cumulative_loss = tdf["loss_absorbed"].sum()
        results["tranches"][t.name] = {
            "cashflows": tdf,
            "original_notional": notional,
            "total_interest_received": float(tdf["interest_paid"].sum()),
            "total_principal_received": float(tdf["principal_paid"].sum()),
            "cumulative_loss_absorbed": float(cumulative_loss),
            "pct_of_notional_lost": float(cumulative_loss / notional) if notional > 0 else float("nan"),
            "annualized_yield": float(annual_yield),
            "ending_balance": float(tdf["ending_balance"].iloc[-1]) if len(tdf) else notional,
        }

    results["pool_cashflows"] = pool_cashflows
    results["tranche_order"] = [t.name for t in ordered]
    return results


def main() -> None:
    import os

    from .loan_portfolio import generate_synthetic_portfolio, portfolio_summary
    from .historical_loss_analysis import generate_vintage_data, derive_base_case_assumptions, MATURITY_MOB

    port_df = generate_synthetic_portfolio()
    summary = portfolio_summary(port_df)

    vintage_df = generate_vintage_data()
    base_case = derive_base_case_assumptions(vintage_df)

    assumptions = PoolAssumptions(
        beginning_balance=summary["total_current_balance"],
        wac=summary["weighted_average_coupon"],
        wam_months=int(summary["weighted_average_life_months"]),
        annual_cdr=annualize_cumulative_rate(
            base_case["base_case_cumulative_default_rate"], MATURITY_MOB / 12
        ),
        lgd=base_case["base_case_lgd"],
    )
    pool_cf = build_pool_cashflows(assumptions)
    results = run_waterfall(pool_cf, assumptions.beginning_balance)

    os.makedirs("output", exist_ok=True)
    pool_cf.to_csv("output/pool_cashflows.csv", index=False)

    for name in results["tranche_order"]:
        t = results["tranches"][name]
        print(
            f"{name}: notional=${t['original_notional']:,.0f} "
            f"yield={t['annualized_yield']:.2%} "
            f"loss_absorbed=${t['cumulative_loss_absorbed']:,.0f} "
            f"({t['pct_of_notional_lost']:.1%} of notional)"
        )
        t["cashflows"].to_csv(f"output/tranche_{name.lower().replace('-', '_')}_cashflows.csv", index=False)


if __name__ == "__main__":
    main()
