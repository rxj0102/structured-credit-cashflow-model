"""
finance_utils.py
================

Small, dependency-light financial math helpers shared across the SRT
model: IRR/NPV solving, MOIC, and a level-payment amortization schedule.
Kept separate from the domain modules so the numerical routines can be
unit tested in isolation.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def npv(rate: float, cashflows: Sequence[float]) -> float:
    """Net present value of a periodic cash flow series.

    ``cashflows[0]`` is treated as occurring at time 0 (undiscounted),
    matching the convention used by Excel's NPV/IRR pair when the initial
    outlay is included as the first element.
    """
    cf = np.asarray(cashflows, dtype=float)
    periods = np.arange(len(cf))
    return float(np.sum(cf / (1.0 + rate) ** periods))


def irr(cashflows: Sequence[float], guess: float = 0.02, tol: float = 1e-9,
        max_iter: int = 200) -> float:
    """Periodic internal rate of return via Newton's method with a
    bisection fallback.

    Returns the periodic (e.g. monthly) IRR that zeroes the NPV of
    ``cashflows``. Raises ``ValueError`` if no sign change exists in the
    cash flow series (no solvable IRR) or if the root finder fails to
    converge.
    """
    cf = np.asarray(cashflows, dtype=float)
    if len(cf) < 2:
        raise ValueError("Need at least two cash flows to compute an IRR")
    # A total loss of capital (an outflow followed by zero inflows forever)
    # has no algebraic IRR root, but its economically meaningful limit is
    # -100% per period; report that directly rather than raising.
    if np.any(cf < 0) and not np.any(cf > 0):
        return -1.0
    if not (np.any(cf > 0) and np.any(cf < 0)):
        raise ValueError("Cash flow series must contain both inflows and outflows")

    rate = guess
    periods = np.arange(len(cf))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(max_iter):
            # Guard against a rate that has drifted to <= -1 (division blow-up).
            if rate <= -1.0 + 1e-9:
                rate = -1.0 + 1e-9
            denom = (1.0 + rate) ** periods
            f = np.sum(cf / denom)
            df = np.sum(-periods * cf / ((1.0 + rate) ** (periods + 1)))
            if not np.isfinite(f) or not np.isfinite(df) or abs(df) < 1e-14:
                break
            new_rate = rate - f / df
            if not np.isfinite(new_rate):
                break
            if abs(new_rate - rate) < tol:
                return float(new_rate)
            rate = new_rate

    # Newton failed to converge cleanly -- fall back to bisection over a
    # wide, sane bracket.
    lo, hi = -0.9999, 10.0
    f_lo, f_hi = npv(lo, cf), npv(hi, cf)
    if f_lo * f_hi > 0:
        raise ValueError("IRR did not converge and no bracket was found")
    for _ in range(200):
        mid = (lo + hi) / 2.0
        f_mid = npv(mid, cf)
        if abs(f_mid) < tol:
            return float(mid)
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return float((lo + hi) / 2.0)


def annualize(periodic_rate: float, periods_per_year: int = 12) -> float:
    """Convert a periodic (e.g. monthly) rate to an annualized (compounded) rate."""
    return (1.0 + periodic_rate) ** periods_per_year - 1.0


def periodic_rate_from_annual(annual_rate: float, periods_per_year: int = 12) -> float:
    """Convert an annual (e.g. CDR/CPR) rate to the equivalent periodic rate."""
    return 1.0 - (1.0 - annual_rate) ** (1.0 / periods_per_year)


def annualize_cumulative_rate(cumulative_rate: float, years: float) -> float:
    """Convert a cumulative rate observed/projected over ``years`` (e.g. a
    lifetime cumulative default rate from a vintage curve) into the
    equivalent constant annual rate, using the same compounding
    convention as ``periodic_rate_from_annual``."""
    return 1.0 - (1.0 - cumulative_rate) ** (1.0 / years)


def moic(contributions: Sequence[float], distributions: Sequence[float]) -> float:
    """Multiple on invested capital = total distributions / total contributions."""
    total_in = float(np.sum(np.abs(contributions)))
    total_out = float(np.sum(distributions))
    if total_in == 0:
        return float("nan")
    return total_out / total_in


def level_payment_amortization(balance: float, annual_rate: float, n_periods: int) -> float:
    """Monthly level payment (principal + interest) that fully amortizes
    ``balance`` over ``n_periods`` months at ``annual_rate`` (annual, paid
    monthly). Falls back to straight-line if the rate is ~0."""
    r = annual_rate / 12.0
    if n_periods <= 0:
        return balance
    if abs(r) < 1e-12:
        return balance / n_periods
    return balance * (r * (1 + r) ** n_periods) / ((1 + r) ** n_periods - 1)
