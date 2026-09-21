"""
loan_portfolio.py
=================

Generates a synthetic, obligor-level pool of commercial loans resembling
the reference portfolio a bank would put forward for a significant risk
transfer (SRT) transaction, and computes standard portfolio-level
underwriting statistics.

ALL DATA IS SYNTHETIC. See README.md for the full disclaimer.

Design notes / simplifying assumptions
---------------------------------------
* One loan per obligor (no multi-facility obligors), so ``loan_id`` and
  ``obligor_id`` are 1:1. This is a common simplification for a first-pass
  reference-pool model.
* Ratings are expressed on an internal 8-notch scale typical of a bank's
  SME/mid-market book (BBB down to CCC+), each mapped to a 1-year PD.
* LGD is driven primarily by collateral type/seniority, with idiosyncratic
  noise layered on top.
* ``current_balance`` reflects schedule-only amortization (no defaults or
  prepayments) from origination to the portfolio "as of" date.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .finance_utils import level_payment_amortization

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

INDUSTRY_SECTORS = [
    "Healthcare & Life Sciences",
    "Manufacturing & Industrials",
    "Technology & Software",
    "Retail & Consumer Products",
    "Energy & Utilities",
    "Commercial Real Estate",
    "Financial Services",
    "Transportation & Logistics",
    "Telecommunications & Media",
    "Business & Professional Services",
    "Hospitality & Leisure",
    "Agriculture & Food Processing",
]
# Weights create realistic (uneven) sector concentration in the pool.
SECTOR_WEIGHTS = np.array(
    [0.12, 0.14, 0.09, 0.11, 0.06, 0.13, 0.07, 0.08, 0.05, 0.08, 0.04, 0.03]
)
SECTOR_WEIGHTS = SECTOR_WEIGHTS / SECTOR_WEIGHTS.sum()

# Internal rating scale -> 1-year probability of default (annualized).
RATING_PD_MAP: Dict[str, float] = {
    "BBB": 0.0030,
    "BB+": 0.0060,
    "BB": 0.0100,
    "BB-": 0.0150,
    "B+": 0.0220,
    "B": 0.0320,
    "B-": 0.0480,
    "CCC+": 0.0750,
}
RATINGS = list(RATING_PD_MAP.keys())
# Weight of each rating notch in the pool (skewed to BB/B, typical of a
# bank SME/mid-market book being optimized for SRT relief).
RATING_WEIGHTS = np.array([0.05, 0.10, 0.15, 0.15, 0.20, 0.15, 0.12, 0.08])
RATING_WEIGHTS = RATING_WEIGHTS / RATING_WEIGHTS.sum()

COLLATERAL_LGD_MAP: Dict[str, float] = {
    "Senior Secured - All Assets": 0.35,
    "Secured - Real Estate": 0.28,
    "Secured - Equipment": 0.45,
    "Secured - Receivables/Inventory": 0.42,
    "Unsecured": 0.65,
}
COLLATERAL_TYPES = list(COLLATERAL_LGD_MAP.keys())
COLLATERAL_WEIGHTS = np.array([0.30, 0.18, 0.17, 0.15, 0.20])
COLLATERAL_WEIGHTS = COLLATERAL_WEIGHTS / COLLATERAL_WEIGHTS.sum()

SENIORITY_BY_COLLATERAL: Dict[str, str] = {
    "Senior Secured - All Assets": "Senior Secured",
    "Secured - Real Estate": "Senior Secured",
    "Secured - Equipment": "Senior Secured",
    "Secured - Receivables/Inventory": "Senior Secured",
    "Unsecured": "Senior Unsecured",
}

AMORTIZATION_TYPES = ["Amortizing", "Bullet"]
AMORTIZATION_WEIGHTS = np.array([0.65, 0.35])

RATE_TYPES = ["Fixed", "Floating"]
RATE_TYPE_WEIGHTS = np.array([0.40, 0.60])

DEFAULT_AS_OF_DATE = date(2026, 9, 30)


@dataclass(frozen=True)
class PortfolioConfig:
    """Configuration for synthetic portfolio generation."""

    n_loans: int = 3000
    as_of_date: date = DEFAULT_AS_OF_DATE
    min_balance: float = 1_000_000.0
    max_balance: float = 75_000_000.0
    balance_mu: float = 15.3   # lognormal(mu, sigma) in $ before clipping
    balance_sigma: float = 0.85
    base_rate: float = 0.045   # reference/base rate (e.g. SOFR proxy) for floaters
    seed: Optional[int] = 42


def _sample_term_months(rng: np.random.Generator, n: int) -> np.ndarray:
    # Commercial loan terms cluster around 3-7 years.
    choices = np.array([36, 48, 60, 72, 84])
    weights = np.array([0.15, 0.25, 0.30, 0.20, 0.10])
    return rng.choice(choices, size=n, p=weights)


def _months_between(d1: date, d2: date) -> int:
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


def _add_months(d: date, months: int) -> date:
    month_index = d.month - 1 + months
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    day = min(d.day, 28)
    return date(year, month, day)


def generate_synthetic_portfolio(config: Optional[PortfolioConfig] = None) -> pd.DataFrame:
    """Generate a synthetic obligor-level commercial loan pool.

    Returns a DataFrame with one row per loan/obligor and the fields
    described in the module docstring.
    """
    cfg = config or PortfolioConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_loans

    sectors = rng.choice(INDUSTRY_SECTORS, size=n, p=SECTOR_WEIGHTS)
    ratings = rng.choice(RATINGS, size=n, p=RATING_WEIGHTS)
    collateral = rng.choice(COLLATERAL_TYPES, size=n, p=COLLATERAL_WEIGHTS)
    amort_type = rng.choice(AMORTIZATION_TYPES, size=n, p=AMORTIZATION_WEIGHTS)
    rate_type = rng.choice(RATE_TYPES, size=n, p=RATE_TYPE_WEIGHTS)

    raw_balance = rng.lognormal(mean=cfg.balance_mu, sigma=cfg.balance_sigma, size=n)
    original_balance = np.clip(raw_balance, cfg.min_balance, cfg.max_balance)
    original_balance = np.round(original_balance, -3)  # round to nearest $1,000

    term_months = _sample_term_months(rng, n)

    # Origination dates spread over the trailing ~4 years before as_of_date.
    max_seasoning_months = 48
    seasoning_months = rng.integers(1, max_seasoning_months, size=n)
    # Loans that would already be past maturity are re-seasoned to stay current.
    seasoning_months = np.minimum(seasoning_months, term_months - 1)
    origination_dates = [
        _add_months(cfg.as_of_date, -int(m)) for m in seasoning_months
    ]
    maturity_dates = [
        _add_months(od, int(t)) for od, t in zip(origination_dates, term_months)
    ]

    # Base PD from rating, with idiosyncratic multiplicative noise.
    base_pd = np.array([RATING_PD_MAP[r] for r in ratings])
    pd_noise = rng.lognormal(mean=0.0, sigma=0.20, size=n)
    pd_1yr = np.clip(base_pd * pd_noise, 0.0005, 0.35)

    # Base LGD from collateral type, with idiosyncratic noise, bounded [0.05, 0.95].
    base_lgd = np.array([COLLATERAL_LGD_MAP[c] for c in collateral])
    lgd_noise = rng.normal(loc=0.0, scale=0.06, size=n)
    lgd = np.clip(base_lgd + lgd_noise, 0.05, 0.95)

    seniority = np.array([SENIORITY_BY_COLLATERAL[c] for c in collateral])

    # Coupon: spread over base rate, wider for lower ratings.
    rating_rank = np.array([RATINGS.index(r) for r in ratings])  # 0=best .. 7=worst
    credit_spread = 0.015 + rating_rank * 0.0075
    spread_noise = rng.normal(0.0, 0.0035, size=n)
    all_in_rate = np.clip(cfg.base_rate + credit_spread + spread_noise, 0.02, 0.25)

    # Current balance = original balance amortized on schedule (no defaults/
    # prepayments) from origination through as_of_date. Bullet loans carry
    # their full original balance until maturity.
    current_balance = np.empty(n)
    for i in range(n):
        if amort_type[i] == "Bullet":
            current_balance[i] = original_balance[i]
        else:
            payment = level_payment_amortization(
                original_balance[i], all_in_rate[i], int(term_months[i])
            )
            bal = original_balance[i]
            r = all_in_rate[i] / 12.0
            for _ in range(int(seasoning_months[i])):
                interest = bal * r
                principal = max(payment - interest, 0.0)
                bal = max(bal - principal, 0.0)
            current_balance[i] = bal

    loan_id = [f"L{idx + 1:05d}" for idx in range(n)]
    obligor_id = [f"OBL{idx + 1:05d}" for idx in range(n)]

    df = pd.DataFrame(
        {
            "loan_id": loan_id,
            "obligor_id": obligor_id,
            "industry_sector": sectors,
            "origination_date": origination_dates,
            "maturity_date": maturity_dates,
            "term_months": term_months,
            "seasoning_months": seasoning_months,
            "original_balance": original_balance,
            "current_balance": np.round(current_balance, 2),
            "amortization_type": amort_type,
            "interest_rate_type": rate_type,
            "interest_rate": np.round(all_in_rate, 5),
            "rating": ratings,
            "pd_1yr": np.round(pd_1yr, 5),
            "collateral_type": collateral,
            "seniority": seniority,
            "lgd": np.round(lgd, 4),
        }
    )
    return df


def portfolio_summary(df: pd.DataFrame) -> Dict[str, object]:
    """Compute standard portfolio-level underwriting statistics.

    All weighted averages are weighted by ``current_balance``.
    """
    total_balance = float(df["current_balance"].sum())
    w = df["current_balance"] / total_balance

    wac = float((df["interest_rate"] * w).sum())
    wa_pd = float((df["pd_1yr"] * w).sum())
    wa_lgd = float((df["lgd"] * w).sum())
    wa_expected_loss_rate = float(((df["pd_1yr"] * df["lgd"]) * w).sum())

    wam_months = float((df["term_months"] - df["seasoning_months"]).clip(lower=0).mul(w).sum())

    sector_conc = (
        df.groupby("industry_sector")["current_balance"].sum() / total_balance
    ).sort_values(ascending=False)

    rating_dist = (
        df.groupby("rating")["current_balance"].sum() / total_balance
    ).reindex(RATINGS).fillna(0.0)

    collateral_dist = (
        df.groupby("collateral_type")["current_balance"].sum() / total_balance
    ).sort_values(ascending=False)

    amort_dist = (
        df.groupby("amortization_type")["current_balance"].sum() / total_balance
    )

    obligor_balances = df.groupby("obligor_id")["current_balance"].sum().sort_values(
        ascending=False
    )
    top10_obligor_conc = float(obligor_balances.head(10).sum() / total_balance)
    largest_obligor_conc = float(obligor_balances.iloc[0] / total_balance)

    return {
        "n_loans": int(len(df)),
        "n_obligors": int(df["obligor_id"].nunique()),
        "total_current_balance": total_balance,
        "total_original_balance": float(df["original_balance"].sum()),
        "avg_loan_size": float(df["current_balance"].mean()),
        "weighted_average_coupon": wac,
        "weighted_average_life_months": wam_months,
        "weighted_average_pd_1yr": wa_pd,
        "weighted_average_lgd": wa_lgd,
        "weighted_average_annual_expected_loss_rate": wa_expected_loss_rate,
        "sector_concentration": sector_conc.to_dict(),
        "top5_sector_concentration": float(sector_conc.head(5).sum()),
        "rating_distribution": rating_dist.to_dict(),
        "collateral_distribution": collateral_dist.to_dict(),
        "amortization_distribution": amort_dist.to_dict(),
        "top10_obligor_concentration": top10_obligor_conc,
        "largest_obligor_concentration": largest_obligor_conc,
    }


def main() -> None:
    df = generate_synthetic_portfolio()
    summary = portfolio_summary(df)

    import os

    os.makedirs("output", exist_ok=True)
    df.to_csv("output/loan_portfolio.csv", index=False)

    print(f"Generated {summary['n_loans']} loans / {summary['n_obligors']} obligors")
    print(f"Total current balance: ${summary['total_current_balance']:,.0f}")
    print(f"Weighted average coupon: {summary['weighted_average_coupon']:.2%}")
    print(f"Weighted average 1yr PD: {summary['weighted_average_pd_1yr']:.2%}")
    print(f"Weighted average LGD: {summary['weighted_average_lgd']:.2%}")
    print(f"Top-5 sector concentration: {summary['top5_sector_concentration']:.1%}")
    print("Saved loan-level detail to output/loan_portfolio.csv")


if __name__ == "__main__":
    main()
