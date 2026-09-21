"""
historical_loss_analysis.py
============================

Builds a synthetic vintage-based historical performance dataset (annual
origination vintages, observed over monthly "months on book") and derives
base-case cumulative default rate (CDR) and loss-given-default (LGD)
assumptions for underwriting the current portfolio, following the
standard static-pool "vintage curve" methodology used by ABS/SRT analysts.

ALL DATA IS SYNTHETIC. See README.md for the full disclaimer.

Methodology
-----------
1. Each annual vintage has an "ultimate" cumulative default rate (the
   cumulative default rate the vintage would reach if seasoned to full
   maturity), varying by vintage to reflect the credit cycle (e.g. a
   stressed vintage originated just before a downturn).
2. Defaults emerge over time following a Weibull-shaped seasoning curve
   (slow ramp-up, peak mid-life, long tail) -- a common stylized shape
   for commercial loan default timing curves.
3. Vintages are right-censored at the analysis "as of" date: younger
   vintages have only partially seasoned and their observed cumulative
   default rate understates their ultimate rate.
4. A dollar-weighted "vintage curve" (fraction of ultimate defaults
   reached by a given month-on-book) is built from the more seasoned
   vintages and used to project the ultimate default rate for immature
   vintages (development-factor extrapolation, i.e. chain-ladder style).
5. LGD is realized (simulated) per defaulting dollar with vintage-level
   variation (stressed vintages realize weaker recoveries).
6. Base-case assumptions = balance-weighted average of each vintage's
   projected ultimate CDR / realized LGD.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import weibull_min

MATURITY_MOB = 84  # months on book considered "fully seasoned" (7 years)


@dataclass(frozen=True)
class VintageConfig:
    n_vintages: int = 8
    as_of_date: date = date(2026, 9, 30)
    first_vintage_year: Optional[int] = None  # defaults to as_of_year - n_vintages
    weibull_k: float = 1.6          # shape: >1 => defaults peak mid-life
    weibull_lambda: float = 34.0    # scale: months to ~63% of ultimate defaults
    seed: Optional[int] = 7


def _ultimate_cdr_by_vintage(n_vintages: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate vintage-level ultimate cumulative default rates.

    A macro cycle is layered on top of an idiosyncratic base rate so that
    one or two vintages look like they were originated into a downturn
    (higher ultimate CDR), consistent with real bank SME portfolios.

    The base rate is calibrated so that its implied annual default rate
    (once annualized over the full ``MATURITY_MOB``-month seasoning
    window) lines up with the origination-time, rating-implied PDs used
    in ``loan_portfolio.py`` -- i.e. the bottom-up (obligor rating -> PD)
    and top-down (vintage curve) views of portfolio credit risk are
    calibrated to be broadly consistent with one another, as a real
    analyst would want to reconcile before using either as a base case.
    """
    base = 0.155  # long-run average ultimate CDR over a ~7yr seasoning window
    # Deterministic-looking cycle: a stress bump around the middle vintages.
    cycle = np.array(
        [0.0, 0.10, 0.35, 0.20, -0.05, -0.15, -0.10, 0.0, 0.05, 0.05][:n_vintages]
    )
    if len(cycle) < n_vintages:
        cycle = np.pad(cycle, (0, n_vintages - len(cycle)))
    noise = rng.normal(0.0, 0.006, size=n_vintages)
    ultimate = base * (1.0 + cycle) + noise
    return np.clip(ultimate, 0.01, 0.30)


def _lgd_by_vintage(n_vintages: int, ultimate_cdr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stressed (higher-CDR) vintages realize somewhat weaker recoveries."""
    base_lgd = 0.38
    stress_sensitivity = 1.5
    cycle_lgd_bump = stress_sensitivity * (ultimate_cdr - ultimate_cdr.mean())
    noise = rng.normal(0.0, 0.02, size=n_vintages)
    lgd = base_lgd + cycle_lgd_bump + noise
    return np.clip(lgd, 0.15, 0.75)


def generate_vintage_data(config: Optional[VintageConfig] = None) -> pd.DataFrame:
    """Generate a synthetic monthly vintage performance panel.

    Returns a long-format DataFrame with one row per (vintage_year, mob)
    combination, right-censored at ``as_of_date`` so younger vintages have
    fewer observed months.
    """
    cfg = config or VintageConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_vintages
    first_year = cfg.first_vintage_year or (cfg.as_of_date.year - n)
    vintage_years = np.arange(first_year, first_year + n)

    ultimate_cdr = _ultimate_cdr_by_vintage(n, rng)
    ultimate_lgd = _lgd_by_vintage(n, ultimate_cdr, rng)

    # Origination balance per vintage (growing book over time), used as
    # weights for portfolio-level aggregation.
    origination_balance = 400_000_000 * (1.06 ** np.arange(n)) * rng.uniform(0.9, 1.1, size=n)

    rows = []
    for i, vy in enumerate(vintage_years):
        vintage_origination_date = date(int(vy), 6, 30)  # mid-year convention
        months_observed = min(
            MATURITY_MOB,
            (cfg.as_of_date.year - vintage_origination_date.year) * 12
            + (cfg.as_of_date.month - vintage_origination_date.month),
        )
        months_observed = max(months_observed, 1)
        mobs = np.arange(1, months_observed + 1)

        # Weibull CDF gives the fraction of ULTIMATE defaults reached by mob.
        frac_of_ultimate = weibull_min.cdf(mobs, c=cfg.weibull_k, scale=cfg.weibull_lambda)
        cum_default_rate = ultimate_cdr[i] * frac_of_ultimate

        # Add small month-to-month observation noise, keeping the series
        # monotonically non-decreasing (cumulative by construction).
        noise = rng.normal(0.0, 0.0006, size=len(mobs)).cumsum() * 0.0  # kept 0: pure curve
        cum_default_rate = np.clip(cum_default_rate + noise, 0.0, ultimate_cdr[i])
        cum_default_rate = np.maximum.accumulate(cum_default_rate)

        incremental_default_rate = np.diff(cum_default_rate, prepend=0.0)

        # Realized LGD per period: noisy around the vintage's ultimate LGD,
        # only meaningful where defaults actually occurred.
        lgd_noise = rng.normal(0.0, 0.03, size=len(mobs))
        realized_lgd = np.clip(ultimate_lgd[i] + lgd_noise, 0.10, 0.90)

        cumulative_loss_rate = np.cumsum(incremental_default_rate * realized_lgd)

        for j, mob in enumerate(mobs):
            rows.append(
                {
                    "vintage_year": int(vy),
                    "mob": int(mob),
                    "origination_balance": float(origination_balance[i]),
                    "ultimate_cdr_true": float(ultimate_cdr[i]),  # latent (for validation only)
                    "cumulative_default_rate": float(cum_default_rate[j]),
                    "incremental_default_rate": float(incremental_default_rate[j]),
                    "realized_lgd": float(realized_lgd[j]),
                    "cumulative_loss_rate": float(cumulative_loss_rate[j]),
                }
            )

    return pd.DataFrame(rows)


def compute_vintage_curve(df: pd.DataFrame, mature_mob_threshold: int = MATURITY_MOB) -> pd.DataFrame:
    """Build a dollar-weighted average default-timing curve by mob.

    Uses only vintages that have reached at least ``mature_mob_threshold``
    months of seasoning at a given mob to avoid biasing the curve shape
    with censored (immature) data at long tenors.
    """
    max_mob_by_vintage = df.groupby("vintage_year")["mob"].max()
    mature_vintages = max_mob_by_vintage[max_mob_by_vintage >= mature_mob_threshold].index

    curve_source = df[df["vintage_year"].isin(mature_vintages)]
    if curve_source.empty:
        # Fall back to whatever data exists if nothing is fully mature yet.
        curve_source = df

    grouped = curve_source.groupby("mob").apply(
        lambda g: pd.Series(
            {
                "avg_cumulative_default_rate": np.average(
                    g["cumulative_default_rate"], weights=g["origination_balance"]
                )
            }
        ),
        include_groups=False,
    )
    grouped = grouped.sort_index()
    ultimate_level = grouped["avg_cumulative_default_rate"].iloc[-1]
    grouped["pct_of_ultimate"] = grouped["avg_cumulative_default_rate"] / ultimate_level
    return grouped.reset_index()


def derive_base_case_assumptions(
    df: pd.DataFrame, curve: Optional[pd.DataFrame] = None
) -> Dict[str, object]:
    """Project each vintage's ultimate CDR via development-factor
    extrapolation off the vintage curve, then take a balance-weighted
    average across vintages to set base-case underwriting assumptions.
    """
    curve = curve if curve is not None else compute_vintage_curve(df)
    curve_lookup = curve.set_index("mob")["pct_of_ultimate"]
    max_curve_mob = int(curve_lookup.index.max())

    latest = df.loc[df.groupby("vintage_year")["mob"].idxmax()].copy()

    def pct_of_ultimate_at(mob: int) -> float:
        mob_capped = min(int(mob), max_curve_mob)
        # nearest available mob at or below the requested one
        available = curve_lookup.index[curve_lookup.index <= mob_capped]
        if len(available) == 0:
            return float(curve_lookup.iloc[0])
        return float(curve_lookup.loc[available.max()])

    latest["pct_of_ultimate"] = latest["mob"].apply(pct_of_ultimate_at)
    latest["projected_ultimate_cdr"] = (
        latest["cumulative_default_rate"] / latest["pct_of_ultimate"]
    ).clip(upper=0.5)

    total_balance = latest["origination_balance"].sum()
    base_case_cdr = float(
        np.average(latest["projected_ultimate_cdr"], weights=latest["origination_balance"])
    )

    # Weight LGD by realized defaulted-dollar exposure (incremental default
    # rate * balance) rather than plain vintage balance, since LGD is only
    # meaningfully observed on defaulted dollars.
    defaulted_dollars = df["incremental_default_rate"] * df["origination_balance"]
    base_case_lgd = float(np.average(df["realized_lgd"], weights=defaulted_dollars.clip(lower=1e-9)))

    base_case_annual_loss_rate = base_case_cdr * base_case_lgd

    return {
        "base_case_cumulative_default_rate": base_case_cdr,
        "base_case_lgd": base_case_lgd,
        "base_case_cumulative_loss_rate": base_case_annual_loss_rate,
        "per_vintage_projection": latest[
            ["vintage_year", "mob", "cumulative_default_rate", "projected_ultimate_cdr"]
        ].reset_index(drop=True),
        "n_vintages_used": int(total_balance > 0) and int(len(latest)),
    }


def main() -> None:
    import os

    df = generate_vintage_data()
    curve = compute_vintage_curve(df)
    base_case = derive_base_case_assumptions(df, curve)

    os.makedirs("output", exist_ok=True)
    df.to_csv("output/vintage_performance.csv", index=False)
    curve.to_csv("output/vintage_curve.csv", index=False)

    print(f"Base-case cumulative default rate: {base_case['base_case_cumulative_default_rate']:.2%}")
    print(f"Base-case LGD: {base_case['base_case_lgd']:.2%}")
    print(f"Base-case cumulative loss rate: {base_case['base_case_cumulative_loss_rate']:.2%}")
    print("Per-vintage projection:")
    print(base_case["per_vintage_projection"])


if __name__ == "__main__":
    main()
