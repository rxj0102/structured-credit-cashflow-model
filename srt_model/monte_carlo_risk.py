"""
monte_carlo_risk.py
=====================

Monte Carlo simulation of correlated obligor defaults using a single-factor
Gaussian copula, producing a portfolio loss distribution and tail-risk
metrics for the first-loss and mezzanine SRT tranches specifically (the
tranches whose investors bear most of the credit risk).

ALL DATA IS SYNTHETIC. See README.md for the full disclaimer.

Methodology
-----------
For each simulation path:
  1. Draw a single systemic factor ``Z ~ N(0,1)`` shared by every obligor
     in the pool (the "single factor" in single-factor Gaussian copula).
  2. Draw an idiosyncratic shock ``eps_i ~ N(0,1)`` independently for each
     obligor.
  3. Obligor i's latent asset return is
     ``A_i = sqrt(rho) * Z + sqrt(1 - rho) * eps_i``, where ``rho`` is the
     asset correlation (a single scalar in the base case; can be
     overridden per loan for a "sector-varying" run).
  4. Obligor i defaults over the analysis horizon if
     ``A_i < Phi^-1(p_i)``, where ``p_i`` is obligor i's cumulative default
     probability over the horizon (derived from its 1-year PD) and
     ``Phi^-1`` is the inverse standard normal CDF -- this reproduces
     marginal default probability p_i exactly while correlating defaults
     across obligors through the shared factor Z (the Vasicek/Gaussian
     copula single-factor model used industry-wide for CDO/SRT capital).
  5. Loss-given-default is drawn per defaulting loan from a Beta
     distribution centered on the loan's LGD assumption, adding LGD
     uncertainty on top of default-timing uncertainty.

Portfolio loss per path is then mapped onto tranche losses using the
tranche attachment/detachment points from ``cashflow_waterfall``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from .cashflow_waterfall import Tranche, DEFAULT_TRANCHES, tranche_notional


@dataclass(frozen=True)
class MonteCarloConfig:
    n_paths: int = 10_000
    horizon_years: float = 5.0
    rho: float = 0.15               # single-factor asset correlation
    lgd_beta_vol: float = 0.15      # relative std-dev of the LGD beta draw
    seed: Optional[int] = 2024


def _beta_params_from_mean_vol(mean: np.ndarray, rel_vol: float):
    """Method-of-moments Beta(a, b) parameters from a target mean and a
    relative volatility (std / mean), clipped to keep the distribution
    well-defined on (0, 1)."""
    mean = np.clip(mean, 0.01, 0.99)
    std = np.clip(mean * rel_vol, 1e-4, mean * (1 - mean) * 0.99 ** 0.5)
    var = std ** 2
    common = mean * (1 - mean) / var - 1.0
    common = np.clip(common, 1e-3, None)
    a = mean * common
    b = (1 - mean) * common
    return np.clip(a, 1e-3, None), np.clip(b, 1e-3, None)


def simulate_portfolio_losses(
    portfolio: pd.DataFrame,
    config: Optional[MonteCarloConfig] = None,
) -> Dict[str, object]:
    """Run the single-factor Gaussian copula Monte Carlo simulation.

    ``portfolio`` must have columns ``current_balance``, ``pd_1yr``,
    ``lgd`` (as produced by ``loan_portfolio.generate_synthetic_portfolio``).

    Returns a dict with the simulated portfolio loss vector (one value per
    path, in dollars), the default-count vector, and the config used.
    """
    cfg = config or MonteCarloConfig()
    rng = np.random.default_rng(cfg.seed)

    balances = portfolio["current_balance"].to_numpy()
    pd_1yr = portfolio["pd_1yr"].to_numpy()
    lgd_mean = portfolio["lgd"].to_numpy()
    n_loans = len(portfolio)

    # Cumulative default probability over the horizon, from the annual PD.
    p_horizon = 1.0 - (1.0 - pd_1yr) ** cfg.horizon_years
    p_horizon = np.clip(p_horizon, 1e-6, 1 - 1e-6)
    default_threshold = norm.ppf(p_horizon)  # Phi^-1(p_i)

    sqrt_rho = np.sqrt(cfg.rho)
    sqrt_1m_rho = np.sqrt(1.0 - cfg.rho)

    a_lgd, b_lgd = _beta_params_from_mean_vol(lgd_mean, cfg.lgd_beta_vol)

    n_paths = cfg.n_paths
    portfolio_losses = np.empty(n_paths)
    default_counts = np.empty(n_paths, dtype=int)

    # Chunked simulation to bound peak memory for large pools * many paths.
    chunk_size = max(1, min(n_paths, 2000))
    path_start = 0
    while path_start < n_paths:
        chunk = min(chunk_size, n_paths - path_start)
        z = rng.standard_normal(chunk)
        eps = rng.standard_normal((chunk, n_loans))
        asset_return = sqrt_rho * z[:, None] + sqrt_1m_rho * eps
        defaults = asset_return < default_threshold[None, :]

        lgd_draws = rng.beta(a_lgd[None, :], b_lgd[None, :], size=(chunk, n_loans))
        losses = np.where(defaults, balances[None, :] * lgd_draws, 0.0)

        portfolio_losses[path_start:path_start + chunk] = losses.sum(axis=1)
        default_counts[path_start:path_start + chunk] = defaults.sum(axis=1)
        path_start += chunk

    return {
        "portfolio_losses": portfolio_losses,
        "default_counts": default_counts,
        "total_balance": float(balances.sum()),
        "config": cfg,
    }


def _tail_metrics(losses: np.ndarray, notional: float) -> Dict[str, float]:
    mean_loss = float(np.mean(losses))
    p95 = float(np.percentile(losses, 95))
    p99 = float(np.percentile(losses, 99))
    es99 = float(losses[losses >= p99].mean()) if np.any(losses >= p99) else p99
    es95 = float(losses[losses >= p95].mean()) if np.any(losses >= p95) else p95
    prob_any_loss = float(np.mean(losses > 1e-6))
    prob_full_loss = float(np.mean(losses >= notional - 1e-6)) if notional > 0 else 0.0
    return {
        "expected_loss": mean_loss,
        "expected_loss_pct": mean_loss / notional if notional > 0 else float("nan"),
        "var_95": p95,
        "var_95_pct": p95 / notional if notional > 0 else float("nan"),
        "var_99": p99,
        "var_99_pct": p99 / notional if notional > 0 else float("nan"),
        "expected_shortfall_95": es95,
        "expected_shortfall_95_pct": es95 / notional if notional > 0 else float("nan"),
        "expected_shortfall_99": es99,
        "expected_shortfall_99_pct": es99 / notional if notional > 0 else float("nan"),
        "prob_any_loss": prob_any_loss,
        "prob_full_wipeout": prob_full_loss,
    }


def tranche_loss_distribution(
    portfolio_losses: np.ndarray,
    original_pool_balance: float,
    tranche: Tranche,
) -> np.ndarray:
    """Map simulated portfolio-level losses onto a single tranche's loss,
    given its attachment/detachment points (both as % of original pool
    balance)."""
    attach = tranche.attachment_pct * original_pool_balance
    detach = tranche.detachment_pct * original_pool_balance
    tranche_size = detach - attach
    tranche_losses = np.clip(portfolio_losses - attach, 0.0, tranche_size)
    return tranche_losses


def analyze_tranche_risk(
    portfolio: pd.DataFrame,
    original_pool_balance: Optional[float] = None,
    tranches: Optional[List[Tranche]] = None,
    config: Optional[MonteCarloConfig] = None,
) -> Dict[str, object]:
    """Run the Monte Carlo simulation and compute tail-risk metrics for
    every tranche, with particular focus on first-loss and mezzanine
    (the tranches an SRT protection buyer/investor is exposed to)."""
    tranches = tranches or DEFAULT_TRANCHES
    original_pool_balance = original_pool_balance or float(portfolio["current_balance"].sum())

    sim = simulate_portfolio_losses(portfolio, config)
    losses = sim["portfolio_losses"]

    portfolio_metrics = _tail_metrics(losses, original_pool_balance)

    tranche_results = {}
    for t in tranches:
        notional = tranche_notional(t, original_pool_balance)
        t_losses = tranche_loss_distribution(losses, original_pool_balance, t)
        metrics = _tail_metrics(t_losses, notional)
        metrics["notional"] = notional
        tranche_results[t.name] = metrics

    return {
        "portfolio_loss_metrics": portfolio_metrics,
        "tranche_loss_metrics": tranche_results,
        "portfolio_losses": losses,
        "default_counts": sim["default_counts"],
        "n_paths": sim["config"].n_paths,
    }


def main() -> None:
    from .loan_portfolio import generate_synthetic_portfolio

    portfolio = generate_synthetic_portfolio()
    results = analyze_tranche_risk(portfolio)

    print(f"Simulated {results['n_paths']:,} paths")
    pm = results["portfolio_loss_metrics"]
    print(f"Portfolio expected loss: {pm['expected_loss_pct']:.2%}")
    print(f"Portfolio 99% VaR: {pm['var_99_pct']:.2%}")

    for name, metrics in results["tranche_loss_metrics"].items():
        print(
            f"{name}: EL={metrics['expected_loss_pct']:.2%} "
            f"VaR95={metrics['var_95_pct']:.2%} VaR99={metrics['var_99_pct']:.2%} "
            f"ES99={metrics['expected_shortfall_99_pct']:.2%}"
        )


if __name__ == "__main__":
    main()
