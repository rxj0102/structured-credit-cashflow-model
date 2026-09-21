"""Shared pytest fixtures. Portfolios are kept small (n_loans) to keep the
test suite fast; the numerical methodology is identical to production
settings."""
import numpy as np
import pandas as pd
import pytest

from srt_model.loan_portfolio import PortfolioConfig, generate_synthetic_portfolio
from srt_model.historical_loss_analysis import (
    VintageConfig,
    generate_vintage_data,
    compute_vintage_curve,
    derive_base_case_assumptions,
)
from srt_model.cashflow_waterfall import PoolAssumptions, build_pool_cashflows


@pytest.fixture
def small_portfolio() -> pd.DataFrame:
    return generate_synthetic_portfolio(PortfolioConfig(n_loans=250, seed=123))


@pytest.fixture
def vintage_data() -> pd.DataFrame:
    return generate_vintage_data(VintageConfig(seed=11))


@pytest.fixture
def vintage_curve(vintage_data):
    return compute_vintage_curve(vintage_data)


@pytest.fixture
def base_case_assumptions(vintage_data, vintage_curve):
    return derive_base_case_assumptions(vintage_data, vintage_curve)


@pytest.fixture
def pool_assumptions() -> PoolAssumptions:
    return PoolAssumptions(
        beginning_balance=1_000_000_000.0,
        wac=0.08,
        wam_months=60,
        annual_cdr=0.02,
        lgd=0.40,
        annual_cpr=0.05,
    )


@pytest.fixture
def pool_cashflows(pool_assumptions):
    return build_pool_cashflows(pool_assumptions)


@pytest.fixture
def rng():
    return np.random.default_rng(999)
