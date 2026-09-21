import pytest

from srt_model.ic_memo_draft import MemoConfig, generate_memo
from srt_model.fund_rollup import FundConfig
from srt_model.monte_carlo_risk import MonteCarloConfig


@pytest.fixture(scope="module")
def small_memo_config():
    return MemoConfig(
        portfolio_seed=1,
        vintage_seed=2,
        mc_seed=3,
        n_loans=150,
        mc_paths=400,
        fund_config=FundConfig(),
    )


@pytest.fixture(scope="module")
def memo_text(small_memo_config):
    return generate_memo(small_memo_config)


def test_memo_contains_all_required_sections(memo_text):
    required_headings = [
        "Transaction Overview",
        "Collateral / Obligor Summary",
        "Historical Loss Analysis",
        "Structure and Tranching",
        "Monte Carlo Risk Summary",
        "Scenario Sensitivity",
        "Fund-Level Economics",
        "Recommendation",
    ]
    for heading in required_headings:
        assert heading in memo_text


def test_memo_contains_synthetic_data_disclaimer(memo_text):
    assert "SYNTHETIC" in memo_text.upper()
    assert "not a real transaction" in memo_text.lower() or "not a real" in memo_text.lower()


def test_memo_reflects_configured_loan_count(small_memo_config, memo_text):
    assert f"{small_memo_config.n_loans:,}" in memo_text


def test_memo_generation_is_deterministic_given_same_config(small_memo_config):
    memo1 = generate_memo(small_memo_config, as_of=__import__("datetime").date(2026, 1, 1))
    memo2 = generate_memo(small_memo_config, as_of=__import__("datetime").date(2026, 1, 1))
    assert memo1 == memo2
