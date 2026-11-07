# structured-credit-cashflow-model

[![CI](https://github.com/rxj0102/structured-credit-cashflow-model/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/structured-credit-cashflow-model/actions/workflows/ci.yml)

A synthetic, end-to-end model of the workflow used by a **structured credit /
significant risk transfer (SRT) investment analyst** to underwrite a bank
commercial loan portfolio — from reference-pool construction through
historical loss analysis, cash flow waterfall modeling, Monte Carlo risk
simulation, scenario sensitivity, fund-level economics, and a draft
Investment Committee (IC) memo.

> ## ⚠️ Synthetic Data Disclaimer
>
> **Every number in this repository — every loan, obligor, industry sector,
> rating, default, recovery, and cash flow — is randomly generated.** There
> is no real bank, no real portfolio, and no real transaction behind any of
> it. The methodology (vintage curve analysis, Gaussian copula Monte Carlo,
> tranche waterfalls, fund fee mechanics) mirrors techniques used in real
> SRT/structured credit practice, but the code and its outputs are for
> **educational and illustrative purposes only**. Nothing here is
> investment advice, and none of the figures produced by this repository
> should be used to make an actual investment, underwriting, or risk
> decision.

## The scenario

A bank wants to transfer the credit risk of a portion of its commercial loan
book to third-party investors via a **significant risk transfer (SRT)**
transaction. An investor/fund analyst is asked to underwrite the deal:
understand the collateral, size expected losses off historical performance,
model how cash flows and losses move through the proposed tranche structure,
stress-test the assumptions, work out what the deal means once leverage,
FX hedging, and fund fees are layered on top, and write up the findings for
an investment committee. This repository implements each step of that
workflow as an independent, testable Python module.

## Project structure

```
srt_model/
├── loan_portfolio.py          # 1. Synthetic obligor-level loan pool + portfolio stats
├── historical_loss_analysis.py# 2. Vintage-based default/loss curves -> base-case assumptions
├── cashflow_waterfall.py      # 3. Pool cash flows -> tranche waterfall, yields, loss allocation
├── monte_carlo_risk.py        # 4. Single-factor Gaussian copula Monte Carlo tail-risk metrics
├── scenario_sensitivity.py    # 5. Base/Stress/Severe Stress scenario grid
├── fund_rollup.py             # 6. Leverage + FX hedge + fee/carry waterfall -> net LP IRR/MOIC
├── ic_memo_draft.py           # 7. Orchestrates 1-6 into a structured IC memo (Markdown)
└── finance_utils.py           # Shared IRR/NPV/amortization helpers

tests/                         # pytest suite (54 tests)
output/                        # CSV/Markdown artifacts written by each module's `main()`
```

Each numbered module can be run standalone (`python -m srt_model.<module>`)
and also imported as a library by downstream modules — `ic_memo_draft.py`
imports and calls every other module directly, so every figure in the memo
is computed live rather than hard-coded.

## Methodology summary

1. **`loan_portfolio.py`** — Generates 2,000–5,000 synthetic commercial
   loans (default 3,000), one per obligor, with industry sector, balance,
   coupon, origination/maturity, rating (an 8-notch internal scale mapped to
   a 1-year PD), LGD (driven by collateral type/seniority), and
   amortization type. Computes weighted-average coupon, sector
   concentration, rating distribution, and single-name concentration.

2. **`historical_loss_analysis.py`** — Builds a synthetic 8-vintage
   (annual-origination) static-pool performance panel with a Weibull-shaped
   default-timing curve and vintage-level credit-cycle variation. Younger
   vintages are right-censored at the "as of" date, exactly as real
   performance data would be. A dollar-weighted vintage development curve
   is fit from the more seasoned vintages and used to extrapolate each
   vintage's *ultimate* cumulative default rate (a chain-ladder-style
   development-factor approach), then balance-weighted into base-case CDR
   and LGD assumptions.

3. **`cashflow_waterfall.py`** — Projects aggregate pool cash flows
   (interest, scheduled principal, defaults, recoveries net of a recovery
   lag, prepayments) using standard ABS/CLO cash flow mechanics, then
   allocates them through a 3-tranche structure (Senior / Mezzanine /
   First-Loss) with senior-first interest and principal pay and
   reverse-seniority loss allocation. Reports each tranche's realized yield
   (via IRR), interest/principal received, and cumulative loss absorbed.

4. **`monte_carlo_risk.py`** — Simulates 5,000–10,000 correlated default
   paths with a single-factor Gaussian copula (a shared systemic factor
   plus obligor-level idiosyncratic risk, calibrated so each loan's
   marginal default probability matches its origination PD) and stochastic
   LGD (Beta-distributed around each loan's LGD assumption). Produces the
   portfolio loss distribution and tranche-level expected loss, 95th/99th
   percentile loss (VaR), and expected shortfall (CVaR), with particular
   focus on the mezzanine and first-loss tranches.

5. **`scenario_sensitivity.py`** — Runs Base / Stress / Severe Stress
   scenarios that jointly move default rate (up to 3.5x), LGD (up to
   1.45x), and prepayment speed (down to 0.3x, reflecting slower voluntary
   refinancing in a downturn), and reports the resulting tranche-level
   yield impact versus base case.

6. **`fund_rollup.py`** — Rolls the mezzanine tranche's cash flows up into
   a fund-level model: a cash-sweep financing facility (leverage), an
   annualized bps FX-hedging cost drag for a non-USD LP share class, and a
   European (whole-fund) fee waterfall — management fee, return of
   capital, compounding preferred return/hurdle, 100% GP catch-up, and a
   residual carry split — producing net LP IRR and MOIC under each
   scenario.

7. **`ic_memo_draft.py`** — Runs the full pipeline end-to-end and formats
   the results into a structured Markdown IC memo (transaction overview,
   collateral summary, historical loss analysis, structure/tranching,
   Monte Carlo risk summary, scenario sensitivity, fund economics, and a
   draft recommendation), with every figure pulled directly from the other
   six modules.

### A note on internal consistency

The Monte Carlo module (bottom-up, obligor-rating-implied PDs) and the
historical vintage curve (top-down, realized default experience) are two
independent lenses on the same portfolio's credit risk, calibrated to be
broadly consistent with each other but not identical — exactly the kind of
reconciliation a real analyst would need to do between an origination-time
view and a realized-performance view. Don't expect the two to match
exactly; the memo surfaces both.

## Installation

```bash
pip install -r requirements.txt
```

## Running the model

```bash
# Run each stage independently (writes CSV/Markdown to output/)
python -m srt_model.loan_portfolio
python -m srt_model.historical_loss_analysis
python -m srt_model.cashflow_waterfall
python -m srt_model.monte_carlo_risk
python -m srt_model.scenario_sensitivity
python -m srt_model.fund_rollup

# Run the full pipeline and generate the IC memo
python -m srt_model.ic_memo_draft
```

## Running the tests

```bash
python -m pytest -v
```

The suite (54 tests) covers portfolio generation and summary statistics,
vintage curve construction, waterfall allocation logic (loss sequencing,
notional conservation, interest/principal pay order), Monte Carlo
reproducibility/convergence and tail-risk metric ordering, scenario math
(monotonicity of stress impacts), and fund roll-up arithmetic (leverage,
fee waterfall ordering, IRR/MOIC edge cases).

## Key simplifying assumptions

- One loan per obligor (no multi-facility obligors).
- The loan pool is modeled at the aggregate (pool-factor) level for cash
  flow projection, characterized by its weighted average coupon and life,
  rather than loan-by-loan cash flows.
- Losses are allocated to tranches as permanent write-downs (no
  write-up from later recoveries).
- The fund's LP is assumed to fund 100% of fund equity (no separate GP
  co-invest tranche).
- FX hedging cost is modeled as a flat annualized bps drag on NAV rather
  than derived from a forward curve/rate differential.

See the module docstrings in `srt_model/` for the full methodology and
assumptions behind each step.
