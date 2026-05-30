---
name: markov-regime
description: Observable Markov regime model for trading research on any ticker. Use when the user asks Codex to run Markov-regime analysis, build a Bull/Bear/Sideways transition matrix, compute stationary regime distribution, forecast n-step transitions, project the next trading-session regime, export readable reports, run a walk-forward Sharpe/max-drawdown backtest, or optionally fit an HMM to returns.
---

# Markov Regime

Use this skill to run the bundled Python research tool against market tickers.
It fetches daily OHLCV with `yfinance`, labels regimes from rolling returns, estimates a 3x3 observable Markov transition matrix, solves the stationary distribution, projects the next trading-session regime probabilities, and runs a no-lookahead walk-forward backtest. It also keeps a separate regime filter overlay for position sizing, risk reduction, mean-reversion blocking, trend/volatility filters, optional macro/risk filters, optional breadth filters, and grid tests across tickers/windows/thresholds.

## Quick Start

Run from the skill directory:

```powershell
cd C:\Users\patri\.codex\skills\markov-regime
uv run python -m markov_regime.run --ticker SPY --years 10
```

If `uv` is installed but not on PATH in the current shell, use:

```powershell
C:\Users\patri\.local\bin\uv.exe run python -m markov_regime.run --ticker SPY --years 10
```

Common variants:

```powershell
uv run python -m markov_regime.run --ticker AAPL --years 5 --window 60
uv run python -m markov_regime.run --ticker BTC-USD --years 4 --threshold 0.05
uv run python -m markov_regime.run --ticker QQQ --no-hmm
uv run python -m markov_regime.run --ticker SPY --export-dir reports\spy
uv run python -m markov_regime.run --ticker SPY --macro-ticker ^VIX --breadth-tickers XLB,XLC,XLE,XLF,XLI,XLK,XLP,XLRE,XLU,XLV,XLY
uv run python -m markov_regime.run --grid --tickers SPY,QQQ,IWM --windows 10,20,40,60 --thresholds 0.01,0.02,0.03 --export-dir reports\grid
```

If `uv` or dependencies are missing, install them before running:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
cd C:\Users\patri\.codex\skills\markov-regime
uv python install 3.12
uv sync
uv pip install "hmmlearn>=0.3"
```

The HMM layer is optional. If `hmmlearn` fails to install, run with `--no-hmm`; the observable Markov model, stationary distribution, next-session projection, and walk-forward backtest still work.

## Outputs

Every successful run prints:

- ticker, date range, and fetched row count
- current/latest observed regime
- next trading-session projection with Bear, Sideways, and Bull probabilities
- separate regime filter overlay with target exposure and rule explanations
- transition matrix with rows as current regime and columns as next regime
- persistence diagonal for Bear, Sideways, and Bull
- stationary distribution as the long-run regime mix
- walk-forward annualized Sharpe, max drawdown, and number of evaluated trades
- optional HMM mean daily returns by hidden state when `hmmlearn` is available

Use `--export-dir <folder>` to write:

- `summary.json`
- `transition_matrix.csv`
- `stationary_distribution.csv`
- `forecast_matrix.csv`
- `next_session_projection.csv`
- `filter_overlay.csv`
- `report.html`

Open `report.html` in a browser for the easiest view of the matrix, forecast, and next-session projection.

## Implementation Notes

- Regimes are labeled from `close.pct_change(window)`: Bull if above `threshold`, Bear if below `-threshold`, otherwise Sideways.
- Transition probabilities are maximum-likelihood counts from adjacent regime labels.
- The next-session projection is the transition-matrix row for the latest observed regime.
- The filter overlay is separate from the core matrix: it converts persistence, trend, volatility, macro/risk, and breadth evidence into long-only target exposure.
- Mean-reversion longs are disabled when the latest regime is Bear and Bear persistence exceeds `--bear-persistence-high`.
- Use `--no-filters` to show only the core regime model.
- Use `--grid --tickers ... --windows ... --thresholds ...` to compare parameter stability across markets and regime definitions.
- Walk-forward backtest re-estimates the matrix at each timestep using only labels available before the traded return.
- The simple directional score is `P(next=Bull | current) - P(next=Bear | current)`.
- This is research tooling, not financial advice. Backtests are historical and not forward-looking.
