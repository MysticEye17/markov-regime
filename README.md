# Markov Regime

Observable Markov regime model for trading research on market tickers.

The tool fetches daily OHLCV data with `yfinance`, labels each day as Bear, Sideways, or Bull from a rolling return, estimates a 3x3 transition matrix, projects the next trading-session regime, computes the stationary distribution, and runs a walk-forward backtest.

## Install

Install `uv` first if needed:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then from this repository:

```powershell
uv python install 3.12
uv sync
```

Optional HMM support:

```powershell
uv pip install "hmmlearn>=0.3"
```

## Run

```powershell
uv run python -m markov_regime.run --ticker SPY --years 10
```

Examples:

```powershell
uv run python -m markov_regime.run --ticker AAPL --years 5 --window 60
uv run python -m markov_regime.run --ticker BTC-USD --years 4 --threshold 0.05
uv run python -m markov_regime.run --ticker QQQ --no-hmm
uv run python -m markov_regime.run --ticker SPY --export-dir reports\spy
```

## Output

Each run prints:

- latest/current observed regime
- next trading-session regime projection
- Bear, Sideways, and Bull next-session probabilities
- transition matrix
- persistence diagonal
- stationary distribution
- n-step transition forecast
- walk-forward Sharpe, max drawdown, and trade count
- optional HMM regime mean returns

Use `--export-dir` to write CSV files, `summary.json`, and `report.html`.

## Notes

This is research tooling, not financial advice. Backtests are historical and not forward-looking.
