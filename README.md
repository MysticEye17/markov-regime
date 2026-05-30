# Markov Regime

Observable Markov regime model for trading research on market tickers.

The tool fetches daily OHLCV data with `yfinance`, labels each day as Bear, Sideways, or Bull from a rolling return, estimates a 3x3 transition matrix, projects the next trading-session regime, computes the stationary distribution, and runs a walk-forward backtest.

The core regime model stays separate from the filter overlay. The overlay converts persistence, trend, volatility, macro/risk, and breadth checks into a target exposure and a separate long-only walk-forward backtest.

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

## Regime Filter Overlay

The filter overlay is enabled by default and prints a separate section from the core transition matrix.

It implements:

- larger exposure when Bull persistence is high
- reduced or zero exposure when Bear persistence is high
- disabling mean-reversion longs during persistent Bear regimes
- trend confirmation using a moving average
- volatility throttling using realized-volatility percentile
- optional macro/risk filter, such as VIX above its moving average
- optional breadth filter using a basket of tickers and percent above moving average

Example with VIX as the macro/risk filter and sector ETFs as breadth:

```powershell
uv run python -m markov_regime.run --ticker SPY --years 10 --macro-ticker ^VIX --breadth-tickers XLB,XLC,XLE,XLF,XLI,XLK,XLP,XLRE,XLU,XLV,XLY --export-dir reports\spy-filters
```

Skip the overlay:

```powershell
uv run python -m markov_regime.run --ticker SPY --no-filters
```

Key knobs:

```text
--bull-persistence-high 0.85
--bear-persistence-high 0.80
--trend-window 200
--vol-window 20
--vol-high-percentile 0.80
--macro-ticker ^VIX
--macro-window 50
--breadth-tickers XLF,XLK,XLV
--breadth-window 50
```

## Grid Testing

Use grid mode to test multiple tickers, rolling windows, and regime thresholds.

```powershell
uv run python -m markov_regime.run --grid --tickers SPY,QQQ,IWM --windows 10,20,40,60 --thresholds 0.01,0.02,0.03 --years 10 --export-dir reports\grid
```

Grid mode writes `grid_results.csv` when `--export-dir` is set.

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
- filter-overlay target exposure, rule explanations, Sharpe, max drawdown, and average exposure
- optional HMM regime mean returns

Use `--export-dir` to write CSV files, `summary.json`, and `report.html`.

## Notes

This is research tooling, not financial advice.
