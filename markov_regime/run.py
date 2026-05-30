"""CLI entry point for the Markov regime model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .filters import FilterConfig, decide_regime_filter, walk_forward_filter_backtest
from .regime import (
    STATES,
    build_transition_matrix,
    label_regimes,
    n_step_forecast,
    stationary_distribution,
    walk_forward_backtest,
)


def _fetch_with_retry(ticker: str, years: int) -> pd.DataFrame:
    """Fetch daily adjusted OHLCV through yfinance with one retry."""
    import yfinance as yf

    end = pd.Timestamp.now(tz="UTC").normalize()
    start = end - pd.DateOffset(years=years)

    for attempt in (1, 2):
        try:
            df = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:
            print(f"  ! yfinance error on attempt {attempt}: {exc}")
            df = pd.DataFrame()

        if not df.empty:
            return df

        if attempt == 1:
            print("  ! yfinance returned empty data; retrying in 30 seconds.")
            time.sleep(30)

    raise RuntimeError(
        f"yfinance returned empty data for {ticker} after retry. "
        "Yahoo may be rate-limiting; try again in a few minutes."
    )


def _parse_csv(value: str | None, cast=str) -> list:
    if not value:
        return []
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _close_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            close = df["Close"]
        else:
            close = df.xs("Close", axis=1, level=-1)
        if isinstance(close, pd.DataFrame):
            preferred = ticker if ticker in close.columns else close.columns[0]
            close = close[preferred]
        return close.dropna()

    return df["Close"].dropna()


def _print_matrix(title: str, matrix: np.ndarray) -> None:
    print(f"\n{title}")
    print(f"            {STATES[0]:>9s} {STATES[1]:>9s} {STATES[2]:>9s}")
    for i, from_state in enumerate(STATES):
        row = "  ".join(f"{matrix[i, j] * 100:7.2f}%" for j in range(3))
        print(f"  {from_state:>9s}  {row}")


def _matrix_frame(matrix: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=STATES, columns=STATES)


def _current_projection(labels: pd.Series, matrix: np.ndarray) -> dict:
    current_state = int(labels.iloc[-1])
    probabilities = matrix[current_state]
    projected_state = int(np.argmax(probabilities))
    directional_score = float(probabilities[2] - probabilities[0])
    return {
        "as_of": str(labels.index[-1].date()),
        "current_regime": STATES[current_state],
        "projected_next_regime": STATES[projected_state],
        "projected_probability": float(probabilities[projected_state]),
        "bear_probability": float(probabilities[0]),
        "sideways_probability": float(probabilities[1]),
        "bull_probability": float(probabilities[2]),
        "directional_score": directional_score,
    }


def _print_projection(projection: dict) -> None:
    print(f"\nNext trading-session regime projection (from latest close {projection['as_of']}):")
    print(f"  Current regime:          {projection['current_regime']}")
    print(
        "  Projected next regime:   "
        f"{projection['projected_next_regime']} "
        f"({projection['projected_probability'] * 100:.2f}%)"
    )
    print("  Next-session probabilities:")
    print(f"       Bear: {projection['bear_probability'] * 100:6.2f}%")
    print(f"   Sideways: {projection['sideways_probability'] * 100:6.2f}%")
    print(f"       Bull: {projection['bull_probability'] * 100:6.2f}%")
    print(
        "  Directional score "
        f"(Bull probability - Bear probability): {projection['directional_score']:+.3f}"
    )


def _print_filter_decision(decision: dict) -> None:
    print("\nRegime filter overlay:")
    print(f"  Target exposure:              {decision['target_exposure']:.2f}x")
    print(f"  Mean-reversion longs allowed: {decision['mean_reversion_longs_allowed']}")
    print(f"  Bull persistence:             {decision['bull_persistence'] * 100:.2f}%")
    print(f"  Bear persistence:             {decision['bear_persistence'] * 100:.2f}%")
    if decision["trend_ok"] is not None:
        print(f"  Trend filter passed:          {decision['trend_ok']}")
    if decision["vol_percentile"] is not None:
        print(f"  Volatility percentile:        {decision['vol_percentile']:.2f}")
    if decision["macro_risk_on"] is not None:
        print(f"  Macro/risk filter passed:     {decision['macro_risk_on']}")
    if decision["breadth_percent_above_sma"] is not None:
        print(f"  Breadth above SMA:            {decision['breadth_percent_above_sma'] * 100:.1f}%")
    print("  Rules fired:")
    for reason in decision["reasons"]:
        print(f"    - {reason}")


def _write_report(
    export_dir: Path,
    ticker: str,
    years: int,
    window: int,
    threshold: float,
    matrix: np.ndarray,
    forecast: np.ndarray,
    forecast_steps: int,
    stationary: np.ndarray,
    projection: dict,
    backtest: dict,
    filter_decision: dict | None = None,
    filter_backtest: dict | None = None,
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)

    transition_df = _matrix_frame(matrix)
    forecast_df = _matrix_frame(forecast)
    stationary_df = pd.DataFrame(
        {"regime": STATES, "probability": stationary},
    )
    projection_df = pd.DataFrame([projection])
    filter_df = pd.DataFrame([filter_decision]) if filter_decision else None

    transition_df.to_csv(export_dir / "transition_matrix.csv")
    forecast_df.to_csv(export_dir / "forecast_matrix.csv")
    stationary_df.to_csv(export_dir / "stationary_distribution.csv", index=False)
    projection_df.to_csv(export_dir / "next_session_projection.csv", index=False)
    if filter_df is not None:
        filter_df.drop(columns=["reasons"], errors="ignore").to_csv(export_dir / "filter_overlay.csv", index=False)

    summary = {
        "ticker": ticker,
        "years": years,
        "window": window,
        "threshold": threshold,
        "forecast_steps": forecast_steps,
        "projection": projection,
        "backtest": backtest,
        "filter_overlay": filter_decision,
        "filter_backtest": filter_backtest,
    }
    (export_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def percent_frame(df: pd.DataFrame) -> pd.DataFrame:
        return df.map(lambda value: f"{value * 100:.2f}%")

    stationary_html = stationary_df.assign(
        probability=stationary_df["probability"].map(lambda value: f"{value * 100:.2f}%")
    ).to_html(index=False)
    projection_html = projection_df.assign(
        projected_probability=projection_df["projected_probability"].map(lambda value: f"{value * 100:.2f}%"),
        bear_probability=projection_df["bear_probability"].map(lambda value: f"{value * 100:.2f}%"),
        sideways_probability=projection_df["sideways_probability"].map(lambda value: f"{value * 100:.2f}%"),
        bull_probability=projection_df["bull_probability"].map(lambda value: f"{value * 100:.2f}%"),
        directional_score=projection_df["directional_score"].map(lambda value: f"{value:+.3f}"),
    ).to_html(index=False)

    sharpe = backtest["sharpe"]
    max_drawdown = backtest["max_drawdown"]
    sharpe_text = f"{sharpe:.3f}" if np.isfinite(sharpe) else "NaN"
    drawdown_text = f"{max_drawdown * 100:.2f}%" if np.isfinite(max_drawdown) else "NaN"
    filter_html = ""
    if filter_decision and filter_backtest:
        filter_sharpe = filter_backtest["sharpe"]
        filter_drawdown = filter_backtest["max_drawdown"]
        filter_sharpe_text = f"{filter_sharpe:.3f}" if np.isfinite(filter_sharpe) else "NaN"
        filter_drawdown_text = f"{filter_drawdown * 100:.2f}%" if np.isfinite(filter_drawdown) else "NaN"
        reasons = "".join(f"<li>{reason}</li>" for reason in filter_decision["reasons"])
        filter_html = f"""
    <section><h2>Filter Overlay</h2>
      <table>
        <tr><th>Target exposure</th><td>{filter_decision["target_exposure"]:.2f}x</td></tr>
        <tr><th>Mean-reversion longs allowed</th><td>{filter_decision["mean_reversion_longs_allowed"]}</td></tr>
        <tr><th>Overlay Sharpe</th><td>{filter_sharpe_text}</td></tr>
        <tr><th>Overlay max drawdown</th><td>{filter_drawdown_text}</td></tr>
        <tr><th>Average exposure</th><td>{filter_backtest["avg_exposure"]:.2f}x</td></tr>
      </table>
      <ul>{reasons}</ul>
    </section>
"""

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{ticker} Markov Regime Report</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 32px; color: #17202a; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .meta {{ color: #53616f; margin-top: 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 24px; }}
    section {{ border: 1px solid #d8dee4; border-radius: 8px; padding: 16px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d8dee4; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f3f6f8; }}
    .callout {{ font-size: 18px; margin: 8px 0 16px; }}
    .score {{ font-weight: 600; }}
    li {{ margin-bottom: 6px; }}
  </style>
</head>
<body>
  <h1>{ticker} Markov Regime Report</h1>
  <p class="meta">Years: {years} | Window: {window} | Threshold: {threshold:.4f} | Forecast steps: {forecast_steps}</p>
  <p class="callout">Current regime: <strong>{projection["current_regime"]}</strong>.
  Projected next trading-session regime: <strong>{projection["projected_next_regime"]}</strong>
  ({projection["projected_probability"] * 100:.2f}%).</p>
  <p class="score">Directional score: {projection["directional_score"]:+.3f}</p>
  <div class="grid">
    <section><h2>Next Session</h2>{projection_html}</section>
    <section><h2>Transition Matrix</h2>{percent_frame(transition_df).to_html()}</section>
    <section><h2>{forecast_steps}-Step Forecast</h2>{percent_frame(forecast_df).to_html()}</section>
    <section><h2>Stationary Distribution</h2>{stationary_html}</section>
    <section><h2>Walk-Forward Backtest</h2>
      <table>
        <tr><th>Sharpe</th><td>{sharpe_text}</td></tr>
        <tr><th>Max drawdown</th><td>{drawdown_text}</td></tr>
        <tr><th>Trades evaluated</th><td>{backtest["n_trades"]}</td></tr>
      </table>
    </section>
    {filter_html}
  </div>
</body>
</html>
"""
    (export_dir / "report.html").write_text(html, encoding="utf-8")


def _hmm_available() -> bool:
    try:
        import hmmlearn  # noqa: F401
    except ImportError:
        return False
    return True


def _fetch_optional_close(ticker: str | None, years: int) -> pd.Series | None:
    if not ticker:
        return None
    df = _fetch_with_retry(ticker, years)
    return _close_series(df, ticker)


def _fetch_breadth_closes(tickers: list[str], years: int) -> dict[str, pd.Series]:
    closes: dict[str, pd.Series] = {}
    for ticker in tickers:
        print(f"  fetching breadth ticker {ticker}...")
        closes[ticker] = _close_series(_fetch_with_retry(ticker, years), ticker)
    return closes


def _run_grid(args: argparse.Namespace, config: FilterConfig) -> int:
    tickers = _parse_csv(args.tickers, str) or [args.ticker]
    windows = _parse_csv(args.windows, int) or [args.window]
    thresholds = _parse_csv(args.thresholds, float) or [args.threshold]
    rows: list[dict] = []

    for ticker in tickers:
        print(f"\nGrid ticker: {ticker}")
        close = _close_series(_fetch_with_retry(ticker, args.years), ticker)
        for window in windows:
            for threshold in thresholds:
                labels = label_regimes(close, window=window, threshold=threshold)
                matrix = build_transition_matrix(labels)
                projection = _current_projection(labels, matrix)
                filter_decision = decide_regime_filter(close, labels, matrix, config=config)
                baseline = walk_forward_backtest(close, labels, min_train=args.min_train)
                filtered = walk_forward_filter_backtest(close, labels, config=config, min_train=args.min_train)
                rows.append(
                    {
                        "ticker": ticker,
                        "window": window,
                        "threshold": threshold,
                        "current_regime": projection["current_regime"],
                        "projected_next_regime": projection["projected_next_regime"],
                        "projected_probability": projection["projected_probability"],
                        "target_exposure": filter_decision["target_exposure"],
                        "baseline_sharpe": baseline["sharpe"],
                        "baseline_max_drawdown": baseline["max_drawdown"],
                        "filtered_sharpe": filtered["sharpe"],
                        "filtered_max_drawdown": filtered["max_drawdown"],
                        "filtered_avg_exposure": filtered["avg_exposure"],
                    }
                )

    results = pd.DataFrame(rows)
    print("\nWindow/threshold/ticker grid:")
    display = results.copy()
    for column in [
        "projected_probability",
        "baseline_sharpe",
        "baseline_max_drawdown",
        "filtered_sharpe",
        "filtered_max_drawdown",
        "filtered_avg_exposure",
    ]:
        display[column] = display[column].map(lambda value: f"{value:.3f}" if np.isfinite(value) else "NaN")
    print(display.to_string(index=False))

    if args.export_dir:
        args.export_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.export_dir / "grid_results.csv", index=False)
        print(f"\nGrid results written to: {(args.export_dir / 'grid_results.csv').resolve()}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="markov-regime")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--forecast-steps", type=int, default=5)
    parser.add_argument("--min-train", type=int, default=252)
    parser.add_argument("--export-dir", type=Path, help="Write CSV files plus report.html to this folder")
    parser.add_argument("--no-filters", action="store_true", help="Skip the separate regime filter overlay")
    parser.add_argument("--bull-persistence-high", type=float, default=0.85)
    parser.add_argument("--bear-persistence-high", type=float, default=0.80)
    parser.add_argument("--trend-window", type=int, default=200)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--vol-high-percentile", type=float, default=0.80)
    parser.add_argument("--macro-ticker", help="Optional risk/macro ticker such as ^VIX; above SMA reduces exposure")
    parser.add_argument("--macro-window", type=int, default=50)
    parser.add_argument("--breadth-tickers", help="Comma-separated breadth tickers; percent above SMA drives breadth filter")
    parser.add_argument("--breadth-window", type=int, default=50)
    parser.add_argument("--grid", action="store_true", help="Test ticker/window/threshold combinations")
    parser.add_argument("--tickers", help="Comma-separated ticker list for --grid")
    parser.add_argument("--windows", help="Comma-separated rolling-return windows for --grid")
    parser.add_argument("--thresholds", help="Comma-separated regime thresholds for --grid")
    parser.add_argument("--no-hmm", action="store_true")
    args = parser.parse_args()
    config = FilterConfig(
        bull_persistence_high=args.bull_persistence_high,
        bear_persistence_high=args.bear_persistence_high,
        trend_window=args.trend_window,
        vol_window=args.vol_window,
        vol_high_percentile=args.vol_high_percentile,
        macro_window=args.macro_window,
        breadth_window=args.breadth_window,
    )

    if args.grid:
        return _run_grid(args, config)

    print(
        "\nmarkov-regime "
        f"ticker={args.ticker} years={args.years} window={args.window} "
        f"threshold={args.threshold}"
    )
    print(f"  fetching {args.ticker} from Yahoo Finance...")
    df = _fetch_with_retry(args.ticker, args.years)
    close = _close_series(df, args.ticker)
    print(f"  fetched {len(close)} rows | {close.index.min().date()} -> {close.index.max().date()}")

    labels = label_regimes(close, window=args.window, threshold=args.threshold)
    matrix = build_transition_matrix(labels)
    stationary = stationary_distribution(matrix)
    projection = _current_projection(labels, matrix)
    macro_close = None
    breadth_closes: dict[str, pd.Series] = {}
    if not args.no_filters and args.macro_ticker:
        print(f"  fetching macro/risk ticker {args.macro_ticker}...")
        macro_close = _fetch_optional_close(args.macro_ticker, args.years)
    if not args.no_filters and args.breadth_tickers:
        breadth_closes = _fetch_breadth_closes(_parse_csv(args.breadth_tickers, str), args.years)

    _print_projection(projection)
    filter_decision = None
    filter_result = None
    if not args.no_filters:
        filter_decision = decide_regime_filter(
            close=close,
            labels=labels,
            matrix=matrix,
            config=config,
            macro_close=macro_close,
            breadth_closes=breadth_closes,
        )
        _print_filter_decision(filter_decision)

    _print_matrix("Transition matrix (rows = from, cols = to):", matrix)

    print("\nPersistence diagonal:")
    for i, state in enumerate(STATES):
        print(f"  {state} -> {state}: {matrix[i, i] * 100:.2f}%")

    print("\nStationary distribution (long-run regime mix):")
    for state, probability in zip(STATES, stationary):
        print(f"  {state:>9s}: {probability * 100:.2f}%")

    forecast = n_step_forecast(matrix, args.forecast_steps)
    _print_matrix(f"{args.forecast_steps}-step transition forecast:", forecast)

    print("\nWalk-forward backtest (re-estimating matrix at every step, no lookahead)...")
    result = walk_forward_backtest(close, labels, min_train=args.min_train)
    sharpe = result["sharpe"]
    max_drawdown = result["max_drawdown"]
    if np.isfinite(sharpe):
        print(f"  Sharpe (annualized, walk-forward): {sharpe:.3f}")
    else:
        print("  Sharpe: NaN (insufficient data or zero return volatility)")
    if np.isfinite(max_drawdown):
        print(f"  Max drawdown:                        {max_drawdown * 100:.2f}%")
    else:
        print("  Max drawdown: NaN")
    print(f"  Trades evaluated: {result['n_trades']}")
    if not args.no_filters:
        print("\nFilter-overlay backtest (long-only exposure sizing, no lookahead)...")
        filter_result = walk_forward_filter_backtest(
            close=close,
            labels=labels,
            config=config,
            min_train=args.min_train,
            macro_close=macro_close,
            breadth_closes=breadth_closes,
        )
        filter_sharpe = filter_result["sharpe"]
        filter_mdd = filter_result["max_drawdown"]
        if np.isfinite(filter_sharpe):
            print(f"  Sharpe (annualized, filter overlay): {filter_sharpe:.3f}")
        else:
            print("  Sharpe: NaN (insufficient data or zero return volatility)")
        if np.isfinite(filter_mdd):
            print(f"  Max drawdown:                         {filter_mdd * 100:.2f}%")
        else:
            print("  Max drawdown: NaN")
        print(f"  Average exposure:                     {filter_result['avg_exposure']:.2f}x")
        print(f"  Trades evaluated:                     {filter_result['n_trades']}")

    if args.export_dir:
        _write_report(
            export_dir=args.export_dir,
            ticker=args.ticker,
            years=args.years,
            window=args.window,
            threshold=args.threshold,
            matrix=matrix,
            forecast=forecast,
            forecast_steps=args.forecast_steps,
            stationary=stationary,
            projection=projection,
            backtest=result,
            filter_decision=filter_decision,
            filter_backtest=filter_result,
        )
        print(f"\nReport files written to: {args.export_dir.resolve()}")
        print(f"Open this file to view the results: {(args.export_dir / 'report.html').resolve()}")

    if not args.no_hmm and _hmm_available():
        print("\nFitting Hidden Markov Model (Baum-Welch + Viterbi via hmmlearn)...")
        try:
            from .hmm_extension import fit_hmm

            returns = close.pct_change().dropna()
            model, _hidden = fit_hmm(returns, n_components=3)
            if model is None:
                print("  HMM extension skipped; hmmlearn import failed at runtime.")
            else:
                means = np.array([model.means_[k][0] for k in range(model.n_components)])
                order = np.argsort(means)
                names = ["Bear (lowest mean return)", "Sideways", "Bull (highest mean return)"]
                print("  HMM regime mean daily returns (sorted):")
                for rank, state_index in enumerate(order):
                    print(f"    {names[rank]:<30s} state {state_index}: {means[state_index] * 100:+.3f}% per day")
                print("  Note: Baum-Welch finds local maxima; production work should fit several seeds.")
        except Exception as exc:
            print(f"  HMM extension skipped at runtime: {exc}")
    else:
        print("\nHMM extension skipped; observable Markov model completed successfully.")

    print("\nBacktests are historical, not forward-looking. This is research tooling, not financial advice.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
