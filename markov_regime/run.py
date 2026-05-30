"""CLI entry point for the Markov regime model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

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
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)

    transition_df = _matrix_frame(matrix)
    forecast_df = _matrix_frame(forecast)
    stationary_df = pd.DataFrame(
        {"regime": STATES, "probability": stationary},
    )
    projection_df = pd.DataFrame([projection])

    transition_df.to_csv(export_dir / "transition_matrix.csv")
    forecast_df.to_csv(export_dir / "forecast_matrix.csv")
    stationary_df.to_csv(export_dir / "stationary_distribution.csv", index=False)
    projection_df.to_csv(export_dir / "next_session_projection.csv", index=False)

    summary = {
        "ticker": ticker,
        "years": years,
        "window": window,
        "threshold": threshold,
        "forecast_steps": forecast_steps,
        "projection": projection,
        "backtest": backtest,
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


def main() -> int:
    parser = argparse.ArgumentParser(prog="markov-regime")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--forecast-steps", type=int, default=5)
    parser.add_argument("--min-train", type=int, default=252)
    parser.add_argument("--export-dir", type=Path, help="Write CSV files plus report.html to this folder")
    parser.add_argument("--no-hmm", action="store_true")
    args = parser.parse_args()

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

    _print_projection(projection)

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
