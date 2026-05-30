"""Risk overlay filters layered on top of the core Markov regime model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .regime import STATES, build_transition_matrix


@dataclass(frozen=True)
class FilterConfig:
    bull_persistence_high: float = 0.85
    bear_persistence_high: float = 0.80
    bull_large_size: float = 1.25
    bull_normal_size: float = 1.00
    sideways_size: float = 0.50
    bear_reduced_size: float = 0.25
    bear_defensive_size: float = 0.00
    max_exposure: float = 1.50
    trend_window: int = 200
    vol_window: int = 20
    vol_high_percentile: float = 0.80
    macro_window: int = 50
    breadth_window: int = 50
    breadth_weak_threshold: float = 0.40
    breadth_strong_threshold: float = 0.60


def realized_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Annualized realized volatility from daily close-to-close returns."""
    return close.pct_change().rolling(window).std() * np.sqrt(252)


def _latest_at_or_before(series: pd.Series, index_value) -> float | None:
    available = series.dropna().loc[:index_value]
    if available.empty:
        return None
    return float(available.iloc[-1])


def _breadth_percent_above_sma(
    breadth_closes: dict[str, pd.Series],
    as_of,
    window: int,
) -> float | None:
    if not breadth_closes:
        return None

    flags: list[bool] = []
    for close in breadth_closes.values():
        history = close.dropna().loc[:as_of]
        if len(history) < window:
            continue
        flags.append(bool(history.iloc[-1] > history.rolling(window).mean().iloc[-1]))

    if not flags:
        return None
    return float(np.mean(flags))


def decide_regime_filter(
    close: pd.Series,
    labels: pd.Series,
    matrix: np.ndarray,
    config: FilterConfig | None = None,
    macro_close: pd.Series | None = None,
    breadth_closes: dict[str, pd.Series] | None = None,
) -> dict[str, float | str | bool | None | list[str]]:
    """Return current risk overlay decision separated from regime estimation."""
    config = config or FilterConfig()
    as_of = labels.index[-1]
    current_state = int(labels.iloc[-1])
    current_regime = STATES[current_state]
    bull_persistence = float(matrix[2, 2])
    bear_persistence = float(matrix[0, 0])

    reasons: list[str] = []
    if current_regime == "Bull":
        if bull_persistence >= config.bull_persistence_high:
            exposure = config.bull_large_size
            reasons.append("Bull regime with high Bull persistence: size above normal.")
        else:
            exposure = config.bull_normal_size
            reasons.append("Bull regime without high-persistence boost: normal long size.")
    elif current_regime == "Bear":
        if bear_persistence >= config.bear_persistence_high:
            exposure = config.bear_defensive_size
            reasons.append("Bear regime with high Bear persistence: defensive risk setting.")
        else:
            exposure = config.bear_reduced_size
            reasons.append("Bear regime without high persistence: reduced long size.")
    else:
        exposure = config.sideways_size
        reasons.append("Sideways regime: partial exposure.")

    mean_reversion_longs_allowed = not (
        current_regime == "Bear" and bear_persistence >= config.bear_persistence_high
    )
    if not mean_reversion_longs_allowed:
        reasons.append("Mean-reversion longs disabled during persistent Bear regime.")

    trend_sma = close.rolling(config.trend_window).mean()
    latest_close = _latest_at_or_before(close, as_of)
    latest_sma = _latest_at_or_before(trend_sma, as_of)
    trend_ok = None
    if latest_close is not None and latest_sma is not None:
        trend_ok = latest_close >= latest_sma
        if not trend_ok:
            exposure = min(exposure, 0.50)
            reasons.append(f"Trend filter: close below {config.trend_window}-day SMA, cap exposure at 0.50.")
        else:
            reasons.append(f"Trend filter: close above {config.trend_window}-day SMA.")

    vol = realized_volatility(close, config.vol_window).dropna()
    vol_percentile = None
    if not vol.empty:
        latest_vol = _latest_at_or_before(vol, as_of)
        if latest_vol is not None:
            vol_percentile = float((vol.loc[:as_of] <= latest_vol).mean())
            if vol_percentile >= config.vol_high_percentile:
                exposure *= 0.50
                reasons.append(
                    f"Volatility filter: realized vol percentile {vol_percentile:.2f}, cut exposure by half."
                )
            else:
                reasons.append(f"Volatility filter: realized vol percentile {vol_percentile:.2f}.")

    macro_risk_on = None
    if macro_close is not None:
        macro_sma = macro_close.rolling(config.macro_window).mean()
        latest_macro = _latest_at_or_before(macro_close, as_of)
        latest_macro_sma = _latest_at_or_before(macro_sma, as_of)
        if latest_macro is not None and latest_macro_sma is not None:
            macro_risk_on = latest_macro <= latest_macro_sma
            if not macro_risk_on:
                exposure *= 0.50
                reasons.append(
                    f"Macro filter: macro/risk ticker above {config.macro_window}-day SMA, cut exposure by half."
                )
            else:
                reasons.append(f"Macro filter: macro/risk ticker below {config.macro_window}-day SMA.")

    breadth_percent = _breadth_percent_above_sma(
        breadth_closes or {},
        as_of=as_of,
        window=config.breadth_window,
    )
    breadth_state = None
    if breadth_percent is not None:
        if breadth_percent < config.breadth_weak_threshold:
            breadth_state = "weak"
            exposure *= 0.50
            reasons.append(
                f"Breadth filter: {breadth_percent:.0%} above SMA, weak breadth cuts exposure by half."
            )
        elif breadth_percent > config.breadth_strong_threshold and current_regime == "Bull":
            breadth_state = "strong"
            exposure *= 1.10
            reasons.append(
                f"Breadth filter: {breadth_percent:.0%} above SMA, strong breadth adds a small Bull boost."
            )
        else:
            breadth_state = "neutral"
            reasons.append(f"Breadth filter: {breadth_percent:.0%} above SMA.")

    exposure = float(np.clip(exposure, 0.0, config.max_exposure))
    return {
        "as_of": str(as_of.date()),
        "current_regime": current_regime,
        "target_exposure": exposure,
        "bull_persistence": bull_persistence,
        "bear_persistence": bear_persistence,
        "mean_reversion_longs_allowed": mean_reversion_longs_allowed,
        "trend_ok": trend_ok,
        "vol_percentile": vol_percentile,
        "macro_risk_on": macro_risk_on,
        "breadth_percent_above_sma": breadth_percent,
        "breadth_state": breadth_state,
        "reasons": reasons,
    }


def walk_forward_filter_backtest(
    close: pd.Series,
    labels: pd.Series,
    config: FilterConfig | None = None,
    min_train: int = 252,
    macro_close: pd.Series | None = None,
    breadth_closes: dict[str, pd.Series] | None = None,
) -> dict[str, float | int]:
    """Long-only walk-forward backtest for the filter overlay.

    This is separate from the core regime signal test. At each timestep it
    rebuilds the transition matrix and computes exposure using only data
    available through that day, then applies that exposure to the next return.
    """
    config = config or FilterConfig()
    daily_returns = close.pct_change().dropna()
    common_index = labels.index.intersection(daily_returns.index)
    labels = labels.loc[common_index]
    daily_returns = daily_returns.loc[common_index]

    if len(labels) < min_train + 30:
        return {
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "n_trades": 0,
            "avg_exposure": float("nan"),
        }

    strategy_returns: list[float] = []
    exposures: list[float] = []
    for t in range(min_train, len(labels) - 1):
        as_of = labels.index[t]
        matrix_t = build_transition_matrix(labels.iloc[:t])
        decision = decide_regime_filter(
            close=close.loc[:as_of],
            labels=labels.iloc[: t + 1],
            matrix=matrix_t,
            config=config,
            macro_close=macro_close.loc[:as_of] if macro_close is not None else None,
            breadth_closes={
                ticker: series.loc[:as_of]
                for ticker, series in (breadth_closes or {}).items()
            },
        )
        exposure = float(decision["target_exposure"])
        exposures.append(exposure)
        strategy_returns.append(exposure * float(daily_returns.iloc[t + 1]))

    returns = np.array(strategy_returns, dtype=float)
    volatility = returns.std(ddof=1)
    if volatility == 0 or not np.isfinite(volatility):
        sharpe = float("nan")
    else:
        sharpe = float(returns.mean() / volatility * np.sqrt(252))

    equity = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) else float("nan")

    return {
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "n_trades": int(len(returns)),
        "avg_exposure": float(np.mean(exposures)) if exposures else float("nan"),
    }
