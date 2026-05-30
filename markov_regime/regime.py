"""Observable Markov regime model utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

STATES = ["Bear", "Sideways", "Bull"]


def label_regimes(
    close: pd.Series,
    window: int = 20,
    threshold: float = 0.02,
) -> pd.Series:
    """Label each day as Bull, Bear, or Sideways from rolling return."""
    rolling_return = close.pct_change(window).dropna()
    labels = pd.Series(1, index=rolling_return.index, dtype=int)
    labels[rolling_return > threshold] = 2
    labels[rolling_return < -threshold] = 0
    return labels


def build_transition_matrix(labels: pd.Series) -> np.ndarray:
    """Estimate the 3x3 transition matrix from adjacent regime labels."""
    n_states = 3
    counts = np.zeros((n_states, n_states), dtype=float)
    arr = labels.dropna().astype(int).to_numpy()

    for i in range(len(arr) - 1):
        from_state = arr[i]
        to_state = arr[i + 1]
        if 0 <= from_state < n_states and 0 <= to_state < n_states:
            counts[from_state, to_state] += 1.0

    matrix = np.zeros_like(counts)
    for i in range(n_states):
        row_sum = counts[i].sum()
        if row_sum > 0:
            matrix[i] = counts[i] / row_sum
        else:
            matrix[i, i] = 1.0
    return matrix


def stationary_distribution(matrix: np.ndarray) -> np.ndarray:
    """Solve pi P = pi with sum(pi) = 1."""
    n_states = matrix.shape[0]
    lhs = np.vstack([matrix.T - np.eye(n_states), np.ones(n_states)])
    rhs = np.zeros(n_states + 1)
    rhs[-1] = 1.0
    solution, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    solution = np.clip(np.real(solution), 0.0, None)

    total = solution.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full(n_states, 1.0 / n_states)
    return solution / total


def n_step_forecast(matrix: np.ndarray, n_steps: int) -> np.ndarray:
    """Return the n-step transition matrix P^n."""
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    return np.linalg.matrix_power(matrix, n_steps)


def signal_from_matrix(matrix: np.ndarray, current_state: int) -> float:
    """Compute signed signal: P(next=Bull|current) - P(next=Bear|current)."""
    return float(matrix[current_state, 2] - matrix[current_state, 0])


def walk_forward_backtest(
    close: pd.Series,
    labels: pd.Series,
    min_train: int = 252,
) -> dict[str, float | int]:
    """Re-estimate the matrix through time and score next-day returns."""
    daily_returns = close.pct_change().dropna()
    common_index = labels.index.intersection(daily_returns.index)
    labels = labels.loc[common_index]
    daily_returns = daily_returns.loc[common_index]

    if len(labels) < min_train + 30:
        return {"sharpe": float("nan"), "max_drawdown": float("nan"), "n_trades": 0}

    strategy_returns: list[float] = []
    for t in range(min_train, len(labels) - 1):
        matrix_t = build_transition_matrix(labels.iloc[:t])
        current_state = int(labels.iloc[t])
        signal = signal_from_matrix(matrix_t, current_state)
        position = float(np.sign(signal))
        strategy_returns.append(position * float(daily_returns.iloc[t + 1]))

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
    }
