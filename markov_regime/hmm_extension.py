"""Optional Hidden Markov Model layer."""

from __future__ import annotations

import pandas as pd


def fit_hmm(returns: pd.Series, n_components: int = 3, random_state: int = 42):
    """Fit a Gaussian HMM on daily returns and return model plus states."""
    try:
        from hmmlearn import hmm
    except ImportError:
        return None, None

    x = returns.dropna().to_numpy().reshape(-1, 1)
    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        n_iter=200,
        random_state=random_state,
    )
    model.fit(x)
    hidden_states = model.predict(x)
    return model, hidden_states
