"""OLS: 36 J_ij edges -> questionnaire outcome."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def zscore_columns(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score each J_ij column across subjects (axis=0)."""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, mu.ravel(), sd.ravel()


def parse_edge_name(edge_name: str) -> tuple[int, int]:
    nums = re.findall(r"\d+", str(edge_name))
    if len(nums) == 1 and len(nums[0]) == 2:
        i = int(nums[0][0]) - 1
        j = int(nums[0][1]) - 1
    elif len(nums) == 2:
        i = int(nums[0]) - 1
        j = int(nums[1]) - 1
    else:
        raise ValueError(f"Could not parse item pair from feature name: {edge_name}")
    if not (0 <= i < 9 and 0 <= j < 9 and i != j):
        raise ValueError(f"Invalid PHQ-9 item pair inferred from: {edge_name}")
    return i, j


def coef_vector_to_9x9(coef_vec: np.ndarray, edge_names: list[str], n_items: int = 9) -> np.ndarray:
    W = np.zeros((n_items, n_items), dtype=float)
    for beta, name in zip(coef_vec, edge_names):
        i, j = parse_edge_name(name)
        W[i, j] = beta
        W[j, i] = beta
    return W


@dataclass
class JijOLSResult:
    outcome: str
    n_subjects: int
    intercept_raw: float
    intercept_z: float
    r2_raw: float
    r2_z: float
    r_raw: float
    p_raw: float
    r_z: float
    p_z: float
    rmse_raw: float
    rmse_z: float
    mae_raw: float
    mae_z: float
    beta_raw: np.ndarray
    beta_z: np.ndarray
    col_sd: np.ndarray
    y_observed: np.ndarray
    y_pred_raw: np.ndarray
    y_pred_z: np.ndarray
    W_raw: np.ndarray
    W_z: np.ndarray


def fit_jij_ols(
    X_raw: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    outcome: str = "outcome",
) -> JijOLSResult:
    X_raw = np.asarray(X_raw, dtype=float)
    y = np.asarray(y, dtype=float)

    if X_raw.shape[0] != len(y):
        raise ValueError(f"Subject count mismatch: X={X_raw.shape[0]}, y={len(y)}")
    if X_raw.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature count mismatch: X={X_raw.shape[1]}, names={len(feature_names)}"
        )

    X_z, _mu, col_sd = zscore_columns(X_raw)

    model_raw = LinearRegression().fit(X_raw, y)
    model_z = LinearRegression().fit(X_z, y)

    y_pred_raw = model_raw.predict(X_raw)
    y_pred_z = model_z.predict(X_z)

    r_raw, p_raw = pearsonr(y, y_pred_raw)
    r_z, p_z = pearsonr(y, y_pred_z)
    # r は観測 vs 予測（y = x からのずれ）。赤い表示用回帰線とは無関係。

    beta_raw = model_raw.coef_
    beta_z = model_z.coef_

    return JijOLSResult(
        outcome=outcome,
        n_subjects=len(y),
        intercept_raw=float(model_raw.intercept_),
        intercept_z=float(model_z.intercept_),
        r2_raw=float(r2_score(y, y_pred_raw)),
        r2_z=float(r2_score(y, y_pred_z)),
        r_raw=float(r_raw),
        p_raw=float(p_raw),
        r_z=float(r_z),
        p_z=float(p_z),
        rmse_raw=float(np.sqrt(mean_squared_error(y, y_pred_raw))),
        rmse_z=float(np.sqrt(mean_squared_error(y, y_pred_z))),
        mae_raw=float(mean_absolute_error(y, y_pred_raw)),
        mae_z=float(mean_absolute_error(y, y_pred_z)),
        beta_raw=beta_raw,
        beta_z=beta_z,
        col_sd=col_sd,
        y_observed=y,
        y_pred_raw=y_pred_raw,
        y_pred_z=y_pred_z,
        W_raw=coef_vector_to_9x9(beta_raw, feature_names),
        W_z=coef_vector_to_9x9(beta_z, feature_names),
    )


def _stable_sigmoid(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    out = np.empty_like(u)
    pos = u >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-u[pos]))
    exp_u = np.exp(u[~pos])
    out[~pos] = exp_u / (1.0 + exp_u)
    return out


def _unpack_one_neuron_params(params: np.ndarray, n_features: int) -> tuple[np.ndarray, float, float, float, float, float]:
    weights = params[:n_features]
    bias = float(params[n_features])
    amplitude, steepness, center, offset = params[n_features + 1 : n_features + 5]
    return weights, bias, float(amplitude), float(steepness), float(center), float(offset)


def _one_neuron_forward(
    X: np.ndarray, params: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Forward pass: y_hat = a * sigmoid(b * (X @ w + b0 - c)) + d."""
    n_features = X.shape[1]
    weights, bias, amplitude, steepness, center, offset = _unpack_one_neuron_params(params, n_features)

    linear = X @ weights + bias
    pre_activation = steepness * (linear - center)
    activation = _stable_sigmoid(pre_activation)
    y_hat = amplitude * activation + offset

    cache = {
        "X": X,
        "weights": weights,
        "bias": bias,
        "amplitude": amplitude,
        "steepness": steepness,
        "center": center,
        "offset": offset,
        "linear": linear,
        "pre_activation": pre_activation,
        "activation": activation,
    }
    return y_hat, cache


def _one_neuron_backward(
    y_obs: np.ndarray,
    y_hat: np.ndarray,
    cache: dict[str, np.ndarray | float],
) -> np.ndarray:
    """Backpropagation for mean squared error."""
    X = cache["X"]
    n_samples = X.shape[0]
    grad_y_hat = 2.0 * (y_hat - y_obs) / n_samples

    grad_offset = float(np.sum(grad_y_hat))
    grad_amplitude = float(np.sum(grad_y_hat * cache["activation"]))
    grad_activation = grad_y_hat * cache["amplitude"]
    activation = cache["activation"]
    grad_pre_activation = grad_activation * activation * (1.0 - activation)
    linear = cache["linear"]
    center = cache["center"]
    steepness = cache["steepness"]
    grad_steepness = float(np.sum(grad_pre_activation * (linear - center)))
    grad_center = float(np.sum(grad_pre_activation * (-steepness)))
    grad_linear = grad_pre_activation * steepness
    grad_bias = float(np.sum(grad_linear))
    grad_weights = X.T @ grad_linear

    return np.concatenate(
        [
            grad_weights,
            np.asarray([grad_bias, grad_amplitude, grad_steepness, grad_center, grad_offset], dtype=float),
        ]
    )


def _adam_update(
    params: np.ndarray,
    grad: np.ndarray,
    state: dict[str, np.ndarray],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    m = state.setdefault("m", np.zeros_like(params))
    v = state.setdefault("v", np.zeros_like(params))
    m = beta1 * m + (1.0 - beta1) * grad
    v = beta2 * v + (1.0 - beta2) * (grad ** 2)
    m_hat = m / (1.0 - beta1 ** step)
    v_hat = v / (1.0 - beta2 ** step)
    params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
    state["m"] = m
    state["v"] = v
    return params, state


def fit_jij_one_neuron_backprop(
    X_raw: np.ndarray,
    y_obs: np.ndarray,
    *,
    res_init: JijOLSResult | None = None,
    n_epochs: int = 8000,
    lr: float = 0.01,
    patience: int = 800,
    seed: int = 42,
) -> dict[str, object]:
    """
    1-layer 1-neuron network trained with backpropagation (Adam).

    Architecture
    ------------
    linear:  z = X_z @ w + b0          (36 inputs -> 1 neuron)
    output:  y_hat = a * sigmoid(b * (z - c)) + d

    Same functional form as the previous scipy sigmoid fit, but optimized
    by gradient descent instead of L-BFGS-B.
    """
    X_z, _, _ = zscore_columns(np.asarray(X_raw, dtype=float))
    y = np.asarray(y_obs, dtype=float)
    mask = np.isfinite(y) & np.all(np.isfinite(X_z), axis=1)
    X_z = X_z[mask]
    y = y[mask]

    if len(y) < 10:
        raise ValueError("Not enough valid samples")

    n_features = X_z.shape[1]
    rng = np.random.default_rng(seed)

    if res_init is not None:
        weights = np.asarray(res_init.beta_z, dtype=float)
        bias = float(res_init.intercept_z)
        linear_init = X_z @ weights + bias
    else:
        model = LinearRegression().fit(X_z, y)
        weights = model.coef_
        bias = float(model.intercept_)
        linear_init = model.predict(X_z)

    y_range = float(np.ptp(y))
    linear_range = float(np.ptp(linear_init))
    params = np.concatenate(
        [
            weights,
            np.asarray(
                [
                    bias,
                    y_range,
                    4.0 / (linear_range + 1e-8),
                    float(np.median(linear_init)),
                    float(np.min(y)),
                ],
                dtype=float,
            ),
        ]
    )

    best_params = params.copy()
    best_loss = np.inf
    stale = 0
    adam_state: dict[str, np.ndarray] = {}

    for epoch in range(1, n_epochs + 1):
        y_hat, cache = _one_neuron_forward(X_z, params)
        if not np.all(np.isfinite(y_hat)):
            raise RuntimeError("Non-finite prediction during training")

        loss = float(np.mean((y - y_hat) ** 2))
        if loss + 1e-10 < best_loss:
            best_loss = loss
            best_params = params.copy()
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

        grad = _one_neuron_backward(y, y_hat, cache)
        if not np.all(np.isfinite(grad)):
            raise RuntimeError("Non-finite gradient during training")

        params, adam_state = _adam_update(
            params,
            grad,
            adam_state,
            lr=lr,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            step=epoch,
        )

        # Keep output-sigmoid parameters in a reasonable range.
        _, _, amplitude, steepness, center, offset = _unpack_one_neuron_params(params, n_features)
        amplitude = float(np.clip(amplitude, 0.0, 3.0 * max(y_range, 1e-6)))
        steepness = float(np.clip(steepness, -50.0, 50.0))
        center = float(np.clip(center, -20.0, 20.0))
        offset = float(np.clip(offset, float(np.min(y) - y_range), float(np.max(y) + y_range)))
        params[n_features + 1 : n_features + 5] = [amplitude, steepness, center, offset]

    y_hat, _ = _one_neuron_forward(X_z, best_params)
    weights, bias, amplitude, steepness, center, offset = _unpack_one_neuron_params(best_params, n_features)

    return {
        "y_obs": y,
        "y_hat": y_hat,
        "weights": weights,
        "bias": bias,
        "a": amplitude,
        "b": steepness,
        "c": center,
        "d": offset,
        "loss": float(best_loss),
        "n_epochs": epoch,
    }


def fit_jij_linear_backprop(
    X_raw: np.ndarray,
    y_obs: np.ndarray,
    *,
    standardize: str = "none",
    res_init: JijOLSResult | None = None,
    n_epochs: int = 5000,
    lr: float = 0.05,
    patience: int = 500,
    seed: int = 42,
) -> dict[str, object]:
    """
    Simplest 1-neuron network: predict y directly from J_ij with a linear output.

    Architecture
    ------------
    y_hat = w^T x + b0

    where x is either raw J_ij or column-z-scored J_ij.

    Trained with MSE + backpropagation (Adam). At convergence this matches OLS
    with the same input preprocessing.
    """
    if standardize not in {"none", "columns"}:
        raise ValueError("standardize must be 'none' or 'columns'")

    X = np.asarray(X_raw, dtype=float)
    if standardize == "columns":
        X, _, _ = zscore_columns(X)

    y = np.asarray(y_obs, dtype=float)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]

    if len(y) < 10:
        raise ValueError("Not enough valid samples")

    n_features = X.shape[1]

    if res_init is not None:
        if standardize == "columns":
            weights = np.asarray(res_init.beta_z, dtype=float)
            bias = float(res_init.intercept_z)
        else:
            weights = np.asarray(res_init.beta_raw, dtype=float)
            bias = float(res_init.intercept_raw)
    else:
        model = LinearRegression().fit(X, y)
        weights = model.coef_
        bias = float(model.intercept_)

    params = np.concatenate([weights, np.asarray([bias], dtype=float)])
    best_params = params.copy()
    best_loss = np.inf
    stale = 0
    adam_state: dict[str, np.ndarray] = {}

    for epoch in range(1, n_epochs + 1):
        pred_weights = params[:n_features]
        pred_bias = float(params[n_features])
        y_hat = X @ pred_weights + pred_bias

        if not np.all(np.isfinite(y_hat)):
            raise RuntimeError("Non-finite prediction during training")

        loss = float(np.mean((y - y_hat) ** 2))
        if loss + 1e-10 < best_loss:
            best_loss = loss
            best_params = params.copy()
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

        n_samples = X.shape[0]
        grad_y_hat = 2.0 * (y_hat - y) / n_samples
        grad_weights = X.T @ grad_y_hat
        grad_bias = float(np.sum(grad_y_hat))
        grad = np.concatenate([grad_weights, np.asarray([grad_bias], dtype=float)])

        params, adam_state = _adam_update(
            params,
            grad,
            adam_state,
            lr=lr,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            step=epoch,
        )

    pred_weights = best_params[:n_features]
    pred_bias = float(best_params[n_features])
    y_hat = X @ pred_weights + pred_bias

    return {
        "y_obs": y,
        "y_hat": y_hat,
        "weights": pred_weights,
        "bias": pred_bias,
        "loss": float(best_loss),
        "n_epochs": epoch,
        "standardize": standardize,
    }


def _short_outcome_label(outcome: str) -> str:
    mapping = {
        "PHQ-9": "PHQ-9",
        "HQ-25 (total)": "HQ-25",
        "HQ-25 social": "HQ-25\nsocial",
        "HQ-25 emotional": "HQ-25\nemotional",
        "HQ-25 isolation": "HQ-25\nisolation",
        "IAT": "IAT",
        "IPS-22": "IPS-22",
        "TACS-22": "TACS-22",
    }
    return mapping.get(outcome, outcome)


def _scatter_jij_pred_vs_score_mean_ax(
    ax,
    res: JijOLSResult,
    score_mean: np.ndarray,
    *,
    use_z: bool,
    panel_label: str,
) -> None:
    """J_ij OLS: x = wave-mean score, y = predicted wave-mean score."""
    import seaborn as sns

    y_pred = res.y_pred_z if use_z else res.y_pred_raw
    x = np.asarray(score_mean, dtype=float)

    ax.scatter(
        x,
        y_pred,
        s=26,
        c="#4C72B0",
        alpha=0.9,
        edgecolors="k",
        linewidths=0.8,
    )

    lo = float(min(x.min(), y_pred.min()))
    hi = float(max(x.max(), y_pred.max()))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    lims = (lo - pad, hi + pad)
    x_line = np.linspace(lims[0], lims[1], 100)
    slope, intercept = np.polyfit(x, y_pred, 1)
    ax.plot(
        x_line,
        slope * x_line + intercept,
        color="red",
        linewidth=1.5,
        zorder=2,
    )
    ax.plot(lims, lims, linestyle="--", color="gray", linewidth=1.0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"{panel_label}\n{_short_outcome_label(res.outcome)}",
        fontsize=11,
    )
    ax.tick_params(labelsize=9)
    sns.despine(ax=ax)


def _scatter_obs_vs_pred_ax(ax, res: JijOLSResult, *, use_z: bool, panel_label: str) -> None:
    import seaborn as sns

    y_obs = res.y_observed
    y_pred = res.y_pred_z if use_z else res.y_pred_raw

    ax.scatter(
        y_obs,
        y_pred,
        s=26,
        c="#4C72B0",
        alpha=0.9,
        edgecolors="k",
        linewidths=0.8,
    )

    lo = float(min(y_obs.min(), y_pred.min()))
    hi = float(max(y_obs.max(), y_pred.max()))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    lims = (lo - pad, hi + pad)
    x_line = np.linspace(lims[0], lims[1], 100)
    slope, intercept = np.polyfit(y_obs, y_pred, 1)
    ax.plot(
        x_line,
        slope * x_line + intercept,
        color="red",
        linewidth=1.5,
        zorder=2,
    )
    ax.plot(lims, lims, linestyle="--", color="gray", linewidth=1.0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"{panel_label}\n{_short_outcome_label(res.outcome)}",
        fontsize=11,
    )
    ax.tick_params(labelsize=9)
    sns.despine(ax=ax)


def _sigmoid_func(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    return a / (1 + np.exp(-b * (x - c))) + d


def _fit_sigmoid_curve(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.optimize import curve_fit

    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    x_grid = np.linspace(xv.min(), xv.max(), 200)

    y_min, y_max = float(yv.min()), float(yv.max())
    y_range = y_max - y_min
    x_mid = float(np.mean(xv))
    x_range = float(xv.max() - xv.min())
    p0 = [y_range, 4.0 / (x_range + 1e-10), x_mid, y_min]

    popt, _ = curve_fit(_sigmoid_func, xv, yv, p0=p0, maxfev=5000)
    y_pred = _sigmoid_func(xv, *popt)
    y_grid = _sigmoid_func(x_grid, *popt)
    return x_grid, y_grid, y_pred


def _scatter_sa_ax(
    ax,
    df: pd.DataFrame,
    x_col: str,
    panel_label: str,
    outcome: str,
    *,
    y_col: str = "vs_group_JS",
) -> float:
    import seaborn as sns

    d = df[[x_col, y_col]].dropna()
    xv = d[x_col].to_numpy(dtype=float)
    yv = d[y_col].to_numpy(dtype=float)

    ax.scatter(
        xv,
        yv,
        s=26,
        c="#4C72B0",
        alpha=0.9,
        edgecolors="k",
        linewidths=0.8,
    )

    x_grid, y_grid, y_pred = _fit_sigmoid_curve(xv, yv)
    ax.plot(x_grid, y_grid, color="red", linewidth=1.5, zorder=2)

    ax.set_title(
        f"{panel_label}\n{_short_outcome_label(outcome)}",
        fontsize=11,
    )
    ax.tick_params(labelsize=9)
    sns.despine(ax=ax)
    # ScoreAssociation.plot_regression と同様: 観測 y とモデル予測 y の Pearson r
    return float(pearsonr(yv, y_pred)[0])


def pearson_obs_vs_pred(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Pearson r between observed and predicted (identity-line reference: y = x)."""
    return float(pearsonr(np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float))[0])


def obs_pred_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Summary metrics for observed-vs-predicted scatter (same arrays as the plot)."""
    y = np.asarray(observed, dtype=float)
    y_hat = np.asarray(predicted, dtype=float)
    if len(y) != len(y_hat):
        raise ValueError("observed and predicted must have the same length")
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r = pearson_obs_vs_pred(y, y_hat)
    return {
        "pearson_r": r,
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "rmse": float(np.sqrt(np.mean((y - y_hat) ** 2))),
    }


def scatter_obs_vs_pred_ax(
    ax,
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    panel_label: str,
    outcome: str,
    metrics: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Scatter: x = observed, y = predicted, gray dashed line = y = x.

    Returns obs_pred_metrics for the plotted points.
    """
    import seaborn as sns

    x_obs = np.asarray(observed, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    if metrics is None:
        metrics = obs_pred_metrics(x_obs, y_pred)

    ax.scatter(
        x_obs,
        y_pred,
        s=26,
        c="#4C72B0",
        alpha=0.9,
        edgecolors="k",
        linewidths=0.8,
    )

    lo = float(min(x_obs.min(), y_pred.min()))
    hi = float(max(x_obs.max(), y_pred.max()))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    lims = (lo - pad, hi + pad)
    ax.plot(lims, lims, linestyle="--", color="gray", linewidth=1.0, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")

    r_val = metrics["pearson_r"]
    r2_val = metrics["r2"]
    ax.set_title(
        f"{panel_label}: {_short_outcome_label(outcome)}\n"
        f"$r={r_val:.2f}$, $R^2={r2_val:.2f}$",
        fontsize=11,
    )
    ax.set_xlabel("Observed score mean", fontsize=10)
    ax.set_ylabel("Predicted score mean", fontsize=10)
    ax.tick_params(labelsize=9)
    sns.despine(ax=ax)
    return metrics


def _plot_r_panel(
    ax,
    r_vals: list[float],
    panel_labels: list[str],
    *,
    title: str = "F",
) -> None:
    import seaborn as sns

    x_pos = np.arange(len(panel_labels), dtype=float)
    ax.scatter(
        x_pos,
        r_vals,
        s=70,
        c="#4C72B0",
        alpha=0.9,
        edgecolors="k",
        linewidths=0.8,
        zorder=3,
    )
    for x, r_val in zip(x_pos, r_vals):
        ax.text(
            x,
            r_val + 0.03,
            f"{r_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ymax = max(max(r_vals) + 0.12, 0.2)
    ax.set_xlim(-0.5, len(panel_labels) - 0.5)
    ax.set_ylim(0.0, min(1.0, ymax))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(panel_labels)
    ax.set_title(f"{title}\n$r$ (obs vs pred)", fontsize=11)
    ax.set_ylabel("Pearson $r$", fontsize=10)
    ax.tick_params(labelsize=9)
    sns.despine(ax=ax)


def build_sa_merged_df(results: dict, score_means_df: pd.DataFrame) -> pd.DataFrame:
    """Build df_feat merged with SA score means (07_PHQ9 pattern)."""
    from ELA_analysis import resultsAnalyzer

    mu = np.asarray(results["mu_all"][:, 0, :], dtype=float)
    eta = np.asarray(results["eta"], dtype=float)
    analyzer = resultsAnalyzer(n=9)
    df_feat = analyzer.feature_distance_table(mu=mu, eta=eta)
    return df_feat.merge(score_means_df, on="subject_id", how="left")


def plot_obs_vs_pred_panels(
    results: list[JijOLSResult],
    outcome_order: list[str],
    *,
    use_z: bool = True,
    figsize: tuple[float, float] = (16.8, 3.0),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """1 x N horizontal scatter panels."""
    import seaborn as sns

    if len(outcome_order) == 0:
        raise ValueError("outcome_order must not be empty")

    res_map = {res.outcome: res for res in results}
    missing = [name for name in outcome_order if name not in res_map]
    if missing:
        raise ValueError(f"Missing OLS results for: {missing}")

    n_panels = len(outcome_order)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    for ax, outcome in zip(axes, outcome_order):
        _scatter_obs_vs_pred_ax(ax, res_map[outcome], use_z=use_z, panel_label="")

    axes[0].set_ylabel("Predicted", fontsize=10)
    for ax in axes:
        ax.set_xlabel("Observed", fontsize=10)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_fig6_obs_vs_pred_panels(
    results: list[JijOLSResult],
    outcome_order: list[str],
    *,
    use_z: bool = True,
    panel_labels: list[str] | None = None,
    figsize: tuple[float, float] = (19.2, 3.2),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Fig6-style 1 x 6 layout:
      A-E: observed vs predicted scatter
      F: dot plot of Pearson r for A-E
    """
    if len(outcome_order) != 5:
        raise ValueError("outcome_order must contain exactly 5 outcomes for panels A-E")

    if panel_labels is None:
        panel_labels = ["A", "B", "C", "D", "E"]
    if len(panel_labels) != 5:
        raise ValueError("panel_labels must contain exactly 5 labels")

    res_map = {res.outcome: res for res in results}
    missing = [name for name in outcome_order if name not in res_map]
    if missing:
        raise ValueError(f"Missing OLS results for: {missing}")

    fig, axes = plt.subplots(1, 6, figsize=figsize, constrained_layout=True)

    for ax, outcome, label in zip(axes[:5], outcome_order, panel_labels):
        _scatter_obs_vs_pred_ax(ax, res_map[outcome], use_z=use_z, panel_label=label)

    axes[0].set_ylabel("Predicted", fontsize=10)
    for ax in axes[:5]:
        ax.set_xlabel("Observed", fontsize=10)

    r_vals = [
        res_map[outcome].r_z if use_z else res_map[outcome].r_raw
        for outcome in outcome_order
    ]
    _plot_r_panel(axes[5], r_vals, panel_labels)
    axes[5].set_xlabel("Panel", fontsize=10)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_fig6_combined_2x6(
    ols_results: list[JijOLSResult],
    sa_merged_df: pd.DataFrame,
    panel_specs: list[dict[str, str]],
    *,
    use_z: bool = True,
    figsize: tuple[float, float] = (19.2, 6.4),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    2 x 6 layout:
      row 1: J_ij OLS observed vs predicted (A-E) + r dots (F)
      row 2: ScoreAssociation score_mean vs vs_group_JS (A-E) + r dots (F)
    """
    if len(panel_specs) != 5:
        raise ValueError("panel_specs must contain exactly 5 entries")

    res_map = {res.outcome: res for res in ols_results}
    outcome_order = [spec["outcome"] for spec in panel_specs]
    panel_labels = [spec["label"] for spec in panel_specs]

    missing = [name for name in outcome_order if name not in res_map]
    if missing:
        raise ValueError(f"Missing OLS results for: {missing}")

    fig, axes = plt.subplots(2, 6, figsize=figsize, constrained_layout=True)

    # row 0: J_ij OLS (x/y とも wave 平均スコア)
    jij_r: list[float] = []
    for ax, spec in zip(axes[0, :5], panel_specs):
        res = res_map[spec["outcome"]]
        x = sa_merged_df[spec["sa_col"]].to_numpy(dtype=float)
        y_pred = res.y_pred_z if use_z else res.y_pred_raw
        _scatter_jij_pred_vs_score_mean_ax(
            ax,
            res,
            x,
            use_z=use_z,
            panel_label=spec["label"],
        )
        jij_r.append(pearson_obs_vs_pred(x, y_pred))
    axes[0, 0].set_ylabel("Score mean\n(predicted)", fontsize=10)
    for ax in axes[0, :5]:
        ax.set_xlabel("Score mean (observed)", fontsize=10)

    _plot_r_panel(axes[0, 5], jij_r, panel_labels)
    axes[0, 5].set_xlabel("Panel", fontsize=10)

    # row 1: ScoreAssociation (x = wave 平均, y = D_JS)
    sa_r: list[float] = []
    for ax, spec in zip(axes[1, :5], panel_specs):
        r_val = _scatter_sa_ax(
            ax,
            sa_merged_df,
            spec["sa_col"],
            spec["label"],
            spec["outcome"],
        )
        sa_r.append(r_val)
    axes[1, 0].set_ylabel("$D_{JS}$", fontsize=10)
    for ax in axes[1, :5]:
        ax.set_xlabel("Score mean", fontsize=10)

    _plot_r_panel(axes[1, 5], sa_r, panel_labels)
    axes[1, 5].set_xlabel("Panel", fontsize=10)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def results_to_summary_df(results: list[JijOLSResult]) -> pd.DataFrame:
    rows = []
    for res in results:
        rows.append(
            {
                "outcome": res.outcome,
                "n_subjects": res.n_subjects,
                "r_raw": res.r_raw,
                "p_raw": res.p_raw,
                "r2_raw": res.r2_raw,
                "r_z": res.r_z,
                "p_z": res.p_z,
                "r2_z": res.r2_z,
                "rmse_z": res.rmse_z,
                "mae_z": res.mae_z,
            }
        )
    return pd.DataFrame(rows)
