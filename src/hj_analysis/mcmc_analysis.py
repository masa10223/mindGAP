"""h/J 解析の MCMC 部分を切り出したモジュール."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp

from .constants import DEFAULT_TARGET_ITEMS_0B


@dataclass(frozen=True)
class MCMCAnalysisResult:
    """MCMC解析の主要出力."""

    all_samples: np.ndarray
    mcmc_metric_df: pd.DataFrame
    exact_metric_df: pd.DataFrame
    subject_group_summary_df: pd.DataFrame
    subject_group_wide_df: pd.DataFrame
    paired_test_df: pd.DataFrame
    item_summary_df: pd.DataFrame
    null_distributions: Dict[str, np.ndarray]


def make_edge_pairs_0b(n_items: int = 9) -> list[Tuple[int, int]]:
    return [(i, j) for i in range(n_items) for j in range(i + 1, n_items)]


def jvec_to_jmat(j_vec: np.ndarray, n_items: int = 9) -> np.ndarray:
    """上三角 vector を対称行列へ戻す."""
    j_mat = np.zeros((n_items, n_items), dtype=float)
    idx = 0
    for i in range(n_items):
        for j in range(i + 1, n_items):
            j_mat[i, j] = j_vec[idx]
            j_mat[j, i] = j_vec[idx]
            idx += 1
    return j_mat


def gibbs_sample_binary_pmem(
    h_vec: np.ndarray,
    j_vec: np.ndarray,
    n_keep: int = 10_000,
    burn_in: int = 5_000,
    thin: int = 1,
    rng: np.random.Generator | None = None,
    init_state: np.ndarray | None = None,
    logp_sign: int = +1,
) -> np.ndarray:
    """0/1 二値 pairwise maximum entropy model の Gibbs sampler."""
    if rng is None:
        rng = np.random.default_rng()

    h_vec = np.asarray(h_vec, dtype=float)
    j_mat = jvec_to_jmat(j_vec, n_items=len(h_vec))
    n_items = len(h_vec)

    if init_state is None:
        state = rng.integers(0, 2, size=n_items).astype(np.int8)
    else:
        state = np.asarray(init_state, dtype=np.int8).copy()

    samples = np.zeros((n_keep, n_items), dtype=np.int8)
    total_sweeps = burn_in + n_keep * thin
    keep_idx = 0

    for sweep in range(total_sweeps):
        for i in rng.permutation(n_items):
            field_i = h_vec[i] + np.dot(j_mat[i, :], state)
            p_on = expit(logp_sign * field_i)
            state[i] = rng.random() < p_on

        if sweep >= burn_in and ((sweep - burn_in) % thin == 0):
            samples[keep_idx] = state
            keep_idx += 1

    return samples


def exact_binary_pmem_marginals(
    h_vec: np.ndarray, j_vec: np.ndarray, logp_sign: int = +1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """n=9 前提で全512状態を列挙した厳密周辺分布."""
    h_vec = np.asarray(h_vec, dtype=float)
    n_items = len(h_vec)
    edge_pairs = make_edge_pairs_0b(n_items=n_items)

    states = np.array(
        [[(s >> i) & 1 for i in range(n_items)] for s in range(2**n_items)],
        dtype=np.int8,
    )
    linear = states @ h_vec

    pair_term = np.zeros(states.shape[0], dtype=float)
    for k, (i, j) in enumerate(edge_pairs):
        pair_term += j_vec[k] * states[:, i] * states[:, j]

    logw = logp_sign * (linear + pair_term)
    logz = logsumexp(logw)
    prob = np.exp(logw - logz)

    p_on = prob @ states
    fixation = np.maximum(p_on, 1 - p_on)
    entropy = binary_entropy_from_p(p_on)
    return p_on, fixation, entropy


def binary_entropy_from_p(p: np.ndarray) -> np.ndarray:
    """二値エントロピー."""
    p = np.asarray(p, dtype=float)
    eps = 1e-12
    return -(p * np.log2(p + eps) + (1 - p) * np.log2(1 - p + eps))


def mean_dwell_length(binary_series: np.ndarray) -> float:
    """同一状態連続長の平均."""
    x = np.asarray(binary_series, dtype=np.int8)
    if len(x) == 0:
        return float("nan")
    change_points = np.where(np.diff(x) != 0)[0] + 1
    segments = np.split(x, change_points)
    lengths = np.array([len(seg) for seg in segments], dtype=float)
    return float(lengths.mean())


def compute_mcmc_metrics(samples: np.ndarray) -> Dict[str, np.ndarray]:
    """MCMCサンプルから項目別指標を算出."""
    p_on = samples.mean(axis=0)
    fixation = np.maximum(p_on, 1 - p_on)
    entropy = binary_entropy_from_p(p_on)
    flips = samples[1:, :] != samples[:-1, :]
    flip_rate = flips.mean(axis=0)
    dwell = np.array([mean_dwell_length(samples[:, i]) for i in range(samples.shape[1])])
    return {
        "p_on": p_on,
        "fixation": fixation,
        "entropy": entropy,
        "flip_rate": flip_rate,
        "mean_dwell": dwell,
    }


def paired_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str = "two-sided",
    n_perm: int = 10_000,
    random_seed: int = 12_345,
) -> Tuple[float, float, np.ndarray]:
    """対応あり置換検定 (y - x)."""
    rng = np.random.default_rng(random_seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    diff = y[mask] - x[mask]
    observed = float(diff.mean())

    null = np.empty(n_perm, dtype=float)
    for b in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diff))
        null[b] = (diff * signs).mean()

    if alternative == "greater":
        p = (np.sum(null >= observed) + 1) / (n_perm + 1)
    elif alternative == "less":
        p = (np.sum(null <= observed) + 1) / (n_perm + 1)
    elif alternative == "two-sided":
        p = (np.sum(np.abs(null) >= np.abs(observed)) + 1) / (n_perm + 1)
    else:
        raise ValueError("alternative must be greater, less, or two-sided.")
    return observed, float(p), null


def run_subject_mcmc_analysis(
    h: np.ndarray,
    j: np.ndarray,
    target_items_0b: Sequence[int] = DEFAULT_TARGET_ITEMS_0B,
    logp_sign: int = +1,
    random_seed: int = 12_345,
    burn_in_sweeps: int = 5_000,
    n_keep_sweeps: int = 10_000,
    thin: int = 1,
    n_perm: int = 10_000,
) -> MCMCAnalysisResult:
    """ノートブックの MCMC 集計処理を一括実行する."""
    n_subj, n_items = h.shape
    all_samples = np.zeros((n_subj, n_keep_sweeps, n_items), dtype=np.int8)
    target_set = set(int(x) for x in target_items_0b)

    mcmc_rows = []
    exact_rows = []
    for s in range(n_subj):
        rng = np.random.default_rng(random_seed + s)
        samples_s = gibbs_sample_binary_pmem(
            h_vec=h[s],
            j_vec=j[s],
            n_keep=n_keep_sweeps,
            burn_in=burn_in_sweeps,
            thin=thin,
            rng=rng,
            init_state=None,
            logp_sign=logp_sign,
        )
        all_samples[s] = samples_s
        metrics_s = compute_mcmc_metrics(samples_s)
        exact_p_on, exact_fixation, exact_entropy = exact_binary_pmem_marginals(
            h_vec=h[s], j_vec=j[s], logp_sign=logp_sign
        )
        for item in range(n_items):
            group = "target_6_8" if item in target_set else "background_0_5"
            mcmc_rows.append(
                {
                    "subject": s,
                    "item_0b": item,
                    "item_1b": item + 1,
                    "item_group": group,
                    "p_on_mcmc": metrics_s["p_on"][item],
                    "fixation_mcmc": metrics_s["fixation"][item],
                    "entropy_mcmc": metrics_s["entropy"][item],
                    "flip_rate_mcmc": metrics_s["flip_rate"][item],
                    "mean_dwell_mcmc": metrics_s["mean_dwell"][item],
                }
            )
            exact_rows.append(
                {
                    "subject": s,
                    "item_0b": item,
                    "item_1b": item + 1,
                    "item_group": group,
                    "p_on_exact": exact_p_on[item],
                    "fixation_exact": exact_fixation[item],
                    "entropy_exact": exact_entropy[item],
                }
            )

    mcmc_metric_df = pd.DataFrame(mcmc_rows)
    exact_metric_df = pd.DataFrame(exact_rows)

    subject_group_summary_df = (
        mcmc_metric_df.groupby(["subject", "item_group"])
        .agg(
            p_on_mcmc_mean=("p_on_mcmc", "mean"),
            fixation_mcmc_mean=("fixation_mcmc", "mean"),
            entropy_mcmc_mean=("entropy_mcmc", "mean"),
            flip_rate_mcmc_mean=("flip_rate_mcmc", "mean"),
            mean_dwell_mcmc_mean=("mean_dwell_mcmc", "mean"),
        )
        .reset_index()
    )

    wide = subject_group_summary_df.pivot(
        index="subject",
        columns="item_group",
        values=[
            "p_on_mcmc_mean",
            "fixation_mcmc_mean",
            "entropy_mcmc_mean",
            "flip_rate_mcmc_mean",
            "mean_dwell_mcmc_mean",
        ],
    )
    wide.columns = [f"{metric}_{group}" for metric, group in wide.columns]
    subject_group_wide_df = wide.reset_index()

    test_specs = [
        ("fixation_mcmc_mean", "greater"),
        ("mean_dwell_mcmc_mean", "greater"),
        ("flip_rate_mcmc_mean", "less"),
        ("entropy_mcmc_mean", "less"),
        ("p_on_mcmc_mean", "two-sided"),
    ]
    null_distributions: Dict[str, np.ndarray] = {}
    test_rows = []
    for metric, alternative in test_specs:
        x_bg = subject_group_wide_df[f"{metric}_background_0_5"].values
        y_tg = subject_group_wide_df[f"{metric}_target_6_8"].values
        observed, p, null = paired_permutation_test(
            x=x_bg,
            y=y_tg,
            alternative=alternative,
            n_perm=n_perm,
            random_seed=random_seed,
        )
        test_rows.append(
            {
                "metric": metric,
                "background_mean": np.nanmean(x_bg),
                "target_mean": np.nanmean(y_tg),
                "observed_target_minus_background": observed,
                "alternative": alternative,
                "p_value": p,
            }
        )
        null_distributions[f"MCMC_null_{metric}_target_minus_background"] = null

    paired_test_df = pd.DataFrame(test_rows)
    item_summary_df = (
        mcmc_metric_df.groupby(["item_0b", "item_1b", "item_group"])
        .agg(
            p_on_mean=("p_on_mcmc", "mean"),
            p_on_sd=("p_on_mcmc", "std"),
            fixation_mean=("fixation_mcmc", "mean"),
            fixation_sd=("fixation_mcmc", "std"),
            flip_rate_mean=("flip_rate_mcmc", "mean"),
            flip_rate_sd=("flip_rate_mcmc", "std"),
            entropy_mean=("entropy_mcmc", "mean"),
            entropy_sd=("entropy_mcmc", "std"),
            mean_dwell_mean=("mean_dwell_mcmc", "mean"),
            mean_dwell_sd=("mean_dwell_mcmc", "std"),
        )
        .reset_index()
    )

    return MCMCAnalysisResult(
        all_samples=all_samples,
        mcmc_metric_df=mcmc_metric_df,
        exact_metric_df=exact_metric_df,
        subject_group_summary_df=subject_group_summary_df,
        subject_group_wide_df=subject_group_wide_df,
        paired_test_df=paired_test_df,
        item_summary_df=item_summary_df,
        null_distributions=null_distributions,
    )
