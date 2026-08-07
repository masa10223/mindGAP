"""J_ij の2次解析をまとめたモジュール."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .constants import DEFAULT_TARGET_ITEMS_0B


STRATIFIED_GES_PAIRS: List[Tuple[str, str, str]] = [
    ("background", "background", "BG→BG"),
    ("background", "target_non_direct", "BG→ND"),
    ("background", "direct_target_triangle", "BG→D"),
    ("target_non_direct", "target_non_direct", "ND→ND"),
    ("target_non_direct", "direct_target_triangle", "ND→D"),
    ("direct_target_triangle", "direct_target_triangle", "D→D"),
]

STRATIFIED_GES_PAIR_LABELS: List[str] = [p[2] for p in STRATIFIED_GES_PAIRS]


def all_stratified_ges_pairwise_comparisons(
    pair_labels: Sequence[str] | None = None,
) -> List[Tuple[str, str]]:
    """stratified GES の全ペアワise比較 (n カテゴリ → n(n-1)/2 通り) を返す."""
    order = list(pair_labels) if pair_labels is not None else STRATIFIED_GES_PAIR_LABELS
    return [(order[i], order[j]) for i in range(len(order)) for j in range(i + 1, len(order))]


@dataclass(frozen=True)
class JijEmbeddingOutputs:
    """J_ij 埋め込み解析の主要出力."""

    edge_score_df: pd.DataFrame
    explained_df: pd.DataFrame
    subject_pca_df: pd.DataFrame
    j_corr_df_0b: pd.DataFrame
    j_corr_df_1b: pd.DataFrame


def make_edge_pairs_0b(n_items: int = 9) -> List[Tuple[int, int]]:
    """上三角順序 (0,1), (0,2), ... の edge pair を作る."""
    return [(i, j) for i in range(n_items) for j in range(i + 1, n_items)]


def build_edge_dataframe(
    n_items: int = 9,
    target_items_0b: Sequence[int] = DEFAULT_TARGET_ITEMS_0B,
) -> pd.DataFrame:
    """edge定義とカテゴリ列を持つ DataFrame を作る."""
    target_set = set(int(x) for x in target_items_0b)
    edge_pairs = make_edge_pairs_0b(n_items=n_items)

    edge_df = pd.DataFrame(
        {
            "edge_index": np.arange(len(edge_pairs)),
            "item_i_0b": [e[0] for e in edge_pairs],
            "item_j_0b": [e[1] for e in edge_pairs],
        }
    )
    edge_df["item_i_1b"] = edge_df["item_i_0b"] + 1
    edge_df["item_j_1b"] = edge_df["item_j_0b"] + 1
    edge_df["edge_label_0b"] = [
        f"J0b_{row.item_i_0b}_{row.item_j_0b}" for row in edge_df.itertuples()
    ]
    edge_df["edge_label_1b"] = [
        f"J{row.item_i_1b}{row.item_j_1b}" for row in edge_df.itertuples()
    ]

    edge_df["n_target_endpoints"] = [
        int(a in target_set) + int(b in target_set) for a, b in edge_pairs
    ]
    edge_df["is_direct_target_triangle"] = edge_df["n_target_endpoints"] == 2
    edge_df["is_target_non_direct"] = edge_df["n_target_endpoints"] == 1
    edge_df["is_background"] = edge_df["n_target_endpoints"] == 0

    for item0 in target_items_0b:
        item0 = int(item0)
        edge_df[f"is_item0b{item0}_non_direct"] = [
            ((item0 in edge) and not ((edge[0] in target_set) and (edge[1] in target_set)))
            for edge in edge_pairs
        ]

    edge_df["edge_category"] = np.select(
        [
            edge_df["is_direct_target_triangle"],
            edge_df["is_target_non_direct"],
            edge_df["is_background"],
        ],
        [
            "direct_target_triangle",
            "target_non_direct",
            "background",
        ],
        default="other",
    )
    return edge_df


def _mean_corr_to_indices(
    corr_mat: np.ndarray, row_idx: int, col_indices: Iterable[int], absolute: bool
) -> float:
    cols = np.asarray(list(col_indices), dtype=int)
    cols = cols[cols != row_idx]
    if len(cols) == 0:
        return float("nan")
    vals = corr_mat[row_idx, cols]
    if absolute:
        vals = np.abs(vals)
    return float(np.nanmean(vals))


def compute_jij_embedding_outputs(
    j_matrix: np.ndarray,
    edge_df: pd.DataFrame,
    n_pc_save: int = 5,
) -> JijEmbeddingOutputs:
    """J(被験者 x edge) から埋め込み関連の主要表を算出する."""
    n_edges = int(j_matrix.shape[1])
    if n_edges != len(edge_df):
        raise ValueError("j_matrix の edge 次元と edge_df の行数が一致しません。")

    j_corr = np.corrcoef(j_matrix, rowvar=False)
    j_corr_df_1b = pd.DataFrame(
        j_corr,
        index=edge_df["edge_label_1b"],
        columns=edge_df["edge_label_1b"],
    )
    j_corr_df_0b = pd.DataFrame(
        j_corr,
        index=edge_df["edge_label_0b"],
        columns=edge_df["edge_label_0b"],
    )

    all_idx = np.arange(n_edges)
    background_idx = edge_df.index[edge_df["is_background"]].to_numpy()
    target_non_direct_idx = edge_df.index[edge_df["is_target_non_direct"]].to_numpy()

    edge_score_df = edge_df.copy()
    edge_score_df["GES_all_abs"] = [
        _mean_corr_to_indices(j_corr, i, all_idx, absolute=True) for i in all_idx
    ]
    edge_score_df["GES_all_signed"] = [
        _mean_corr_to_indices(j_corr, i, all_idx, absolute=False) for i in all_idx
    ]
    edge_score_df["GES_background_abs"] = [
        _mean_corr_to_indices(j_corr, i, background_idx, absolute=True) for i in all_idx
    ]
    edge_score_df["GES_background_signed"] = [
        _mean_corr_to_indices(j_corr, i, background_idx, absolute=False) for i in all_idx
    ]
    edge_score_df["GES_to_target_non_direct_abs"] = [
        _mean_corr_to_indices(j_corr, i, target_non_direct_idx, absolute=True)
        for i in all_idx
    ]
    edge_score_df["GES_to_target_non_direct_signed"] = [
        _mean_corr_to_indices(j_corr, i, target_non_direct_idx, absolute=False)
        for i in all_idx
    ]

    j_z = StandardScaler().fit_transform(j_matrix)
    pca = PCA()
    j_pca_scores = pca.fit_transform(j_z)
    n_pc = min(n_pc_save, j_pca_scores.shape[1])

    m = np.vstack([j_z.T, j_pca_scores[:, :n_pc].T])
    r = np.corrcoef(m)
    pc_corr_loading = r[:n_edges, n_edges : n_edges + n_pc]

    if np.nanmean(pc_corr_loading[:, 0]) < 0:
        j_pca_scores[:, 0] *= -1
        pca.components_[0, :] *= -1
        pc_corr_loading[:, 0] *= -1

    for k in range(n_pc):
        edge_score_df[f"PC{k+1}_weight"] = pca.components_[k, :]
        edge_score_df[f"PC{k+1}_corr_loading"] = pc_corr_loading[:, k]

    explained_df = pd.DataFrame(
        {
            "PC": [f"PC{k+1}" for k in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    subject_pca_df = pd.DataFrame(
        j_pca_scores[:, :n_pc], columns=[f"PC{k+1}_score" for k in range(n_pc)]
    )

    return JijEmbeddingOutputs(
        edge_score_df=edge_score_df,
        explained_df=explained_df,
        subject_pca_df=subject_pca_df,
        j_corr_df_0b=j_corr_df_0b,
        j_corr_df_1b=j_corr_df_1b,
    )


def create_embedding_group_summary(
    edge_score_df: pd.DataFrame, target_items_0b: Sequence[int] = DEFAULT_TARGET_ITEMS_0B
) -> pd.DataFrame:
    """カテゴリ別の記述統計表を作成する."""
    metrics = [
        "GES_all_abs",
        "GES_all_signed",
        "GES_background_abs",
        "GES_background_signed",
        "GES_to_target_non_direct_abs",
        "GES_to_target_non_direct_signed",
        "PC1_corr_loading",
    ]
    group_masks = {
        "background_edges": edge_score_df["is_background"].values,
        "target_non_direct_edges": edge_score_df["is_target_non_direct"].values,
        "direct_target_triangle_edges": edge_score_df["is_direct_target_triangle"].values,
    }
    for item0 in target_items_0b:
        group_masks[f"item0b{int(item0)}_non_direct_edges"] = edge_score_df[
            f"is_item0b{int(item0)}_non_direct"
        ].values

    rows: List[Dict[str, float]] = []
    for metric in metrics:
        for group_name, mask in group_masks.items():
            vals = edge_score_df.loc[mask, metric].dropna().values
            rows.append(
                {
                    "metric": metric,
                    "group": group_name,
                    "n_edges": len(vals),
                    "mean": np.mean(vals),
                    "median": np.median(vals),
                    "sd": np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
                    "min": np.min(vals),
                    "max": np.max(vals),
                }
            )
    return pd.DataFrame(rows)


def create_direct_edge_detail(edge_score_df: pd.DataFrame) -> pd.DataFrame:
    """direct target triangle の詳細表を作る."""
    cols = [
        "edge_index",
        "edge_label_0b",
        "edge_label_1b",
        "item_i_0b",
        "item_j_0b",
        "item_i_1b",
        "item_j_1b",
        "GES_all_abs",
        "GES_all_signed",
        "GES_background_abs",
        "GES_background_signed",
        "GES_to_target_non_direct_abs",
        "GES_to_target_non_direct_signed",
        "PC1_corr_loading",
        "PC2_corr_loading",
    ]
    detail_df = edge_score_df.loc[edge_score_df["is_direct_target_triangle"], cols].copy()
    for metric in [
        "GES_all_abs",
        "GES_background_abs",
        "GES_to_target_non_direct_abs",
        "PC1_corr_loading",
    ]:
        vals = edge_score_df[metric].values
        detail_df[f"{metric}_percentile_among_all_edges"] = [
            100 * np.mean(vals <= x) for x in detail_df[metric].values
        ]
    return detail_df


def permutation_group_vs_random(
    scores: np.ndarray,
    observed_idx: np.ndarray,
    rng: np.random.Generator,
    n_perm: int = 10_000,
    alternative: str = "greater",
) -> Tuple[float, np.ndarray, float]:
    """ランダムサンプルとの平均値比較置換検定."""
    scores = np.asarray(scores, dtype=float)
    observed_idx = np.asarray(observed_idx, dtype=int)
    observed_idx = observed_idx[~np.isnan(scores[observed_idx])]
    valid_idx = np.where(~np.isnan(scores))[0]

    observed_mean = float(np.mean(scores[observed_idx]))
    null_means = np.empty(n_perm, dtype=float)
    for b in range(n_perm):
        sampled = rng.choice(valid_idx, size=len(observed_idx), replace=False)
        null_means[b] = np.mean(scores[sampled])

    if alternative == "greater":
        p = (np.sum(null_means >= observed_mean) + 1) / (n_perm + 1)
    elif alternative == "less":
        p = (np.sum(null_means <= observed_mean) + 1) / (n_perm + 1)
    elif alternative == "two-sided":
        null_center = np.mean(null_means)
        p = (
            np.sum(np.abs(null_means - null_center) >= np.abs(observed_mean - null_center))
            + 1
        ) / (n_perm + 1)
    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'.")
    return observed_mean, null_means, float(p)


def permutation_difference_within_universe(
    scores: np.ndarray,
    high_idx: np.ndarray,
    low_idx: np.ndarray,
    universe_idx: np.ndarray,
    rng: np.random.Generator,
    n_perm: int = 10_000,
    alternative: str = "greater",
) -> Tuple[float, np.ndarray, float]:
    """指定universe内で high-low 差を評価する置換検定."""
    scores = np.asarray(scores, dtype=float)
    high_idx = np.asarray(high_idx, dtype=int)
    low_idx = np.asarray(low_idx, dtype=int)
    universe_idx = np.asarray(universe_idx, dtype=int)

    valid_universe = universe_idx[~np.isnan(scores[universe_idx])]
    observed_diff = float(np.mean(scores[high_idx]) - np.mean(scores[low_idx]))

    null_diffs = np.empty(n_perm, dtype=float)
    for b in range(n_perm):
        sampled_low = rng.choice(valid_universe, size=len(low_idx), replace=False)
        sampled_high = np.setdiff1d(valid_universe, sampled_low, assume_unique=False)
        null_diffs[b] = np.mean(scores[sampled_high]) - np.mean(scores[sampled_low])

    if alternative == "greater":
        p = (np.sum(null_diffs >= observed_diff) + 1) / (n_perm + 1)
    elif alternative == "less":
        p = (np.sum(null_diffs <= observed_diff) + 1) / (n_perm + 1)
    elif alternative == "two-sided":
        null_center = np.mean(null_diffs)
        p = (
            np.sum(np.abs(null_diffs - null_center) >= np.abs(observed_diff - null_center))
            + 1
        ) / (n_perm + 1)
    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'.")
    return observed_diff, null_diffs, float(p)


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR 補正."""
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full_like(pvals, np.nan, dtype=float)

    valid = ~np.isnan(pvals)
    p = pvals[valid]
    n = len(p)
    if n == 0:
        return qvals

    order = np.argsort(p)
    ranked = p[order]

    q = np.empty(n, dtype=float)
    running_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        running_min = min(running_min, ranked[i] * n / rank)
        q[i] = running_min

    q_original_order = np.empty(n, dtype=float)
    q_original_order[order] = np.minimum(q, 1.0)
    qvals[valid] = q_original_order
    return qvals


def run_embedding_permutation_tests(
    edge_score_df: pd.DataFrame,
    target_items_0b: Sequence[int] = DEFAULT_TARGET_ITEMS_0B,
    metrics: Sequence[str] = (
        "GES_all_abs",
        "GES_background_abs",
        "GES_to_target_non_direct_abs",
        "PC1_corr_loading",
    ),
    n_perm: int = 10_000,
    random_seed: int = 12_345,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """ノートブックの J_ij 置換検定群を一括実行する."""
    rng = np.random.default_rng(random_seed)
    null_map: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, float]] = []

    target_non_direct_idx = edge_score_df.index[edge_score_df["is_target_non_direct"]].to_numpy()
    direct_target_idx = edge_score_df.index[
        edge_score_df["is_direct_target_triangle"]
    ].to_numpy()
    target_related_idx = edge_score_df.index[
        edge_score_df["is_target_non_direct"] | edge_score_df["is_direct_target_triangle"]
    ].to_numpy()

    for metric in metrics:
        scores = edge_score_df[metric].values
        test_specs = {
            "target_non_direct_greater_than_random": (target_non_direct_idx, "greater"),
            "direct_target_triangle_less_than_random": (direct_target_idx, "less"),
            "direct_target_triangle_greater_than_random": (direct_target_idx, "greater"),
        }
        for test_name, (idx, alt) in test_specs.items():
            observed, null_vals, p = permutation_group_vs_random(
                scores=scores,
                observed_idx=idx,
                rng=rng,
                n_perm=n_perm,
                alternative=alt,
            )
            rows.append(
                {
                    "metric": metric,
                    "test": test_name,
                    "n_observed_edges": len(idx),
                    "observed_value": observed,
                    "null_mean": np.mean(null_vals),
                    "null_sd": np.std(null_vals, ddof=1),
                    "p_value": p,
                    "alternative": alt,
                }
            )
            null_map[f"null_{metric}_{test_name}"] = null_vals

        observed, null_vals, p = permutation_difference_within_universe(
            scores=scores,
            high_idx=target_non_direct_idx,
            low_idx=direct_target_idx,
            universe_idx=target_related_idx,
            rng=rng,
            n_perm=n_perm,
            alternative="greater",
        )
        rows.append(
            {
                "metric": metric,
                "test": "target_non_direct_minus_direct_within_target_related",
                "n_observed_edges": len(target_related_idx),
                "observed_value": observed,
                "null_mean": np.mean(null_vals),
                "null_sd": np.std(null_vals, ddof=1),
                "p_value": p,
                "alternative": "greater",
            }
        )
        null_map[f"null_{metric}_target_non_direct_minus_direct"] = null_vals

        for item0 in target_items_0b:
            item0 = int(item0)
            item_non_direct_idx = edge_score_df.index[
                edge_score_df[f"is_item0b{item0}_non_direct"]
            ].to_numpy()
            item_direct_idx = edge_score_df.index[
                edge_score_df["is_direct_target_triangle"]
                & (
                    (edge_score_df["item_i_0b"] == item0)
                    | (edge_score_df["item_j_0b"] == item0)
                )
            ].to_numpy()
            item_universe_idx = np.concatenate([item_non_direct_idx, item_direct_idx])

            observed, null_vals, p = permutation_difference_within_universe(
                scores=scores,
                high_idx=item_non_direct_idx,
                low_idx=item_direct_idx,
                universe_idx=item_universe_idx,
                rng=rng,
                n_perm=n_perm,
                alternative="greater",
            )
            rows.append(
                {
                    "metric": metric,
                    "test": f"item0b{item0}_non_direct_minus_item0b{item0}_direct",
                    "n_observed_edges": len(item_universe_idx),
                    "observed_value": observed,
                    "null_mean": np.mean(null_vals),
                    "null_sd": np.std(null_vals, ddof=1),
                    "p_value": p,
                    "alternative": "greater",
                }
            )
            null_map[f"null_{metric}_item0b{item0}_non_direct_minus_direct"] = null_vals

    perm_df = pd.DataFrame(rows)
    perm_df["q_value_BH_FDR"] = fdr_bh(perm_df["p_value"].values)
    return perm_df, null_map


def _category_index_map(edge_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """J edge カテゴリごとの index 配列を返す."""
    return {
        "background": edge_df.index[edge_df["is_background"]].to_numpy(),
        "target_non_direct": edge_df.index[edge_df["is_target_non_direct"]].to_numpy(),
        "direct_target_triangle": edge_df.index[
            edge_df["is_direct_target_triangle"]
        ].to_numpy(),
    }


def create_stratified_ges_long_df(
    j_corr: np.ndarray,
    edge_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    ソースカテゴリ × ターゲットカテゴリの 6 種類 stratified GES を long 形式で返す。

    各 J_{i,j}（ソースカテゴリ src）について、ターゲットカテゴリ tgt 内の
    他 edge との |corr| 平均を GES(src→tgt) として定義する。
    """
    cat_idx = _category_index_map(edge_df)
    rows: List[Dict[str, object]] = []

    for src_cat, tgt_cat, pair_label in STRATIFIED_GES_PAIRS:
        src_indices = edge_df.index[edge_df["edge_category"] == src_cat].to_numpy()
        tgt_indices = cat_idx[tgt_cat]
        for i in src_indices:
            rows.append(
                {
                    "edge_index": int(i),
                    "edge_label_0b": edge_df.loc[i, "edge_label_0b"],
                    "edge_label_1b": edge_df.loc[i, "edge_label_1b"],
                    "source_category": src_cat,
                    "target_category": tgt_cat,
                    "ges_pair": pair_label,
                    "GES_stratified_abs": _mean_corr_to_indices(
                        j_corr, int(i), tgt_indices, absolute=True
                    ),
                }
            )

    return pd.DataFrame(rows)


def create_stratified_ges_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """stratified GES long 表から ges_pair 別の記述統計を作成する."""
    rows: List[Dict[str, float]] = []
    for pair_label in long_df["ges_pair"].unique():
        vals = long_df.loc[long_df["ges_pair"] == pair_label, "GES_stratified_abs"].dropna()
        rows.append(
            {
                "ges_pair": pair_label,
                "n_edges": len(vals),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        )
    order = [p[2] for p in STRATIFIED_GES_PAIRS]
    summary = pd.DataFrame(rows)
    summary["ges_pair"] = pd.Categorical(summary["ges_pair"], categories=order, ordered=True)
    return summary.sort_values("ges_pair").reset_index(drop=True)


def _wilcoxon_safe(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import wilcoxon

    diff = x - y
    if len(diff) < 2 or np.allclose(diff, 0):
        return np.nan
    try:
        return float(wilcoxon(diff, alternative="two-sided").pvalue)
    except ValueError:
        return np.nan


def _mannwhitney_safe(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import mannwhitneyu

    if len(x) < 1 or len(y) < 1:
        return np.nan
    return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)


def run_stratified_ges_pairwise_tests(
    long_df: pd.DataFrame,
    comparisons: Sequence[Tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """
    stratified GES のペア比較検定。

    同一 source カテゴリ内（例: BG→BG vs BG→ND）は同じ edge 上の値なので Wilcoxon。
    異なる source カテゴリ間は Mann-Whitney U。
    """
    if comparisons is None:
        comparisons = all_stratified_ges_pairwise_comparisons()

    pair_to_src = {p[2]: p[0] for p in STRATIFIED_GES_PAIRS}
    rows: List[Dict[str, object]] = []

    for pair_a, pair_b in comparisons:
        a_df = long_df.loc[long_df["ges_pair"] == pair_a].set_index("edge_index")
        b_df = long_df.loc[long_df["ges_pair"] == pair_b].set_index("edge_index")
        src_a = pair_to_src[pair_a]
        src_b = pair_to_src[pair_b]

        if src_a == src_b:
            shared = a_df.index.intersection(b_df.index)
            x = a_df.loc[shared, "GES_stratified_abs"].values
            y = b_df.loc[shared, "GES_stratified_abs"].values
            test_name = "wilcoxon_signed_rank"
            p = _wilcoxon_safe(x, y)
            observed = float(np.mean(x - y))
        else:
            x = a_df["GES_stratified_abs"].dropna().values
            y = b_df["GES_stratified_abs"].dropna().values
            test_name = "mannwhitney_u"
            p = _mannwhitney_safe(x, y)
            observed = float(np.mean(x) - np.mean(y))

        rows.append(
            {
                "pair_a": pair_a,
                "pair_b": pair_b,
                "test": test_name,
                "n_a": len(x),
                "n_b": len(y),
                "observed_diff": observed,
                "p_value": p,
            }
        )

    out = pd.DataFrame(rows)
    out["q_value_BH_FDR"] = fdr_bh(out["p_value"].values)
    return out
