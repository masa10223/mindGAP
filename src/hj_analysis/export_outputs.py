"""h/J 解析結果の CSV / figure 出力."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from .constants import DEFAULT_TARGET_ITEMS_0B
from .jij_secondary import (
    JijEmbeddingOutputs,
    STRATIFIED_GES_PAIRS,
    all_stratified_ges_pairwise_comparisons,
)
from .mcmc_analysis import MCMCAnalysisResult


@dataclass(frozen=True)
class HjPaperOutputDirs:
    """論文用 CSV / figure の出力先."""

    csv_dir: Path
    fig_dir: Path


def default_paper_output_dirs(base_dir: Path | str = ".") -> HjPaperOutputDirs:
    """ノートブック cwd からのデフォルト出力パスを返す."""
    base = Path(base_dir)
    return HjPaperOutputDirs(
        csv_dir=base / "csvs_for_paper" / "h_J_analysis",
        fig_dir=base / "figs_for_paper" / "h_J_analysis",
    )


def _with_suffix(name: str, suffix: str) -> str:
    if suffix and not name.endswith(f"_{suffix}"):
        return f"{name}_{suffix}"
    return name


def save_hj_analysis_csvs(
    *,
    embedding: JijEmbeddingOutputs,
    summary_df: pd.DataFrame,
    direct_detail_df: pd.DataFrame,
    perm_df: pd.DataFrame,
    mcmc_result: MCMCAnalysisResult,
    output_dir: Path | str,
    suffix: str = "REFACTORED",
) -> List[Path]:
    """h/J 解析の CSV を一括保存する."""
    csv_dir = Path(output_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    saved.append(_save_csv(embedding.edge_score_df, csv_dir, "J_edge_embedding_scores_with_PCA", suffix))
    saved.append(_save_csv(summary_df, csv_dir, "J_embedding_group_summary", suffix))
    saved.append(_save_csv(direct_detail_df, csv_dir, "J_direct_target_triangle_edge_detail", suffix))
    saved.append(_save_csv(perm_df, csv_dir, "J_embedding_permutation_tests", suffix))

    saved.append(_save_csv(mcmc_result.mcmc_metric_df, csv_dir, "MCMC_item_metrics_long", suffix))
    saved.append(_save_csv(mcmc_result.exact_metric_df, csv_dir, "exact_item_metrics_long", suffix))
    saved.append(
        _save_csv(
            mcmc_result.subject_group_summary_df,
            csv_dir,
            "MCMC_subject_target_vs_background_summary",
            suffix,
        )
    )
    saved.append(
        _save_csv(
            mcmc_result.subject_group_wide_df,
            csv_dir,
            "MCMC_subject_target_vs_background_wide",
            suffix,
        )
    )
    saved.append(
        _save_csv(
            mcmc_result.paired_test_df,
            csv_dir,
            "MCMC_target_vs_background_paired_permutation_tests",
            suffix,
        )
    )
    saved.append(_save_csv(mcmc_result.item_summary_df, csv_dir, "MCMC_item_summary", suffix))

    for key, values in mcmc_result.null_distributions.items():
        saved.append(
            _save_csv(
                pd.DataFrame({"null_target_minus_background": values}),
                csv_dir,
                key,
                suffix,
            )
        )

    return saved


def configure_paper_plot_style(font_family: str = "Arial") -> None:
    """論文用 figure の seaborn / matplotlib スタイルを設定する."""
    sns.set_theme(style="white", font=font_family)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font_family, "DejaVu Sans", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_hj_analysis_figures(
    *,
    embedding: JijEmbeddingOutputs,
    mcmc_result: MCMCAnalysisResult,
    output_dir: Path | str,
    target_items_0b: Sequence[int] = DEFAULT_TARGET_ITEMS_0B,
    suffix: str = "REFACTORED",
    n_items: int = 9,
    random_seed: int = 12_345,
    network_abs_r_threshold: float = 0.50,
    n_traj_plot: int = 300,
    n_representative_subjects: int = 5,
    font_family: str = "Arial",
) -> List[Path]:
    """h/J 解析の PDF figure を一括保存する."""
    configure_paper_plot_style(font_family=font_family)
    fig_dir = Path(output_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    target_items = tuple(int(x) for x in target_items_0b)
    edge_score_df = embedding.edge_score_df.copy()
    explained_df = embedding.explained_df
    j_corr = embedding.j_corr_df_1b.values
    labels_1b = edge_score_df["edge_label_1b"].tolist()
    rng_plot = np.random.default_rng(random_seed)

    saved: List[Path] = []
    saved.extend(
        _save_embedding_figures(
            edge_score_df=edge_score_df,
            explained_df=explained_df,
            j_corr=j_corr,
            labels_1b=labels_1b,
            target_items_0b=target_items,
            fig_dir=fig_dir,
            suffix=suffix,
            random_seed=random_seed,
            network_abs_r_threshold=network_abs_r_threshold,
        )
    )
    saved.extend(
        _save_mcmc_figures(
            mcmc_result=mcmc_result,
            target_items_0b=target_items,
            fig_dir=fig_dir,
            suffix=suffix,
            n_items=n_items,
            rng_plot=rng_plot,
            n_traj_plot=n_traj_plot,
            n_representative_subjects=n_representative_subjects,
        )
    )
    return saved


def _save_csv(df: pd.DataFrame, output_dir: Path, stem: str, suffix: str) -> Path:
    path = output_dir / f"{_with_suffix(stem, suffix)}.csv"
    df.to_csv(path, index=False)
    return path


def _save_fig(fig: plt.Figure, output_dir: Path, stem: str, suffix: str, **savefig_kwargs) -> Path:
    path = output_dir / f"{_with_suffix(stem, suffix)}.pdf"
    fig.savefig(path, **savefig_kwargs)
    plt.close(fig)
    return path


def _clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _p_to_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _lookup_pairwise_p_value(
    test_df: pd.DataFrame, pair_a: str, pair_b: str
) -> float | None:
    """test_df から pair_a vs pair_b の p 値を取得する（順序は不問）。"""
    row = test_df.loc[
        (test_df["pair_a"] == pair_a) & (test_df["pair_b"] == pair_b)
    ]
    if row.empty:
        row = test_df.loc[
            (test_df["pair_a"] == pair_b) & (test_df["pair_b"] == pair_a)
        ]
    if row.empty:
        return None
    return float(row.iloc[0]["p_value"])


def _assign_sig_bracket_levels(
    x_pairs: Sequence[tuple[float, float]],
) -> list[int]:
    """
    bracket の x 区間が重ならないよう、低いレベルから順に割り当てる。

    狭い span を先に置くことで、同レベル内の重なりを減らす。
    """
    indexed = sorted(
        [(min(x1, x2), max(x1, x2), i) for i, (x1, x2) in enumerate(x_pairs)],
        key=lambda t: (t[1] - t[0], t[0]),
    )
    levels = [-1] * len(x_pairs)
    level_intervals: list[list[tuple[float, float]]] = []

    for x1, x2, orig_i in indexed:
        for lvl, intervals in enumerate(level_intervals):
            if all(x2 < a or x1 > b for a, b in intervals):
                intervals.append((x1, x2))
                levels[orig_i] = lvl
                break
        else:
            level_intervals.append([(x1, x2)])
            levels[orig_i] = len(level_intervals) - 1

    return levels


def _add_sig_text(ax: plt.Axes, x1: float, x2: float, y: float, h: float, text: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], linewidth=1, color="black")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=8)


def _annotate_pairwise_significance_brackets(
    ax: plt.Axes,
    *,
    order: Sequence[str],
    test_df: pd.DataFrame,
    comparisons: Sequence[tuple[str, str]],
    y_data_max: float,
    y_step: float = 0.034,
    bracket_h: float = 0.011,
    base_offset: float = 0.018,
) -> None:
    """全ペア比較の bracket と有意性記号を、重ならない高さで描画する。"""
    entries: list[tuple[float, float, str, float]] = []
    for pair_a, pair_b in comparisons:
        if pair_a not in order or pair_b not in order:
            continue
        p = _lookup_pairwise_p_value(test_df, pair_a, pair_b)
        if p is None:
            continue
        x1 = float(order.index(pair_a) + 1)
        x2 = float(order.index(pair_b) + 1)
        entries.append((x1, x2, _p_to_stars(p), p))

    if not entries:
        return

    x_pairs = [(x1, x2) for x1, x2, _, _ in entries]
    levels = _assign_sig_bracket_levels(x_pairs)
    bracket_y = y_data_max + base_offset

    for (x1, x2, stars, _), level in zip(entries, levels):
        y = bracket_y + level * y_step
        _add_sig_text(ax, x1, x2, y, bracket_h, stars)

    top = bracket_y + max(levels) * y_step + bracket_h + 0.012
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, max(ymax, top))


def _subject_color_map(subject_ids: Sequence[int]) -> dict[int, tuple[float, float, float, float]]:
    """被験者ごとの色を返す."""
    unique_ids = sorted(set(int(x) for x in subject_ids))
    cmap = plt.get_cmap("tab20")
    return {sid: cmap(i % 20) for i, sid in enumerate(unique_ids)}


def _save_embedding_figures(
    *,
    edge_score_df: pd.DataFrame,
    explained_df: pd.DataFrame,
    j_corr: np.ndarray,
    labels_1b: List[str],
    target_items_0b: Sequence[int],
    fig_dir: Path,
    suffix: str,
    random_seed: int,
    network_abs_r_threshold: float,
) -> List[Path]:
    saved: List[Path] = []
    category_order = ["background", "target_non_direct", "direct_target_triangle"]

    for metric in [
        "GES_all_abs",
        "GES_background_abs",
        "GES_to_target_non_direct_abs",
        "PC1_corr_loading",
    ]:
        data = [
            edge_score_df.loc[edge_score_df["edge_category"] == cat, metric].dropna().values
            for cat in category_order
        ]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot(data, labels=category_order, showmeans=True)

        direct_df = edge_score_df.loc[edge_score_df["is_direct_target_triangle"]].copy()
        x_pos = category_order.index("direct_target_triangle") + 1
        ax.scatter(np.repeat(x_pos, len(direct_df)), direct_df[metric], zorder=3)
        for _, row in direct_df.iterrows():
            ax.text(
                x_pos + 0.05,
                row[metric],
                f"{row['edge_label_0b']} / {row['edge_label_1b']}",
                fontsize=8,
                va="center",
            )

        ax.set_ylabel(metric)
        ax.set_title(f"Target-edge embedding comparison: {metric}")
        _clean_axes(ax)
        fig.tight_layout()
        saved.append(
            _save_fig(fig, fig_dir, f"J_embedding_category_comparison_{metric}", suffix)
        )

    for metric in ["GES_all_abs", "GES_background_abs", "PC1_corr_loading"]:
        plot_groups = []
        plot_labels = []
        for item0 in target_items_0b:
            mask = edge_score_df[f"is_item0b{item0}_non_direct"]
            plot_groups.append(edge_score_df.loc[mask, metric].dropna().values)
            plot_labels.append(f"item {item0} non-direct\n0b={item0}, 1b={item0 + 1}")

        plot_groups.append(
            edge_score_df.loc[edge_score_df["is_direct_target_triangle"], metric].dropna().values
        )
        plot_labels.append("direct target\ntriangle")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.boxplot(plot_groups, labels=plot_labels, showmeans=True)
        direct_df = edge_score_df.loc[edge_score_df["is_direct_target_triangle"]]
        x_pos = len(plot_groups)
        ax.scatter(np.repeat(x_pos, len(direct_df)), direct_df[metric], zorder=3)
        for _, row in direct_df.iterrows():
            ax.text(
                x_pos + 0.05,
                row[metric],
                f"{row['edge_label_0b']} / {row['edge_label_1b']}",
                fontsize=8,
                va="center",
            )

        ax.set_ylabel(metric)
        ax.set_title(f"Item-specific non-direct vs direct target triangle: {metric}")
        _clean_axes(ax)
        fig.tight_layout()
        saved.append(_save_fig(fig, fig_dir, f"J_item_specific_vs_direct_{metric}", suffix))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(
        np.arange(1, min(11, len(explained_df) + 1)),
        explained_df["explained_variance_ratio"].values[:10],
        marker="o",
    )
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("J-edge PCA explained variance")
    _clean_axes(ax)
    fig.tight_layout()
    saved.append(_save_fig(fig, fig_dir, "J_PCA_explained_variance", suffix))

    plot_load = edge_score_df.sort_values("PC1_corr_loading").copy()
    fig, ax = plt.subplots(figsize=(8, 7))
    y = np.arange(len(plot_load))
    ax.barh(y, plot_load["PC1_corr_loading"])
    ax.set_yticks(y)
    ax.set_yticklabels(plot_load["edge_label_1b"], fontsize=8)
    for yi, (_, row) in zip(y, plot_load.iterrows()):
        if row["is_direct_target_triangle"]:
            ax.text(
                row["PC1_corr_loading"],
                yi,
                f"  {row['edge_label_0b']} / {row['edge_label_1b']}",
                va="center",
                fontsize=8,
            )
    pc1_var = explained_df.loc[0, "explained_variance_ratio"] * 100
    ax.set_xlabel("PC1 correlation loading")
    ax.set_title(f"J-edge PCA: PC1 loadings\nExplained variance = {pc1_var:.1f}%")
    _clean_axes(ax)
    fig.tight_layout()
    saved.append(_save_fig(fig, fig_dir, "J_PCA_PC1_corr_loadings", suffix))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(edge_score_df["PC1_corr_loading"], edge_score_df["PC2_corr_loading"], alpha=0.7)
    for _, row in edge_score_df.iterrows():
        if row["is_direct_target_triangle"]:
            ax.text(
                row["PC1_corr_loading"],
                row["PC2_corr_loading"],
                f"{row['edge_label_0b']} / {row['edge_label_1b']}",
                fontsize=8,
            )
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("PC1 correlation loading")
    ax.set_ylabel("PC2 correlation loading")
    ax.set_title("PCA loading map of J edges")
    _clean_axes(ax)
    fig.tight_layout()
    saved.append(_save_fig(fig, fig_dir, "J_PCA_PC1_PC2_loading_map", suffix))

    graph = nx.Graph()
    for _, row in edge_score_df.iterrows():
        graph.add_node(
            row["edge_label_1b"],
            edge_label_0b=row["edge_label_0b"],
            edge_category=row["edge_category"],
        )
    for a in range(len(labels_1b)):
        for b in range(a + 1, len(labels_1b)):
            r = j_corr[a, b]
            if np.abs(r) >= network_abs_r_threshold:
                graph.add_edge(
                    labels_1b[a],
                    labels_1b[b],
                    weight=float(np.abs(r)),
                    signed_r=float(r),
                )

    if graph.number_of_edges() > 0:
        fig, ax = plt.subplots(figsize=(7, 6))
        pos = nx.spring_layout(graph, seed=random_seed, weight="weight")
        nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.3, width=0.8)
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=80)
        direct_nodes = edge_score_df.loc[
            edge_score_df["is_direct_target_triangle"], "edge_label_1b"
        ].tolist()
        nx.draw_networkx_labels(
            graph, pos, labels={node: node for node in direct_nodes}, font_size=8, ax=ax
        )
        ax.set_title(f"Edge-edge network, |r| >= {network_abs_r_threshold}")
        ax.axis("off")
        fig.tight_layout()
        saved.append(_save_fig(fig, fig_dir, "J_edge_edge_network_community", suffix))

    return saved


def _save_mcmc_figures(
    *,
    mcmc_result: MCMCAnalysisResult,
    target_items_0b: Sequence[int],
    fig_dir: Path,
    suffix: str,
    n_items: int,
    rng_plot: np.random.Generator,
    n_traj_plot: int,
    n_representative_subjects: int,
) -> List[Path]:
    saved: List[Path] = []
    mcmc_metric_df = mcmc_result.mcmc_metric_df
    exact_metric_df = mcmc_result.exact_metric_df
    wide = mcmc_result.subject_group_wide_df
    test_df = mcmc_result.paired_test_df
    all_samples = mcmc_result.all_samples
    target_set = set(int(x) for x in target_items_0b)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    metrics_to_plot = [
        ("fixation_mcmc", "Fixation\nmax[P(0), P(1)]"),
        ("flip_rate_mcmc", "Flip rate\nper saved sweep"),
        ("entropy_mcmc", "Binary entropy"),
        ("mean_dwell_mcmc", "Mean dwell length"),
    ]
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics_to_plot):
        for item in range(n_items):
            vals = mcmc_metric_df.loc[mcmc_metric_df["item_0b"] == item, metric].values
            x = np.full(len(vals), item) + rng_plot.uniform(-0.15, 0.15, size=len(vals))
            if item in target_set:
                marker, size, alpha = "*", 55, 0.70
            else:
                marker, size, alpha = "^", 28, 0.45
            ax.scatter(
                x, vals, s=size, marker=marker, alpha=alpha, edgecolors="black", linewidths=0.25
            )
        med = mcmc_metric_df.groupby("item_0b")[metric].median()
        ax.plot(np.arange(n_items), med.values, marker="o", linewidth=1.2)
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(n_items))
        ax.set_xticklabels([str(i) for i in range(n_items)])
        _clean_axes(ax)
    axes[-1, 0].set_xlabel("Item index, 0-based")
    axes[-1, 1].set_xlabel("Item index, 0-based")
    fig.suptitle("MCMC-derived item stability metrics\nTarget items = 0-based 6, 7, 8", y=1.02)
    fig.tight_layout()
    saved.append(
        _save_fig(fig, fig_dir, "MCMC_item_stability_metrics", suffix, bbox_inches="tight")
    )

    metrics_group_plot = [
        ("fixation_mcmc_mean", "Fixation"),
        ("flip_rate_mcmc_mean", "Flip rate"),
        ("entropy_mcmc_mean", "Entropy"),
        ("mean_dwell_mcmc_mean", "Mean dwell length"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics_group_plot):
        valid = wide[
            ["subject", f"{metric}_background_0_5", f"{metric}_target_6_8"]
        ].dropna()
        subject = valid["subject"].astype(int).values
        bg = valid[f"{metric}_background_0_5"].values
        tg = valid[f"{metric}_target_6_8"].values
        color_map = _subject_color_map(subject)

        # superplot: subject-level means, color-coded and paired
        for sid, bgi, tgi in zip(subject, bg, tg):
            c = color_map[int(sid)]
            ax.plot([1, 2], [bgi, tgi], color=c, alpha=0.40, linewidth=0.9, zorder=1)
            ax.scatter(
                1 + rng_plot.uniform(-0.04, 0.04),
                bgi,
                marker="o",
                s=20,
                alpha=0.80,
                edgecolors="black",
                linewidths=0.2,
                color=c,
                zorder=2,
            )
            ax.scatter(
                2 + rng_plot.uniform(-0.04, 0.04),
                tgi,
                marker="o",
                s=20,
                alpha=0.80,
                edgecolors="black",
                linewidths=0.2,
                color=c,
                zorder=2,
            )

        # median + IQR error bars
        bg_median, tg_median = np.median(bg), np.median(tg)
        bg_q1, bg_q3 = np.percentile(bg, [25, 75])
        tg_q1, tg_q3 = np.percentile(tg, [25, 75])
        ax.errorbar(
            [1, 2],
            [bg_median, tg_median],
            yerr=[
                [bg_median - bg_q1, tg_median - tg_q1],
                [bg_q3 - bg_median, tg_q3 - tg_median],
            ],
            fmt="D",
            markersize=6,
            color="black",
            ecolor="black",
            elinewidth=1.2,
            capsize=4,
            zorder=4,
        )
        ax.hlines(bg_median, 0.82, 1.18, color="black", linewidth=1.8)
        ax.hlines(tg_median, 1.82, 2.18, color="black", linewidth=1.8)
        p = test_df.loc[test_df["metric"] == metric, "p_value"].iloc[0]
        y_max = np.nanmax([bg.max(), tg.max()])
        y_min = np.nanmin([bg.min(), tg.min()])
        y_range = y_max - y_min if y_max > y_min else 1.0
        ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.20 * y_range)
        _add_sig_text(
            ax, 1, 2, y_max + 0.05 * y_range, 0.03 * y_range, f"{_p_to_stars(p)}  p={p:.3g}"
        )
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["background\n0–5", "target\n6–8"])
        ax.set_ylabel(ylabel)
        _clean_axes(ax)
    fig.suptitle("Subject-level MCMC comparison: background vs target items", y=1.02)
    fig.tight_layout()
    saved.append(
        _save_fig(
            fig, fig_dir, "MCMC_target_vs_background_paired_metrics", suffix, bbox_inches="tight"
        )
    )

    sort_score = (
        wide["fixation_mcmc_mean_target_6_8"] - wide["fixation_mcmc_mean_background_0_5"]
    ).values
    sort_order = np.argsort(sort_score)[::-1]
    fixation_mat = mcmc_metric_df.pivot(
        index="subject", columns="item_0b", values="fixation_mcmc"
    ).values[sort_order, :]
    flip_mat = mcmc_metric_df.pivot(
        index="subject", columns="item_0b", values="flip_rate_mcmc"
    ).values[sort_order, :]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    im0 = axes[0].imshow(
        fixation_mat, aspect="auto", interpolation="nearest", cmap="vlag",
    )
    axes[0].set_title("Fixation")
    axes[0].set_xlabel("Item index, 0-based")
    axes[0].set_ylabel("Subjects\nsorted by target fixation advantage")
    axes[0].set_xticks(np.arange(n_items))
    axes[0].set_xticklabels([str(i) for i in range(n_items)])
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        flip_mat, aspect="auto", interpolation="nearest", cmap="vlag"
    )
    axes[1].set_title("Flip rate")
    axes[1].set_xlabel("Item index, 0-based")
    axes[1].set_xticks(np.arange(n_items))
    axes[1].set_xticklabels([str(i) for i in range(n_items)])
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle("Subject × item stability from MCMC", y=1.02)
    fig.tight_layout()
    saved.append(
        _save_fig(
            fig, fig_dir, "MCMC_subject_item_stability_heatmaps", suffix, bbox_inches="tight"
        )
    )

    rep_subjects = wide.iloc[sort_order[:n_representative_subjects]]["subject"].astype(int).tolist()
    fig, axes = plt.subplots(
        len(rep_subjects), 1, figsize=(12, 1.6 * len(rep_subjects)), sharex=True
    )
    if len(rep_subjects) == 1:
        axes = [axes]
    for ax, subj in zip(axes, rep_subjects):
        traj = all_samples[subj, -n_traj_plot:, :].T
        ax.imshow(traj, aspect="auto", interpolation="nearest", cmap="vlag")
        ax.set_ylabel(f"subj {subj}\nitem")
        ax.set_yticks(np.arange(n_items))
        ax.set_yticklabels([str(i) for i in range(1, n_items + 1)])
        # separator between background items 1–6 and target items 7–9 (0-based row 5.5)
        ax.axhline(5.5, color="white", linewidth=1.2)
    axes[-1].set_xlabel("MCMC saved sweep")
    fig.suptitle(
        "Representative MCMC binary trajectories\nitems 1–6 vs target items 7–9", y=1.01
    )
    fig.tight_layout()
    saved.append(
        _save_fig(
            fig, fig_dir, "MCMC_representative_binary_trajectories", suffix, bbox_inches="tight"
        )
    )

    merged_exact_mcmc = mcmc_metric_df.merge(
        exact_metric_df,
        on=["subject", "item_0b", "item_1b", "item_group"],
        how="left",
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        merged_exact_mcmc["fixation_exact"],
        merged_exact_mcmc["fixation_mcmc"],
        s=18,
        alpha=0.5,
        edgecolors="black",
        linewidths=0.2,
    )
    ax.plot([0.5, 1.0], [0.5, 1.0], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Exact fixation")
    ax.set_ylabel("MCMC fixation")
    ax.set_title("Exact vs MCMC fixation check")
    _clean_axes(ax)
    fig.tight_layout()
    saved.append(
        _save_fig(fig, fig_dir, "MCMC_vs_exact_fixation_check", suffix, bbox_inches="tight")
    )

    return saved


STRATIFIED_GES_PAIR_ORDER = ["BG→BG", "BG→ND", "BG→D", "ND→ND", "ND→D", "D→D"]

# Okabe–Ito 系の色覚多様性向けアンカー（07-1 colorblind figure と整合）
COL_BG = "#9E9E9E"   # background: gray
COL_TND = "#E69F00"  # target non-direct: orange
COL_DIR = "#0072B2"  # direct target triangle: blue

STRATIFIED_GES_CATEGORY_COLORS = {
    "background": COL_BG,
    "target_non_direct": COL_TND,
    "direct_target_triangle": COL_DIR,
}


def _blend_hex_colors(c1: str, c2: str, weight_c2: float = 0.5) -> str:
    """sRGB 上で 2 色を線形合成する（weight_c2=1 で c2）。"""
    w2 = float(weight_c2)
    w1 = 1.0 - w2
    rgb = w1 * np.array(mcolors.to_rgb(c1)) + w2 * np.array(mcolors.to_rgb(c2))
    return mcolors.to_hex(np.clip(rgb, 0.0, 1.0))


def build_stratified_ges_colors(
    *,
    cross_blend: float = 0.5,
) -> dict[str, str]:
    """
    stratified GES 6 分類の色を返す。

    対角（BG→BG, ND→ND, D→D）はカテゴリ純色。
    クロス（BG→ND, BG→D, ND→D）は source / target アンカー色の合成色。
    """
    bg, nd, d = COL_BG, COL_TND, COL_DIR
    return {
        "BG→BG": bg,
        "BG→ND": _blend_hex_colors(bg, nd, cross_blend),
        "BG→D": _blend_hex_colors(bg, d, cross_blend),
        "ND→ND": nd,
        "ND→D": _blend_hex_colors(nd, d, cross_blend),
        "D→D": d,
    }


# 6 種類それぞれ: 対角=純色, クロス=アンサンブル色 + 固有マーカー
STRATIFIED_GES_COLORS = build_stratified_ges_colors()

STRATIFIED_GES_MARKERS = {
    "BG→BG": "^",   # 三角
    "BG→ND": "*",   # 星
    "BG→D": "o",    # 丸
    "ND→ND": "s",   # 四角
    "ND→D": "D",    # ダイヤ
    "D→D": "v",     # 逆三角
}

STRATIFIED_GES_MARKER_SIZES = {
    "BG→BG": 28,
    "BG→ND": 55,
    "BG→D": 36,
    "ND→ND": 36,
    "ND→D": 36,
    "D→D": 36,
}


def _stratified_ges_marker_style(ges_pair: str) -> tuple[str, int, float]:
    """ges_pair から (marker, size, alpha) を返す."""
    return (
        STRATIFIED_GES_MARKERS[ges_pair],
        STRATIFIED_GES_MARKER_SIZES[ges_pair],
        0.85,
    )


def _stripplot_offsets(n: int, width: float = 0.22) -> np.ndarray:
    """カテゴリ内で点が重ならないよう決定的な x オフセットを返す."""
    if n <= 1:
        return np.array([0.0])
    return np.linspace(-width, width, n)


def plot_stratified_ges_boxplot(
    long_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    *,
    pair_order: Sequence[str] | None = None,
    comparisons_to_annotate: Sequence[tuple[str, str]] | None = None,
    figsize: tuple[float, float] = (9, 5),
    title: str = "Stratified Global Embedding Score (GES)",
    annotate_direct_labels: bool = True,
) -> plt.Figure:
    """
    stratified GES を透明 boxplot + 色付きマーカーで描画する。

    対角ペア（BG→BG, ND→ND, D→D）はカテゴリ純色、
    クロスペア（BG→ND, BG→D, ND→D）は source/target 色の合成色。
    マーカー形状も 6 種類で区別する（色覚多様性向け）。

    test_df を渡すとペアワise検定結果を bracket で表示する。
    comparisons_to_annotate 未指定時は order 内の全ペア（既定 15 通り）を描画する。
    bracket の高さは x 区間の重なりを避けるよう自動で段階配置する。
    """
    order = list(pair_order) if pair_order is not None else STRATIFIED_GES_PAIR_ORDER
    data = [
        long_df.loc[long_df["ges_pair"] == pair, "GES_stratified_abs"].dropna().values
        for pair in order
    ]

    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(1, len(order) + 1)
    bp = ax.boxplot(
        data,
        positions=positions,
        labels=order,
        patch_artist=True,
        showmeans=False,
        medianprops={"color": "#333333", "linewidth": 1.2},
        boxprops={"linewidth": 1.0, "edgecolor": "black"},
        whiskerprops={"linewidth": 1.0, "color": "black"},
        capprops={"linewidth": 1.0, "color": "black"},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("none")
        patch.set_edgecolor("black")

    for i, pair in enumerate(order):
        pair_df = long_df.loc[long_df["ges_pair"] == pair].sort_values("edge_label_1b")
        vals = pair_df["GES_stratified_abs"].dropna().values
        if len(vals) == 0:
            continue
        x_center = positions[i]
        x_offsets = _stripplot_offsets(len(vals))
        color = STRATIFIED_GES_COLORS[pair]
        marker, size, alpha = _stratified_ges_marker_style(pair)
        ax.scatter(
            x_center + x_offsets,
            vals,
            c=color,
            marker=marker,
            s=size,
            alpha=alpha,
            edgecolors="black",
            linewidths=0.25,
            zorder=3,
        )
        if annotate_direct_labels and pair == "D→D":
            for x_off, (_, row) in zip(x_offsets, pair_df.iterrows()):
                ax.text(
                    x_center + x_off + 0.10,
                    row["GES_stratified_abs"],
                    row["edge_label_1b"],
                    fontsize=8,
                    color=color,
                    va="center",
                )

    ax.set_ylabel("Global embedding score (GES)")
    ax.set_xlabel("Source → target category")
    ax.set_title(title)
    for tick, pair in zip(ax.get_xticklabels(), order):
        tick.set_color(STRATIFIED_GES_COLORS[pair])
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    _clean_axes(ax)

    if test_df is not None and len(test_df):
        if comparisons_to_annotate is None:
            comparisons_to_annotate = all_stratified_ges_pairwise_comparisons(order)
        y_max = float(np.nanmax(long_df.loc[long_df["ges_pair"].isin(order), "GES_stratified_abs"]))
        n_pairs = len(comparisons_to_annotate)
        y_step = 0.028 if n_pairs > 6 else 0.034
        _annotate_pairwise_significance_brackets(
            ax,
            order=order,
            test_df=test_df,
            comparisons=comparisons_to_annotate,
            y_data_max=y_max,
            y_step=y_step,
        )

    fig.tight_layout()
    return fig


STRATIFIED_GES_CATEGORY_LABELS = {
    "background": "BG",
    "target_non_direct": "ND",
    "direct_target_triangle": "D",
}

STRATIFIED_GES_PAIR_TO_CATEGORIES = {
    pair_label: (src, tgt) for src, tgt, pair_label in STRATIFIED_GES_PAIRS
}


def _edge_category_key(i_1b: int, j_1b: int, target_items_1b: Sequence[int]) -> str:
    """1-based 項目番号から edge カテゴリ key を返す."""
    target_set = set(int(x) for x in target_items_1b)
    n_tgt = int(i_1b in target_set) + int(j_1b in target_set)
    if n_tgt == 0:
        return "background"
    if n_tgt == 1:
        return "target_non_direct"
    return "direct_target_triangle"


def plot_jij_edge_category_matrix(
    *,
    n_items: int = 9,
    target_items_0b: Sequence[int] = DEFAULT_TARGET_ITEMS_0B,
    highlight_ges_pair: str | None = None,
    figsize: tuple[float, float] = (7.2, 6.4),
    title: str = "Matrix of $J_{i,j}$",
) -> plt.Figure:
    """
    9×9 上三角 J 行列を BG / ND / D のカテゴリ色で示す schematic。

    highlight_ges_pair を指定すると、source / target ブロックを強調し矢印を描く。
    """
    target_items_1b = [int(i) + 1 for i in target_items_0b]
    cat_colors = STRATIFIED_GES_CATEGORY_COLORS

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, n_items - 0.5)
    ax.set_ylim(n_items - 0.5, -0.5)
    ax.set_aspect("equal")

    highlight_src: str | None = None
    highlight_tgt: str | None = None
    if highlight_ges_pair is not None:
        highlight_src, highlight_tgt = STRATIFIED_GES_PAIR_TO_CATEGORIES[highlight_ges_pair]

    for row in range(n_items):
        for col in range(n_items):
            i_1b, j_1b = row + 1, col + 1
            x, y = col - 0.5, row - 0.5
            if row == col:
                face = "#F5F5F5"
                edgecolor = "#CCCCCC"
                lw = 0.8
                label = "—"
                text_color = "#666666"
            elif row > col:
                face = "white"
                edgecolor = "white"
                lw = 0.0
                label = ""
                text_color = "black"
            else:
                cat = _edge_category_key(i_1b, j_1b, target_items_1b)
                face = cat_colors[cat]
                edgecolor = "black"
                lw = 0.8
                label = rf"$J_{{{i_1b},{j_1b}}}$"
                text_color = "black"
                if highlight_ges_pair is not None:
                    if cat == highlight_src or cat == highlight_tgt:
                        lw = 2.4
                        if cat == highlight_src and cat == highlight_tgt:
                            edgecolor = STRATIFIED_GES_COLORS[highlight_ges_pair]
                        elif cat == highlight_src:
                            edgecolor = cat_colors[highlight_src]
                        else:
                            edgecolor = cat_colors[highlight_tgt]
                    else:
                        face = mcolors.to_hex(
                            np.clip(np.array(mcolors.to_rgb(face)) * 0.42 + 0.58, 0.0, 1.0)
                        )
                        lw = 0.5
                        edgecolor = "#BBBBBB"

            rect = Rectangle(
                (x, y), 1, 1, facecolor=face, edgecolor=edgecolor, linewidth=lw
            )
            ax.add_patch(rect)
            if label:
                ax.text(
                    col,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7 if row < col else 8,
                    color=text_color,
                )

    # background | target の区切り（0-based 5.5 = 項目 6 | 7）
    split = min(target_items_1b) - 1.5
    ax.axhline(split, color="black", linewidth=1.5, linestyle="--", alpha=0.55)
    ax.axvline(split, color="black", linewidth=1.5, linestyle="--", alpha=0.55)
    ax.text(2.5, -0.85, "items 1–6", ha="center", va="bottom", fontsize=8, color="#555555")
    ax.text(7.0, -0.85, "items 7–9", ha="center", va="bottom", fontsize=8, color="#555555")

    if highlight_ges_pair is not None:
        src_cat, tgt_cat = STRATIFIED_GES_PAIR_TO_CATEGORIES[highlight_ges_pair]
        src_center = _category_block_center(n_items, target_items_1b, src_cat)
        tgt_center = _category_block_center(n_items, target_items_1b, tgt_cat)
        arrow = FancyArrowPatch(
            src_center,
            tgt_center,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.6,
            color=STRATIFIED_GES_COLORS[highlight_ges_pair],
            connectionstyle="arc3,rad=0.12",
            zorder=5,
        )
        ax.add_patch(arrow)
        ax.text(
            0.02,
            0.02,
            f"GES example: {highlight_ges_pair}\n"
            f"source={STRATIFIED_GES_CATEGORY_LABELS[src_cat]}, "
            f"target={STRATIFIED_GES_CATEGORY_LABELS[tgt_cat]}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#CCCCCC"},
        )

    ax.set_xticks(np.arange(n_items))
    ax.set_yticks(np.arange(n_items))
    ax.set_xticklabels([str(i + 1) for i in range(n_items)])
    ax.set_yticklabels([str(i + 1) for i in range(n_items)])
    ax.set_xlabel("Item $j$")
    ax.set_ylabel("Item $i$")
    ax.set_title(title)
    _add_jij_category_legend(ax)
    _clean_axes(ax)
    fig.tight_layout()
    return fig


def _category_block_center(
    n_items: int, target_items_1b: Sequence[int], category: str
) -> tuple[float, float]:
    """カテゴリブロックの代表中心 (col, row) を 0-based 座標で返す."""
    pts: list[tuple[float, float]] = []
    for row in range(n_items):
        for col in range(n_items):
            if row >= col:
                continue
            cat = _edge_category_key(row + 1, col + 1, target_items_1b)
            if cat == category:
                pts.append((float(col), float(row)))
    if not pts:
        return (n_items / 2, n_items / 2)
    xs, ys = zip(*pts)
    return (float(np.mean(xs)), float(np.mean(ys)))


def _add_jij_category_legend(ax: plt.Axes) -> None:
    """BG / ND / D の凡例を axes 外右上に追加."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markersize=10,
            markerfacecolor=STRATIFIED_GES_CATEGORY_COLORS[key],
            markeredgecolor="black",
            markeredgewidth=0.6,
        )
        for key in ("background", "target_non_direct", "direct_target_triangle")
    ]
    labels = [
        f"{STRATIFIED_GES_CATEGORY_LABELS[k]} ({k.replace('_', ' ')})"
        for k in ("background", "target_non_direct", "direct_target_triangle")
    ]
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        fontsize=8,
        title="Edge category",
        title_fontsize=8,
    )


def plot_stratified_ges_source_target_grid(
    *,
    figsize: tuple[float, float] = (5.8, 4.8),
    title: str = "Stratified GES: source $\\rightarrow$ target",
) -> plt.Figure:
    """6 種類の GES(src→tgt) を source×target 表で示す概念図。"""
    cat_order = ["background", "target_non_direct", "direct_target_triangle"]
    cat_short = [STRATIFIED_GES_CATEGORY_LABELS[c] for c in cat_order]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    for row, src_cat in enumerate(cat_order):
        for col, tgt_cat in enumerate(cat_order):
            pair_label = _ges_pair_label(src_cat, tgt_cat)
            x, y = col, row
            if pair_label is None:
                face = "#F0F0F0"
                edgecolor = "#DDDDDD"
                text = "—"
                text_color = "#AAAAAA"
                lw = 0.8
            else:
                face = STRATIFIED_GES_COLORS[pair_label]
                edgecolor = "black"
                text = pair_label
                text_color = "black"
                lw = 1.2
            ax.add_patch(
                Rectangle(
                    (x + 0.06, y + 0.06),
                    0.88,
                    0.88,
                    facecolor=face,
                    edgecolor=edgecolor,
                    linewidth=lw,
                )
            )
            ax.text(
                x + 0.5,
                y + 0.5,
                text,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold" if pair_label else "normal",
                color=text_color,
            )

    ax.set_xticks(np.arange(3) + 0.5)
    ax.set_yticks(np.arange(3) + 0.5)
    ax.set_xticklabels([f"target {s}" for s in cat_short])
    ax.set_yticklabels([f"source {s}" for s in cat_short])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Target edge category")
    ax.set_ylabel("Source edge category")
    ax.set_title(title)
    ax.text(
        0.5,
        -0.22,
        "Each cell: mean |corr| between source-category edges\n"
        "and other edges in the target category",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="#444444",
    )
    _clean_axes(ax)
    fig.tight_layout()
    return fig


def _ges_pair_label(src_cat: str, tgt_cat: str) -> str | None:
    for src, tgt, label in STRATIFIED_GES_PAIRS:
        if src == src_cat and tgt == tgt_cat:
            return label
    return None


def plot_stratified_ges_explainer(
    *,
    highlight_ges_pair: str = "BG→ND",
    n_items: int = 9,
    target_items_0b: Sequence[int] = DEFAULT_TARGET_ITEMS_0B,
    figsize: tuple[float, float] = (13.5, 5.8),
) -> plt.Figure:
    """Panel A: J カテゴリ行列 + Panel B: source→target 表の説明図。"""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.85], wspace=0.45)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    _draw_jij_category_matrix_on_ax(
        ax_a,
        n_items=n_items,
        target_items_0b=target_items_0b,
        highlight_ges_pair=highlight_ges_pair,
        panel_label="A",
    )
    _draw_source_target_grid_on_ax(ax_b, panel_label="B")
    fig.suptitle(
        "Stratified GES: which $J_{i,j}$ edges enter each comparison?",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def _draw_jij_category_matrix_on_ax(
    ax: plt.Axes,
    *,
    n_items: int,
    target_items_0b: Sequence[int],
    highlight_ges_pair: str | None,
    panel_label: str | None = None,
) -> None:
    """plot_jij_edge_category_matrix の描画本体（既存 Axes 上）。"""
    target_items_1b = [int(i) + 1 for i in target_items_0b]
    cat_colors = STRATIFIED_GES_CATEGORY_COLORS
    highlight_src = highlight_tgt = None
    if highlight_ges_pair is not None:
        highlight_src, highlight_tgt = STRATIFIED_GES_PAIR_TO_CATEGORIES[highlight_ges_pair]

    ax.set_xlim(-0.5, n_items - 0.5)
    ax.set_ylim(n_items - 0.5, -0.5)
    ax.set_aspect("equal")

    for row in range(n_items):
        for col in range(n_items):
            i_1b, j_1b = row + 1, col + 1
            x, y = col - 0.5, row - 0.5
            if row == col:
                face, edgecolor, lw, label, text_color = "#F5F5F5", "#CCCCCC", 0.8, "—", "#666666"
            elif row > col:
                face, edgecolor, lw, label, text_color = "white", "white", 0.0, "", "black"
            else:
                cat = _edge_category_key(i_1b, j_1b, target_items_1b)
                face = cat_colors[cat]
                edgecolor, lw, label, text_color = "black", 0.8, rf"$J_{{{i_1b},{j_1b}}}$", "black"
                if highlight_ges_pair is not None:
                    if cat == highlight_src or cat == highlight_tgt:
                        lw = 2.4
                        edgecolor = (
                            STRATIFIED_GES_COLORS[highlight_ges_pair]
                            if highlight_src == highlight_tgt and cat == highlight_src
                            else cat_colors[cat]
                        )
                    else:
                        face = mcolors.to_hex(
                            np.clip(np.array(mcolors.to_rgb(face)) * 0.42 + 0.58, 0.0, 1.0)
                        )
                        lw, edgecolor = 0.5, "#BBBBBB"
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=face, edgecolor=edgecolor, linewidth=lw))
            if label:
                ax.text(col, row, label, ha="center", va="center", fontsize=6.5, color=text_color)

    split = min(target_items_1b) - 1.5
    ax.axhline(split, color="black", linewidth=1.2, linestyle="--", alpha=0.55)
    ax.axvline(split, color="black", linewidth=1.2, linestyle="--", alpha=0.55)

    if highlight_ges_pair is not None:
        src_cat, tgt_cat = STRATIFIED_GES_PAIR_TO_CATEGORIES[highlight_ges_pair]
        arrow = FancyArrowPatch(
            _category_block_center(n_items, target_items_1b, src_cat),
            _category_block_center(n_items, target_items_1b, tgt_cat),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.5,
            color=STRATIFIED_GES_COLORS[highlight_ges_pair],
            connectionstyle="arc3,rad=0.12",
            zorder=5,
        )
        ax.add_patch(arrow)

    ax.set_xticks(np.arange(n_items))
    ax.set_yticks(np.arange(n_items))
    ax.set_xticklabels([str(i + 1) for i in range(n_items)])
    ax.set_yticklabels([str(i + 1) for i in range(n_items)])
    ax.set_xlabel("Item $j$")
    ax.set_ylabel("Item $i$")
    title = "Matrix of $J_{i,j}$ by edge category"
    if panel_label:
        title = f"{panel_label}. {title}"
    ax.set_title(title, loc="left", fontsize=10)
    if highlight_ges_pair:
        src_cat, tgt_cat = STRATIFIED_GES_PAIR_TO_CATEGORIES[highlight_ges_pair]
        ax.text(
            0.0,
            1.02,
            f"Example {highlight_ges_pair}: "
            f"{STRATIFIED_GES_CATEGORY_LABELS[src_cat]} $\\rightarrow$ "
            f"{STRATIFIED_GES_CATEGORY_LABELS[tgt_cat]}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color=STRATIFIED_GES_COLORS[highlight_ges_pair],
        )
    _add_jij_category_legend(ax)
    _clean_axes(ax)


def _draw_source_target_grid_on_ax(ax: plt.Axes, *, panel_label: str | None = None) -> None:
    cat_order = ["background", "target_non_direct", "direct_target_triangle"]
    cat_short = [STRATIFIED_GES_CATEGORY_LABELS[c] for c in cat_order]
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    for row, src_cat in enumerate(cat_order):
        for col, tgt_cat in enumerate(cat_order):
            pair_label = _ges_pair_label(src_cat, tgt_cat)
            x, y = col, row
            if pair_label is None:
                face, edgecolor, text, text_color, lw = "#F0F0F0", "#DDDDDD", "—", "#AAAAAA", 0.8
            else:
                face = STRATIFIED_GES_COLORS[pair_label]
                edgecolor, text, text_color, lw = "black", pair_label, "black", 1.2
            ax.add_patch(
                Rectangle(
                    (x + 0.06, y + 0.06), 0.88, 0.88,
                    facecolor=face, edgecolor=edgecolor, linewidth=lw,
                )
            )
            ax.text(
                x + 0.5, y + 0.5, text,
                ha="center", va="center", fontsize=9,
                fontweight="bold" if pair_label else "normal",
                color=text_color,
            )

    ax.set_xticks(np.arange(3) + 0.5)
    ax.set_yticks(np.arange(3) + 0.5)
    ax.set_xticklabels([f"tgt {s}" for s in cat_short], fontsize=8)
    ax.set_yticklabels([f"src {s}" for s in cat_short], fontsize=8)
    ax.xaxis.tick_top()
    title = "Six stratified GES definitions"
    if panel_label:
        title = f"{panel_label}. {title}"
    ax.set_title(title, loc="left", fontsize=10)
    _clean_axes(ax)


def save_stratified_ges_schematic_figures(
    *,
    output_dirs: HjPaperOutputDirs,
    suffix: str = "STRATIFIED_GES",
    highlight_pairs: Sequence[str] | None = None,
) -> list[Path]:
    """stratified GES 説明用 schematic を PDF 保存する."""
    if highlight_pairs is None:
        highlight_pairs = ["BG→ND", "BG→D", "ND→D"]

    saved: list[Path] = []
    fig_dir = output_dirs.fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig_matrix = plot_jij_edge_category_matrix(highlight_ges_pair=None)
    saved.append(_save_fig(fig_matrix, fig_dir, "J_stratified_GES_category_matrix", suffix))

    fig_grid = plot_stratified_ges_source_target_grid()
    saved.append(_save_fig(fig_grid, fig_dir, "J_stratified_GES_source_target_grid", suffix))

    fig_explainer = plot_stratified_ges_explainer(highlight_ges_pair="BG→ND")
    saved.append(_save_fig(fig_explainer, fig_dir, "J_stratified_GES_explainer_BG_ND", suffix))

    for pair in highlight_pairs:
        safe = pair.replace("→", "_to_")
        fig_h = plot_jij_edge_category_matrix(
            highlight_ges_pair=pair,
            title=f"Matrix of $J_{{i,j}}$ — example {pair}",
        )
        saved.append(_save_fig(fig_h, fig_dir, f"J_stratified_GES_category_matrix_{safe}", suffix))

    return saved


def save_stratified_ges_outputs(
    *,
    long_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dirs: HjPaperOutputDirs,
    suffix: str = "STRATIFIED_GES",
) -> list[Path]:
    """stratified GES の CSV と boxplot を保存する."""
    saved: list[Path] = []
    csv_dir = output_dirs.csv_dir
    fig_dir = output_dirs.fig_dir
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    saved.append(_save_csv(long_df, csv_dir, "J_stratified_GES_long", suffix))
    saved.append(_save_csv(summary_df, csv_dir, "J_stratified_GES_summary", suffix))
    saved.append(_save_csv(test_df, csv_dir, "J_stratified_GES_pairwise_tests", suffix))

    fig = plot_stratified_ges_boxplot(long_df, test_df)
    saved.append(_save_fig(fig, fig_dir, "J_stratified_GES_boxplot", suffix))
    saved.extend(save_stratified_ges_schematic_figures(output_dirs=output_dirs, suffix=suffix))
    return saved
