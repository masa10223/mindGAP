"""
Simple hub-focused submatrix analysis for 36 Jij variables.

Goal
----
Quantify statements like:
"If J_5,8 is high, J_3,8 is also high" (shared hub = 8).

Input
-----
- (N,36) raw subject x edge matrix, or
- 36x36 covariance/correlation matrix of Jij variables.
- Optional feature names (the row/column order of the matrix).

Output
------
- CSV tables with hub coactivation metrics.
- Simple figures (hub-specific 6x6 heatmaps, hub score bars).

Notes
-----
- The script intentionally avoids complex modularity algorithms.
- It relies on submatrix summaries and simple statistics.
"""

from __future__ import annotations

import argparse
import re
from itertools import combinations
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import pinv

BACKGROUND = tuple(range(1, 7))  # 1..6
HUBS = (7, 8, 9)


def edge_name(i: int, j: int) -> str:
    i, j = sorted((int(i), int(j)))
    if i == j:
        raise ValueError(f"i and j must be different: ({i}, {j})")
    return f"J_{i},{j}"


def parse_edge_name(name: str) -> str:
    values = re.findall(r"\d+", str(name))
    if len(values) != 2:
        raise ValueError(f"Cannot parse edge label from {name!r}")
    return edge_name(int(values[0]), int(values[1]))


def canonical_feature_names() -> list[str]:
    return [edge_name(i, j) for i, j in combinations(range(1, 10), 2)]


def _normalize_feature_names(feature_names: Sequence[str] | None, n: int) -> list[str] | None:
    if feature_names is None:
        return None
    if len(feature_names) != n:
        raise ValueError(f"feature_names length mismatch: {len(feature_names)} vs {n}")
    labels = [parse_edge_name(x) for x in feature_names]
    if len(set(labels)) != len(labels):
        raise ValueError("feature_names contains duplicates")
    return labels


def as_square_dataframe(matrix, feature_names: Sequence[str] | None = None) -> pd.DataFrame:
    A = np.asarray(matrix, dtype=float)
    if A.shape != (36, 36):
        raise ValueError(f"Expected 36x36 matrix, got {A.shape}")

    labels = _normalize_feature_names(feature_names, 36)
    if labels is None:
        labels = canonical_feature_names()
    df = pd.DataFrame(A, index=labels, columns=labels)
    return df


def cov_to_corr(cov_df: pd.DataFrame) -> pd.DataFrame:
    A = cov_df.to_numpy(dtype=float)
    A = (A + A.T) / 2.0
    sd = np.sqrt(np.diag(A))
    if np.any(sd <= 0) or np.any(~np.isfinite(sd)):
        raise ValueError("Covariance diagonal must be positive finite values")
    R = A / np.outer(sd, sd)
    np.fill_diagonal(R, 1.0)
    R = np.clip(R, -1.0, 1.0)
    return pd.DataFrame(R, index=cov_df.index, columns=cov_df.columns)


def to_correlation(matrix, matrix_kind: str, feature_names: Sequence[str] | None = None) -> pd.DataFrame:
    matrix_kind = matrix_kind.lower()
    if matrix_kind == "raw":
        X = np.asarray(matrix, dtype=float)
        if X.ndim != 2 or X.shape[1] != 36:
            raise ValueError(f"matrix_kind='raw' expects (N,36), got {X.shape}")
        labels = _normalize_feature_names(feature_names, 36)
        if labels is None:
            labels = canonical_feature_names()
        R = np.corrcoef(X, rowvar=False)
        return pd.DataFrame(R, index=labels, columns=labels)

    df = as_square_dataframe(matrix, feature_names=feature_names)
    if matrix_kind == "cov":
        return cov_to_corr(df)
    if matrix_kind == "corr":
        R = (df + df.T) / 2.0
        np.fill_diagonal(R.values, 1.0)
        return R.clip(-1.0, 1.0)
    raise ValueError("matrix_kind must be 'raw', 'cov' or 'corr'")


def offdiag_values(sub_df: pd.DataFrame) -> np.ndarray:
    A = sub_df.to_numpy(dtype=float)
    return A[np.triu_indices(A.shape[0], k=1)]


def hub_pair_table(R: pd.DataFrame, hub: int) -> pd.DataFrame:
    rows = []
    for a, b in combinations(BACKGROUND, 2):
        e1 = edge_name(a, hub)
        e2 = edge_name(b, hub)
        rows.append(
            {
                "hub": hub,
                "edge_1": e1,
                "edge_2": e2,
                "r": float(R.loc[e1, e2]),
            }
        )
    return pd.DataFrame(rows).sort_values("r", ascending=False).reset_index(drop=True)


def hub_summary(R: pd.DataFrame) -> pd.DataFrame:
    all_labels = list(R.index)
    rows = []
    for hub in HUBS:
        star = [edge_name(b, hub) for b in BACKGROUND]  # S7 / S8 / S9, each size 6
        outside = [x for x in all_labels if x not in star]

        within = float(np.mean(offdiag_values(R.loc[star, star])))
        to_outside = float(R.loc[star, outside].to_numpy().mean())

        # "Hub coactivation index": larger -> stronger shared-hub synchrony
        hci = within - to_outside

        rows.append(
            {
                "hub": hub,
                "mean_within_star_r": within,
                "mean_to_outside_r": to_outside,
                "hub_coactivation_index": hci,
            }
        )
    return pd.DataFrame(rows).sort_values("hub_coactivation_index", ascending=False).reset_index(drop=True)


def triangle_attachment(R: pd.DataFrame) -> pd.DataFrame:
    triangle = [edge_name(7, 8), edge_name(7, 9), edge_name(8, 9)]
    S7 = [edge_name(b, 7) for b in BACKGROUND]
    S8 = [edge_name(b, 8) for b in BACKGROUND]
    S9 = [edge_name(b, 9) for b in BACKGROUND]
    B = [edge_name(i, j) for i, j in combinations(BACKGROUND, 2)]

    rows = []
    for t in triangle:
        rows.append(
            {
                "triangle_edge": t,
                "mean_r_to_S7": float(R.loc[t, S7].mean()),
                "mean_r_to_S8": float(R.loc[t, S8].mean()),
                "mean_r_to_S9": float(R.loc[t, S9].mean()),
                "mean_r_to_B": float(R.loc[t, B].mean()),
            }
        )
    return pd.DataFrame(rows).set_index("triangle_edge")


def module_definitions() -> dict[str, list[str]]:
    return {
        "BG": [edge_name(i, j) for i, j in combinations(BACKGROUND, 2)],
        "S7": [edge_name(b, 7) for b in BACKGROUND],
        "S8": [edge_name(b, 8) for b in BACKGROUND],
        "S9": [edge_name(b, 9) for b in BACKGROUND],
        "T789": [edge_name(7, 8), edge_name(7, 9), edge_name(8, 9)],
    }


def edge_ges_table(R: pd.DataFrame) -> pd.DataFrame:
    mods = module_definitions()
    rows = []
    labels = list(R.index)
    for e in labels:
        others = [x for x in labels if x != e]
        ges_all_abs = float(np.abs(R.loc[e, others]).mean())
        group = "Other"
        for g, members in mods.items():
            if e in members:
                group = g
                break
        rows.append({"edge": e, "group": group, "GES_all_abs": ges_all_abs})
    return pd.DataFrame(rows)


def within_star_pair_ges(R: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hub in HUBS:
        labels = [edge_name(b, hub) for b in BACKGROUND]
        sub = R.loc[labels, labels]
        for a, b in combinations(labels, 2):
            rows.append(
                {
                    "group": f"S{hub}_within",
                    "pair": f"{a}~{b}",
                    "GES_within_abs": float(abs(sub.loc[a, b])),
                    "GES_within_signed": float(sub.loc[a, b]),
                }
            )
    return pd.DataFrame(rows)


def module_within_pair_distribution(
    R: pd.DataFrame,
    module_keys: Sequence[str] = ("S7", "S8", "S9"),
) -> pd.DataFrame:
    mods = module_definitions()
    rows = []
    for k in module_keys:
        labels = mods[k]
        if len(labels) < 2:
            continue
        for a, b in combinations(labels, 2):
            v = float(R.loc[a, b])
            rows.append(
                {
                    "group": f"{k}_within",
                    "pair": f"{a}~{b}",
                    "abs_r": abs(v),
                    "signed_r": v,
                }
            )
    return pd.DataFrame(rows)


def bg_link_distribution(R: pd.DataFrame) -> pd.DataFrame:
    mods = module_definitions()
    BG = mods["BG"]
    rows = []
    for k in ["S7", "S8", "S9", "T789"]:
        for a in BG:
            for b in mods[k]:
                v = float(R.loc[a, b])
                rows.append(
                    {
                        "group": f"BG_to_{k}",
                        "edge_bg": a,
                        "edge_other": b,
                        "abs_r": abs(v),
                        "signed_r": v,
                    }
                )
    return pd.DataFrame(rows)


def module_strength_matrix_abs(R: pd.DataFrame) -> pd.DataFrame:
    mods = module_definitions()
    keys = ["BG", "S7", "S8", "S9", "T789"]
    out = pd.DataFrame(np.nan, index=keys, columns=keys)
    for a in keys:
        for b in keys:
            A = mods[a]
            B = mods[b]
            if a == b:
                vals = offdiag_values(R.loc[A, A]).astype(float)
            else:
                vals = R.loc[A, B].to_numpy().ravel().astype(float)
            out.loc[a, b] = float(np.mean(np.abs(vals)))
    return out


def module_edge_table_from_strength(W: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(W.index):
        for j, b in enumerate(W.columns):
            if j <= i:
                continue
            rows.append({"module_a": a, "module_b": b, "mean_abs_r": float(W.loc[a, b])})
    return pd.DataFrame(rows).sort_values("mean_abs_r", ascending=False).reset_index(drop=True)


def partial_corr_from_square_matrix(
    matrix,
    matrix_kind: str,
    feature_names: Sequence[str] | None = None,
    ridge: float = 1e-6,
) -> pd.DataFrame:
    """
    Compute partial correlation among 36 edges.
    - raw: matrix is (N,36), uses covariance(X)
    - cov: matrix is (36,36) covariance
    - corr: matrix is (36,36) correlation
    """
    mk = matrix_kind.lower()
    if mk == "raw":
        X = np.asarray(matrix, dtype=float)
        if X.ndim != 2 or X.shape[1] != 36:
            raise ValueError(f"matrix_kind='raw' expects (N,36), got {X.shape}")
        cov = np.cov(X, rowvar=False)
        labels = _normalize_feature_names(feature_names, 36)
        if labels is None:
            labels = canonical_feature_names()
    elif mk == "cov":
        cov_df = as_square_dataframe(matrix, feature_names=feature_names)
        cov = cov_df.to_numpy(dtype=float)
        labels = list(cov_df.index)
    elif mk == "corr":
        corr_df = as_square_dataframe(matrix, feature_names=feature_names)
        cov = corr_df.to_numpy(dtype=float)
        labels = list(corr_df.index)
    else:
        raise ValueError("matrix_kind must be 'raw', 'cov' or 'corr'")

    A = (cov + cov.T) / 2.0 + ridge * np.eye(cov.shape[0])
    K = pinv(A)
    d = np.sqrt(np.diag(K))
    P = -K / np.outer(d, d)
    np.fill_diagonal(P, 1.0)
    P = np.clip(P, -1.0, 1.0)
    return pd.DataFrame(P, index=labels, columns=labels)


def s8_pair_table(R: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    labels = [edge_name(b, 8) for b in BACKGROUND]  # J_1,8 ... J_6,8
    rows = []
    for a, b in combinations(labels, 2):
        raw = float(R.loc[a, b])
        partial = float(P.loc[a, b])
        rows.append(
            {
                "edge_u": a,
                "edge_v": b,
                "raw_r": raw,
                "abs_raw_r": abs(raw),
                "partial_r": partial,
                "abs_partial_r": abs(partial),
                "delta_abs_raw_minus_partial": abs(raw) - abs(partial),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_raw_r", ascending=False).reset_index(drop=True)


def plot_s8_raw_partial_network(R: pd.DataFrame, P: pd.DataFrame, out_file: Path) -> None:
    """
    Two-layer network for hub=8:
    - upper: raw correlation among {J_1,8..J_6,8}
    - lower: partial correlation among same nodes
    """
    nodes = [edge_name(b, 8) for b in BACKGROUND]
    theta = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    pos = {n: (np.cos(t), np.sin(t)) for n, t in zip(nodes, theta)}

    def draw_layer(ax, M: pd.DataFrame, title: str) -> None:
        vals = [abs(float(M.loc[a, b])) for a, b in combinations(nodes, 2)]
        vmax = max(vals) if vals else 1.0
        for a, b in combinations(nodes, 2):
            v = float(M.loc[a, b])
            w = 0.4 + 5.0 * (abs(v) / vmax if vmax > 0 else 0.0)
            color = "#d62728" if v >= 0 else "#1f77b4"
            xa, ya = pos[a]
            xb, yb = pos[b]
            ax.plot([xa, xb], [ya, yb], color=color, linewidth=w, alpha=0.75)
            ax.text((xa + xb) / 2, (ya + yb) / 2, f"{v:.2f}", fontsize=8, ha="center", va="center")

        for n in nodes:
            x, y = pos[n]
            ax.scatter([x], [y], s=900, c="#f4a261", edgecolors="black", linewidths=1.1, zorder=3)
            ax.text(x, y, n, fontsize=9, fontweight="bold", ha="center", va="center")

        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")

    fig, axes = plt.subplots(2, 1, figsize=(7.8, 11.2))
    draw_layer(axes[0], R.loc[nodes, nodes], "S8 network (raw correlation): J_1,8 ... J_6,8")
    draw_layer(axes[1], P.loc[nodes, nodes], "S8 network (partial correlation): controlling other edges")
    fig.tight_layout()
    fig.savefig(out_file, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_hub_heatmaps(R: pd.DataFrame, out_file: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    vmin, vmax = -1.0, 1.0

    for ax, hub in zip(axes, HUBS):
        labels = [edge_name(b, hub) for b in BACKGROUND]
        sub = R.loc[labels, labels].to_numpy()
        im = ax.imshow(sub, cmap="coolwarm", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks(np.arange(6))
        ax.set_yticks(np.arange(6))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f"Shared hub {hub}: corr(J_i,{hub}, J_j,{hub})")

    fig.colorbar(im, ax=axes, shrink=0.8, label="r")
    fig.suptitle("Within-star coactivation submatrices (1..6 with hub 7/8/9)")
    fig.savefig(out_file, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_hub_scores(summary_df: pd.DataFrame, out_file: Path) -> None:
    x = summary_df["hub"].astype(str).tolist()
    y = summary_df["hub_coactivation_index"].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, y)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("hub")
    ax.set_ylabel("Hub coactivation index")
    ax.set_title("Hub strength from submatrix statistics")
    fig.tight_layout()
    fig.savefig(out_file, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_ges_box_jitter(edge_ges_df: pd.DataFrame, out_file: Path) -> None:
    groups = ["BG", "S7", "S8", "S9", "T789"]
    data = [edge_ges_df.loc[edge_ges_df["group"] == g, "GES_all_abs"].to_numpy() for g in groups]
    colors = ["#9e9e9e", "#e76f51", "#f4a261", "#2a9d8f", "#457b9d"]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bp = ax.boxplot(data, tick_labels=groups, patch_artist=True, widths=0.6, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor("white")
        patch.set_edgecolor(c)
        patch.set_linewidth(1.5)

    rng = np.random.default_rng(2026)
    for i, vals in enumerate(data, start=1):
        if len(vals) == 0:
            continue
        x = i + rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(x, vals, s=30, alpha=0.75, color=colors[i - 1], edgecolors="black", linewidths=0.25)

    ax.set_ylabel("GES (mean |corr to all other edges|)")
    ax.set_title("GES by edge group (includes 7/8/9 within groups)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_file, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_within_star_box_jitter(within_df: pd.DataFrame, out_file: Path) -> None:
    groups = ["S7_within", "S8_within", "S9_within"]
    data = [within_df.loc[within_df["group"] == g, "GES_within_abs"].to_numpy() for g in groups]
    colors = ["#e76f51", "#f4a261", "#2a9d8f"]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bp = ax.boxplot(data, tick_labels=groups, patch_artist=True, widths=0.6, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor("white")
        patch.set_edgecolor(c)
        patch.set_linewidth(1.5)

    rng = np.random.default_rng(2027)
    for i, vals in enumerate(data, start=1):
        x = i + rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(x, vals, s=32, alpha=0.8, color=colors[i - 1], edgecolors="black", linewidths=0.25)

    ax.set_ylabel("within-star |corr|")
    ax.set_title("7/8/9 within-star coactivation (pair-level)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_file, dpi=250, bbox_inches="tight")
    plt.close(fig)


def permutation_test_two_groups(x, y, n_perm: int = 10000, seed: int = 12345) -> tuple[float, float]:
    """Two-sided permutation test for mean difference between two samples."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan

    observed = float(np.mean(y) - np.mean(x))
    pooled = np.concatenate([x, y])
    n_x = len(x)
    null = np.empty(n_perm)
    for b in range(n_perm):
        perm = rng.permutation(pooled)
        null[b] = np.mean(perm[n_x:]) - np.mean(perm[:n_x])
    p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)
    return observed, float(p)


def p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def add_sig_bar(ax, x1: float, x2: float, y: float, h: float, text: str, fontsize: int = 9) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="black", clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=fontsize, clip_on=False)


def plot_module_within_box_jitter(within_module_df: pd.DataFrame, out_file: Path) -> None:
    preferred = ["S7_within", "S8_within", "S9_within", "T789_within", "BG_within"]
    present = [g for g in preferred if g in set(within_module_df["group"])]
    groups = present if present else sorted(within_module_df["group"].unique().tolist())
    data = [within_module_df.loc[within_module_df["group"] == g, "abs_r"].to_numpy() for g in groups]
    color_map = {
        "S7_within": "#e76f51",
        "S8_within": "#f4a261",
        "S9_within": "#2a9d8f",
        "BG_within": "#9e9e9e",
        "T789_within": "#457b9d",
    }
    colors = [color_map.get(g, "#666666") for g in groups]
    label_map = {
        "S7_within": "7 within\nJ1,7-J6,7",
        "S8_within": "8 within\nJ1,8-J6,8",
        "S9_within": "9 within\nJ1,9-J6,9",
        "T789_within": "789 closed\nJ7,8/J7,9/J8,9",
        "BG_within": "background\nJ1,2-J5,6",
    }
    tick_labels = [label_map.get(g, g) for g in groups]

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    bp = ax.boxplot(data, tick_labels=tick_labels, patch_artist=True, widths=0.62, showfliers=False, zorder=1)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor("white")
        patch.set_edgecolor(c)
        patch.set_linewidth(1.5)

    marker_map = {
        "S7_within": "o",
        "S8_within": "s",
        "S9_within": "D",
        "BG_within": "^",
        "T789_within": "P",
    }

    rng = np.random.default_rng(2028)
    for i, (vals, g) in enumerate(zip(data, groups), start=1):
        if len(vals) == 0:
            continue
        x = i + rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(
            x,
            vals,
            s=42,
            alpha=0.92,
            color=colors[i - 1],
            marker=marker_map.get(g, "o"),
            edgecolors="black",
            linewidths=0.35,
            zorder=4,
        )

    # All pairwise significance bars (permutation test, 07-1 style)
    group_to_x = {g: i + 1 for i, g in enumerate(groups)}
    comparison_specs = list(combinations(groups, 2))
    p_rows = []
    for k, (g1, g2) in enumerate(comparison_specs):
        xvals = within_module_df.loc[within_module_df["group"] == g1, "abs_r"].to_numpy()
        yvals = within_module_df.loc[within_module_df["group"] == g2, "abs_r"].to_numpy()
        diff, p = permutation_test_two_groups(xvals, yvals, seed=12345 + k)
        p_rows.append({"group_1": g1, "group_2": g2, "diff": diff, "p_value": p, "stars": p_to_stars(p)})

    if data and p_rows:
        y_max = max(np.max(v) for v in data if len(v) > 0)
        y_min = min(np.min(v) for v in data if len(v) > 0)
        y_range = max(y_max - y_min, 0.05)
        bar_h = y_range * 0.022
        bar_gap = y_range * 0.065
        n_bars = len(p_rows)
        ax.set_ylim(y_min - y_range * 0.05, y_max + bar_gap * (n_bars + 1.2))

        # Wider pairs first (bottom), narrower pairs higher -> less visual crossing
        bar_order = sorted(
            p_rows,
            key=lambda r: abs(group_to_x[r["group_2"]] - group_to_x[r["group_1"]]),
            reverse=True,
        )
        for k, row in enumerate(bar_order):
            x1 = group_to_x[row["group_1"]]
            x2 = group_to_x[row["group_2"]]
            y = y_max + bar_gap * (k + 0.45)
            add_sig_bar(ax, x1, x2, y, bar_h, f"{row['stars']}  p={row['p_value']:.3g}", fontsize=8)

    ax.set_ylabel("J-edge similarity across participants (|r|)")
    ax.set_title(
        "Within-module co-variation of interaction edges\n"
        "Each point = |corr(J_a, J_b)| for two J edges in the same module"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_file, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_bg_link_box_jitter(bg_link_df: pd.DataFrame, out_file: Path) -> None:
    groups = ["BG_to_S7", "BG_to_S8", "BG_to_S9", "BG_to_T789"]
    data = [bg_link_df.loc[bg_link_df["group"] == g, "abs_r"].to_numpy() for g in groups]
    colors = ["#e76f51", "#f4a261", "#2a9d8f", "#457b9d"]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bp = ax.boxplot(data, tick_labels=groups, patch_artist=True, widths=0.62, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor("white")
        patch.set_edgecolor(c)
        patch.set_linewidth(1.5)

    rng = np.random.default_rng(2029)
    for i, vals in enumerate(data, start=1):
        x = i + rng.uniform(-0.14, 0.14, size=len(vals))
        ax.scatter(x, vals, s=20, alpha=0.65, color=colors[i - 1], edgecolors="black", linewidths=0.2)

    ax.set_ylabel("cross-module |corr|")
    ax.set_title("BG links to S7/S8/S9/T789")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_file, dpi=250, bbox_inches="tight")
    plt.close(fig)


def write_derivation_note(out: Path, W: pd.DataFrame, within_mod_df: pd.DataFrame, bg_link_df: pd.DataFrame) -> None:
    def mean_abs(group_name: str) -> float:
        vals = within_mod_df.loc[within_mod_df["group"] == group_name, "abs_r"].to_numpy()
        return float(np.mean(vals))

    lines = []
    lines.append("Derivation of mod4b and GES plots")
    lines.append("=================================")
    lines.append("")
    lines.append("[1] Current ges_box_jitter_groups.pdf definition")
    lines.append("  For each edge e: GES_all_abs(e) = mean_{f!=e} |corr(e,f)|")
    lines.append("  Then grouped by edge category (BG / S7 / S8 / S9 / T789).")
    lines.append("  -> This is NOT the same as S8-within submatrix average.")
    lines.append("")
    lines.append("[2] S8 within-submatrix definition (your intended definition)")
    lines.append("  S8 = {J_1,8, ..., J_6,8}")
    lines.append("  S8_within_mean = mean_{a<b, a,b in S8} |corr(a,b)|")
    lines.append("  (J_7,8 and J_8,9 are NOT included)")
    lines.append(f"  Observed mean(S8_within) = {mean_abs('S8_within'):.3f}")
    lines.append("")
    lines.append("[3] mod4b network edge derivation")
    lines.append("  Node set = {BG, S7, S8, S9, T789}")
    lines.append("  Edge weight between module A and B:")
    lines.append("    w(A,B) = mean_{a in A, b in B} |corr(a,b)|  (A!=B)")
    lines.append("    w(A,A) would be within-module mean |corr| (not used as drawn edges).")
    lines.append("")
    lines.append("  Current inter-module means:")
    for _, r in (
        W.stack()
        .reset_index()
        .rename(columns={"level_0": "A", "level_1": "B", 0: "mean_abs_r"})
        .query("A != B")
        .sort_values("mean_abs_r", ascending=False)
        .iterrows()
    ):
        lines.append(f"    {r['A']}-{r['B']}: {r['mean_abs_r']:.3f}")
    lines.append("")
    lines.append("[4] Why BG links may look small")
    lines.append("  BG has 15 edges among {1..6}.")
    lines.append("  BG-to-S8 uses 15x6=90 cross terms, many with weak relation, so mean shrinks.")
    lines.append("  In this dataset, BG_to_S8 > BG_to_T789, and BG_to_{S7,S8,S9} are all > BG_to_T789.")
    lines.append("  So the statement 'S7/S8/S9 reflect BG more than T789 does' can still hold.")
    lines.append("")
    lines.append("[5] Current key output")
    lines.append("  - ges_box_jitter_module_within.pdf")
    lines.append("  - mod4b_module_network.pdf")
    lines.append("  - module_within_pair_distribution.csv")
    lines.append("  - bg_link_distribution.csv")

    (out / "derivation_note_mod4b.txt").write_text("\n".join(lines), encoding="utf-8")


def plot_module_network_4b(R: pd.DataFrame, out_file: Path) -> pd.DataFrame:
    """
    4b-style module network:
    - BG, S7, S8, S9, T789 as nodes
    - edge weight = mean absolute correlation between modules
    """
    W = module_strength_matrix_abs(R)
    keys = ["BG", "S7", "S8", "S9", "T789"]

    positions = {
        "BG": (-1.2, 0.0),
        "S7": (0.0, 0.9),
        "S8": (0.6, 0.0),
        "S9": (0.0, -0.9),
        "T789": (1.7, 0.0),
    }
    node_color = {"BG": "#9e9e9e", "S7": "#e76f51", "S8": "#f4a261", "S9": "#2a9d8f", "T789": "#457b9d"}
    node_size = {"BG": 2300, "S7": 1500, "S8": 1500, "S9": 1500, "T789": 1200}

    all_vals = []
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if j <= i:
                continue
            all_vals.append(float(W.loc[a, b]))
    med = float(np.median(all_vals))

    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if j <= i:
                continue
            w = float(W.loc[a, b])
            xa, ya = positions[a]
            xb, yb = positions[b]
            edge_color = "#2a9d8f" if w >= med else "#9aa0a6"
            width = 0.6 + 6.0 * w
            alpha = 0.5 + 0.5 * w
            ax.plot([xa, xb], [ya, yb], color=edge_color, linewidth=width, alpha=alpha, zorder=1)
            ax.text((xa + xb) / 2, (ya + yb) / 2, f"{w:.2f}", fontsize=9, ha="center", va="center")

    for n in keys:
        x, y = positions[n]
        ax.scatter(
            [x],
            [y],
            s=node_size[n],
            c=node_color[n],
            alpha=0.95,
            edgecolors="black",
            linewidths=1.2,
            zorder=2,
        )
        ax.text(x, y, n, fontsize=12, fontweight="bold", ha="center", va="center", color="black")

    ax.set_title("4b-style module network: BG vs S7/S8/S9 vs T789 (edge labels = mean |r|)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_file, dpi=250, bbox_inches="tight")
    plt.close(fig)

    edge_rows = []
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if j <= i:
                continue
            edge_rows.append({"module_a": a, "module_b": b, "mean_abs_r": float(W.loc[a, b])})
    return pd.DataFrame(edge_rows).sort_values("mean_abs_r", ascending=False).reset_index(drop=True)


def plot_hub_coactivation_network(R: pd.DataFrame, out_file: Path) -> None:
    """
    Aggregated 4b-style hub coactivation network.

    Nodes:
      - one aggregated "1-6" node
      - hub nodes 7, 8, 9

    Edge widths:
      - 1-6 -> h is mean within-submatrix corr among {J_1,h, ..., J_6,h}
        This captures statements like:
        corr(J_3,8, J_5,8) is high, so interactions sharing hub 8 co-move.
      - 7/8/9 triangle edges use corr among J_7,8 / J_7,9 / J_8,9,
        showing whether the closed 789 triangle itself co-moves.
    """
    from matplotlib.patches import FancyArrowPatch

    pos = {
        "BG": (0.0, 0.0),
        "7": (2.9, 1.75),
        "8": (4.0, 0.0),
        "9": (2.9, -1.75),
    }
    node_colors = {"BG": "#9e9e9e", "7": "#2f66c5", "8": "#e64b35", "9": "#2f7d3b"}

    def star_mean(h: int) -> float:
        vals = [
            float(R.loc[edge_name(i, h), edge_name(j, h)])
            for i, j in combinations(BACKGROUND, 2)
        ]
        return float(np.mean(vals))

    bg_to_hub = {str(h): star_mean(h) for h in (7, 8, 9)}

    triangle = [edge_name(7, 8), edge_name(7, 9), edge_name(8, 9)]
    triangle_weights = {
        ("7", "8"): float(abs(R.loc[edge_name(7, 8), edge_name(7, 9)])),
        ("7", "9"): float(abs(R.loc[edge_name(7, 8), edge_name(8, 9)])),
        ("8", "9"): float(abs(R.loc[edge_name(7, 9), edge_name(8, 9)])),
    }

    all_weights = list(bg_to_hub.values()) + list(triangle_weights.values())
    max_w = max(all_weights) if all_weights else 1.0

    def width(w: float) -> float:
        return 0.8 + 9.0 * (w / max_w if max_w > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(8.2, 5.7))

    # Main 1-6 -> 7/8/9 edges.
    for hub_label, w in bg_to_hub.items():
        x1, y1 = pos["BG"]
        x2, y2 = pos[hub_label]
        rad = {"7": 0.10, "8": 0.0, "9": -0.10}[hub_label]
        patch = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-",
            linewidth=width(w),
            color=node_colors[hub_label],
            alpha=0.72,
            zorder=1,
            capstyle="round",
        )
        ax.add_patch(patch)
        ax.text(
            (x1 + x2) / 2 - 0.15,
            (y1 + y2) / 2 + (0.12 if hub_label != "9" else -0.18),
            f"S{hub_label} within\nmean r={w:.2f}",
            color=node_colors[hub_label],
            fontsize=9,
            ha="center",
            va="center",
        )

    # Closed 789 triangle: drawn thinner/dashed.
    for (a, b), w in triangle_weights.items():
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        patch = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            connectionstyle="arc3,rad=0.08",
            arrowstyle="-",
            linewidth=max(0.8, width(w) * 0.55),
            color="#3b7d3b",
            alpha=0.45,
            linestyle="--",
            zorder=0,
            capstyle="round",
        )
        ax.add_patch(patch)
        ax.text((x1 + x2) / 2 + 0.12, (y1 + y2) / 2, f"{a}-{b}\n{w:.2f}", fontsize=8, color="#2f5f34")

    # Nodes.
    node_sizes = {"BG": 3000, "7": 1900, "8": 2100, "9": 1900}
    node_labels = {"BG": "1-6", "7": "7", "8": "8", "9": "9"}
    for node, (x, y) in pos.items():
        ax.scatter(
            [x],
            [y],
            s=node_sizes[node],
            color=node_colors[node],
            edgecolors="black",
            linewidths=1.4,
            zorder=3,
        )
        ax.text(
            x,
            y,
            node_labels[node],
            fontsize=18 if node != "BG" else 16,
            fontweight="bold",
            ha="center",
            va="center",
            color="white" if node != "BG" else "black",
            zorder=4,
        )

    ax.text(
        2.2,
        2.65,
        "Edge width from 1-6 to hub h = mean corr(J_i,h, J_j,h), i,j in 1..6",
        fontsize=10.5,
        ha="center",
    )
    ax.text(
        4.25,
        -2.65,
        "Dashed green edges: co-movement inside the closed 7-8-9 triangle",
        fontsize=9,
        color="#2f5f34",
        ha="center",
    )

    ax.set_xlim(-1.1, 5.25)
    ax.set_ylim(-3.05, 3.05)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_file, dpi=250, bbox_inches="tight")
    plt.close(fig)


def load_matrix(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        A = np.load(path)
        return np.asarray(A, dtype=float)
    if path.suffix.lower() in {".csv", ".txt"}:
        # Keep it simple: plain numeric file only.
        A = np.loadtxt(path, delimiter=",")
        return np.asarray(A, dtype=float)
    raise ValueError("input must be .npy or .csv/.txt")


def load_feature_names(path: Path) -> list[str]:
    txt = path.read_text(encoding="utf-8").strip().splitlines()
    names = [line.strip() for line in txt if line.strip()]
    return names


def run_analysis(
    matrix,
    matrix_kind: str = "raw",
    feature_names: Sequence[str] | None = None,
    outdir: str | Path = "figs_for_paper/hub_submatrix_analysis",
) -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    R = to_correlation(matrix, matrix_kind=matrix_kind, feature_names=feature_names)
    summary = hub_summary(R)
    pair_tables = {hub: hub_pair_table(R, hub) for hub in HUBS}
    # within-submatrix definition, now including the closed triangle (T789) and background (BG)
    within_mod_df = module_within_pair_distribution(
        R, module_keys=("S7", "S8", "S9", "T789", "BG")
    )
    network_df = module_edge_table_from_strength(module_strength_matrix_abs(R))

    # Save a small set of CSVs only
    summary.to_csv(out / "hub_summary.csv", index=False)
    within_mod_df.to_csv(out / "module_within_pair_distribution.csv", index=False)

    # Save figures
    plot_module_within_box_jitter(within_mod_df, out / "ges_box_jitter_module_within.pdf")
    plot_hub_coactivation_network(R, out / "hub_coactivation_network.pdf")

    return {
        "R": R,
        "hub_summary": summary,
        "pair_tables": pair_tables,
        "module_within_distribution": within_mod_df,
        "network_edges": network_df,
        "outdir": out,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Simple Jij hub submatrix analysis")
    p.add_argument("--input", type=str, required=True, help="Path to matrix (.npy or .csv)")
    p.add_argument(
        "--matrix-kind",
        type=str,
        default="raw",
        choices=["raw", "cov", "corr"],
        help="Interpret input as raw(N,36), covariance(36,36), or correlation(36,36)",
    )
    p.add_argument(
        "--feature-names",
        type=str,
        default=None,
        help="Optional txt file: one Jij label per line in matrix order",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="figs_for_paper/hub_submatrix_analysis",
        help="Output directory",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    input_path = Path(args.input)
    A = load_matrix(input_path)

    names = None
    if args.feature_names is not None:
        names = load_feature_names(Path(args.feature_names))

    res = run_analysis(
        A,
        matrix_kind=args.matrix_kind,
        feature_names=names,
        outdir=args.outdir,
    )
    print("analysis completed")
    print(f"outdir: {res['outdir']}")
    print(res["hub_summary"])


if __name__ == "__main__":
    main()
