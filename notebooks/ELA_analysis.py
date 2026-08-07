"""
Energy Landscape Analysis (ELA) - 汎用的な結果解析と可視化モジュール

このモジュールは、NumPyベースの汎用的なランドスケープ解析と可視化を提供します。
VEM-MEMモデル（VEM_MEM.py）とは独立して使用可能です。

主な機能:
- LandscapeEngine: NumPyベースのランドスケープ計算
- resultsAnalyzer: 結果解析と可視化
- TrajectoryAnalyzer: 軌跡解析（観測vs理論の比較）
- TrajectoryComparator: 個人vs集団ランドスケープの比較（VEM-MEMのLandscapeAnalyzer使用）
- TrajectoryVisualizer: 遷移行列・ネットワーク可視化
- ScoreAssociation: スコア関連付け
- plot_net_flux_arrows: ネットフラックス可視化
- build_dual_traj_df_all: 全被験者の個人vs集団軌跡解析（TrajectoryComparatorを使用）
- per_subject_kinetics: 被験者ごとの動力学統計計算

Note: VEM-MEMモデル（JAXベース）の解析には VEM_MEM.py の LandscapeAnalyzer を使用してください。
TrajectoryComparator と build_dual_traj_df_all は VEM-MEM の LandscapeAnalyzer を使用しますが、
このファイル（ELA_analysis.py）に配置されています。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os

import math
import heapq
import numpy as np

# plotting
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import ListedColormap
import seaborn as sns
import networkx as nx

# optional tables
try:
    import pandas as pd
except Exception:
    pd = None

sns.set_theme(style="white", context="talk")


# =========================================================
# 0) State codec (string/binary/index)
# =========================================================
class StateCodec:
    """
    State encoding convention:
      - state index s is an integer in [0, 2^n)
      - bit i corresponds to item i
      - by default item0 is LSB (little-endian)
    Example (n=9):
      s=5 -> bits 000000101 (item0=1, item2=1)
    """

    def __init__(self, n: int, bit_order: str = "little"):
        if bit_order not in ("little", "big"):
            raise ValueError("bit_order must be 'little' or 'big'")
        self.n = n
        self.bit_order = bit_order

    def index_to_bits(self, s: int) -> np.ndarray:
        bits = np.zeros(self.n, dtype=np.int8)
        for i in range(self.n):
            if self.bit_order == "little":
                bits[i] = (s >> i) & 1
            else:
                bits[i] = (s >> (self.n - 1 - i)) & 1
        return bits

    def bits_to_index(self, bits: np.ndarray) -> int:
        b = np.asarray(bits).astype(int).reshape(-1)
        if b.size != self.n:
            raise ValueError(f"bits length mismatch: got {b.size}, expected {self.n}")
        s = 0
        for i in range(self.n):
            if self.bit_order == "little":
                s |= (int(b[i]) & 1) << i
            else:
                s |= (int(b[i]) & 1) << (self.n - 1 - i)
        return int(s)

    def str_to_index(self, bitstr: str) -> int:
        bitstr = bitstr.strip()
        if len(bitstr) != self.n or any(c not in "01" for c in bitstr):
            raise ValueError(f"bitstr must be length {self.n} of 0/1, got: {bitstr}")
        # interpret bitstr as item0..item(n-1) in order
        bits = np.array([int(c) for c in bitstr], dtype=np.int8)
        return self.bits_to_index(bits)

    def index_to_str(self, s: int) -> str:
        bits = self.index_to_bits(s)
        return "".join(str(int(x)) for x in bits)


# =========================================================
# 1) Landscape engine (enumeration, basins, transitions, barriers)
# =========================================================
def _neighbors_hamming1(n: int) -> List[List[int]]:
    N = 1 << n
    neigh = [[] for _ in range(N)]
    for s in range(N):
        for i in range(n):
            neigh[s].append(s ^ (1 << i))
    return neigh

def _enumerate_states(n: int) -> np.ndarray:
    S = np.zeros((1 << n, n), dtype=np.int8)
    for s in range(1 << n):
        for i in range(n):
            S[s, i] = (s >> i) & 1
    return S

def _build_design_matrix(S: np.ndarray) -> np.ndarray:
    """
    X columns: [sigma_0..sigma_{n-1}, sigma_0*sigma_1, sigma_0*sigma_2, ..., sigma_{n-2}*sigma_{n-1}]
    If theta = (h, J_upper) with that pair ordering, then:
      E(s) = - X(s) @ theta
    """
    S = S.astype(np.float32)
    n = S.shape[1]
    m = n * (n - 1) // 2
    X = np.zeros((S.shape[0], n + m), dtype=np.float32)
    X[:, :n] = S
    idx = n
    for i in range(n - 1):
        for j in range(i + 1, n):
            X[:, idx] = S[:, i] * S[:, j]
            idx += 1
    return X

def _energies(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float32).reshape(-1)
    if theta.size != X.shape[1]:
        raise ValueError(f"theta length mismatch: got {theta.size}, expected {X.shape[1]}")
    return (-(X @ theta)).astype(np.float64)

def _boltzmann_probs(E: np.ndarray) -> np.ndarray:
    Emin = np.min(E)
    w = np.exp(-(E - Emin))
    return w / np.sum(w)

def _local_minima(E: np.ndarray, neigh: List[List[int]]) -> np.ndarray:
    mins = []
    for s in range(E.size):
        e = E[s]
        if all(e <= E[t] for t in neigh[s]):
            mins.append(s)
    return np.array(mins, dtype=int)

def _assign_basins_steepest(E: np.ndarray, neigh: List[List[int]]) -> np.ndarray:
    """
    Deterministic steepest descent. basin_min[s] is the attractor state's index.
    """
    N = E.size
    next_state = np.arange(N, dtype=int)
    for s in range(N):
        best = s
        bestE = E[s]
        for t in neigh[s]:
            if E[t] < bestE - 1e-15:
                best = t
                bestE = E[t]
        next_state[s] = best

    basin_min = np.full(N, -1, dtype=int)

    def find_attr(s: int) -> int:
        path = []
        while basin_min[s] == -1:
            path.append(s)
            ns = next_state[s]
            if ns == s:
                basin_min[s] = s
                break
            s = ns
        attr = basin_min[s]
        for u in path:
            basin_min[u] = attr
        return attr

    for s in range(N):
        if basin_min[s] == -1:
            find_attr(s)
    return basin_min

def _basin_stats(E: np.ndarray, P: np.ndarray, basin_min: np.ndarray) -> Dict[int, Dict[str, float]]:
    stats: Dict[int, Dict[str, float]] = {}
    mins = np.unique(basin_min)
    for m in mins:
        idx = np.where(basin_min == m)[0]
        stats[int(m)] = {
            "size": float(idx.size),
            "mass": float(np.sum(P[idx])),
            "Emin": float(E[int(m)]),
        }
    return stats

def _scalar_features(E: np.ndarray, P: np.ndarray, bstats: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    mins = sorted(bstats.keys(), key=lambda m: (bstats[m]["Emin"], -bstats[m]["mass"]))
    n_min = len(mins)
    g = mins[0]
    global_E = bstats[g]["Emin"]
    global_mass = bstats[g]["mass"]
    global_size = bstats[g]["size"]

    if n_min >= 2:
        second_E = bstats[mins[1]]["Emin"]
        deltaE = second_E - global_E
    else:
        second_E = float("nan")
        deltaE = float("nan")

    masses = np.array([bstats[m]["mass"] for m in mins], dtype=float)
    masses = masses[masses > 0]
    H = float(-np.sum(masses * np.log(masses))) if masses.size else 0.0

    return {
        "n_minima": float(n_min),
        "global_min_E": float(global_E),
        "second_min_E": float(second_E),
        "deltaE_2_1": float(deltaE),
        "global_basin_mass": float(global_mass),
        "global_basin_size": float(global_size),
        "basin_mass_entropy": float(H),
    }

def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = (p > 0) & (q > 0)
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

def _js(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

def _ecorr(E1: np.ndarray, E2: np.ndarray) -> float:
    x = E1 - E1.mean()
    y = E2 - E2.mean()
    denom = np.sqrt(np.sum(x*x) * np.sum(y*y))
    return float(np.sum(x*y) / denom) if denom > 0 else float("nan")


# ---- minimax (minimum barrier) path
def _minimax_costs_prev(E: np.ndarray, neigh: List[List[int]], start: int) -> Tuple[np.ndarray, np.ndarray]:
    N = E.size
    cost = np.full(N, np.inf, dtype=float)
    prev = np.full(N, -1, dtype=int)
    cost[start] = E[start]
    pq = [(cost[start], start)]
    while pq:
        c, u = heapq.heappop(pq)
        if c > cost[u] + 1e-15:
            continue
        for v in neigh[u]:
            nc = max(c, E[v])
            if nc < cost[v] - 1e-15:
                cost[v] = nc
                prev[v] = u
                heapq.heappush(pq, (nc, v))
    return cost, prev

def _reconstruct(prev: np.ndarray, start: int, goal: int) -> List[int]:
    path = []
    v = goal
    while v != -1 and v != start:
        path.append(v)
        v = prev[v]
    if v == -1:
        return []
    path.append(start)
    path.reverse()
    return path

def _barrier_between(E: np.ndarray, neigh: List[List[int]], a: int, b: int) -> Dict[str, Any]:
    cost, prev = _minimax_costs_prev(E, neigh, a)
    saddle_energy = float(cost[b])
    path = _reconstruct(prev, a, b)
    if not path:
        return {"saddle_energy": np.inf, "barrier_from_a": np.inf, "barrier_from_b": np.inf,
                "path": [], "saddle_state": None}
    epath = E[path]
    saddle_state = int(path[int(np.argmax(epath))])
    return {
        "saddle_energy": saddle_energy,
        "barrier_from_a": saddle_energy - float(E[a]),
        "barrier_from_b": saddle_energy - float(E[b]),
        "path": path,
        "saddle_state": saddle_state,
    }

def _coarse_basin_transition(E: np.ndarray, P: np.ndarray, basin_min: np.ndarray, neigh: List[List[int]]) -> Tuple[np.ndarray, np.ndarray, Dict[int,int], List[int]]:
    """
    Metropolis single-bit-flip proposal; aggregate stationary flux into basin transitions.
    Output:
      T_basin (B,B), pi_basin (B,), basin_index (attractor->id), mins (list of attractor indices)
    """
    mins = np.unique(basin_min).astype(int).tolist()
    B = len(mins)
    basin_index = {m: k for k, m in enumerate(mins)}
    basin_id = np.array([basin_index[int(basin_min[s])] for s in range(E.size)], dtype=int)

    pi_basin = np.zeros(B, dtype=float)
    for s in range(E.size):
        pi_basin[basin_id[s]] += P[s]

    flux = np.zeros((B, B), dtype=float)
    deg = len(neigh[0])

    for i in range(E.size):
        a = basin_id[i]
        out = 0.0
        for j in neigh[i]:
            dE = E[j] - E[i]
            acc = 1.0 if dE <= 0 else math.exp(-dE)
            p = (1.0 / deg) * acc
            b = basin_id[j]
            flux[a, b] += P[i] * p
            out += p
        flux[a, a] += P[i] * (1.0 - out)

    T_basin = np.zeros((B, B), dtype=float)
    for a in range(B):
        if pi_basin[a] > 0:
            T_basin[a, :] = flux[a, :] / pi_basin[a]

    return T_basin, pi_basin, basin_index, mins


@dataclass
class Landscape:
    theta: np.ndarray
    E: np.ndarray
    P: np.ndarray
    basin_min: np.ndarray                 # per-state -> attractor (state index)
    basin_stats: Dict[int, Dict[str, float]]
    minima: np.ndarray                    # local minima state indices
    # basin-level (computed on demand)
    T_basin: Optional[np.ndarray] = None
    pi_basin: Optional[np.ndarray] = None
    basin_index: Optional[Dict[int, int]] = None
    basin_attractors: Optional[List[int]] = None


class LandscapeEngine:
    """
    Computes landscapes on the full 2^n state space (n=9 => 512 states).
    NumPyベースの汎用的なランドスケープ計算エンジン。
    
    theta ordering:
      [h_0..h_{n-1}, J_{0,1}, J_{0,2}, ..., J_{0,n-1}, J_{1,2}, ..., J_{n-2,n-1}]
    
    Note: VEM-MEMモデル（JAXベース）を使用する場合は、
    VEM_MEM.py の LandscapeAnalyzer を使用してください。
    """

    def __init__(self, n: int = 9, bit_order: str = "little"):
        self.n = n
        self.codec = StateCodec(n=n, bit_order=bit_order)
        self.neigh = _neighbors_hamming1(n)
        self.S = _enumerate_states(n)          # little-endian bits for energy calculation
        self.X = _build_design_matrix(self.S)  # for E = -X@theta

    def build(self, theta: np.ndarray, compute_basin_transition: bool = True) -> Landscape:
        E = _energies(theta, self.X)
        P = _boltzmann_probs(E)
        basin_min = _assign_basins_steepest(E, self.neigh)
        bstats = _basin_stats(E, P, basin_min)
        mins = _local_minima(E, self.neigh)

        L = Landscape(theta=np.asarray(theta).reshape(-1), E=E, P=P,
                      basin_min=basin_min, basin_stats=bstats, minima=mins)

        if compute_basin_transition:
            T_basin, pi_basin, basin_index, mins_list = _coarse_basin_transition(E, P, basin_min, self.neigh)
            L.T_basin = T_basin
            L.pi_basin = pi_basin
            L.basin_index = basin_index
            L.basin_attractors = mins_list
        return L

    def features(self, L: Landscape) -> Dict[str, float]:
        return _scalar_features(L.E, L.P, L.basin_stats)

    def distances(self, A: Landscape, B: Landscape) -> Dict[str, float]:
        return {
            "JS": _js(A.P, B.P),
            "KL_A_to_B": _kl(A.P, B.P),
            "KL_B_to_A": _kl(B.P, A.P),
            "E_corr": _ecorr(A.E, B.E),
            "delta_n_minima": float(len(A.minima) - len(B.minima)),
            "delta_global_min_E": float(np.min(A.E) - np.min(B.E)),
        }

    def barrier(self, L: Landscape, a: int, b: int) -> Dict[str, Any]:
        return _barrier_between(L.E, self.neigh, a, b)

    def barrier_matrix_top_minima(self, L: Landscape, top: int = 8, by: str = "mass") -> Dict[str, Any]:
        mins = list(L.basin_stats.keys())
        if by == "mass":
            mins_sorted = sorted(mins, key=lambda m: (-L.basin_stats[m]["mass"], L.basin_stats[m]["Emin"]))
        elif by == "energy":
            mins_sorted = sorted(mins, key=lambda m: (L.basin_stats[m]["Emin"], -L.basin_stats[m]["mass"]))
        else:
            raise ValueError("by must be 'mass' or 'energy'")

        mins_sel = mins_sorted[:min(top, len(mins_sorted))]
        B = len(mins_sel)
        saddle = np.zeros((B, B), dtype=float)
        barrier = np.zeros((B, B), dtype=float)

        for i, a in enumerate(mins_sel):
            cost, _ = _minimax_costs_prev(L.E, self.neigh, int(a))
            for j, b in enumerate(mins_sel):
                sE = float(cost[int(b)])
                saddle[i, j] = sE
                barrier[i, j] = sE - float(L.E[int(a)])

        return {"minima": mins_sel, "saddle_energy": saddle, "barrier_from_row_min": barrier}


# =========================================================
# 2) Batch analyzer for Results 3.1-3.3 (+ plots)
# =========================================================
class resultsAnalyzer:
    """
    Produces:
      3.1 group landscape summaries & figures
      3.2 distributions of individual features
      3.3 individual-vs-group distances (JS etc)
    """

    def __init__(self, n: int = 9, bit_order: str = "little"):
        self.engine = LandscapeEngine(n=n, bit_order=bit_order)
        self.group: Optional[Landscape] = None
        self.subjects: Dict[int, Landscape] = {}

    def set_group(self, eta: np.ndarray) -> None:
        self.group = self.engine.build(eta, compute_basin_transition=True)

    def build_subjects(self, mu: np.ndarray, subject_ids: Optional[List[int]] = None, compute_basin_transition: bool = False) -> None:
        if subject_ids is None:
            subject_ids = list(range(mu.shape[0]))
        for k in subject_ids:
            if k not in self.subjects:
                self.subjects[k] = self.engine.build(mu[k], compute_basin_transition=compute_basin_transition)

    # ---------- tables ----------
    def feature_distance_table(
        self,
        mu: np.ndarray,
        eta: np.ndarray,
        subject_ids: Optional[List[int]] = None,
        compute_basin_transition_subjects: bool = False,
    ):
        if self.group is None:
            self.set_group(eta)
        self.build_subjects(mu, subject_ids=subject_ids, compute_basin_transition=compute_basin_transition_subjects)

        g = self.group
        g_feat = self.engine.features(g)

        rows = []
        for k, Lk in self.subjects.items():
            f = self.engine.features(Lk)
            d = self.engine.distances(Lk, g)
            row = {"subject_id": k}
            for kk, vv in f.items():
                row[kk] = vv
                row[f"{kk}_minus_group"] = vv - g_feat[kk]
            for kk, vv in d.items():
                row[f"vs_group_{kk}"] = vv
            rows.append(row)

        if pd is not None:
            return pd.DataFrame(rows).sort_values("subject_id")
        return rows

    # ---------- group summaries ----------
    def group_top_basins(self, top: int = 10, by: str = "mass"):
        if self.group is None:
            raise ValueError("Group not set. Call set_group(eta).")
        g = self.group
        mins = list(g.basin_stats.keys())
        if by == "mass":
            mins = sorted(mins, key=lambda m: (-g.basin_stats[m]["mass"], g.basin_stats[m]["Emin"]))
        else:
            mins = sorted(mins, key=lambda m: (g.basin_stats[m]["Emin"], -g.basin_stats[m]["mass"]))
        mins = mins[:min(top, len(mins))]

        rows = []
        for m in mins:
            rows.append({
                "attractor_state": int(m),
                "bitstring": self.engine.codec.index_to_str(int(m)),
                "mass": g.basin_stats[int(m)]["mass"],
                "size": g.basin_stats[int(m)]["size"],
                "Emin": g.basin_stats[int(m)]["Emin"],
            })
        if pd is not None:
            return pd.DataFrame(rows)
        return rows

    # ---------- plotting (seaborn) ----------
    def plot_group_basins(self, top: int = 10, by: str = "mass", savefig: bool = False, filename: Optional[str] = None) -> None:
        tab = self.group_top_basins(top=top, by=by)
        if pd is None:
            raise RuntimeError("pandas is recommended for plotting tables. Please install pandas.")
        df = tab.copy()

        plt.figure(figsize=(max(8, 0.7 * len(df)), 5))
        ax = sns.barplot(data=df, x="attractor_state", y="mass", hue="Emin", dodge=False, palette="vlag")
        ax.set_title(f"Group: top basins ({by})")
        ax.set_xlabel("attractor state index")
        ax.set_ylabel("basin probability mass")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="E(min)", frameon=True)
        sns.despine()
        plt.tight_layout()
        if savefig:
            if filename is None:
                filename = f"group_basins_top{top}_{by}.pdf"
            os.makedirs('./figs_for_paper/', exist_ok=True)
            plt.savefig(f'./figs_for_paper/{filename}', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_group_transition_heatmap(
        self,
        top: int = 12,
        normalize: str = "row",   # "row"|"none"
        annotate: bool = False,
        diag_zero: bool = True,
        savefig: bool = False,
        filename: Optional[str] = None,
    ) -> None:
        if self.group is None or self.group.T_basin is None:
            raise ValueError("Group transition not computed. Call set_group(eta) first.")
        T = self.group.T_basin
        pi = self.group.pi_basin
        mins = self.group.basin_attractors

        idx = np.argsort(-pi)[:min(top, len(pi))]
        Tsub = T[np.ix_(idx, idx)].astype(float)
        labels = [str(mins[i]) for i in idx]

        if normalize == "row":
            denom = Tsub.sum(axis=1, keepdims=True); denom[denom == 0] = 1.0
            Tshow = Tsub / denom
            cbar_label = "P(to | from) row-normalized"
        else:
            Tshow = Tsub
            cbar_label = "transition probability"

        if diag_zero:
            Tshow = Tshow.copy()
            np.fill_diagonal(Tshow, 0.0)

        fig, ax = plt.subplots(figsize=(max(7, 0.6 * len(idx)), max(6, 0.6 * len(idx))))
        sns.heatmap(
            Tshow, ax=ax, cmap="coolwarm", square=True,
            linewidths=0.5, linecolor="white",
            annot=annotate, fmt=".2f",
            vmin=0.0, vmax=None,
            cbar_kws={"label": cbar_label, "shrink": 0.85},
        )
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels, rotation=0)
        ax.set_title("Group: coarse-grained basin transitions")
        ax.set_xlabel("to basin (attractor index)")
        ax.set_ylabel("from basin (attractor index)")
        plt.tight_layout()
        if savefig:
            if filename is None:
                filename = f"group_transition_heatmap_top{top}_norm{normalize}_diagzero{diag_zero}.pdf"
            os.makedirs('./figs_for_paper/', exist_ok=True)
            plt.savefig(f'./figs_for_paper/{filename}', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_feature_distributions(self, df_feat, features: List[str], savefig: bool = False, filename: Optional[str] = None) -> None:
        """
        df_feat: output of feature_distance_table()
        """
        if pd is None:
            raise RuntimeError("pandas required for this plot.")
        df = df_feat.copy()
        long = df.melt(id_vars=["subject_id"], value_vars=features, var_name="feature", value_name="value")

        plt.figure(figsize=(max(9, 1.2 * len(features)), 6))
        ax = sns.violinplot(data=long, x="feature", y="value", inner="quartile", cut=0)
        ax.set_title("Individual-level feature distributions")
        ax.tick_params(axis="x", rotation=30)
        sns.despine()
        plt.tight_layout()
        if savefig:
            if filename is None:
                feat_str = "_".join(features[:3])  # 最初の3つをファイル名に使用
                if len(features) > 3:
                    feat_str += "_etc"
                filename = f"feature_distributions_{feat_str}.pdf"
            os.makedirs('./figs_for_paper/', exist_ok=True)
            plt.savefig(f'./figs_for_paper/{filename}', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_distance_distribution(self, df_feat, col: str = "vs_group_JS", savefig: bool = False, filename: Optional[str] = None) -> None:
        if pd is None:
            raise RuntimeError("pandas required for this plot.")
        plt.figure(figsize=(8, 5))
        ax = sns.histplot(df_feat[col], bins=30, kde=True)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        sns.despine()
        plt.tight_layout()
        if savefig:
            if filename is None:
                filename = f"distance_distribution_{col}.pdf"
            os.makedirs('./figs_for_paper/', exist_ok=True)
            plt.savefig(f'./figs_for_paper/{filename}', dpi=300, bbox_inches='tight')
        plt.show()

    def pick_cases(self, df_feat, col: str = "vs_group_JS", n_each: int = 3) -> Dict[str, List[int]]:
        """
        Choose exemplars:
          - most different from group (top n)
          - most similar to group (bottom n)
        """
        if pd is None:
            # df_feat may be list[dict]
            vals = [(r["subject_id"], r[col]) for r in df_feat]
            vals_sorted = sorted(vals, key=lambda x: x[1])
            return {
                "most_similar": [k for k, _ in vals_sorted[:n_each]],
                "most_different": [k for k, _ in vals_sorted[-n_each:]][::-1],
            }
        df = df_feat.sort_values(col)
        return {
            "most_similar": df["subject_id"].head(n_each).astype(int).tolist(),
            "most_different": df["subject_id"].tail(n_each).astype(int).tolist()[::-1],
        }

    def plot_disconnectivity_like(self, use_group: bool = True, k: Optional[int] = None, top: int = 8, by: str = "mass", savefig: bool = False, filename: Optional[str] = None) -> None:
        """
        Dendrogram if scipy available, else saddle-energy heatmap.
        """
        if use_group:
            if self.group is None:
                raise ValueError("Group not set.")
            L = self.group
            title = "Group"
            prefix = "group"
        else:
            if k is None or k not in self.subjects:
                raise ValueError("Subject not built.")
            L = self.subjects[k]
            title = f"Subject {k}"
            prefix = f"subject{k}"

        res = self.engine.barrier_matrix_top_minima(L, top=top, by=by)
        mins = res["minima"]
        saddle = res["saddle_energy"]

        try:
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import squareform
            D = 0.5 * (saddle + saddle.T)
            cond = squareform(D, checks=False)
            Z = linkage(cond, method="average")
            plt.figure(figsize=(9, 5))
            dendrogram(Z, labels=[str(m) for m in mins])
            plt.ylabel("saddle energy")
            plt.title(f"{title}: disconnectivity-like dendrogram (top={top}, by={by})")
            plt.tight_layout()
            if savefig:
                if filename is None:
                    filename = f"{prefix}_disconnectivity_top{top}_{by}.pdf"
                os.makedirs('./figs_for_paper/', exist_ok=True)
                plt.savefig(f'./figs_for_paper/{filename}', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception:
            plt.figure(figsize=(7, 6))
            ax = sns.heatmap(saddle, cmap="rocket_r", square=True, cbar_kws={"label": "saddle energy"})
            ax.set_xticklabels([str(m) for m in mins], rotation=45, ha="right")
            ax.set_yticklabels([str(m) for m in mins], rotation=0)
            ax.set_title(f"{title}: saddle energy matrix (fallback)")
            plt.tight_layout()
            if savefig:
                if filename is None:
                    filename = f"{prefix}_saddle_energy_matrix_top{top}_{by}.pdf"
                os.makedirs('./figs_for_paper/', exist_ok=True)
                plt.savefig(f'./figs_for_paper/{filename}', dpi=300, bbox_inches='tight')
            plt.show()

    def basin_transition(self, use_group: bool = True, k: Optional[int] = None) -> Dict[str, Any]:
        """
        Returns T_basin, pi_basin, and mins for group or individual subject.
        """
        if use_group:
            if self.group is None:
                raise ValueError("Group not set. Call set_group(eta) first.")
            L = self.group
        else:
            if k is None or k not in self.subjects:
                raise ValueError(f"Subject {k} not built.")
            L = self.subjects[k]
        
        if L.T_basin is None or L.pi_basin is None or L.basin_attractors is None:
            raise ValueError("Basin transition not computed. Ensure compute_basin_transition=True when building landscape.")
        
        return {
            "T_basin": L.T_basin,
            "pi_basin": L.pi_basin,
            "mins": L.basin_attractors,
        }

    def plot_net_flux_arrows(
        self,
        use_group: bool = True,
        k: Optional[int] = None,
        top_k_edges: int | None = None,
        min_abs_net: float | None = None,
        curvature: float = 0.25,
        node_size: float = 800,
        node_alpha: float = 0.95,
        width_scale: float = 50.0,
        text_scale: float = 11,
        annotate_edges: bool = True,
        arrow_color: str = "black",
        min_linewidth: float = 1.5,  # 最小線幅
        max_linewidth: float | None = None,  # 最大線幅（Noneの場合は制限なし）
        width_power: float = 1.0,  # 線幅のスケーリング（1.0=線形、2.0=2乗、0.5=平方根など）
        savefig: bool = False,
        filename: Optional[str] = None,
        ax=None,
    ) -> None:
        """
        Draw net flux arrows for coarse-grained basins.
        
        Args:
            use_group: If True, use group landscape; if False, use subject k
            k: Subject ID (required if use_group=False)
            top_k_edges: Keep only largest |net flux| edges (i<j)
            min_abs_net: Threshold on |net flux|
            curvature: Arrow curvature
            node_size: Node size for scatter plot
            node_alpha: Node alpha transparency
            width_scale: Scales arrow thickness (base scale)
            text_scale: Font size for node labels
            annotate_edges: Whether to annotate edges with net flux values
            arrow_color: Arrow color (e.g., "black", "red", "#FF0000", "blue", etc.)
            min_linewidth: Minimum line width (default: 1.5)
            max_linewidth: Maximum line width (None = no limit)
            width_power: Scaling power for line width (1.0=linear, 2.0=squared, 0.5=sqrt)
            savefig: Whether to save the figure
            filename: Filename for saving (default: auto-generated)
            ax: Matplotlib axes (if None, creates new figure)
        """
        # Get basin transition data
        out = self.basin_transition(use_group=use_group, k=k)
        T = out["T_basin"]
        pi = out["pi_basin"]
        mins = out["mins"]
        
        labels = [str(m) for m in mins]
        
        # Set default filename if not provided
        if savefig and filename is None:
            prefix = "group" if use_group else f"subject{k}"
            filename = f"{prefix}_net_flux_arrows.pdf"
        
        # Call the standalone function
        plot_net_flux_arrows(
            T=T,
            pi=pi,
            labels=labels,
            pos=None,
            top_k_edges=top_k_edges,
            min_abs_net=min_abs_net,
            curvature=curvature,
            node_size=node_size,
            node_alpha=node_alpha,
            width_scale=width_scale,
            text_scale=text_scale,
            annotate_edges=annotate_edges,
            arrow_color=arrow_color,
            min_linewidth=min_linewidth,
            max_linewidth=max_linewidth,
            width_power=width_power,
            savefig=savefig,
            filename=filename,
            ax=ax,
        )


# =========================================================
# 2.5) Net flux visualization (helper functions)
# =========================================================
def circular_layout(n: int, radius: float = 1.0, start_angle: float = np.pi/2):
    """Return dict: node_id -> (x,y) on a circle."""
    pos = {}
    for i in range(n):
        ang = start_angle - 2*np.pi*i/n
        pos[i] = (radius*np.cos(ang), radius*np.sin(ang))
    return pos


def plot_net_flux_arrows(
    T: np.ndarray,
    pi: np.ndarray,
    labels=None,                 # list[str] length B
    pos=None,                    # dict[int,(x,y)] length B; if None, circular layout
    top_k_edges: int | None = None,   # keep only largest |net flux| edges (i<j)
    min_abs_net: float | None = None, # threshold on |net flux|
    curvature: float = 0.25,     # arrow curvature
    node_size: float = 800,
    node_alpha: float = 0.95,
    width_scale: float = 50.0,   # scales arrow thickness
    text_scale: float = 11,
    annotate_edges: bool = True,
    arrow_color: str = "black",  # arrow color (can be color name, hex, or RGB tuple)
    min_linewidth: float = 1.5,  # 最小線幅
    max_linewidth: float | None = None,  # 最大線幅（Noneの場合は制限なし）
    width_power: float = 1.0,  # 線幅のスケーリング（1.0=線形、2.0=2乗、0.5=平方根など）
    savefig: bool = False,
    filename: Optional[str] = None,
    ax=None,
):
    """
    Draw net flux arrows for coarse-grained basins.

    Inputs
      T: (B,B) basin transition matrix (row-stochastic)
      pi: (B,) stationary basin mass (sum=1)
    """
    T = np.asarray(T, dtype=float)
    pi = np.asarray(pi, dtype=float).reshape(-1)
    B = T.shape[0]
    if T.shape != (B, B):
        raise ValueError("T must be square (B,B)")
    if pi.size != B:
        raise ValueError("pi must have length B")

    if labels is None:
        labels = [str(i) for i in range(B)]
    if len(labels) != B:
        raise ValueError("labels must have length B")

    if pos is None:
        pos = circular_layout(B, radius=1.0)

    # flux and net flux
    F = pi[:, None] * T
    net = F - F.T  # antisymmetric

    # collect undirected pairs (i<j) with magnitude
    pairs = []
    for i in range(B):
        for j in range(i+1, B):
            val = net[i, j]
            pairs.append((abs(val), i, j, val))
    pairs.sort(reverse=True, key=lambda x: x[0])

    if top_k_edges is not None:
        pairs = pairs[:top_k_edges]
    if min_abs_net is not None:
        pairs = [p for p in pairs if p[0] >= min_abs_net]

    # set up axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.axis("off")

    # draw nodes
    xs = [pos[i][0] for i in range(B)]
    ys = [pos[i][1] for i in range(B)]
    ax.scatter(xs, ys, s=node_size, alpha=node_alpha)

    # node labels
    for i in range(B):
        x, y = pos[i]
        ax.text(x, y, labels[i], ha="center", va="center", fontsize=text_scale, fontweight="bold")

    # scale arrow widths
    max_abs = pairs[0][0] if pairs else 1.0
    max_abs = max(max_abs, 1e-12)

    # draw arrows
    for mag, i, j, val_ij in pairs:
        # direction: net positive goes i->j, else j->i
        if val_ij >= 0:
            src, dst, net_val = i, j, val_ij
            rad = curvature
        else:
            src, dst, net_val = j, i, -val_ij
            rad = -curvature

        x1, y1 = pos[src]
        x2, y2 = pos[dst]

        # net fluxの値に応じて線の太さを計算（正規化された値を使用）
        normalized_val = net_val / max_abs if max_abs > 0 else 0.0
        # べき乗スケーリングを適用（デフォルトは1.0で線形）
        scaled_val = normalized_val ** width_power
        lw = width_scale * scaled_val
        
        # 最小・最大線幅の制限を適用
        lw = max(lw, min_linewidth)
        if max_linewidth is not None:
            lw = min(lw, max_linewidth)

        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12 + 18*(net_val / max_abs),
            linewidth=lw,
            connectionstyle=f"arc3,rad={rad}",
            color=arrow_color,
            alpha=0.9,
        )
        ax.add_patch(arrow)

        if annotate_edges:
            # place text near middle of the curved edge
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            # small normal offset for readability
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            norm = (nx*nx + ny*ny) ** 0.5 + 1e-12
            nx, ny = nx / norm, ny / norm
            offset = 0.10 * (1 if rad > 0 else -1)
            ax.text(mx + offset*nx, my + offset*ny, f"{net_val:.3g}", fontsize=10, ha="center", va="center")

    ax.set_title("Net flux between basins  (F_ij = pi_i T_ij;  net = F_ij - F_ji)", fontsize=12)
    
    if savefig:
        if filename is None:
            filename = "net_flux_arrows.pdf"
        os.makedirs('./figs_for_paper/', exist_ok=True)
        plt.savefig(f'./figs_for_paper/{filename}', dpi=300, bbox_inches='tight')
    
    return ax


# =========================================================
# 3) Trajectory analysis for Results 3.4 (observed vs theoretical)
# =========================================================
class TrajectoryAnalyzer:
    """
    Takes observed trajectories and compares:
      - empirical basin transitions
      - theoretical basin transitions (group or individual)
    Recommended input dataframe columns (either):
      A) subject_id, time_idx, state_index (int in [0,2^n))
      or
      B) subject_id, time_idx, observed_state (bitstring length n)
    
    Note: このクラスは LandscapeEngine（NumPyベース）と連携します。
    VEM-MEMモデルで個人vs集団ランドスケープを「同一 state_index」上で比較する場合は、
    このファイルの TrajectoryComparator（VEM-MEMのLandscapeAnalyzerを使用）を使用してください。
    """

    def __init__(self, engine: LandscapeEngine):
        self.engine = engine

    def _to_state_index(self, row: Any, n: int) -> int:
        if "state_index" in row and row["state_index"] is not None:
            return int(row["state_index"])
        if "observed_state" in row and row["observed_state"] is not None:
            x = row["observed_state"]
            # bitstring
            if isinstance(x, str):
                return self.engine.codec.str_to_index(x)
            # list/array of bits
            arr = np.asarray(x).reshape(-1)
            if arr.size == n:
                return self.engine.codec.bits_to_index(arr)
        raise ValueError("Need 'state_index' or 'observed_state' to decode state.")

    def add_basin_labels(self, traj_df, landscape: Landscape, prefix: str = ""):
        """
        Adds columns:
          - state_index
          - attractor_state (basin minimum state index)
          - basin_id (0..B-1) using landscape.basin_index
        """
        if pd is None:
            raise RuntimeError("pandas required for trajectory analysis.")
        df = traj_df.copy()

        # ensure state_index
        if "state_index" not in df.columns:
            df["state_index"] = df.apply(lambda r: self._to_state_index(r, self.engine.n), axis=1)

        df[f"{prefix}attractor_state"] = df["state_index"].apply(lambda s: int(landscape.basin_min[int(s)]))

        if landscape.basin_index is None:
            # compute basin mapping if missing
            T_basin, pi_basin, basin_index, mins = _coarse_basin_transition(
                landscape.E, landscape.P, landscape.basin_min, self.engine.neigh
            )
            landscape.T_basin = T_basin
            landscape.pi_basin = pi_basin
            landscape.basin_index = basin_index
            landscape.basin_attractors = mins

        df[f"{prefix}basin_id"] = df[f"{prefix}attractor_state"].apply(lambda a: int(landscape.basin_index[int(a)]))
        return df

    def empirical_transition_counts(self, basin_seq: List[int], B: int) -> np.ndarray:
        C = np.zeros((B, B), dtype=float)
        for t in range(len(basin_seq) - 1):
            i = basin_seq[t]
            j = basin_seq[t + 1]
            C[i, j] += 1.0
        return C

    def empirical_transition_matrix(self, counts: np.ndarray, add_pseudocount: float = 0.0) -> np.ndarray:
        C = counts.copy()
        if add_pseudocount > 0:
            C += add_pseudocount
        row = C.sum(axis=1, keepdims=True)
        row[row == 0] = 1.0
        return C / row

    def sequence_loglik(self, basin_seq: List[int], T_basin: np.ndarray, pi_basin: Optional[np.ndarray] = None, eps: float = 1e-12) -> float:
        """
        log p(seq) approx:
          log pi[b0] + sum_t log T[b_t, b_{t+1}]
        """
        ll = 0.0
        if len(basin_seq) == 0:
            return float("nan")
        if pi_basin is not None:
            ll += math.log(max(pi_basin[basin_seq[0]], eps))
        for t in range(len(basin_seq) - 1):
            ll += math.log(max(T_basin[basin_seq[t], basin_seq[t + 1]], eps))
        return float(ll)

    def fit_table_group_vs_individual(
        self,
        traj_df,
        group_landscape: Landscape,
        individual_landscapes: Dict[int, Landscape],
    ):
        """
        Returns per subject:
          - empirical transition entropy / self-loop rate (simple summaries)
          - loglik under group T_basin
          - loglik under individual T_basin
          - delta_ll = ll_indiv - ll_group
        """
        if pd is None:
            raise RuntimeError("pandas required.")
        out_rows = []

        # label for group and individuals
        df_g = self.add_basin_labels(traj_df, group_landscape, prefix="g_")

        for k, Lk in individual_landscapes.items():
            df_k = traj_df[traj_df["subject_id"] == k].copy()
            if df_k.empty:
                continue
            df_k = self.add_basin_labels(df_k, group_landscape, prefix="g_")   # group basins
            df_k = self.add_basin_labels(df_k, Lk, prefix="i_")               # individual basins

            # sequences
            seq_g = df_k.sort_values("time_idx")["g_basin_id"].astype(int).tolist()
            seq_i = df_k.sort_values("time_idx")["i_basin_id"].astype(int).tolist()

            # empirical transitions (for descriptive stats)
            Bg = group_landscape.T_basin.shape[0] if group_landscape.T_basin is not None else int(df_g["g_basin_id"].max() + 1)
            Bi = Lk.T_basin.shape[0] if Lk.T_basin is not None else int(df_k["i_basin_id"].max() + 1)

            Cg = self.empirical_transition_counts(seq_g, Bg)
            Tg_emp = self.empirical_transition_matrix(Cg, add_pseudocount=0.0)
            self_loop_g = float(np.trace(Tg_emp))

            Ci = self.empirical_transition_counts(seq_i, Bi)
            Ti_emp = self.empirical_transition_matrix(Ci, add_pseudocount=0.0)
            self_loop_i = float(np.trace(Ti_emp))

            # likelihood under theoretical T_basin
            ll_group = self.sequence_loglik(seq_g, group_landscape.T_basin, group_landscape.pi_basin)  # type: ignore
            ll_indiv = self.sequence_loglik(seq_i, Lk.T_basin, Lk.pi_basin)  # type: ignore

            out_rows.append({
                "subject_id": int(k),
                "n_steps": int(max(len(seq_g) - 1, 0)),
                "emp_selfloop_group": self_loop_g,
                "emp_selfloop_indiv": self_loop_i,
                "ll_group": ll_group,
                "ll_indiv": ll_indiv,
                "delta_ll": float(ll_indiv - ll_group),
            })

        return pd.DataFrame(out_rows).sort_values("delta_ll", ascending=False)

    def plot_empirical_vs_theoretical_heatmap(
        self,
        traj_df,
        landscape: Landscape,
        top: int = 12,
        use_group_labels: bool = True,
        title: str = "",
        savefig: bool = False,
        filename: Optional[str] = None,
    ) -> None:
        """
        Shows:
          left: empirical basin transition (row-normalized)
          right: theoretical basin transition (row-normalized)
        """
        if pd is None:
            raise RuntimeError("pandas required.")
        df = self.add_basin_labels(traj_df, landscape, prefix="b_")
        if landscape.T_basin is None:
            raise ValueError("landscape must have T_basin computed")

        pi = landscape.pi_basin
        mins = landscape.basin_attractors
        idx = np.argsort(-pi)[:min(top, len(pi))]

        # empirical
        seq = df.sort_values(["subject_id", "time_idx"])["b_basin_id"].astype(int).tolist()
        C = self.empirical_transition_counts(seq, landscape.T_basin.shape[0])
        Temp = self.empirical_transition_matrix(C, add_pseudocount=0.0)
        Temp = Temp[np.ix_(idx, idx)]

        # theoretical
        Tth = landscape.T_basin[np.ix_(idx, idx)].copy()
        # row normalize for display
        Temp = Temp / np.maximum(Temp.sum(axis=1, keepdims=True), 1e-12)
        Tth  = Tth  / np.maximum(Tth.sum(axis=1, keepdims=True), 1e-12)

        labels = [str(mins[i]) for i in idx]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.heatmap(Temp, ax=axes[0], cmap="coolwarm", square=True, cbar_kws={"label": "empirical"}, annot= True)
        sns.heatmap(Tth,  ax=axes[1], cmap="coolwarm", square=True, cbar_kws={"label": "theoretical"}, annot= True)
        for ax in axes:
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels, rotation=0)
            ax.set_xlabel("to")
            ax.set_ylabel("from")
        axes[0].set_title("Empirical basin transitions")
        axes[1].set_title("Theoretical basin transitions")
        fig.suptitle(title or "Empirical vs theoretical transitions", y=1.02)
        plt.tight_layout()
        if savefig:
            if filename is None:
                filename = f"empirical_vs_theoretical_heatmap_top{top}.pdf"
            os.makedirs('./figs_for_paper/', exist_ok=True)
            plt.savefig(f'./figs_for_paper/{filename}', dpi=300, bbox_inches='tight')
        plt.show()


# =========================================================
# 3.5) Trajectory Comparator (個人 vs 集団ランドスケープ比較)
# =========================================================
# Note: このクラスは VEM-MEM モデルの LandscapeAnalyzer を使用します。
# 個人ランドスケープと集団ランドスケープを「同一 state_index」上で比較します。

class TrajectoryComparator:
    """
    個人ランドスケープ(theta_indiv)と集団ランドスケープ(theta_group)の
    "同一 state_index "上でのラベル付けを行い、二重座標系の traj_df を作る。
    
    VEM-MEMモデルの LandscapeAnalyzer を使用する場合は、
    VEM_MEM.py から LandscapeAnalyzer をインポートしてください。
    """
    def __init__(self, analyzer, theta_group, eps=1e-12):
        self.analyzer = analyzer
        self.theta_group = np.asarray(theta_group)
        self.eps = eps

        # 集団ランドスケープは固定なので一度だけ
        self.group_land = self.analyzer.compute_energy_landscape(self.theta_group, show_progress=False)
        self.group_rank = self._basin_rank_map(self.group_land)

    @staticmethod
    def _basin_rank_map(land):
        """
        basin_id は local minimum の state_index（VEM_MEM.py 実装）なので、
        それを energy の低い順に rank 付けして map を返す。
        """
        lm = np.asarray(land["local_minima_indices"])
        if lm.size == 0:
            return {}
        lm_e = np.asarray(land["energies"])[lm]
        order = np.argsort(lm_e)
        return {int(lm[order[i]]): int(i) for i in range(len(lm))}

    def analyze_trajectory_dual(self, data_timeseries, theta_indiv, use_jax_match=True, show_progress=False):
        """
        Returns columns:
          time_idx, observed_state, state_index,
          basin_group, basin_indiv,
          basin_group_rank, basin_indiv_rank,
          energy_group, prob_group, energy_indiv, prob_indiv,
          energy_delta, log_prob_delta
        """
        data_timeseries = np.asarray(data_timeseries)
        theta_indiv = np.asarray(theta_indiv)

        ind_land = self.analyzer.compute_energy_landscape(theta_indiv, show_progress=show_progress)
        ind_rank = self._basin_rank_map(ind_land)

        # 共通：state_index を一回だけ求める
        T, d = data_timeseries.shape
        if use_jax_match:
            try:
                import jax.numpy as jnp
                obs_states_jax = jnp.array(data_timeseries.astype(int))
                state_indices = self.analyzer._match_states_jax(obs_states_jax, self.analyzer._states_jax)
                state_indices = np.array(state_indices)
            except (AttributeError, ImportError):
                # JAXが使えない、またはanalyzerがJAX対応していない場合
                use_jax_match = False
        
        if not use_jax_match:
            states = np.asarray(ind_land["states"])
            state_indices = []
            for t in range(T):
                obs_state = data_timeseries[t].astype(int)
                match = np.all(states == obs_state, axis=1)
                if np.any(match):
                    state_indices.append(int(np.where(match)[0][0]))
                else:
                    # ハミング最小にフォールバック
                    hd = np.sum(states != obs_state, axis=1)
                    state_indices.append(int(np.argmin(hd)))
            state_indices = np.array(state_indices)

        # ラベル付け
        recs = []
        for t in range(T):
            sidx = int(state_indices[t])
            obs = data_timeseries[t].astype(int)
            obs_str = "".join(map(str, obs.tolist()))

            bg = int(self.group_land["basin_mapping"][sidx])
            bi = int(ind_land["basin_mapping"][sidx])

            eg = float(self.group_land["energies"][sidx])
            pg = float(self.group_land["probabilities"][sidx])
            ei = float(ind_land["energies"][sidx])
            pi = float(ind_land["probabilities"][sidx])

            recs.append({
                "time_idx": t,
                "observed_state": obs_str,
                "state_index": sidx,

                "basin_group": bg,
                "basin_indiv": bi,
                "basin_group_rank": self.group_rank.get(bg, np.nan),
                "basin_indiv_rank": ind_rank.get(bi, np.nan),

                "energy_group": eg,
                "prob_group": pg,
                "energy_indiv": ei,
                "prob_indiv": pi,

                "energy_delta": eg - ei,
                "log_prob_delta": np.log(pg + self.eps) - np.log(pi + self.eps),
            })

        return pd.DataFrame(recs) if pd is not None else recs

    def analyze_all_subjects(self, data_all, mu_all, subject_axis=0):
        """
        data_all: (K, T, d)
        mu_all:   (K, D)  または (K, 1, D)
        """
        data_all = np.asarray(data_all)
        mu_all = np.asarray(mu_all)
        if mu_all.ndim == 3:
            mu_all = mu_all[:, 0, :]

        K, T, d = data_all.shape
        out = []
        for k in range(K):
            dfk = self.analyze_trajectory_dual(data_all[k], mu_all[k], show_progress=False)
            if pd is not None:
                dfk["subject_id"] = k
                out.append(dfk)
            else:
                for rec in dfk:
                    rec["subject_id"] = k
                out.extend(dfk)
        
        if pd is not None:
            return pd.concat(out, ignore_index=True)
        return out

    @staticmethod
    def empirical_transition_matrix(traj_df, basin_col, max_rank=None, normalize="row"):
        """
        subjectごとに time_idx で並べ、連続時点の遷移を数える。
        basin_col は basin_group_rank / basin_indiv_rank などを想定。
        """
        if pd is None:
            raise RuntimeError("pandas required for this method.")
        # ランクの最大を自動推定（NaN除外）
        vals = traj_df[basin_col].dropna().astype(int)
        if max_rank is None:
            max_rank = int(vals.max()) if len(vals) else 0
        n = max_rank + 1

        C = np.zeros((n, n), dtype=float)
        for sid, df in traj_df.groupby("subject_id"):
            df = df.sort_values("time_idx")
            seq = df[basin_col].values
            for a, b in zip(seq[:-1], seq[1:]):
                if np.isnan(a) or np.isnan(b):
                    continue
                a = int(a); b = int(b)
                if a <= max_rank and b <= max_rank:
                    C[a, b] += 1.0

        if normalize == "row":
            row = C.sum(axis=1, keepdims=True)
            C = np.divide(C, row, out=np.zeros_like(C), where=row > 0)
        elif normalize == "all":
            s = C.sum()
            if s > 0:
                C = C / s
        return C

    @staticmethod
    def basin_overlap_matrix(theta_indiv_land, theta_group_land, weight="indiv"):
        """
        個人盆地×群盆地の"重なり"を state 空間上で定義。
        例：W(i,g)=Σ_{state: bi= i, bg= g} p_indiv(state)
        """
        bi = np.asarray(theta_indiv_land["basin_mapping"]).astype(int)
        bg = np.asarray(theta_group_land["basin_mapping"]).astype(int)
        if weight == "indiv":
            w = np.asarray(theta_indiv_land["probabilities"])
        else:
            w = np.asarray(theta_group_land["probabilities"])

        ui = np.unique(bi)
        ug = np.unique(bg)
        i_map = {b:i for i,b in enumerate(ui)}
        g_map = {b:j for j,b in enumerate(ug)}
        W = np.zeros((len(ui), len(ug)), dtype=float)

        for s in range(len(bi)):
            W[i_map[bi[s]], g_map[bg[s]]] += float(w[s])

        return W, ui, ug


# =========================================================
# 3.5.1) Trajectory Comparator ユーティリティ関数
# =========================================================

def _mu_aggregate(mu_all: np.ndarray) -> np.ndarray:
    """
    (N,1,D) -> (N,D) に正規化。すでに (N,D) ならそのまま。
    
    Args:
        mu_all: パラメータ配列 (N,1,D) または (N,D)
    
    Returns:
        正規化されたパラメータ配列 (N,D)
    """
    mu_all = np.asarray(mu_all)
    if mu_all.ndim == 3 and mu_all.shape[1] == 1:
        return mu_all[:, 0, :]
    if mu_all.ndim == 2:
        return mu_all
    raise ValueError(f"mu_all shape unexpected for aggregate: {mu_all.shape}")


def build_dual_traj_df_all(data_all: np.ndarray, model, eta: np.ndarray, mu_indiv: np.ndarray) -> pd.DataFrame:
    """
    全被験者について、各時点の観測状態を
    (groupとindivの basin_rank / energy / prob / delta) に写像したDFを作る。
    
    Args:
        data_all: 時系列データ (N, T, d)
        model: VEM-MEMモデル（VEMMEMインスタンス）
        eta: 集団パラメータ (D,)
        mu_indiv: 個人パラメータ (N, D) または (N, 1, D)
    
    Returns:
        dual_df: 各時点での個人・集団ランドスケープ情報を含むDataFrame
        列: subject_id, time_idx, observed_state, state_index,
            basin_group, basin_indiv, basin_group_rank, basin_indiv_rank,
            energy_group, prob_group, energy_indiv, prob_indiv,
            energy_delta, log_prob_delta
    
    Note:
        VEM_MEM.py から LandscapeAnalyzer をインポートする必要があります:
        from notebooks.VEM_MEM import LandscapeAnalyzer
    """
    if pd is None:
        raise RuntimeError("pandas required for this function.")
    
    # LandscapeAnalyzerをインポート（循環依存を避けるため、関数内でインポート）
    try:
        from notebooks.VEM_MEM import LandscapeAnalyzer
    except ImportError:
        # 相対インポートを試す
        try:
            from VEM_MEM import LandscapeAnalyzer
        except ImportError:
            raise ImportError(
                "LandscapeAnalyzer could not be imported. "
                "Please ensure VEM_MEM.py is in the Python path."
            )
    
    analyzer = LandscapeAnalyzer(model)
    TC = TrajectoryComparator(analyzer, theta_group=eta)

    rows = []
    N = data_all.shape[0]
    mu_indiv = _mu_aggregate(mu_indiv)  # (N, D) に正規化
    
    for k in range(N):
        dfk = TC.analyze_trajectory_dual(
            data_timeseries=data_all[k],
            theta_indiv=mu_indiv[k],
            show_progress=False,
        )
        dfk["subject_id"] = k
        rows.append(dfk)

    dual_df = pd.concat(rows, ignore_index=True)
    return dual_df


def per_subject_kinetics(dual_df: pd.DataFrame, col_rank: str = "basin_group_rank") -> pd.DataFrame:
    """
    "全被験者のそれぞれの動力学解析"の最小セット：
      - 訪問basin数
      - basin遷移回数
    
    Args:
        dual_df: build_dual_traj_df_all()の出力
        col_rank: 使用するランク列（"basin_group_rank" または "basin_indiv_rank"）
    
    Returns:
        DataFrame with columns: subject_id, n_transitions, n_unique_basins
    """
    if pd is None:
        raise RuntimeError("pandas required for this function.")
    
    out = []
    for sid, g in dual_df.sort_values(["subject_id", "time_idx"]).groupby("subject_id"):
        seq = g[col_rank].to_numpy()
        n_trans = int(np.sum(seq[1:] != seq[:-1])) if len(seq) >= 2 else 0
        n_unique = int(len(np.unique(seq)))
        out.append({"subject_id": int(sid), "n_transitions": n_trans, "n_unique_basins": n_unique})
    return pd.DataFrame(out)


# =========================================================
# 3.6) Trajectory Visualizer (遷移行列・ネットワーク可視化)
# =========================================================

class TrajectoryVisualizer:
    """
    遷移行列やネットワークの可視化を行うクラス。
    """
    def __init__(self):
        sns.set(style="white", context="talk")

    def plot_transition_heatmap(self, P, title="", ax=None, annot=False, fmt=".2f"):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        sns.heatmap(
            P, ax=ax, cmap="viridis", square=True,
            cbar_kws={"label": "Transition prob."},
            annot=annot, fmt=fmt, linewidths=0.5, linecolor="white",
            vmin=0.0, vmax=np.nanmax(P) if np.isfinite(P).any() else 1.0
        )
        ax.set_title(title)
        ax.set_xlabel("to")
        ax.set_ylabel("from")
        return ax

    def plot_net_flux(self, P_counts_or_probs, title="", ax=None, node_scale=2000, thr=0.02):
        """
        net flux = max(0, F_ij - F_ji). ここでは F=P (all-normalized) でも counts でもOK。
        
        Note: より高機能な可視化機能（カーブ、アノテーション、カスタマイズオプション）は
        このファイルの plot_net_flux_arrows() を参照してください。
        """
        F = np.array(P_counts_or_probs, dtype=float)
        # all 正規化にしておくと見やすい
        s = F.sum()
        if s > 0:
            F = F / s
        net = np.maximum(0.0, F - F.T)

        G = nx.DiGraph()
        n = net.shape[0]
        for i in range(n):
            G.add_node(i, mass=float(F[i].sum()))

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if net[i, j] >= thr:
                    G.add_edge(i, j, w=float(net[i, j]))

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        pos = nx.spring_layout(G, seed=0)
        masses = np.array([G.nodes[i]["mass"] for i in range(n)])
        sizes = node_scale * (masses / (masses.max() + 1e-12) + 0.05)

        nx.draw_networkx_nodes(G, pos, node_size=sizes, ax=ax, alpha=0.9)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)

        # edge width
        ws = np.array([G[u][v]["w"] for u, v in G.edges()])
        widths = 10.0 * (ws / (ws.max() + 1e-12))
        nx.draw_networkx_edges(G, pos, ax=ax, width=widths, arrows=True, arrowsize=20, alpha=0.8)

        ax.set_title(title)
        ax.axis("off")
        return ax


# =========================================================
# 4) Score association for Results 3.5
# =========================================================
class ScoreAssociation:
    """
    Merge landscape features/distances with scores and make plots.
    Expects:
      features_df: output of resultsAnalyzer.feature_distance_table()
      score_df: columns include subject_id and a score (e.g., total_score) per subject or per time.
    """

    def __init__(self):
        if pd is None:
            raise RuntimeError("pandas required.")

    def summarize_score_per_subject(self, score_df, score_col: str = "score", how: str = "mean"):
        df = score_df.copy()
        if how == "mean":
            return df.groupby("subject_id")[score_col].mean().reset_index().rename(columns={score_col: f"{score_col}_mean"})
        if how == "sum":
            return df.groupby("subject_id")[score_col].sum().reset_index().rename(columns={score_col: f"{score_col}_sum"})
        if how == "median":
            return df.groupby("subject_id")[score_col].median().reset_index().rename(columns={score_col: f"{score_col}_median"})
        if how == "max":
            return df.groupby("subject_id")[score_col].max().reset_index().rename(columns={score_col: f"{score_col}_max"})
        if how == "min":
            return df.groupby("subject_id")[score_col].min().reset_index().rename(columns={score_col: f"{score_col}_min"})
        if how == "baseline":
            # smallest time_idx
            if "time_idx" not in df.columns:
                raise ValueError("baseline requires time_idx column")
            return (df.sort_values("time_idx")
                      .groupby("subject_id")[score_col]
                      .first()
                      .reset_index()
                      .rename(columns={score_col: f"{score_col}_baseline"}))
        if how == "last":
            if "time_idx" not in df.columns:
                raise ValueError("last requires time_idx column")
            return (df.sort_values("time_idx")
                      .groupby("subject_id")[score_col]
                      .last()
                      .reset_index()
                      .rename(columns={score_col: f"{score_col}_last"}))
        raise ValueError("how must be mean|sum|median|max|min|baseline|last")

    def merge(self, features_df, score_summary_df):
        return features_df.merge(score_summary_df, on="subject_id", how="left")

    def plot_scatter(self, df, x: str, y: str, hue: Optional[str] = None, title: str = "", savefig: bool = False, filename: Optional[str] = None):
        plt.figure(figsize=(7, 6))
        ax = sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.8)
        ax.set_title(title or f"{y} vs {x}")
        sns.despine()
        plt.tight_layout()
        if savefig:
            if filename is None:
                filename = f"scatter_{y}_vs_{x}.pdf"
            os.makedirs('./figs_for_paper/', exist_ok=True)
            plt.savefig(f'./figs_for_paper/{filename}', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_regression(
        self, 
        df, 
        x: str, 
        y: str, 
        title: str = "", 
        reg_type: str = "linear",
        poly_degree: int = 2,
        corr_type: str = "pearson",
        savefig: bool = False, 
        filename: Optional[str] = None,
        ax=None,
    ):
        """
        回帰分析と可視化（線形・多項式・非線形回帰対応）
        
        Args:
            df: DataFrame
            x: 説明変数列名
            y: 目的変数列名
            title: タイトル
            reg_type: 回帰タイプ
                - "linear": 線形回帰
                - "polynomial": 多項式回帰
                - "logistic": ロジスティック回帰（2値化）
                - "exponential": 指数回帰 y = a * exp(b*x)
                - "logarithmic": 対数回帰 y = a + b * log(x)
                - "power": べき乗回帰 y = a * x^b
                - "sigmoid": シグモイド回帰 y = a / (1 + exp(-b*(x-c))) + d
            poly_degree: 多項式回帰の次数（reg_type="polynomial"の場合）
            corr_type: 相関係数のタイプ（後方互換性のため残していますが、現在は使用されていません。
                観測値と予測値の間の相関係数として、Pearson、Chatterjee、Spearman、Kendallを全て計算します）
            savefig: 保存するか
            filename: 保存ファイル名
            ax: matplotlib axes（指定しない場合は新規作成）
        
        Returns:
            ax: matplotlib axes
            stats: 回帰統計情報の辞書。以下のキーを含む:
                - "pearson_r": 観測値と予測値の間のPearson相関係数
                - "chatterjee_xi": 観測値と予測値の間のChatterjee相関係数
                - "spearman_rho": 観測値と予測値の間のSpearman相関係数
                - "spearman_p": Spearman相関のp値
                - "kendall_tau": 観測値と予測値の間のKendall相関係数
                - "kendall_p": Kendall相関のp値
                - "r2": 決定係数（R²）
                - "aic": Akaike情報量基準
                - "model_type": 回帰タイプの文字列
        """
        if pd is None:
            raise RuntimeError("pandas required.")
        
        # データの準備
        d = df[[x, y]].dropna().copy()
        if len(d) == 0:
            raise ValueError(f"No valid data points after dropping NaN for {x} and {y}")
        
        xv = d[x].to_numpy(float)
        yv = d[y].to_numpy(float)
        
        # 図の準備
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))
        
        # 散布図
        sns.scatterplot(data=d, x=x, y=y, ax=ax, alpha=0.99, edgecolors="k", linewidths=2.0)
        
        # 相関係数を計算するヘルパー関数
        def chatterjee_correlation(x, y):
            """
            Chatterjee's correlation coefficient (ξ_n)
            非線形関係も検出できるランクベースの相関係数
            
            ξ_n = 1 - (3 * Σ|r_i - s_i|) / (n^2 - 1)
            ここで、r_iはXのランク、s_iはXでソートした後のYのランク
            """
            n = len(x)
            if n < 2:
                return np.nan
            
            # Xでソート
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_sorted = y[sort_idx]
            
            # Xのランク（元の順序）
            x_ranks = np.argsort(np.argsort(x)) + 1  # 1-indexed
            
            # Yのランク（Xでソートした後の順序）
            y_ranks_sorted = np.argsort(np.argsort(y_sorted)) + 1  # 1-indexed
            
            # 元の順序に戻す
            y_ranks = np.zeros(n, dtype=int)
            y_ranks[sort_idx] = y_ranks_sorted
            
            # |r_i - s_i|の合計
            diff_sum = np.sum(np.abs(x_ranks - y_ranks))
            
            # Chatterjeeの相関係数
            if n == 1:
                return 0.0
            xi = 1.0 - (3.0 * diff_sum) / (n * n - 1.0)
            
            return float(xi)
        
        def pearson_correlation(x, y):
            """ピアソンの積率相関係数"""
            return float(np.corrcoef(x, y)[0, 1])
        
        def spearman_correlation(x, y):
            """Spearmanの順位相関係数とp値（観測値と予測値の間）"""
            try:
                from scipy.stats import spearmanr
                rho, p_value = spearmanr(x, y)
                return float(rho), float(p_value)
            except ImportError:
                # scipyがない場合は手動計算（ランク付け、p値は計算しない）
                def rank_data(data):
                    """手動でランク付け（同順位は平均ランク）"""
                    sorted_indices = np.argsort(data)
                    ranks = np.zeros(len(data))
                    for i, idx in enumerate(sorted_indices):
                        ranks[idx] = i + 1
                    # 同順位の処理（簡易版：平均ランク）
                    unique_vals, counts = np.unique(data, return_counts=True)
                    for val, count in zip(unique_vals, counts):
                        if count > 1:
                            mask = data == val
                            ranks[mask] = ranks[mask].mean()
                    return ranks
                
                x_rank = rank_data(x)
                y_rank = rank_data(y)
                rho = float(np.corrcoef(x_rank, y_rank)[0, 1])
                return rho, np.nan  # p値は計算できない
        
        def kendall_correlation(x, y):
            """Kendallの順位相関係数とp値（観測値と予測値の間）"""
            try:
                from scipy.stats import kendalltau
                tau, p_value = kendalltau(x, y)
                return float(tau), float(p_value)
            except ImportError:
                # scipyがない場合は計算しない
                return np.nan, np.nan
        
        # 回帰タイプに応じた処理
        stats = {}
        x_sorted = np.sort(xv)
        x_grid = np.linspace(xv.min(), xv.max(), 200)
        
        try:
            if reg_type == "linear":
                # 線形回帰（OLS）
                try:
                    from sklearn.linear_model import LinearRegression
                    from sklearn.metrics import r2_score
                    
                    X = xv.reshape(-1, 1)
                    model = LinearRegression()
                    model.fit(X, yv)
                    y_pred = model.predict(X)
                    y_grid = model.predict(x_grid.reshape(-1, 1))
                    
                    r2 = r2_score(yv, y_pred)
                    # 観測値と予測値の間の相関係数を全て計算
                    pearson_r = pearson_correlation(yv, y_pred)
                    chatterjee_xi = chatterjee_correlation(yv, y_pred)
                    spearman_rho, spearman_p = spearman_correlation(yv, y_pred)
                    kendall_tau, kendall_p = kendall_correlation(yv, y_pred)
                    
                    # AIC計算（簡易版: AIC = n*log(RSS/n) + 2*k）
                    n = len(yv)
                    k = 2  # 切片と傾き
                    rss = np.sum((yv - y_pred) ** 2)
                    aic = n * np.log(rss / n) + 2 * k
                    
                    stats = {
                        "pearson_r": pearson_r, 
                        "chatterjee_xi": chatterjee_xi,
                        "spearman_rho": spearman_rho, 
                        "spearman_p": spearman_p,
                        "kendall_tau": kendall_tau,
                        "kendall_p": kendall_p,
                        "r2": r2, 
                        "aic": aic, 
                        "model_type": "linear"
                    }
                    
                    ax.plot(x_grid, y_grid, color="r", linewidth=2, label=f"Linear (AIC={aic:.1f})")
                    
                except ImportError:
                    # sklearnがない場合はseabornのregplotを使用
                    sns.regplot(data=d, x=x, y=y, ax=ax, scatter=False, line_kws=dict(color="r", linewidth=2))
                    # regplotの場合は予測値を取得できないため、観測値同士の相関のみ計算
                    pearson_r = pearson_correlation(xv, yv)
                    chatterjee_xi = chatterjee_correlation(xv, yv)
                    stats = {
                        "pearson_r": pearson_r,
                        "chatterjee_xi": chatterjee_xi,
                        "model_type": "linear"
                    }
            
            elif reg_type == "polynomial":
                # 多項式回帰
                try:
                    from sklearn.preprocessing import PolynomialFeatures
                    from sklearn.linear_model import LinearRegression
                    from sklearn.metrics import r2_score
                    from sklearn.pipeline import Pipeline
                    
                    poly_model = Pipeline([
                        ('poly', PolynomialFeatures(degree=poly_degree)),
                        ('linear', LinearRegression())
                    ])
                    
                    X = xv.reshape(-1, 1)
                    poly_model.fit(X, yv)
                    y_pred = poly_model.predict(X)
                    y_grid = poly_model.predict(x_grid.reshape(-1, 1))
                    
                    r2 = r2_score(yv, y_pred)
                    # 観測値と予測値の間の相関係数を全て計算
                    pearson_r = pearson_correlation(yv, y_pred)
                    chatterjee_xi = chatterjee_correlation(yv, y_pred)
                    spearman_rho, spearman_p = spearman_correlation(yv, y_pred)
                    kendall_tau, kendall_p = kendall_correlation(yv, y_pred)
                    
                    # AIC計算
                    n = len(yv)
                    k = poly_degree + 1  # 多項式の係数数
                    rss = np.sum((yv - y_pred) ** 2)
                    aic = n * np.log(rss / n) + 2 * k
                    
                    stats = {
                        "pearson_r": pearson_r,
                        "chatterjee_xi": chatterjee_xi,
                        "spearman_rho": spearman_rho,
                        "spearman_p": spearman_p,
                        "kendall_tau": kendall_tau,
                        "kendall_p": kendall_p,
                        "r2": r2, 
                        "aic": aic, 
                        "model_type": f"polynomial (degree={poly_degree})"
                    }
                    
                    ax.plot(x_grid, y_grid, color="r", linewidth=2, label=f"Polynomial (deg={poly_degree}, AIC={aic:.1f})")
                    
                except ImportError:
                    raise ImportError("sklearn is required for polynomial regression. Please install scikit-learn.")
            
            elif reg_type == "logistic":
                # ロジスティック回帰（yを0-1に正規化）
                try:
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.metrics import log_loss
                    
                    # yを0-1に正規化（既に0-1の範囲の場合はそのまま）
                    y_min, y_max = yv.min(), yv.max()
                    if y_max > 1.0 or y_min < 0.0:
                        yv_norm = (yv - y_min) / (y_max - y_min + 1e-10)
                    else:
                        yv_norm = yv.copy()
                    
                    # 2値化（0.5を閾値）
                    yv_binary = (yv_norm >= 0.5).astype(int)
                    
                    X = xv.reshape(-1, 1)
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_scaled, yv_binary)
                    
                    # 予測確率
                    X_grid_scaled = scaler.transform(x_grid.reshape(-1, 1))
                    y_proba_grid = model.predict_proba(X_grid_scaled)[:, 1]
                    # 元のスケールに戻す
                    y_grid = y_proba_grid * (y_max - y_min) + y_min
                    
                    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
                    y_pred_binary = model.predict(X_scaled)
                    
                    # AIC計算（ロジスティック回帰用: AIC = 2k - 2*log_likelihood）
                    n = len(yv_binary)
                    k = 2  # 切片と係数
                    # 対数尤度の計算
                    eps = 1e-15
                    y_pred_proba_clipped = np.clip(y_pred_proba, eps, 1 - eps)
                    log_likelihood = np.sum(yv_binary * np.log(y_pred_proba_clipped) + 
                                           (1 - yv_binary) * np.log(1 - y_pred_proba_clipped))
                    aic = 2 * k - 2 * log_likelihood
                    
                    # ロジスティック回帰の場合、y_pred_probaとyv_binaryの相関係数
                    pearson_r = pearson_correlation(yv_binary, y_pred_proba)
                    chatterjee_xi = chatterjee_correlation(yv_binary, y_pred_proba)
                    spearman_rho, spearman_p = spearman_correlation(yv_binary, y_pred_proba)
                    kendall_tau, kendall_p = kendall_correlation(yv_binary, y_pred_proba)
                    
                    stats = {
                        "pearson_r": pearson_r,
                        "chatterjee_xi": chatterjee_xi,
                        "spearman_rho": spearman_rho,
                        "spearman_p": spearman_p,
                        "kendall_tau": kendall_tau,
                        "kendall_p": kendall_p,
                        "aic": aic, 
                        "model_type": "logistic"
                    }
                    
                    ax.plot(x_grid, y_grid, color="r", linewidth=2, label=f"Logistic (AIC={aic:.1f})")
                    
                except ImportError:
                    raise ImportError("sklearn is required for logistic regression. Please install scikit-learn.")
            
            elif reg_type == "exponential":
                # 指数回帰: y = a * exp(b*x)
                try:
                    from scipy.optimize import curve_fit
                    from sklearn.metrics import r2_score
                    
                    def exp_func(x, a, b):
                        return a * np.exp(b * x)
                    
                    # 初期値の推定（y > 0 を仮定）
                    if np.any(yv <= 0):
                        yv_shift = yv - yv.min() + 1e-10
                    else:
                        yv_shift = yv
                    
                    # 初期値: a ≈ y_mean, b ≈ log(y_max/y_min) / (x_max - x_min)
                    a_init = np.mean(yv_shift)
                    b_init = np.log(yv_shift.max() / (yv_shift.min() + 1e-10)) / (xv.max() - xv.min() + 1e-10)
                    
                    popt, _ = curve_fit(exp_func, xv, yv_shift, p0=[a_init, b_init], maxfev=5000)
                    y_pred = exp_func(xv, *popt)
                    y_grid = exp_func(x_grid, *popt)
                    
                    # 元のスケールに戻す
                    if np.any(yv <= 0):
                        y_pred = y_pred + yv.min() - 1e-10
                        y_grid = y_grid + yv.min() - 1e-10
                    
                    r2 = r2_score(yv, y_pred)
                    # 観測値と予測値の間の相関係数を全て計算
                    pearson_r = pearson_correlation(yv, y_pred)
                    chatterjee_xi = chatterjee_correlation(yv, y_pred)
                    spearman_rho, spearman_p = spearman_correlation(yv, y_pred)
                    kendall_tau, kendall_p = kendall_correlation(yv, y_pred)
                    
                    # AIC計算
                    n = len(yv)
                    k = 2  # a, b
                    rss = np.sum((yv - y_pred) ** 2)
                    aic = n * np.log(rss / n) + 2 * k
                    
                    stats = {
                        "pearson_r": pearson_r,
                        "chatterjee_xi": chatterjee_xi,
                        "spearman_rho": spearman_rho,
                        "spearman_p": spearman_p,
                        "kendall_tau": kendall_tau,
                        "kendall_p": kendall_p,
                        "r2": r2, 
                        "aic": aic, 
                        "model_type": "exponential", 
                        "params": {"a": popt[0], "b": popt[1]}
                    }
                    ax.plot(x_grid, y_grid, color="r", linewidth=2, label=f"Exponential (AIC={aic:.1f})")
                    
                except ImportError:
                    raise ImportError("scipy is required for exponential regression. Please install scipy.")
            
            elif reg_type == "logarithmic":
                # 対数回帰: y = a + b * log(x)
                try:
                    from sklearn.linear_model import LinearRegression
                    from sklearn.metrics import r2_score
                    
                    # x > 0 を仮定
                    xv_log = np.log(xv + 1e-10)
                    X = xv_log.reshape(-1, 1)
                    
                    model = LinearRegression()
                    model.fit(X, yv)
                    y_pred = model.predict(X)
                    
                    x_grid_log = np.log(x_grid + 1e-10)
                    y_grid = model.predict(x_grid_log.reshape(-1, 1))
                    
                    r2 = r2_score(yv, y_pred)
                    # 観測値と予測値の間の相関係数を全て計算
                    pearson_r = pearson_correlation(yv, y_pred)
                    chatterjee_xi = chatterjee_correlation(yv, y_pred)
                    spearman_rho, spearman_p = spearman_correlation(yv, y_pred)
                    kendall_tau, kendall_p = kendall_correlation(yv, y_pred)
                    
                    # AIC計算
                    n = len(yv)
                    k = 2  # 切片と傾き
                    rss = np.sum((yv - y_pred) ** 2)
                    aic = n * np.log(rss / n) + 2 * k
                    
                    stats = {
                        "pearson_r": pearson_r,
                        "chatterjee_xi": chatterjee_xi,
                        "spearman_rho": spearman_rho,
                        "spearman_p": spearman_p,
                        "kendall_tau": kendall_tau,
                        "kendall_p": kendall_p,
                        "r2": r2, 
                        "aic": aic, 
                        "model_type": "logarithmic"
                    }
                    ax.plot(x_grid, y_grid, color="r", linewidth=2, label=f"Logarithmic (AIC={aic:.1f})")
                    
                except ImportError:
                    raise ImportError("sklearn is required for logarithmic regression. Please install scikit-learn.")
            
            elif reg_type == "power":
                # べき乗回帰: y = a * x^b
                try:
                    from scipy.optimize import curve_fit
                    from sklearn.metrics import r2_score
                    
                    def power_func(x, a, b):
                        return a * np.power(x + 1e-10, b)
                    
                    # 初期値の推定（x > 0, y > 0 を仮定）
                    if np.any(xv <= 0) or np.any(yv <= 0):
                        xv_shift = xv - xv.min() + 1
                        yv_shift = yv - yv.min() + 1
                    else:
                        xv_shift = xv
                        yv_shift = yv
                    
                    # 初期値: 線形回帰で log(y) ~ log(x) から推定
                    log_x = np.log(xv_shift + 1e-10)
                    log_y = np.log(yv_shift + 1e-10)
                    b_init = np.polyfit(log_x, log_y, 1)[0]
                    a_init = np.exp(np.mean(log_y) - b_init * np.mean(log_x))
                    
                    popt, _ = curve_fit(power_func, xv_shift, yv_shift, p0=[a_init, b_init], maxfev=5000)
                    y_pred = power_func(xv_shift, *popt)
                    y_grid = power_func(x_grid - xv.min() + 1 if np.any(xv <= 0) else x_grid, *popt)
                    
                    # 元のスケールに戻す
                    if np.any(xv <= 0) or np.any(yv <= 0):
                        y_pred = y_pred - 1 + yv.min()
                        y_grid = y_grid - 1 + yv.min()
                    
                    r2 = r2_score(yv, y_pred)
                    # 観測値と予測値の間の相関係数を全て計算
                    pearson_r = pearson_correlation(yv, y_pred)
                    chatterjee_xi = chatterjee_correlation(yv, y_pred)
                    spearman_rho, spearman_p = spearman_correlation(yv, y_pred)
                    kendall_tau, kendall_p = kendall_correlation(yv, y_pred)
                    
                    # AIC計算
                    n = len(yv)
                    k = 2  # a, b
                    rss = np.sum((yv - y_pred) ** 2)
                    aic = n * np.log(rss / n) + 2 * k
                    
                    stats = {
                        "pearson_r": pearson_r,
                        "chatterjee_xi": chatterjee_xi,
                        "spearman_rho": spearman_rho,
                        "spearman_p": spearman_p,
                        "kendall_tau": kendall_tau,
                        "kendall_p": kendall_p,
                        "r2": r2, 
                        "aic": aic, 
                        "model_type": "power", 
                        "params": {"a": popt[0], "b": popt[1]}
                    }
                    ax.plot(x_grid, y_grid, color="r", linewidth=2, label=f"Power (AIC={aic:.1f})")
                    
                except ImportError:
                    raise ImportError("scipy is required for power regression. Please install scipy.")
            
            elif reg_type == "sigmoid":
                # シグモイド回帰: y = a / (1 + exp(-b*(x-c))) + d
                try:
                    from scipy.optimize import curve_fit
                    from sklearn.metrics import r2_score
                    
                    def sigmoid_func(x, a, b, c, d):
                        return a / (1 + np.exp(-b * (x - c))) + d
                    
                    # 初期値の推定
                    y_min, y_max = yv.min(), yv.max()
                    y_range = y_max - y_min
                    x_mid = np.mean(xv)
                    x_range = xv.max() - xv.min()
                    
                    a_init = y_range
                    b_init = 4.0 / (x_range + 1e-10)  # ステップ幅
                    c_init = x_mid
                    d_init = y_min
                    
                    popt, _ = curve_fit(sigmoid_func, xv, yv, p0=[a_init, b_init, c_init, d_init], maxfev=5000)
                    y_pred = sigmoid_func(xv, *popt)
                    y_grid = sigmoid_func(x_grid, *popt)
                    
                    r2 = r2_score(yv, y_pred)
                    # 観測値と予測値の間の相関係数を全て計算
                    pearson_r = pearson_correlation(yv, y_pred)
                    chatterjee_xi = chatterjee_correlation(yv, y_pred)
                    spearman_rho, spearman_p = spearman_correlation(yv, y_pred)
                    kendall_tau, kendall_p = kendall_correlation(yv, y_pred)
                    
                    # AIC計算
                    n = len(yv)
                    k = 4  # a, b, c, d
                    rss = np.sum((yv - y_pred) ** 2)
                    aic = n * np.log(rss / n) + 2 * k
                    
                    stats = {
                        "pearson_r": pearson_r,
                        "chatterjee_xi": chatterjee_xi,
                        "spearman_rho": spearman_rho,
                        "spearman_p": spearman_p,
                        "kendall_tau": kendall_tau,
                        "kendall_p": kendall_p,
                        "r2": r2, 
                        "aic": aic, 
                        "model_type": "sigmoid", 
                        "params": {"a": popt[0], "b": popt[1], "c": popt[2], "d": popt[3]}
                    }
                    ax.plot(x_grid, y_grid, color="r", linewidth=2, label=f"Sigmoid (AIC={aic:.1f})")
                    
                except ImportError:
                    raise ImportError("scipy is required for sigmoid regression. Please install scipy.")
            
            else:
                raise ValueError(f"reg_type must be one of: 'linear', 'polynomial', 'logistic', 'exponential', 'logarithmic', 'power', 'sigmoid'. Got '{reg_type}'")
        
        except Exception as e:
            # フォールバック: seabornのregplot
            print(f"Warning: {reg_type} regression failed: {e}. Falling back to linear regression.")
            sns.regplot(data=d, x=x, y=y, ax=ax, scatter=False, line_kws=dict(color="r", linewidth=2))
            # regplotの場合は予測値を取得できないため、観測値同士の相関のみ計算
            pearson_r = pearson_correlation(xv, yv)
            chatterjee_xi = chatterjee_correlation(xv, yv)
            stats = {
                "pearson_r": pearson_r,
                "chatterjee_xi": chatterjee_xi,
                "model_type": "linear (fallback)"
            }
        
        # タイトルとラベル（相関係数を表示）
        if "aic" in stats:
            aic_str = f", AIC={stats['aic']:.1f}"
        else:
            aic_str = ""
        
        # 相関係数の文字列を構築
        corr_strs = []
        if "pearson_r" in stats and not np.isnan(stats['pearson_r']):
            corr_strs.append(f"r={stats['pearson_r']:.3f}")
        if "chatterjee_xi" in stats and not np.isnan(stats['chatterjee_xi']):
            corr_strs.append(f"ξ={stats['chatterjee_xi']:.3f}")
        if "spearman_rho" in stats and not np.isnan(stats['spearman_rho']):
            if "spearman_p" in stats and not np.isnan(stats['spearman_p']):
                corr_strs.append(f"ρ={stats['spearman_rho']:.3f}(p={stats['spearman_p']:.3f})")
            else:
                corr_strs.append(f"ρ={stats['spearman_rho']:.3f}")
        if "kendall_tau" in stats and not np.isnan(stats['kendall_tau']):
            if "kendall_p" in stats and not np.isnan(stats['kendall_p']):
                corr_strs.append(f"τ={stats['kendall_tau']:.3f}(p={stats['kendall_p']:.3f})")
            else:
                corr_strs.append(f"τ={stats['kendall_tau']:.3f}")
        
        corr_str = ", " + ", ".join(corr_strs) if corr_strs else ""
        
        if title:
            plot_title = f"{title} ({stats.get('model_type', reg_type)}{corr_str}{aic_str})"
        else:
            plot_title = f"{y} ~ {x} ({stats.get('model_type', reg_type)}{corr_str}{aic_str})"
        
        ax.set_title(plot_title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        
        if "label" in locals() and hasattr(ax, "legend"):
            ax.legend()
        
        sns.despine()
        plt.tight_layout()
        
        if savefig:
            if filename is None:
                filename = f"regression_{reg_type}_{y}_vs_{x}.pdf"
            os.makedirs('./figs_for_paper/', exist_ok=True)
            plt.savefig(f'./figs_for_paper/{filename}', dpi=300, bbox_inches='tight')
        
        return ax, stats


# =========================================================
# 5) Minimal "run recipe" (call these from notebook/script)
# =========================================================
def run_results_pipeline(
    mu: np.ndarray,
    eta: np.ndarray,
    traj_df=None,
    score_df=None,
    score_col: str = "score",
):
    """
    Convenience wrapper for notebooks.
    """
    if pd is None:
        raise RuntimeError("pandas required for this wrapper.")
    R = resultsAnalyzer(n=9, bit_order="little")
    R.set_group(eta)

    # 3.2-3.3 tables
    df_feat = R.feature_distance_table(mu=mu, eta=eta)

    # 3.1 group figures
    R.plot_group_basins(top=10, by="mass")
    R.plot_group_transition_heatmap(top=12, normalize="row", diag_zero=True)

    # 3.2 feature distributions
    R.plot_feature_distributions(df_feat, features=["n_minima", "global_basin_mass", "basin_mass_entropy"])

    # 3.3 distance distribution
    R.plot_distance_distribution(df_feat, col="vs_group_JS")

    # pick case subjects
    cases = R.pick_cases(df_feat, col="vs_group_JS", n_each=3)
    print("Case subjects:", cases)

    # 3.4 observed vs theoretical (optional)
    if traj_df is not None:
        # build only needed subjects for trajectories with T_basin
        R.build_subjects(mu, subject_ids=list(set(traj_df["subject_id"].astype(int).tolist())), compute_basin_transition=True)
        TA = TrajectoryAnalyzer(R.engine)
        fit = TA.fit_table_group_vs_individual(traj_df, R.group, R.subjects)  # type: ignore
        print(fit.head(10))

        # show global empirical vs group theoretical (all subjects pooled)
        TA.plot_empirical_vs_theoretical_heatmap(traj_df, R.group, top=12, title="All subjects: group basins")

    # 3.5 score association (optional)
    if score_df is not None:
        SA = ScoreAssociation()
        ssum = SA.summarize_score_per_subject(score_df, score_col=score_col, how="mean")
        merged = SA.merge(df_feat, ssum)
        SA.plot_scatter(merged, x=f"{score_col}_mean", y="vs_group_JS", title="Distance-to-group vs mean score")
        SA.plot_regression(merged, x=f"{score_col}_mean", y="vs_group_JS", title="Regression: JS ~ score")

    return df_feat


class Fig2Option1Builder:
    """
    Figure2 (Option1):
      A) Group-basin trajectory raster (top basins + Other)
      B) Mismatch raster (0/1): group basin != mapped individual basin (projected)
      C) Micro-switch rate vs score (size = mismatch rate, color = JS)
      D) JS vs score (lowess trend)
    """

    def __init__(self, top_n_basins: int = 3):
        self.top_n_basins = top_n_basins

    # ---------- (0) dual trajectory: attach group labels ----------
    @staticmethod
    def attach_group_columns(traj_indiv: pd.DataFrame, group_landscape: dict,
                             indiv_basin_col: str = "basin_id") -> pd.DataFrame:
        """
        traj_indiv: VEM_MEM.analyze_all_trajectories() output
          required columns: subject_id, time_idx, state_index, basin_id
        group_landscape: analyzer.compute_energy_landscape(eta) output
          keys used: basin_mapping, energies, probabilities, local_minima_indices, occupation_times
        """
        df = traj_indiv.copy()

        # rename individual columns for clarity
        if indiv_basin_col != "basin_indiv":
            df = df.rename(columns={indiv_basin_col: "basin_indiv"})
        if "energy" in df.columns:
            df = df.rename(columns={"energy": "energy_indiv"})
        if "probability" in df.columns:
            df = df.rename(columns={"probability": "prob_indiv"})

        sidx = df["state_index"].astype(int).values
        df["basin_group"]  = np.asarray(group_landscape["basin_mapping"], dtype=int)[sidx]
        df["energy_group"] = np.asarray(group_landscape["energies"], dtype=float)[sidx]
        df["prob_group"]   = np.asarray(group_landscape["probabilities"], dtype=float)[sidx]

        # deltas (optional but sometimes handy)
        if "energy_indiv" in df.columns:
            df["energy_delta"] = df["energy_group"] - df["energy_indiv"]
        if "prob_indiv" in df.columns:
            eps = 1e-12
            df["log_prob_delta"] = np.log(df["prob_group"] + eps) - np.log(df["prob_indiv"] + eps)

        return df

    @staticmethod
    def _top_group_basins_from_landscape(group_landscape: dict, n: int = 3) -> tuple:
        lm = np.asarray(group_landscape.get("local_minima_indices", []), dtype=int)
        occ = np.asarray(group_landscape.get("occupation_times", []), dtype=float)
        if lm.size == 0:
            return tuple()
        # occupation_times is aligned to lm order in your implementation
        order = np.argsort(occ)[::-1]
        top = [int(lm[i]) for i in order[:min(n, len(order))]]
        return tuple(top)

    @staticmethod
    def _top_group_basins_from_traj(traj_df_dual: pd.DataFrame, n: int = 3) -> tuple:
        vc = traj_df_dual["basin_group"].value_counts()
        return tuple(vc.index[:min(n, len(vc))].astype(int).tolist())

    # ---------- (1) mapping: individual basin -> group basin ----------
    @staticmethod
    def add_indiv_to_group_mapping(traj_df_dual: pd.DataFrame) -> pd.DataFrame:
        df = traj_df_dual.copy()

        # majority vote: for each subject and indiv basin, pick the most frequent group basin
        map_df = (df.groupby(["subject_id", "basin_indiv"])["basin_group"]
                    .agg(lambda x: x.value_counts().idxmax())
                    .rename("basin_indiv_to_group")
                    .reset_index())

        df = df.merge(map_df, on=["subject_id", "basin_indiv"], how="left")
        df["mismatch"] = (df["basin_indiv_to_group"] != df["basin_group"]).astype(int)
        return df

    # ---------- (2) trajectory-derived subject metrics ----------
    @staticmethod
    def compute_subject_metrics(traj_df_mapped: pd.DataFrame) -> pd.DataFrame:
        df = traj_df_mapped.sort_values(["subject_id", "time_idx"]).copy()
        g = df.groupby("subject_id")

        df["group_next"] = g["basin_group"].shift(-1)
        df["indiv_next"] = g["basin_indiv"].shift(-1)

        valid = df["group_next"].notna() & df["indiv_next"].notna()
        df["group_switch"] = valid & (df["group_next"] != df["basin_group"])
        #df["micro_switch"] = valid & (df["group_next"] == df["basin_group"]) & (df["indiv_next"] != df["basin_indiv"])
        df["micro_switch"] = valid & (df["indiv_next"] != df["basin_indiv"])

        # 植木算: 時間点数がnなら遷移数はn-1
        # 各被験者の時間点数を計算
        n_timepoints = df.groupby("subject_id")["time_idx"].nunique()
        n_transitions = n_timepoints - 1  # 遷移数 = 時間点数 - 1

        # スイッチ回数を集計
        switch_counts = df.groupby("subject_id").agg(
            mismatch_sum=("mismatch", "sum"),
            group_switch_sum=("group_switch", "sum"),
            micro_switch_sum=("micro_switch", "sum"),
        )

        # インデックスを揃えて遷移数で割ってスイッチ率を計算
        out = switch_counts.copy()
        out["mismatch_rate"] = switch_counts["mismatch_sum"] / n_timepoints.reindex(switch_counts.index)
        out["group_switch_rate"] = switch_counts["group_switch_sum"] / n_transitions.reindex(switch_counts.index)
        out["micro_switch_rate"] = switch_counts["micro_switch_sum"] / n_transitions.reindex(switch_counts.index)
        out = out[["mismatch_rate", "group_switch_rate", "micro_switch_rate"]].reset_index()

        # keep subject-level covariates if present
        for col in ["score_mean", "vs_group_JS"]:
            if col in df.columns:
                out = out.merge(df[["subject_id", col]].drop_duplicates("subject_id"),
                                on="subject_id", how="left")
        return out

    # ---------- (3) plotting ----------
    @staticmethod
    def _pivot_matrix(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        # if duplicates exist per (subject_id, time_idx), keep the first
        tmp = df.sort_values(["subject_id", "time_idx"]).drop_duplicates(["subject_id", "time_idx"])
        mat = tmp.pivot(index="subject_id", columns="time_idx", values=value_col)
        mat = mat.reindex(sorted(mat.columns), axis=1)
        return mat

    @staticmethod
    def _sorted_subjects(subj_metrics: pd.DataFrame, sort_by: str = "score_mean") -> np.ndarray:
        if sort_by in subj_metrics.columns and subj_metrics[sort_by].notna().any():
            return subj_metrics.sort_values(sort_by)["subject_id"].values
        if "vs_group_JS" in subj_metrics.columns and subj_metrics["vs_group_JS"].notna().any():
            return subj_metrics.sort_values("vs_group_JS")["subject_id"].values
        return subj_metrics["subject_id"].values

    def plot(self,
             traj_df_dual: pd.DataFrame,
             top_group_basins: tuple | None = None,
             sort_by: str = "score_mean",
             title_prefix: str = "Figure 2"):

        # sanity
        required = {"subject_id", "time_idx", "basin_group", "basin_indiv"}
        missing = required - set(traj_df_dual.columns)
        if missing:
            raise ValueError(f"traj_df_dual is missing required columns: {missing}")

        # choose top basins
        if top_group_basins is None:
            top_group_basins = self._top_group_basins_from_traj(traj_df_dual, n=self.top_n_basins)
        top_group_basins = tuple(int(x) for x in top_group_basins)

        # mapping + metrics
        df = self.add_indiv_to_group_mapping(traj_df_dual)
        subj = self.compute_subject_metrics(df)

        order = self._sorted_subjects(subj, sort_by=sort_by)

        # Panel A: group basin category raster (top + other)
        df["group_cat"] = df["basin_group"].where(df["basin_group"].isin(top_group_basins), -1)
        A = self._pivot_matrix(df, "group_cat").reindex(order)

        # map categories to ints for imshow
        cats = list(top_group_basins) + [-1]
        cat_labels = [str(b) for b in top_group_basins] + ["Other"]
        cat_to_int = {c: i for i, c in enumerate(cats)}
        A_int = A.applymap(lambda v: cat_to_int.get(int(v), cat_to_int[-1]) if pd.notna(v) else cat_to_int[-1]).values

        # Panel B: mismatch raster
        B = self._pivot_matrix(df, "mismatch").reindex(order).fillna(0).values

        # figure layout
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], wspace=0.25, hspace=0.25)

        # ---- A ----
        axA = fig.add_subplot(gs[0, 0])
        palette = sns.color_palette("Set2", n_colors=len(cats))
        cmapA = ListedColormap(palette)

        imA = axA.imshow(A_int, aspect="auto", interpolation="nearest", cmap=cmapA, vmin=0, vmax=len(cats)-1)
        axA.set_title(f"A) Group-basin trajectories (top={list(top_group_basins)} + Other)")
        axA.set_xlabel("time_idx")
        axA.set_ylabel("subjects (sorted)")
        axA.set_xticks(range(A.shape[1]))
        axA.set_xticklabels(list(A.columns))

        # thin y ticks
        yt = np.arange(0, len(order), max(1, len(order)//10))
        axA.set_yticks(yt)
        axA.set_yticklabels([str(order[i]) for i in yt])

        cbarA = fig.colorbar(imA, ax=axA, fraction=0.046, pad=0.02, ticks=np.arange(len(cats)))
        cbarA.set_ticklabels(cat_labels)
        cbarA.set_label("basin_group (category)")

        # ---- B ----
        axB = fig.add_subplot(gs[0, 1])
        imB = axB.imshow(B, aspect="auto", interpolation="nearest", cmap="Reds", vmin=0, vmax=1)
        axB.set_title("B) Mismatch (individual basin projected to group)")
        axB.set_xlabel("time_idx")
        axB.set_ylabel("subjects (sorted)")
        axB.set_xticks(range(A.shape[1]))
        axB.set_xticklabels(list(A.columns))
        axB.set_yticks(yt)
        axB.set_yticklabels([str(order[i]) for i in yt])
        cbarB = fig.colorbar(imB, ax=axB, fraction=0.046, pad=0.02)
        cbarB.set_label("mismatch (0/1)")

        # ---- C ----
        axC = fig.add_subplot(gs[1, 0])
        if "score_mean" in subj.columns:
            hue = "vs_group_JS" if "vs_group_JS" in subj.columns else None
            size = "mismatch_rate"
            sns.scatterplot(
                data=subj, x="score_mean", y="micro_switch_rate",
                hue=hue, size=size, sizes=(20, 250),
                ax=axC, edgecolor="none", alpha=0.85
            )
            # smooth trend (simple lowess)
            try:
                sns.regplot(data=subj, x="score_mean", y="micro_switch_rate",
                            scatter=False, lowess=True, ax=axC)
            except Exception:
                pass
            axC.set_title("C) Micro-switching within group basin vs score_mean\n(size=mismatch_rate, color=JS if available)")
            axC.set_xlabel("score_mean")
            axC.set_ylabel("micro_switch_rate")
            axC.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
        else:
            axC.text(0.5, 0.5, "score_mean not found\n(merge it into traj_df_dual first)",
                     ha="center", va="center")
            axC.axis("off")

        # ---- D ----
        axD = fig.add_subplot(gs[1, 1])
        if "score_mean" in subj.columns and "vs_group_JS" in subj.columns:
            sns.scatterplot(data=subj, x="score_mean", y="vs_group_JS",
                            ax=axD, edgecolor="none", alpha=0.85)
            try:
                sns.regplot(data=subj, x="score_mean", y="vs_group_JS",
                            scatter=False, lowess=True, ax=axD)
            except Exception:
                pass
            axD.set_title("D) Distance-to-group (JS) vs score_mean")
            axD.set_xlabel("score_mean")
            axD.set_ylabel("vs_group_JS")
        else:
            axD.text(0.5, 0.5, "need score_mean and vs_group_JS\n(merge into traj_df_dual or subj table)",
                     ha="center", va="center")
            axD.axis("off")

        fig.suptitle(title_prefix, y=0.99)
        plt.tight_layout()
        return fig, df, subj


# =========================================================
# Example usage for plot_net_flux_arrows
# =========================================================
"""
# Example 1: Plot net flux for group landscape
R = resultsAnalyzer(n=9, bit_order="little")
R.set_group(eta)  # eta is the group-level parameter vector

# Basic usage - show top 6 edges with minimum net flux threshold
R.plot_net_flux_arrows(
    use_group=True,
    top_k_edges=6,
    min_abs_net=1e-3,
    curvature=0.28,
    savefig=True  # Saves to './figs_for_paper/group_net_flux_arrows.pdf'
)
plt.show()

# Example 2: Plot net flux for individual subject
R.build_subjects(mu, subject_ids=[0, 1, 2], compute_basin_transition=True)

# Plot for subject 0
R.plot_net_flux_arrows(
    use_group=False,
    k=0,
    top_k_edges=8,
    min_abs_net=1e-4,
    curvature=0.25,
    savefig=True,  # Saves to './figs_for_paper/subject0_net_flux_arrows.pdf'
    filename="custom_name.pdf"  # Or specify custom filename
)
plt.show()

# Example 3: Using standalone function (if you have T, pi, labels directly)
out = R.basin_transition(use_group=True)
T = out["T_basin"]
pi = out["pi_basin"]
mins = out["mins"]
labels = [str(m) for m in mins]

plot_net_flux_arrows(
    T, pi, labels=labels,
    top_k_edges=6,
    min_abs_net=1e-3,
    curvature=0.28,
    savefig=True,
    filename="net_flux_arrows.pdf"
)
plt.show()

# Example 4: Customize appearance
R.plot_net_flux_arrows(
    use_group=True,
    top_k_edges=10,
    node_size=1200,
    node_alpha=0.8,
    width_scale=30.0,
    text_scale=14,
    annotate_edges=True,
    savefig=True
)
plt.show()
"""


# =========================================================
# 8) Reporting utilities (Group / Individual figures)
# =========================================================
# These helpers intentionally avoid any figure-number naming (Fig1/Fig2...)
# so you can compose them freely in the manuscript.

from matplotlib.backends.backend_pdf import PdfPages


def _require_pandas():
    if pd is None:
        raise ImportError("pandas is required for the reporting utilities. Please install pandas.")


def _to_numpy(x):
    """Convert JAX/NumPy arrays to NumPy without importing JAX explicitly."""
    try:
        import numpy as _np
        return _np.asarray(x)
    except Exception:
        return x


def _theta_for_subject(mu_all, subject_id: int, theta_strategy: str = "mean"):
    """
    mu_all allowed shapes:
      - (K, D)
      - (K, 1, D)
      - (K, T, D)
    Returns a (D,) vector.
    """
    mu = _to_numpy(mu_all)
    if mu.ndim == 2:
        return mu[subject_id]
    if mu.ndim == 3 and mu.shape[1] == 1:
        return mu[subject_id, 0]
    if mu.ndim == 3:
        if theta_strategy == "mean":
            return mu[subject_id].mean(axis=0)
        if theta_strategy == "last":
            return mu[subject_id, -1]
        raise ValueError("theta_strategy must be 'mean' or 'last' for (K,T,D) mu_all")
    raise ValueError(f"Unsupported mu_all shape: {mu.shape}")


def _role_rank_map_for_theta(analyzer, theta, rank_by: str = "mass"):
    """
    Build a mapping: basin_id (local-minimum state_index) -> rank (0,1,2,...)

    rank_by:
      - 'mass'  : occupation_times descending (most-visited basin is LM1)
      - 'energy': local-minimum energy ascending (deepest basin is LM1)

    Requires analyzer.compute_energy_landscape(theta) to return:
      - local_minima_indices : array-like of state_index for minima
      - occupation_times     : array-like aligned to minima list
      - energies             : (2^d,) energies
    """
    land = analyzer.compute_energy_landscape(theta, show_progress=False)
    mins = _to_numpy(land.get("local_minima_indices", []))
    if mins is None or len(mins) == 0:
        return {}, land

    mins = mins.astype(int)

    if rank_by == "mass":
        occ = _to_numpy(land.get("occupation_times", None))
        if occ is None or len(occ) != len(mins):
            # fallback: empirical mass from probabilities if occupation_times absent
            probs = _to_numpy(land.get("probabilities"))
            bas = _to_numpy(land.get("basin_mapping")).astype(int)
            occ = np.array([probs[bas == m].sum() for m in mins], dtype=float)
        order = np.argsort(occ)[::-1]
    elif rank_by == "energy":
        energies = _to_numpy(land.get("energies"))
        e_m = energies[mins]
        order = np.argsort(e_m)
    else:
        raise ValueError("rank_by must be 'mass' or 'energy'")

    rank_map = {int(mins[order[i]]): int(i) for i in range(len(order))}
    return rank_map, land


def add_role_rank_to_traj_df(
    analyzer,
    traj_df: 'pd.DataFrame',
    mu_all,
    basin_col: str = "basin_id",
    rank_by: str = "mass",
    theta_strategy: str = "mean",
    top_roles: int = 6,
) -> 'pd.DataFrame':
    """
    Add role-based basin labels per subject.

    Output columns:
      - basin_rank (int): 0.., rank within subject (LM1=0)
      - role_cat   (int): 0..top_roles-1 and top_roles=Other
      - role_label (str): 'LM1'.. and 'Other'

    Note: This *does not* attempt to match basin IDs across subjects.
          It explicitly discards absolute basin_id and uses within-subject roles.
    """
    _require_pandas()

    df = traj_df.copy()
    if basin_col not in df.columns:
        raise ValueError(f"traj_df must contain '{basin_col}'")
    if "subject_id" not in df.columns:
        raise ValueError("traj_df must contain 'subject_id'")

    df["basin_indiv"] = df[basin_col].astype(int)

    # compute rank map per subject (cached)
    rank_maps = {}
    for sid in df["subject_id"].unique():
        sid_int = int(sid)
        theta = _theta_for_subject(mu_all, sid_int, theta_strategy=theta_strategy)
        rmap, _ = _role_rank_map_for_theta(analyzer, theta, rank_by=rank_by)
        rank_maps[sid_int] = rmap

    def _lookup_rank(row):
        rmap = rank_maps.get(int(row["subject_id"]), {})
        return rmap.get(int(row["basin_indiv"]), np.nan)

    df["basin_rank"] = df.apply(_lookup_rank, axis=1)

    # role category: keep top_roles, else Other
    def _role_cat(x):
        if np.isnan(x):
            return top_roles
        x = int(x)
        return x if x < top_roles else top_roles

    df["role_cat"] = df["basin_rank"].map(_role_cat).astype(int)
    labels = [f"LM{i+1}" for i in range(top_roles)] + ["Other"]
    df["role_label"] = df["role_cat"].map(lambda i: labels[i])
    return df


def plot_role_rank_timeseries(
    traj_rank_df: 'pd.DataFrame',
    subj_df: 'pd.DataFrame' | None = None,
    sort_by: str = "score_mean",
    top_roles: int = 6,
    title: str = "Role-based basin trajectories",
    ax=None,
):
    """Subject x time raster for role_cat (LM1.. + Other)."""
    _require_pandas()

    df = traj_rank_df.copy()
    if subj_df is not None and ("score_mean" in subj_df.columns or "vs_group_JS" in subj_df.columns):
        keep = [c for c in ["subject_id", "score_mean", "vs_group_JS"] if c in subj_df.columns]
        df = df.merge(subj_df[keep].drop_duplicates("subject_id"), on="subject_id", how="left")

    if sort_by in df.columns:
        order = (df[["subject_id", sort_by]].drop_duplicates()
                 .sort_values(sort_by)["subject_id"].astype(int).tolist())
    else:
        order = sorted(df["subject_id"].unique().astype(int).tolist())

    mat = (df.sort_values(["subject_id", "time_idx"])\
             .drop_duplicates(["subject_id", "time_idx"])\
             .pivot(index="subject_id", columns="time_idx", values="role_cat")\
             .reindex(order))

    # discrete colormap
    palette = sns.color_palette("Set2", n_colors=top_roles + 1)
    cmap = ListedColormap(palette)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(mat.values, aspect="auto", interpolation="nearest", cmap=cmap,
                   vmin=0, vmax=top_roles)

    ax.set_title(title)
    ax.set_xlabel("time_idx")
    ax.set_ylabel("subjects")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([str(c) for c in mat.columns])

    # sparse y ticks
    step = max(1, len(order) // 10)
    yt = np.arange(0, len(order), step)
    ax.set_yticks(yt)
    ax.set_yticklabels([str(order[i]) for i in yt])

    # legend-like colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, ticks=np.arange(top_roles + 1))
    labels = [f"LM{i+1}" for i in range(top_roles)] + ["Other"]
    cbar.ax.set_yticklabels(labels)
    cbar.set_label("role")

    return ax


def _lowess(x, y, frac: float = 0.6):
    """Try statsmodels LOWESS; fallback to simple moving-average smoothing."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        out = lowess(y, x, frac=frac, return_sorted=True)
        return out[:, 0], out[:, 1]
    except Exception:
        # fallback: sort and do a simple window average
        order = np.argsort(x)
        xs = x[order]
        ys = y[order]
        w = max(3, int(len(xs) * frac))
        yhat = np.convolve(ys, np.ones(w) / w, mode="same")
        return xs, yhat


def plot_distance_vs_score_with_ci(
    subj_df: 'pd.DataFrame',
    x: str = "score_mean",
    y: str = "vs_group_JS",
    frac: float = 0.6,
    n_boot: int = 1000,
    ci: float = 95,
    seed: int = 0,
    grid_n: int = 200,
    title: str = "Distance-to-group vs score (LOWESS + bootstrap CI)",
    ax=None,
):
    """Scatter + LOWESS curve + bootstrap CI band."""
    _require_pandas()

    d = subj_df[[x, y]].dropna().copy()
    xv = d[x].to_numpy(float)
    yv = d[y].to_numpy(float)

    xgrid = np.linspace(xv.min(), xv.max(), grid_n)

    # fit on full data
    xs_fit, ys_fit = _lowess(xv, yv, frac=frac)
    ygrid = np.interp(xgrid, xs_fit, ys_fit)

    # bootstrap
    rng = np.random.default_rng(seed)
    preds = np.zeros((n_boot, grid_n), float)
    n = len(d)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        xb = xv[idx]
        yb = yv[idx]
        xs_b, ys_b = _lowess(xb, yb, frac=frac)
        preds[b] = np.interp(xgrid, xs_b, ys_b)

    alpha = (100 - ci) / 2
    lo = np.percentile(preds, alpha, axis=0)
    hi = np.percentile(preds, 100 - alpha, axis=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    sns.scatterplot(data=d, x=x, y=y, ax=ax, s=40, edgecolor="none", alpha=0.85)
    ax.plot(xgrid, ygrid)
    ax.fill_between(xgrid, lo, hi, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return ax


def write_individual_report_pdf(
    analyzer,
    traj_df: 'pd.DataFrame',
    mu_all,
    subj_df: 'pd.DataFrame',
    out_pdf: str = "individual_landscape_report.pdf",
    basin_col: str = "basin_id",
    rank_by: str = "mass",
    theta_strategy: str = "mean",
    top_roles: int = 6,
    sort_by: str = "score_mean",
    lowess_frac: float = 0.6,
    n_boot: int = 1000,
    ci: float = 95,
    seed: int = 0,
):
    """
    Write an *individual-level* report PDF:
      - Page1: role-based basin trajectories (LM1.. + Other)
      - Page2: Distance-to-group (vs_group_JS) vs score_mean with CI band

    Note: This is *not* tied to any manuscript figure number.
    """
    _require_pandas()

    df_rank = add_role_rank_to_traj_df(
        analyzer=analyzer,
        traj_df=traj_df,
        mu_all=mu_all,
        basin_col=basin_col,
        rank_by=rank_by,
        theta_strategy=theta_strategy,
        top_roles=top_roles,
    )

    # ensure subject-level columns exist
    need_cols = {"subject_id", "score_mean", "vs_group_JS"}
    missing = need_cols - set(subj_df.columns)
    if missing:
        raise ValueError(f"subj_df missing required columns: {missing}")

    with PdfPages(out_pdf) as pdf:
        # Page 1
        fig1, ax1 = plt.subplots(figsize=(10, 7))
        plot_role_rank_timeseries(
            traj_rank_df=df_rank,
            subj_df=subj_df,
            sort_by=sort_by,
            top_roles=top_roles,
            title=f"Individual trajectories in role space (rank_by={rank_by})",
            ax=ax1,
        )
        fig1.tight_layout()
        pdf.savefig(fig1)
        plt.close(fig1)

        # Page 2
        fig2, ax2 = plt.subplots(figsize=(7, 6))
        plot_distance_vs_score_with_ci(
            subj_df=subj_df,
            x="score_mean",
            y="vs_group_JS",
            frac=lowess_frac,
            n_boot=n_boot,
            ci=ci,
            seed=seed,
            title="Distance-to-group vs score_mean (LOWESS + bootstrap CI)",
            ax=ax2,
        )
        fig2.tight_layout()
        pdf.savefig(fig2)
        plt.close(fig2)

    return out_pdf


def write_group_report_pdf(
    analyzer,
    eta,
    out_pdf: str = "group_landscape_report.pdf",
    top_k_minima: int = 10,
    title: str = "Group energy landscape summary",
):
    """Compact group-level PDF summary."""
    land = analyzer.compute_energy_landscape(eta, show_progress=False)

    mins = _to_numpy(land.get("local_minima_indices", []))
    occ = _to_numpy(land.get("occupation_times", []))
    energies = _to_numpy(land.get("energies", []))

    with PdfPages(out_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(title)

        if mins is not None and len(mins) > 0:
            mins = mins.astype(int)
            if occ is not None and len(occ) == len(mins):
                order = np.argsort(occ)[::-1]
            else:
                # fallback: sort by depth
                order = np.argsort(energies[mins])

            show = order[:min(top_k_minima, len(order))]
            labels = [str(int(mins[i])) for i in show]
            vals = [float(occ[i]) if occ is not None and len(occ)==len(mins) else float(np.exp(-energies[int(mins[i])] ))
                    for i in show]

            ax.bar(range(len(vals)), vals)
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_xlabel("local minima (state_index)")
            ax.set_ylabel("occupation (or proxy)")
        else:
            ax.text(0.5, 0.5, "No minima found", ha="center", va="center")

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    return out_pdf
