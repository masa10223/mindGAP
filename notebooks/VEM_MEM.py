"""
VEM-MEM (Variational EM for Pairwise Maximum Entropy Model) - JAXベースの実装

このモジュールは、JAXを使用した高速なVEM-MEMモデルの実装と、
それに統合されたランドスケープ解析を提供します。

主な機能:
- VEMMEM: VEM-MEMモデルの学習
- LandscapeAnalyzer: JAXベースのランドスケープ解析（VEM-MEMと統合）
- ResultVisualizer: 基本的な可視化

Note: 解析・可視化機能（TrajectoryComparator, TrajectoryVisualizer など）は
ELA_analysis.py に移動しました。汎用的なランドスケープ解析や可視化にも
ELA_analysis.py の LandscapeEngine や resultsAnalyzer を使用してください。
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.special import logsumexp
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from functools import partial
import networkx as nx
from tqdm import tqdm
import math
import seaborn as sns

# ==========================================
# 0. 入力データ要件のメモ
# ==========================================
"""
【入力データ (data_all) の要件】
形式: 3次元の numpy.ndarray または jax.numpy.ndarray
形状: (N, T, d)
    - N (Subjects): 被験者数 (例: 20人)
    - T (Time points): 各被験者の時系列データの長さ (例: 100時点)
    - d (Nodes/ROIs): 脳領域の数 (例: 10箇所) ※計算量的に d=15程度が限界です

値:
    - 2値データ (Binary): 0 または 1 (活動なし/活動あり)
    - もし元のデータが連続値(fMRIのBOLD信号など)の場合は、事前にZスコア化して
      0以上なら1、未満なら0にする等の「2値化処理」を行ってください。

【注意点】
    - d (ノード数) が増えると、状態数は 2^d で爆発的に増えます。
      d=10 -> 1,024状態 (余裕)
      d=15 -> 32,768状態 (重いが計算可能)
      d=20 -> 1,048,576状態 (メモリ不足のリスク大)
"""

# ==========================================
# 1. Config & Core Model (VEMMEM)
# ==========================================

@dataclass
class VEMConfig:
    """
    VEMアルゴリズムのハイパーパラメータ設定クラス
    
    Attributes:
        max_iter (int): 推定の最大反復回数。収束しない場合はこの回数で打ち切ります。
        tol (float): 収束判定の許容誤差。ELBOの変化率がこれを下回ると収束とみなします。
        alpha_h_init (float): 外部磁場(h)に対する事前分布の精度(precision)の初期値。
        alpha_J_init (float): 相互作用(J)に対する事前分布の精度(precision)の初期値。
        reg (float): 数値計算安定化のための正則化項（逆行列計算時の対角成分への加算値）。
    """
    max_iter: int = 1000
    tol: float = 1e-6
    alpha_h_init: float = 5.0
    alpha_J_init: float = 25.0
    reg: float = 1e-5


class VEMMEM:
    """
    VEM-MEM (Variational EM for Pairwise Maximum Entropy Model) のコア実装
    
    論文の "2.2. Bayesian formulation and variational EM algorithm" に基づく。
    集団全体のデータから事前分布(Prior)を学習しつつ、個人のパラメータ(Posterior)を推定します。
    """
    
    def __init__(self, d: int, config: VEMConfig = None):
        """
        Args:
            d (int): ノード数 (ROIの数)。
            config (VEMConfig): 設定オブジェクト。Noneの場合はデフォルト値を使用。
        """
        self.d = d
        # パラメータ総数 D = d (バイアスh) + d(d-1)/2 (相互作用J)
        self.D = d + d * (d - 1) // 2
        self.config = config or VEMConfig()
        
        # 全状態 (2^d 通り) を事前に生成して保持します。
        # これは分配関数 Z(θ) や期待値の計算に毎回使用します。
        self.all_states = self._generate_all_states() # Shape: (2^d, d), dtype=float32
        self.n_states = len(self.all_states)

    def _generate_all_states(self) -> jnp.ndarray:
        """
        0から2^d-1までの整数をビット演算で2進数配列に変換し、全状態パターンを生成します。
        例 (d=2): [[0,0], [0,1], [1,0], [1,1]]
        """
        n_states = 1 << self.d
        # ビットシフトを利用した高速な全状態生成
        s = (jnp.arange(n_states)[:, None] & (1 << jnp.arange(self.d))) > 0
        # GPU XLA では int8 行列演算が unsupported になることがあるため float32 を使う
        return s.astype(jnp.float32)
    
    @staticmethod
    @partial(jit, static_argnums=(1,))
    def _expand_state(sigma: jnp.ndarray, d: int) -> jnp.ndarray:
        """
        状態ベクトル σ (d次元) を、モデルパラメータ θ と内積が取れる形 (D次元) に拡張します。
        拡張ベクトル = [σ_1, ..., σ_d,  σ_1σ_2, σ_1σ_3, ..., σ_{d-1}σ_d]
        
        前半 d個: h (外部磁場) に対応
        後半 D-d個: J (相互作用) に対応
        """
        i, j = jnp.triu_indices(d, k=1) # 上三角行列のインデックスを取得
        sigma = sigma.astype(jnp.float32)
        return jnp.concatenate([sigma, sigma[i] * sigma[j]])

    def _compute_expectations(self, eta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """現在の η に基づく期待値・共分散・log Z。"""
        return _compute_expectations_impl(self.all_states, self.d, eta)

    @partial(jit, static_argnums=(0, 2))
    def _e_step_internal(self, emp_mean_n, T, eta, alpha, mean_eta, cov_eta):
        """
        【E-step (個人パラメータの推定)】
        論文 Eq. (15) & (16)
        
        Args:
            emp_mean_n: その被験者の観測データの平均 (経験的モーメント)
            T: データ点数 (観測の信頼度重みとして機能)
            eta: グループ事前分布の平均
            alpha: グループ事前分布の精度
            mean_eta, cov_eta: グループ平均パラメータでのモデル期待値・共分散
        
        Returns:
            mu_n: 事後分布の平均 (推定された個人のパラメータ)
            beta_n: 事後分布の精度行列の対角成分
        """
        # 事後精度の計算: データ数Tが多いほど、データの共分散(cov_eta)の影響が強くなる
        precision_matrix = T * cov_eta + jnp.diag(alpha)
        
        # 観測データとモデル平均のズレ (residual)
        residual = emp_mean_n - mean_eta
        
        # 補正項の計算 (逆行列の代わりに solve を使用して安定化)
        # mu_n = eta + (Precision)^-1 * T * (Data - Mean)
        # 基本は eta (みんなと同じ) だが、データが十分あれば個人の特徴の方へ修正する
        correction = jnp.linalg.solve(
            precision_matrix + jnp.eye(self.D) * self.config.reg, # 正則化
            T * residual
        )
        mu_n = eta + correction
        
        # 近似的な事後精度 (対角成分のみ計算して保持)
        beta_n = T * jnp.diag(cov_eta) + alpha
        return mu_n, beta_n

    @partial(jit, static_argnums=(0,))
    def _m_step_internal(self, mu_all, beta_all):
        """
        【M-step (グループ事前分布の更新)】
        論文 Eq. (17) および Remark (P.4)
        全員の推定パラメータ(mu_all)を集めて、新しいグループ平均(eta)と精度(alpha)を更新します。
        """
        # 1. 新しいグループ平均 η (全員の平均)
        eta = jnp.mean(mu_all, axis=0)
        
        # 2. ばらつき(分散)の計算
        # (個人のパラメータ - グループ平均)^2 + 推定の不確実性(1/beta)
        variances = jnp.mean((mu_all - eta)**2 + 1.0 / beta_all, axis=0)
        epsilon = 1e-8 # ゼロ除算防止
        
        # 3. 精度の更新 (Remarkに基づき、hとJでそれぞれ1つのスカラーにまとめる)
        # これにより、パラメータ数が多くても過学習を防ぎ安定させる
        alpha_h = 1.0 / (jnp.mean(variances[:self.d]) + epsilon)
        alpha_J = 1.0 / (jnp.mean(variances[self.d:]) + epsilon)
        
        alpha = jnp.concatenate([
            jnp.full(self.d, alpha_h),
            jnp.full(self.D - self.d, alpha_J)
        ])
        return eta, alpha

    @partial(jit, static_argnums=(0, 2))
    def _compute_elbo_internal(self, emp_mean_n, T, mu_n, beta_n, eta, alpha, mean_eta, cov_eta, log_Z):
        """
        ELBO (Evidence Lower Bound) の計算。
        この値が上昇しなくなったら「収束した」とみなします。
        """
        dev = mu_n - eta
        # F1項: データの対数尤度の期待値
        f1  = T * jnp.dot(mu_n, emp_mean_n) - T * log_Z - T * jnp.dot(mean_eta, dev)
        f1 -= 0.5 * T * (jnp.sum(jnp.diag(cov_eta) / beta_n) + jnp.dot(dev, cov_eta @ dev))
        # F2項: 事前分布とのKLダイバージェンス関連
        f2  = 0.5 * jnp.sum(jnp.log(alpha)) - 0.5 * jnp.sum(alpha * (dev**2 + 1.0 / beta_n))
        # F3項: エントロピー項
        f3 = -0.5 * jnp.sum(jnp.log(beta_n))
        return f1 + f2 + f3

    def extract_h_J(self, theta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        1次元のパラメータベクトル θ を、h (バイアスベクトル) と J (相互作用行列) に分解します。
        Returns:
            h: shape (d,)
            J: shape (d, d) 対称行列
        """
        h = theta[:self.d]
        J_vec = theta[self.d:]
        J = jnp.zeros((self.d, self.d))
        idx = 0
        for i in range(self.d):
            for j in range(i+1, self.d):
                J = J.at[i, j].set(J_vec[idx])
                J = J.at[j, i].set(J_vec[idx])
                idx += 1
        return h, J

    def fit(self, data_all: jnp.ndarray, verbose: bool = True, time_mode: str = "aggregate", aggregate_how: str = "mean") -> Dict:
        """
        【メインの学習関数】
        
        Args:
            data_all: 入力データ。Shape (N, T, d)。
            verbose: プログレスバーを表示するかどうか。
            time_mode:
                - "aggregate": (推奨) 被験者ごとに全時点Tをまとめて、1つのパラメータθ_nを推定します。
                  個人の定常的な脳内ネットワーク特徴を知りたい場合はこちら。
                - "per_time": 全時点を独立とみなし、(n,t)ごとにパラメータθ_{n,t}を推定します。
                  瞬間的なダイナミクスを見たい場合に使いますが、推定が不安定になりやすいです。
            aggregate_how: aggregateモード時のデータ集約方法。"mean"推奨。
            
        Returns:
            Dict: 推定結果を格納した辞書
                - eta: グループ平均パラメータ
                - mu_all: 推定された個人のパラメータ
                - final_elbo: 最終的なELBO値
                - ...
        """
        # 入力チェック
        N, T, d = data_all.shape
        if d != self.d: raise ValueError(f"Model d={self.d} but data d={d}.")
        
        # データの拡張: 生データ(0/1) -> 拡張特徴量(0/1の積)
        # vmapを使って、全被験者・全時点を一括変換します
        expand_T = vmap(partial(type(self)._expand_state, d=self.d))
        emp_means_all = vmap(expand_T)(data_all) # Shape: (N, T, K)

        # パラメータの初期化
        key = jax.random.PRNGKey(0)
        eta = jax.random.normal(key, (self.D,), dtype=jnp.float32) * jnp.float32(0.1)
        alpha = jnp.concatenate([
            jnp.full(self.d, self.config.alpha_h_init, dtype=jnp.float32),
            jnp.full(self.D - self.d, self.config.alpha_J_init, dtype=jnp.float32),
        ])
        
        # 事前の期待値計算
        mean_eta, cov_eta, logZ_eta = self._compute_expectations(eta)
        elbo_history = []
        elbo_prev = -jnp.inf
        
        iters = tqdm(
            range(self.config.max_iter), 
            desc=f"VEM ({time_mode})",
            miniters=1,
            mininterval=0.1,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        ) if verbose else range(self.config.max_iter)

        for _ in iters:
            # ---------------------------
            # 1. E-Step: 個人パラメータ(mu)の更新
            # ---------------------------
            if time_mode == "aggregate":
                # 被験者ごとの平均活動を計算
                if aggregate_how == "mean":
                    emp_input = jnp.mean(emp_means_all, axis=1) # (N, K)
                    T_scalar = T # データ数としての重み
                else:
                    emp_input = jnp.sum(emp_means_all, axis=1)
                    T_scalar = 1
                
                # vmapで全員分並列計算
                mu_sub, beta_sub = vmap(self._e_step_internal, in_axes=(0, None, None, None, None, None))(
                    emp_input, T_scalar, eta, alpha, mean_eta, cov_eta
                )
                # 後の処理のためにShapeを合わせる
                mu_flat, beta_flat = mu_sub, beta_sub
                mu_all = mu_sub[:, None, :] # (N, 1, K)
                beta_all = beta_sub[:, None, :]
                
                # ELBO計算
                elbo_func = vmap(self._compute_elbo_internal, in_axes=(0, None, 0, 0, None, None, None, None, None))
                elbo = jnp.sum(elbo_func(emp_input, T_scalar, mu_sub, beta_sub, eta, alpha, mean_eta, cov_eta, logZ_eta))

            else: # per_timeモード
                # (N*T, K) にフラット化して全時点を独立サンプルとして扱う
                emp_flat = emp_means_all.reshape(N*T, self.D)
                mu_flat, beta_flat = vmap(self._e_step_internal, in_axes=(0, None, None, None, None, None))(
                    emp_flat, 1, eta, alpha, mean_eta, cov_eta
                )
                mu_all = mu_flat.reshape(N, T, self.D)
                beta_all = beta_flat.reshape(N, T, self.D)
                
                elbo_func = vmap(self._compute_elbo_internal, in_axes=(0, None, 0, 0, None, None, None, None, None))
                elbo = jnp.sum(elbo_func(emp_flat, 1, mu_flat, beta_flat, eta, alpha, mean_eta, cov_eta, logZ_eta))

            # ---------------------------
            # 2. M-Step: グループパラメータ(eta, alpha)の更新
            # ---------------------------
            eta, alpha = self._m_step_internal(mu_flat, beta_flat)
            
            # ---------------------------
            # 3. 収束判定と準備
            # ---------------------------
            elbo_history.append(float(elbo))
            # 更新された eta に基づいて期待値を再計算
            mean_eta, cov_eta, logZ_eta = self._compute_expectations(eta)
            
            if verbose and hasattr(iters, "set_postfix"):
                iters.set_postfix(elbo=float(elbo))
            
            # ELBOの変化率が閾値未満なら終了
            if abs(elbo - elbo_prev) < abs(self.config.tol * elbo_prev):
                break
            elbo_prev = elbo

        return {
            "eta": np.array(eta), 
            "alpha": np.array(alpha),
            "mu_all": np.array(mu_all), 
            "beta_all": np.array(beta_all),
            "final_elbo": float(elbo), 
            "elbo_history": np.array(elbo_history),
            "n_iterations": len(elbo_history)
        }


@partial(jit, static_argnums=(1,))
def _compute_expectations_impl(
    all_states: jnp.ndarray, d: int, eta: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """η に対するモデル期待値・共分散・log Z（モジュールレベル jit）."""
    expanded_states = vmap(VEMMEM._expand_state, in_axes=(0, None))(
        all_states.astype(jnp.float32), d
    )
    eta = eta.astype(jnp.float32)
    scores = expanded_states @ eta
    log_Z = logsumexp(scores)
    probs = jnp.exp(scores - log_Z)
    mean_eta = jnp.sum(expanded_states * probs[:, None], axis=0)
    centered_states = expanded_states - mean_eta
    cov_eta = (centered_states * probs[:, None]).T @ centered_states
    return mean_eta, cov_eta, log_Z


# ==========================================
# 2. Integrated Landscape Analyzer
# ==========================================

class LandscapeAnalyzer:
    """
    推定されたパラメータを用いてエネルギーランドスケープ解析を行うクラス（JAXベース）
    
    このクラスは VEM-MEM モデル（VEMMEM）と統合されており、JAXによる高速化を活用します。
    
    機能:
        1. エネルギー地形の計算 (全状態のエネルギー算出)
        2. 局所安定状態 (Local Minima) の探索
        3. 誘引流域 (Basin) の特定とサイズ計算
        4. 特徴量抽出 (OCC1, OCC1+2, LM数など)
        5. 状態遷移ネットワーク (Disconnectivity Graph) 用データの構築
        6. 時系列軌跡解析 (各被験者がランドスケープ上でどのように移動したかを追跡)
    
    使用例:
        # 軌跡解析の実行
        analyzer = LandscapeAnalyzer(model)
        trajectory_df = analyzer.analyze_all_trajectories(data_all, mu_all)
        stats_df = analyzer.calculate_trajectory_statistics(trajectory_df)
        
        # CSVで保存
        trajectory_df.to_csv('trajectory_analysis.csv', index=False)
        stats_df.to_csv('trajectory_statistics.csv', index=False)
    
    Note: 汎用的なランドスケープ解析（VEM-MEMモデルを使わない場合）には、
    ELA_analysis.py の LandscapeEngine を使用してください。
    """
    def __init__(self, vem_model: VEMMEM):
        self.model = vem_model
        self.d = vem_model.d
        # 全状態の特徴量を事前に計算 (2^d, K)
        self._expanded_states = vmap(partial(type(self.model)._expand_state, d=self.d))(self.model.all_states) 
        
        # 状態間の「隣接関係」グラフを構築 (ハミング距離=1)
        self.state_graph = self._build_state_space_graph()
        
        # JAX高速化用: 隣接行列を事前計算 (2^d, 2^d)
        self._adjacency_matrix = self._build_adjacency_matrix()
        
        # JAX高速化用: 状態マッチング用の状態配列をJAX配列として保持
        self._states_jax = jnp.array(self.model.all_states)
        
        # ランドスケープ計算結果のキャッシュ（高速化のため）
        self._landscape_cache = {}

    def _build_state_space_graph(self):
        """全状態(2^d個)をノードとし、1ビット違いの状態間にエッジを張ったグラフを作成"""
        G = nx.Graph()
        states_np = np.array(self.model.all_states)
        # NetworkX用に tuple(float) に変換
        state_tuples = [tuple(float(x) for x in s) for s in states_np]
        G.add_nodes_from(range(len(state_tuples))) 
        
        # エッジの追加 (全ペアのハミング距離計算)
        # ※ d>15 の場合は重いので注意
        for i in range(len(state_tuples)):
            for j in range(i + 1, len(state_tuples)):
                if np.sum(np.abs(states_np[i] - states_np[j])) == 1:
                    G.add_edge(i, j)
        return G
    
    def _build_adjacency_matrix(self) -> jnp.ndarray:
        """隣接行列をJAX配列として構築（ハミング距離=1の状態間にエッジ）"""
        n_states = len(self.model.all_states)
        states_jax = jnp.array(self.model.all_states)  # (2^d, d)
        
        # 全ペアのハミング距離を計算
        # states_jax[i] と states_jax[j] のハミング距離
        def hamming_dist(i, j):
            return jnp.sum(states_jax[i] != states_jax[j])
        
        # ベクトル化して隣接行列を構築
        i_indices = jnp.arange(n_states)[:, None]  # (n_states, 1)
        j_indices = jnp.arange(n_states)[None, :]  # (1, n_states)
        
        # ブロードキャストで全ペアのハミング距離を計算
        states_expanded_i = states_jax[i_indices]  # (n_states, n_states, d)
        states_expanded_j = states_jax[j_indices]  # (n_states, n_states, d)
        hamming_dists = jnp.sum(states_expanded_i != states_expanded_j, axis=2)  # (n_states, n_states)
        
        # ハミング距離=1の場合は隣接（1）、それ以外は0
        adj_matrix = (hamming_dists == 1).astype(jnp.int32)
        
        return adj_matrix

    def _compute_basins_fast(
        self, 
        energies: np.ndarray, 
        adj_matrix: np.ndarray,
        show_progress: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        高速化されたBasin探索（Steepest Descent）
        NumPyベースだが、隣接行列を使うことで高速化
        
        Args:
            energies: (2^d,) 各状態のエネルギー
            adj_matrix: (2^d, 2^d) 隣接行列（NumPy配列）
            show_progress: プログレスバーを表示するか
        
        Returns:
            basins: (2^d,) 各状態が属するBasin（Local Minimumのインデックス）
            lm_mask: (2^d,) Local Minimumかどうかのマスク
        """
        n_states = len(energies)
        basins = np.full(n_states, -1, dtype=int)
        lm_indices = []
        
        # エネルギーの低い順にソート
        sorted_indices = np.argsort(energies)
        
        # プログレスバーを設定
        if show_progress and n_states > 100:  # 状態数が少ない場合は表示しない
            iterator = tqdm(
                sorted_indices,
                desc="Basin exploration",
                miniters=max(1, n_states // 100),
                mininterval=0.5,  # 更新間隔を長くして表示のちらつきを減らす
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                leave=False,
                disable=False
            )
        else:
            iterator = sorted_indices
        
        # 隣接行列から各状態の隣接状態を取得（ベクトル化）
        for idx in iterator:
            e_curr = energies[idx]
            
            # 隣接状態のインデックスを取得
            neighbors = np.where(adj_matrix[idx] > 0)[0]
            
            if len(neighbors) == 0:
                best_n = idx
            else:
                neighbor_energies = energies[neighbors]
                min_n_idx = np.argmin(neighbor_energies)
                if neighbor_energies[min_n_idx] < e_curr:
                    best_n = neighbors[min_n_idx]
                else:
                    best_n = idx
            
            if best_n == idx:
                basins[idx] = idx
                lm_indices.append(idx)
            else:
                basins[idx] = basins[best_n]
        
        lm_mask = np.zeros(n_states, dtype=bool)
        lm_mask[lm_indices] = True
        
        return basins, lm_mask
    
    @partial(jit, static_argnums=(0,))
    def _compute_energies_probs_batch(self, thetas: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        バッチ処理用: 複数のthetaに対してエネルギーと確率を計算（JAX高速化）
        
        Args:
            thetas: (batch_size, D) のパラメータ配列
        
        Returns:
            energies: (batch_size, 2^d) 各thetaに対するエネルギー
            probs: (batch_size, 2^d) 各thetaに対する確率
        """
        def compute_for_theta(theta):
            scores = jnp.dot(self._expanded_states, theta)
            energies = -scores
            log_Z = logsumexp(scores)
            probs = jnp.exp(scores - log_Z)
            return energies, probs
        
        energies_batch, probs_batch = vmap(compute_for_theta)(thetas)
        return energies_batch, probs_batch
    
    def compute_energy_landscape(
        self, 
        theta: jnp.ndarray,
        show_progress: bool = True,
        use_cache: bool = True
    ) -> Dict:
        """
        エネルギー、確率、Basin、Local Minimaを一括計算（JAX高速化版）
        
        Args:
            theta: パラメータベクトル
            show_progress: プログレスバーを表示するか
            use_cache: キャッシュを使用するか（デフォルト: True）
        
        Returns:
            Dict: ランドスケープ解析結果
        """
        # キャッシュチェック（thetaをハッシュ可能な形式に変換）
        if use_cache:
            theta_tuple = tuple(float(x) for x in np.array(theta).flatten())
            if theta_tuple in self._landscape_cache:
                return self._landscape_cache[theta_tuple]
        
        # プログレスバーで処理ステップを表示
        if show_progress:
            pbar = tqdm(
                total=4,
                desc="Computing energy landscape",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                leave=True
            )
        
        # 1. エネルギー計算: E = -θ^T * σ (JAXで高速化)
        if show_progress:
            pbar.set_description("Computing energies")
        scores = jnp.dot(self._expanded_states, theta) # (S,)
        energies = -scores
        if show_progress:
            pbar.update(1)
        
        # 2. 確率計算: P ∝ exp(-E) (JAXで高速化)
        if show_progress:
            pbar.set_description("Computing probabilities")
        log_Z = logsumexp(scores)
        probs = jnp.exp(scores - log_Z)
        if show_progress:
            pbar.update(1)
        
        # NumPy配列に変換（Basin探索はNumPyで行う）
        energies_np = np.array(energies)
        probs_np = np.array(probs)
        adj_matrix_np = np.array(self._adjacency_matrix)
        
        # 3. Local Minima & Basin探索 (高速化版: 隣接行列を使用)
        if show_progress:
            pbar.set_description("Exploring basins")
        basins, lm_mask = self._compute_basins_fast(energies_np, adj_matrix_np, show_progress=show_progress)
        lm_indices = np.where(lm_mask)[0]
        if show_progress:
            pbar.update(1)
        
        # Basinサイズと滞在確率を計算
        if show_progress:
            pbar.set_description("Calculating basin statistics")
        basin_sizes = np.array([np.sum(basins == lm) for lm in lm_indices])
        occ_times = np.array([np.sum(probs_np[basins == lm]) for lm in lm_indices])
        if show_progress:
            pbar.update(1)
            pbar.close()
        
        result = {
            'energies': energies_np,
            'probabilities': probs_np,
            'local_minima_indices': lm_indices,
            'basin_mapping': basins,
            'basin_sizes': basin_sizes,
            'occupation_times': occ_times,
            'states': np.array(self.model.all_states)
        }
        
        # キャッシュに保存
        if use_cache:
            theta_tuple = tuple(float(x) for x in np.array(theta).flatten())
            self._landscape_cache[theta_tuple] = result
        
        return result

    def landscape_to_attractor_table(
        self,
        land: Dict[str, Union[np.ndarray, List[int]]],
        subject_id: Optional[int] = None,
        rank_by: str = "energy",
    ) -> pd.DataFrame:
        """
        ランドスケープの局所最小（アトラクター）だけを列挙する。

        Args:
            land: compute_energy_landscape の戻り値
            subject_id: 被験者ID（任意）
            rank_by: 'energy'（深い順=LM1が0）または 'mass'（滞在確率の大きい順）

        Returns:
            columns: subject_id, basin_rank, state_index, state_str, bit_0..bit_{d-1},
                     energy, probability_mass, basin_size, n_minima
        """
        if rank_by not in ("energy", "mass"):
            raise ValueError("rank_by must be 'energy' or 'mass'")

        states = np.asarray(land["states"], dtype=int)
        lm = np.asarray(land["local_minima_indices"], dtype=int)
        occ = np.asarray(land["occupation_times"], dtype=float)
        sizes = np.asarray(land["basin_sizes"], dtype=int)
        energies = np.asarray(land["energies"], dtype=float)

        if lm.size == 0:
            cols = ["basin_rank", "state_index", "state_str"]
            cols += [f"bit_{i}" for i in range(self.d)]
            cols += ["energy", "probability_mass", "basin_size", "n_minima"]
            if subject_id is not None:
                cols = ["subject_id"] + cols
            return pd.DataFrame(columns=cols)

        e_lm = energies[lm]
        if rank_by == "energy":
            order = np.argsort(e_lm)
        else:
            order = np.argsort(-occ)

        rows = []
        n_minima = len(lm)
        for rank, j in enumerate(order):
            sidx = int(lm[j])
            bits = states[sidx]
            row = {
                "basin_rank": int(rank),
                "state_index": sidx,
                "state_str": "".join(str(int(b)) for b in bits),
                "energy": float(e_lm[j]),
                "probability_mass": float(occ[j]),
                "basin_size": int(sizes[j]),
                "n_minima": n_minima,
            }
            for i, b in enumerate(bits):
                row[f"bit_{i}"] = int(b)
            if subject_id is not None:
                row["subject_id"] = int(subject_id)
            rows.append(row)

        df = pd.DataFrame(rows)
        if subject_id is not None:
            df = df[["subject_id", "basin_rank", "state_index", "state_str"]
                    + [f"bit_{i}" for i in range(self.d)]
                    + ["energy", "probability_mass", "basin_size", "n_minima"]]
        return df

    def enumerate_attractors_all_subjects(
        self,
        mu_all: np.ndarray,
        rank_by: str = "energy",
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        全被験者について、アトラクター（局所最小）のみを列挙する。

        Args:
            mu_all: (N, D) または (N, 1, D) または (N, T, D)
            rank_by: basin_rank の付け方（'energy' または 'mass'）
        """
        mu_all = np.asarray(mu_all)
        if mu_all.ndim == 2:
            N = mu_all.shape[0]
        elif mu_all.ndim == 3:
            N = mu_all.shape[0]
        else:
            raise ValueError(f"Unsupported mu_all shape: {mu_all.shape}")

        tables = []
        iterator = range(N)
        if show_progress:
            iterator = tqdm(iterator, desc="Enumerate attractors")

        for sid in iterator:
            if mu_all.ndim == 3:
                theta_i = mu_all[sid, 0]
            else:
                theta_i = mu_all[sid]
            land_i = self.compute_energy_landscape(theta_i, show_progress=False)
            tables.append(
                self.landscape_to_attractor_table(
                    land_i, subject_id=sid, rank_by=rank_by
                )
            )

        if not tables:
            return self.landscape_to_attractor_table(
                self.compute_energy_landscape(mu_all[0], show_progress=False),
                subject_id=0,
                rank_by=rank_by,
            ).iloc[0:0]

        return pd.concat(tables, ignore_index=True)

    def calculate_features(self, theta: jnp.ndarray, show_progress: bool = True) -> Dict[str, float]:
        """
        論文のTable 1にあるような主要な特徴量を抽出します。
        
        Args:
            theta: パラメータベクトル
            show_progress: プログレスバーを表示するか（デフォルト: True）
        
        Returns:
            Dict:
                - nLM: Local Minimaの数
                - OCC1: 最もエネルギーが低い(安定な)状態の滞在確率
                - OCC2: 2番目に安定な状態の滞在確率
                - OCC1+2: OCC1 + OCC2
                - LM1_energy: 最安定状態のエネルギー
                - entropy: シャノンエントロピー
        """
        res = self.compute_energy_landscape(theta, show_progress=show_progress)
        
        lm_indices = res['local_minima_indices']
        energies = res['energies']
        occ_times = res['occupation_times'] # これは lm_indices の順序に対応
        states = res['states']
        
        # LMをエネルギーが低い順（安定順）にソートして、LM1, LM2... を特定する
        lm_energies = energies[lm_indices]
        sorted_rank = np.argsort(lm_energies) # エネルギー昇順のインデックス
        lm_states = states[lm_indices]
        
        # ソート後の指標
        sorted_occ = occ_times[sorted_rank]
        sorted_energies = lm_energies[sorted_rank]
        sorted_states = lm_states[sorted_rank]
        features = {}
        features['nLM'] = len(lm_indices)
        features['entropy'] = -np.sum(res['probabilities'] * np.log(res['probabilities'] + 1e-10))
        
        # LM1 (Global Minimum)
        if len(lm_indices) > 0:
            features['LM1_energy'] = sorted_energies[0]
            features['OCC1'] = sorted_occ[0]
            features['LM1_state'] = sorted_states[0]
        else:
            features['LM1_energy'] = np.nan
            features['OCC1'] = 0.0
            features['LM1_state'] = np.nan
        # LM2 (Second Minimum)
        if len(lm_indices) > 1:
            features['LM2_energy'] = sorted_energies[1]
            features['OCC2'] = sorted_occ[1]
            features['LM2_state'] = sorted_states[1]
            features['OCC1+2'] = sorted_occ[0] + sorted_occ[1]
            features['energy_gap'] = sorted_energies[1] - sorted_energies[0]
        else:
            features['LM2_energy'] = np.nan
            features['OCC2'] = 0.0
            features['LM2_state'] = np.nan
            features['OCC1+2'] = features.get('OCC1', 0.0)
            features['energy_gap'] = 0.0
            
        return features

    @partial(jit, static_argnums=(0,))
    def _compute_landscape_batch(self, thetas: jnp.ndarray) -> Dict:
        """
        バッチ処理用: 複数のthetaに対してエネルギーランドスケープを計算
        
        Args:
            thetas: (batch_size, D) のパラメータ配列
        
        Returns:
            Dict with batched results
        """
        batch_size = thetas.shape[0]
        
        # バッチ処理: 各thetaに対してエネルギーと確率を計算
        def compute_for_theta(theta):
            scores = jnp.dot(self._expanded_states, theta)
            energies = -scores
            log_Z = logsumexp(scores)
            probs = jnp.exp(scores - log_Z)
            return energies, probs
        
        energies_batch, probs_batch = vmap(compute_for_theta)(thetas)
        
        return {
            'energies': energies_batch,  # (batch_size, 2^d)
            'probabilities': probs_batch,  # (batch_size, 2^d)
        }
    
    def analyze_group_features(self, mu_all: np.ndarray, use_jax_batch: bool = True) -> pd.DataFrame:
        """
        全被験者のパラメータ (mu_all) を受け取り、全員分の特徴量を計算してDataFrameで返します。
        JAXバッチ処理で高速化可能。
        
        Args:
            mu_all: (N, T, D) または (N, 1, D) のパラメータ配列
            use_jax_batch: JAXバッチ処理を使用するか（デフォルト: True）
        
        Returns:
            pd.DataFrame: 各行が1被験者(1時点)の特徴量
        """
        # 形状の正規化 (N, T, D)
        if mu_all.ndim == 2:
            mu_all = mu_all[:, None, :]
            
        N, T, D = mu_all.shape
        records = []
        
        print("Calculating features for all subjects...")
        
        if use_jax_batch:
            # JAXバッチ処理で高速化: エネルギーと確率をバッチ計算、Basin探索は個別に
            mu_all_flat = mu_all.reshape(-1, D)  # (N*T, D)
            mu_all_jax = jnp.array(mu_all_flat)
            
            # バッチ処理でエネルギーと確率を計算（JAX高速化）
            energies_batch, probs_batch = self._compute_energies_probs_batch(mu_all_jax)
            energies_batch_np = np.array(energies_batch)  # (N*T, 2^d)
            probs_batch_np = np.array(probs_batch)  # (N*T, 2^d)
            adj_matrix_np = np.array(self._adjacency_matrix)
            
            # 各thetaに対してBasin探索と特徴量計算（tqdmで進捗表示）
            for idx in tqdm(
                range(N * T),
                desc="Computing landscapes",
                miniters=1,
                mininterval=0.1,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            ):
                n = idx // T
                t = idx % T
                
                # バッチ計算済みのエネルギーと確率を使用
                energies = energies_batch_np[idx]
                probs = probs_batch_np[idx]
                
                # Basin探索（プログレスバーは外側で表示するため非表示）
                basins, lm_mask = self._compute_basins_fast(energies, adj_matrix_np, show_progress=False)
                lm_indices = np.where(lm_mask)[0]
                
                # Basinサイズと滞在確率を計算
                basin_sizes = np.array([np.sum(basins == lm) for lm in lm_indices])
                occ_times = np.array([np.sum(probs[basins == lm]) for lm in lm_indices])
                
                # ランドスケープ結果を構築
                landscape = {
                    'energies': energies,
                    'probabilities': probs,
                    'local_minima_indices': lm_indices,
                    'basin_mapping': basins,
                    'basin_sizes': basin_sizes,
                    'occupation_times': occ_times,
                    'states': np.array(self.model.all_states)
                }
                
                # 特徴量を計算
                feats = self._calculate_features_from_landscape(landscape)
                
                # ID情報を追加
                feats['subject_id'] = n
                feats['time_idx'] = t
                records.append(feats)
        else:
            # 従来のループ処理（互換性のため残す、tqdmで進捗表示）
            total_items = N * T
            idx = 0
            for n in tqdm(
                range(N),
                desc="Computing landscapes",
                miniters=1,
                mininterval=0.1,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            ):
                for t in range(T):
                    theta = jnp.array(mu_all[n, t])
                    
                    # compute_energy_landscapeを直接呼び出し
                    landscape = self.compute_energy_landscape(theta, show_progress=True)
                    feats = self._calculate_features_from_landscape(landscape)
                    
                    # ID情報を追加
                    feats['subject_id'] = n
                    feats['time_idx'] = t
                    records.append(feats)
                    idx += 1
                
        return pd.DataFrame(records)
    
    def _calculate_features_from_landscape(self, res: Dict) -> Dict[str, float]:
        """compute_energy_landscapeの結果から特徴量を計算（内部関数）"""
        lm_indices = res['local_minima_indices']
        energies = res['energies']
        occ_times = res['occupation_times']
        states = res['states']
        
        # LMをエネルギーが低い順（安定順）にソート
        lm_energies = energies[lm_indices]
        sorted_rank = np.argsort(lm_energies)
        lm_states = states[lm_indices]
        
        sorted_occ = occ_times[sorted_rank]
        sorted_energies = lm_energies[sorted_rank]
        sorted_states = lm_states[sorted_rank]
        
        features = {}
        features['nLM'] = len(lm_indices)
        features['entropy'] = -np.sum(res['probabilities'] * np.log(res['probabilities'] + 1e-10))
        
        if len(lm_indices) > 0:
            features['LM1_energy'] = sorted_energies[0]
            features['OCC1'] = sorted_occ[0]
            features['LM1_state'] = sorted_states[0]
        else:
            features['LM1_energy'] = np.nan
            features['OCC1'] = 0.0
            features['LM1_state'] = np.nan
            
        if len(lm_indices) > 1:
            features['LM2_energy'] = sorted_energies[1]
            features['OCC2'] = sorted_occ[1]
            features['LM2_state'] = sorted_states[1]
            features['OCC1+2'] = sorted_occ[0] + sorted_occ[1]
            features['energy_gap'] = sorted_energies[1] - sorted_energies[0]
        else:
            features['LM2_energy'] = np.nan
            features['OCC2'] = 0.0
            features['LM2_state'] = np.nan
            features['OCC1+2'] = features.get('OCC1', 0.0)
            features['energy_gap'] = 0.0
            
        return features

    @partial(jit, static_argnums=(0,))
    def _match_states_jax(self, obs_states: jnp.ndarray, all_states: jnp.ndarray) -> jnp.ndarray:
        """
        JAXで高速化された状態マッチング
        
        Args:
            obs_states: (T, d) 観測された状態
            all_states: (2^d, d) 全状態空間
        
        Returns:
            state_indices: (T,) 各観測状態に対応する全状態空間のインデックス
        """
        T = obs_states.shape[0]
        n_states = all_states.shape[0]
        
        # 各観測状態について、全状態との一致をチェック
        # obs_states[t] と all_states[i] の一致判定
        obs_expanded = obs_states[:, None, :]  # (T, 1, d)
        all_expanded = all_states[None, :, :]  # (1, 2^d, d)
        
        # 完全一致をチェック
        matches = jnp.all(obs_expanded == all_expanded, axis=2)  # (T, 2^d)
        
        # 完全一致がある場合はそのインデックス、ない場合はハミング距離が最小のインデックス
        def find_match(match_row, obs_state):
            exact_match = jnp.any(match_row)
            exact_idx = jnp.argmax(match_row)  # 最初の一致
            
            # ハミング距離を計算
            hamming_dists = jnp.sum(all_states != obs_state, axis=1)  # (2^d,)
            min_hamming_idx = jnp.argmin(hamming_dists)
            
            # 完全一致があればそれを使う、なければハミング距離最小
            return jnp.where(exact_match, exact_idx, min_hamming_idx)
        
        state_indices = vmap(find_match)(matches, obs_states)
        return state_indices
    
    def analyze_trajectory(self, data_timeseries: np.ndarray, theta: jnp.ndarray, use_jax_match: bool = True, show_progress: bool = False) -> pd.DataFrame:
        """
        時系列データのランドスケープ上での軌跡を解析します（JAX高速化版）。
        
        Args:
            data_timeseries: 時系列データ (T, d) - 各時点での観測状態
            theta: その被験者のパラメータ（ランドスケープを定義）
            use_jax_match: JAXで状態マッチングを高速化するか（デフォルト: True）
            show_progress: プログレスバーを表示するか（デフォルト: False）
        
        Returns:
            pd.DataFrame: 各時点での軌跡情報
                - time_idx: 時点インデックス
                - observed_state: 観測された状態 (d次元のバイナリベクトル)
                - state_index: 全状態空間での状態インデックス
                - basin_id: 所属するBasin（Local Minimumのインデックス）
                - energy: その状態のエネルギー
                - probability: その状態の出現確率
        """
        T, d = data_timeseries.shape
        if d != self.d:
            raise ValueError(f"Data dimension {d} does not match model dimension {self.d}")
        
        # ランドスケープを計算（キャッシュを使用）
        landscape = self.compute_energy_landscape(theta, show_progress=show_progress)
        basin_mapping = landscape['basin_mapping']  # 各状態がどのBasinに属するか
        energies = landscape['energies']
        probabilities = landscape['probabilities']
        states = landscape['states']  # 全状態 (2^d, d)
        
        # 観測状態を全状態空間とマッチング（JAX高速化版）
        if use_jax_match:
            obs_states_jax = jnp.array(data_timeseries.astype(int))
            state_indices = self._match_states_jax(obs_states_jax, self._states_jax)
            state_indices = np.array(state_indices)
        else:
            # 従来のループ処理（互換性のため残す）
            state_indices = []
            for t in range(T):
                obs_state = data_timeseries[t].astype(int)
                state_match = np.all(states == obs_state, axis=1)
                if not np.any(state_match):
                    hamming_distances = np.sum(states != obs_state, axis=1)
                    state_idx = np.argmin(hamming_distances)
                else:
                    state_idx = np.where(state_match)[0][0]
                state_indices.append(state_idx)
            state_indices = np.array(state_indices)
        
        # 結果をDataFrameにまとめる
        records = []
        for t in range(T):
            obs_state = data_timeseries[t].astype(int)
            state_idx = state_indices[t]
            
            # Basin IDを取得
            basin_id = basin_mapping[state_idx]
            
            # 状態を文字列として保存（CSV出力時に扱いやすい）
            state_str = ''.join(map(str, obs_state.astype(int)))
            
            records.append({
                'time_idx': t,
                'observed_state': state_str,  # 文字列として保存
                'observed_state_array': obs_state,  # 配列も保存（後で使う場合）
                'state_index': int(state_idx),
                'basin_id': int(basin_id),
                'energy': float(energies[state_idx]),
                'probability': float(probabilities[state_idx])
            })
        
        return pd.DataFrame(records)
    
    def analyze_all_trajectories(self, data_all: np.ndarray, mu_all: np.ndarray, show_progress: bool = True) -> pd.DataFrame:
        """
        全被験者の時系列軌跡を一括解析します。
        
        Args:
            data_all: 時系列データ (N, T, d)
            mu_all: 各被験者のパラメータ (N, T, D) または (N, 1, D)
                    aggregateモードの場合は (N, 1, D)、per_timeモードの場合は (N, T, D)
            show_progress: プログレスバーを表示するか（デフォルト: True）
        
        Returns:
            pd.DataFrame: 全被験者・全時点の軌跡情報
                - subject_id: 被験者ID
                - time_idx: 時点インデックス
                - observed_state: 観測された状態
                - state_index: 全状態空間での状態インデックス
                - basin_id: 所属するBasin
                - energy: その状態のエネルギー
                - probability: その状態の出現確率
        """
        N, T, d = data_all.shape
        
        # mu_allの形状を正規化
        if mu_all.ndim == 2:
            mu_all = mu_all[:, None, :]  # (N, D) -> (N, 1, D)
        _, T_theta, D = mu_all.shape
        
        if T_theta == 1:
            # aggregateモード: 全時点で同じパラメータを使用
            time_mode = "aggregate"
        else:
            # per_timeモード: 各時点で異なるパラメータを使用
            time_mode = "per_time"
            if T_theta != T:
                raise ValueError(f"Time dimension mismatch: data T={T}, mu_all T={T_theta}")
        
        all_records = []
        total_items = N * T
        current_item = 0
        
        if show_progress:
            pbar = tqdm(
                total=total_items,
                desc="Analyzing trajectories",
                miniters=max(1, total_items // 100),
                mininterval=0.1,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        
        # aggregateモードの場合、各被験者のthetaを事前に計算してキャッシュ
        if time_mode == "aggregate":
            for n in range(N):
                theta = jnp.array(mu_all[n, 0])  # 全時点で同じ
                # ランドスケープを事前計算（キャッシュされる）
                self.compute_energy_landscape(theta, show_progress=False, use_cache=True)
        
        for n in range(N):
            for t in range(T):
                # パラメータの選択
                if time_mode == "aggregate":
                    theta = jnp.array(mu_all[n, 0])  # 全時点で同じ
                else:
                    theta = jnp.array(mu_all[n, t])  # 各時点で異なる
                
                # その時点のデータ
                data_t = data_all[n, t:t+1]  # (1, d)
                
                # 軌跡解析（1時点分、プログレスバーは表示しない）
                traj_df = self.analyze_trajectory(data_t, theta, show_progress=False)
                traj_df['subject_id'] = n
                
                all_records.append(traj_df)
                
                current_item += 1
                if show_progress:
                    pbar.update(1)
        
        if show_progress:
            pbar.close()
        
        result_df = pd.concat(all_records, ignore_index=True)
        
        # observed_state_arrayがある場合は除外（メモリ節約のため、CSV出力しやすく）
        if 'observed_state_array' in result_df.columns:
            result_df = result_df.drop(columns=['observed_state_array'])
        
        # 列の順序を整理
        cols = ['subject_id', 'time_idx', 'observed_state', 'state_index', 
                'basin_id', 'energy', 'probability']
        # 存在する列のみを選択
        available_cols = [c for c in cols if c in result_df.columns]
        result_df = result_df[available_cols + [c for c in result_df.columns if c not in available_cols]]
        
        return result_df
    
    def calculate_trajectory_statistics(self, trajectory_df: pd.DataFrame) -> pd.DataFrame:
        """
        軌跡データから統計量を計算します。
        
        Args:
            trajectory_df: analyze_all_trajectories()の出力
        
        Returns:
            pd.DataFrame: 被験者ごとの統計量
                - subject_id: 被験者ID
                - n_basins_visited: 訪問したBasinの数
                - n_transitions: Basin間の遷移回数
                - unique_basins: 訪問したBasinのリスト
                - mean_energy: 平均エネルギー
                - mean_probability: 平均確率
                - basin_occupation_times: 各Basinでの滞在時間（時点数）
        """
        stats_records = []
        
        for subject_id in trajectory_df['subject_id'].unique():
            subj_df = trajectory_df[trajectory_df['subject_id'] == subject_id].copy()
            
            # Basin間の遷移を検出
            basin_sequence = subj_df['basin_id'].values
            transitions = np.sum(basin_sequence[1:] != basin_sequence[:-1])
            
            # 訪問したBasin
            unique_basins = sorted(subj_df['basin_id'].unique().tolist())
            
            # 各Basinでの滞在時間
            basin_counts = subj_df['basin_id'].value_counts().to_dict()
            
            stats_records.append({
                'subject_id': subject_id,
                'n_basins_visited': len(unique_basins),
                'n_transitions': int(transitions),
                'unique_basins': unique_basins,
                'mean_energy': float(subj_df['energy'].mean()),
                'mean_probability': float(subj_df['probability'].mean()),
                'basin_occupation_times': basin_counts
            })
        
        return pd.DataFrame(stats_records)

    # ... (find_saddle_points, build_disconnectivity_tree は以前と同じなので省略可) ...
    def find_saddle_points(self, theta: jnp.ndarray, lm_indices: np.ndarray) -> Dict[Tuple[int, int], float]:
        energies = -jnp.dot(self._expanded_states, theta)
        energies_np = np.array(energies)
        saddle_energies = {}
        for i, lm1 in enumerate(lm_indices):
            for j, lm2 in enumerate(lm_indices):
                if i >= j: continue
                try:
                    path = nx.shortest_path(self.state_graph, source=lm1, target=lm2)
                    path_energies = energies_np[path]
                    saddle_e = np.max(path_energies)
                    saddle_energies[(lm1, lm2)] = saddle_e
                except nx.NetworkXNoPath:
                    continue
        return saddle_energies

    def build_disconnectivity_tree(self, theta: jnp.ndarray, n_levels: int = 50, show_progress: bool = True) -> Dict:
        res = self.compute_energy_landscape(theta, show_progress=show_progress)
        energies = res['energies']
        min_e, max_e = np.min(energies), np.max(energies)
        thresholds = np.linspace(min_e, max_e, n_levels)
        tree_structure = []
        for th in thresholds:
            valid_nodes = np.where(energies <= th)[0]
            if len(valid_nodes) == 0:
                tree_structure.append([])
                continue
            subgraph = self.state_graph.subgraph(valid_nodes)
            components = list(nx.connected_components(subgraph))
            tree_structure.append([list(c) for c in components])
        return {
            'thresholds': thresholds,
            'tree_structure': tree_structure,
            'energies': energies,
            'local_minima': res['local_minima_indices']
        }


# ==========================================
# 3. Visualizer
# ==========================================

class ResultVisualizer:
    """
    解析結果の可視化を担当するクラス。
    matplotlibを使用してエネルギー地形やネットワーク図を描画します。
    """
    def __init__(self, analyzer: LandscapeAnalyzer):
        self.analyzer = analyzer

    def plot_energy_landscape_bar(self, theta, title="Energy Landscape"):
        """
        エネルギー地形の概要図。
        上段: 全状態のエネルギー（Local Minimaを赤星で表示）
        下段: 各状態の出現確率
        """
        res = self.analyzer.compute_energy_landscape(theta, show_progress=True)
        energies = res['energies']
        probs = res['probabilities']
        # エネルギーが低い順にソートして表示
        order = np.argsort(energies)
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Energy Plot
        axes[0].plot(energies[order], color='navy', label='Energy')
        # LMのプロット
        lm_indices = res['local_minima_indices']
        lm_ranks = [np.where(order == lm)[0][0] for lm in lm_indices]
        axes[0].scatter(lm_ranks, energies[lm_indices], color='red', marker='*', s=100, zorder=5, label='Local Minima')
        axes[0].set_ylabel('Energy')
        axes[0].set_title(f'{title} (States sorted by Energy)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Probability Plot
        axes[1].bar(range(len(probs)), probs[order], color='skyblue', width=1.0)
        axes[1].set_ylabel('Probability')
        axes[1].set_xlabel('State Rank (Sorted by Energy)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def plot_disconnectivity_graph(self, theta, n_levels=50):
        """
        Disconnectivity Graph (樹形図) の描画。
        縦軸がエネルギー。枝の分岐点は、異なる盆地(Basin)が結合するエネルギー障壁の高さを表します。
        """
        tree_data = self.analyzer.build_disconnectivity_tree(theta, n_levels)
        thresholds = tree_data['thresholds']
        structure = tree_data['tree_structure']
        lm_indices = sorted(list(set(tree_data['local_minima'])))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # X軸上の配置: Local Minimaを等間隔に並べる
        lm_pos = {lm: i for i, lm in enumerate(lm_indices)}
        
        # 描画用の一時変数: 各LMが現在属している「グループの重心位置」
        current_lm_positions = {lm: float(i) for i, lm in enumerate(lm_indices)}
        
        # 下から順に閾値を走査
        for i, th in enumerate(thresholds):
            components = structure[i]
            if not components: continue
            
            for comp in components:
                # この連結成分に含まれるLMたちを探す
                lms_in_comp = [node for node in comp if node in lm_indices]
                if not lms_in_comp: continue
                
                # 重心を計算 (マージ後の新しいX座標)
                center_x = np.mean([lm_pos[lm] for lm in lms_in_comp])
                
                # 線を引く (1つ前の閾値レベルから現在のレベルまで)
                if i > 0:
                    prev_th = thresholds[i-1]
                    for lm in lms_in_comp:
                        old_x = current_lm_positions[lm]
                        ax.plot([old_x, center_x], [prev_th, th], color='black', linewidth=0.8)
                
                # 位置情報の更新
                for lm in lms_in_comp:
                    current_lm_positions[lm] = center_x

        # 葉ノード(LM)のラベル表示
        res = self.analyzer.compute_energy_landscape(theta, show_progress=True)
        energies = res['energies']
        for lm in lm_indices:
            state = res['states'][lm]
            # 実際のエネルギー位置より少し下にラベルを表示
            ax.text(lm_pos[lm], energies[lm]-.4, f"{state}", ha='center', fontsize=8, rotation=0)

        ax.set_ylabel("Energy")
        ax.set_title("Disconnectivity Graph (Simplified)")
        # エネルギーが高いほうが上に来るように表示
        # (論文によっては低いほうが下という表現もあるが、ここでは一般的なY軸方向)
        
        return fig

    def plot_transition_network(self, theta, threshold_rate=0.01, show_state_heatmap=False, heatmap_size=0.15):
        """
        状態遷移ネットワーク図 (State Transition Network)
        ノード: Local Minima (大きさは滞在時間に対応)
        エッジ: 遷移確率 (太さは確率に対応)
        
        Args:
            theta: モデルパラメータ
            threshold_rate: 遷移率の閾値（これ以下の遷移は表示しない）
            show_state_heatmap: 各ノードに状態のヒートマップを表示するか
            heatmap_size: ヒートマップのサイズ（ノード位置からの相対的なオフセット）
        """
        res = self.analyzer.compute_energy_landscape(theta, show_progress=True)
        lm_indices = res['local_minima_indices']
        occ_times = res['occupation_times']
        energies = res['energies']
        states = res['states']  # 全状態 (2^d, d)
        saddles = self.analyzer.find_saddle_points(theta, lm_indices)
        
        G = nx.DiGraph()
        
        # ノード追加
        # statesを確実にNumPy配列に変換
        states_np = np.asarray(states)
        for i, lm in enumerate(lm_indices):
            # ノードサイズを見やすく調整
            size = occ_times[i] * 2000 + 300 
            # 状態を確実にNumPy配列として保存
            state_vec = np.asarray(states_np[lm]).astype(float).flatten()
            G.add_node(i, label=f"LM{lm}", size=size, state=state_vec)
            
        # エッジ追加 (アレニウスの式: Rate ~ exp(-Barrier))
        for (lm_a, lm_b), saddle_e in saddles.items():
            idx_a = np.where(lm_indices == lm_a)[0][0]
            idx_b = np.where(lm_indices == lm_b)[0][0]
            
            # 障壁高さ = サドル点エネルギー - 出発点のエネルギー
            barrier_ab = max(0, saddle_e - energies[lm_a])
            rate_ab = np.exp(-barrier_ab) # 単純化した遷移率
            
            barrier_ba = max(0, saddle_e - energies[lm_b])
            rate_ba = np.exp(-barrier_ba)
            
            # 閾値以上の遷移のみ描画
            if rate_ab > threshold_rate:
                G.add_edge(idx_a, idx_b, weight=rate_ab)
            if rate_ba > threshold_rate:
                G.add_edge(idx_b, idx_a, weight=rate_ba)

        # レイアウト決定 (バネモデル)
        pos = nx.spring_layout(G, k=1.5, seed=42)
        sizes = [nx.get_node_attributes(G, 'size')[n] for n in G.nodes]
        
        fig, ax = plt.subplots(figsize=(9, 9))
        
        # エッジを先に描画（ノードの下に来るように）
        edges = G.edges(data=True)
        if len(edges) > 0:
            weights = [d['weight']*3 for u, v, d in edges]
            nx.draw_networkx_edges(G, pos, width=weights * 10, arrowstyle='->', arrowsize=20, ax=ax, alpha=0.5, connectionstyle='arc3,rad=0.1')
        
        # ノードを描画
        nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color='orange', alpha=0.7, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
        
        # 各ノードに状態のヒートマップを追加
        if show_state_heatmap:
            d = self.analyzer.d
            # 状態を2次元にreshape（可能な場合）
            # dが平方数でない場合は、最も近い長方形に配置
            n_cols = int(math.ceil(math.sqrt(d)))
            n_rows = int(math.ceil(d / n_cols))
            
            # posから範囲を計算（より確実な方法）
            pos_array = np.array(list(pos.values()))
            x_min, x_max = pos_array[:, 0].min(), pos_array[:, 0].max()
            y_min, y_max = pos_array[:, 1].min(), pos_array[:, 1].max()
            x_range = x_max - x_min if x_max != x_min else 1.0
            y_range = y_max - y_min if y_max != y_min else 1.0
            
            # ヒートマップのサイズ（軸単位）
            hm_width = x_range * heatmap_size
            hm_height = y_range * heatmap_size
            
            for node_id in G.nodes():
                # 状態を確実に取得（JAX配列の場合はNumPyに変換）
                state_raw = nx.get_node_attributes(G, 'state')[node_id]
                # JAX配列やその他の形式に対応
                if hasattr(state_raw, '__array__'):
                    state = np.asarray(state_raw).astype(float)
                else:
                    state = np.array(state_raw, dtype=float)
                
                # 状態が1次元配列であることを確認
                state = state.flatten()
                
                # 状態の値を0-1に正規化（既に0/1の場合はそのまま）
                state = np.clip(state, 0, 1)
                
                x, y = pos[node_id]
                
                # 状態を2次元配列に変換
                state_2d = np.zeros((n_rows, n_cols))
                for idx in range(min(len(state), d)):
                    val = state[idx]
                    row = idx // n_cols
                    col = idx % n_cols
                    if row < n_rows and col < n_cols:
                        state_2d[row, col] = val
                
                # ヒートマップの左下の座標（ノードの下に配置）
                hm_x = x - hm_width * 0.5
                hm_y = y - hm_height * 1.2
                
                # ヒートマップを描画（バイナリデータ用のカラーマップ）
                # 0=青、1=赤で表示（明確なコントラストのため）
                # カスタムカラーマップ: 0=青、1=赤
                colors = ['#2166ac', '#d73027']  # 青と赤
                cmap_binary = ListedColormap(colors)
                
                im = ax.imshow(state_2d, cmap=cmap_binary, vmin=0, vmax=1, 
                              extent=[hm_x, hm_x + hm_width, hm_y, hm_y + hm_height],
                              aspect='auto', interpolation='nearest', alpha=0.9)
                
                # 境界線を追加
                rect = Rectangle((hm_x, hm_y), hm_width, hm_height, 
                               linewidth=1.5, edgecolor='black', facecolor='none')
                ax.add_patch(rect)
        
        ax.set_title("Transition Network\n(Node size: Occ.Time, Edge: Transition Rate, Heatmap: State Pattern)")
        plt.axis('off')
        return fig

# ==========================================
# 4. Demo Execution
# ==========================================

def run_demo():
    print("=== VEM-MEM Integrated Demo ===")
    
    # ----------------------------------------------------
    # データ準備 (Data Preparation)
    # 実際の研究では、ここをご自身のfMRIデータ読み込みに置き換えてください
    # ----------------------------------------------------
    np.random.seed(42)
    N = 50   # 被験者数
    T = 100  # 各被験者の時間点数
    d = 8    # 脳領域(ROI)の数 ※8~12くらいが計算しやすい推奨値
    
    print(f"Generating dummy data: N={N}, T={T}, d={d}")
    # 0と1のランダムなバイナリデータ (確率0.3で活動あり)
    data = np.random.binomial(1, 0.3, size=(N, T, d)).astype(float)
    
    # JAX配列に変換 (GPUがあれば自動的にGPUメモリに乗ります)
    data_jax = jnp.array(data)
    
    # ----------------------------------------------------
    # 1. モデル学習 (Model Fitting)
    # ----------------------------------------------------
    print("Fitting model (VEM-MEM)...")
    config = VEMConfig(max_iter=50, tol=1e-4)
    model = VEMMEM(d=d, config=config)
    
    # "aggregate"モード: 被験者の個性を安定的(time-averaged)に推定する
    results = model.fit(data_jax, verbose=True, time_mode="aggregate")
    
    print(f"Converged in {results['n_iterations']} iterations.")
    print(f"Final ELBO: {results['final_elbo']:.2f}")
    
    # ----------------------------------------------------
    # 2. 解析 (Analysis) - グループ平均パラメータ(eta)を使用
    # ----------------------------------------------------
    print("Analyzing landscape using Group Prior (eta)...")
    eta = results['eta'] # グループ全体の傾向を表すパラメータ
    
    analyzer = LandscapeAnalyzer(model)
    landscape_res = analyzer.compute_energy_landscape(eta)
    
    n_lm = len(landscape_res['local_minima_indices'])
    print(f"Found {n_lm} Local Minima in group landscape.")
    
    # ----------------------------------------------------
    # 3. 可視化 (Visualization)
    # ----------------------------------------------------
    viz = ResultVisualizer(analyzer)
    
    # A. バープロット (エネルギー地形の全体像)
    fig1 = viz.plot_energy_landscape_bar(eta, title="Group Energy Landscape")
    
    # B. Disconnectivity Graph (状態遷移の階層構造)
    print("Building Disconnectivity Graph...")
    fig2 = viz.plot_disconnectivity_graph(eta)
    
    # C. Transition Network (主要状態間の遷移ネットワーク)
    print("Building Transition Network...")
    fig3 = viz.plot_transition_network(eta, threshold_rate=0.005)
    
    # ----------------------------------------------------
    # 4. 軌跡解析 (Trajectory Analysis)
    # ----------------------------------------------------
    print("\n=== Trajectory Analysis ===")
    print("Analyzing trajectories for all subjects...")
    
    # 全被験者の軌跡を解析
    trajectory_df = analyzer.analyze_all_trajectories(data, results['mu_all'])
    
    # 統計量を計算
    stats_df = analyzer.calculate_trajectory_statistics(trajectory_df)
    
    print(f"\nTrajectory analysis completed:")
    print(f"  - Total records: {len(trajectory_df)}")
    print(f"  - Subjects analyzed: {len(stats_df)}")
    print(f"\nSample statistics:")
    print(stats_df.head())
    
    # 軌跡データをCSVで保存（オプション）
    # trajectory_df.to_csv('trajectory_analysis.csv', index=False)
    # stats_df.to_csv('trajectory_statistics.csv', index=False)
    
    print("\nDone. Displaying plots...")
    plt.show()
    
    return {
        'results': results,
        'analyzer': analyzer,
        'trajectory_df': trajectory_df,
        'stats_df': stats_df
    }



