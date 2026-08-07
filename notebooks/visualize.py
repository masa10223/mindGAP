import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap
from pathlib import Path

def plot_landscape_patent_style(
    data_in,
    title="Energy Landscape (Patent Style)",
    save_fig=False,
    filename="landscape_patent_style"
):
    # --- 1. データ展開 ---
    energies = data_in['energies']
    # 状態数が512 (2^9) なので9ビットシステムと推定
    num_states = len(energies)
    n_bits = int(np.log2(num_states))
    
    # local_minima_indices の取得
    lm_indices = data_in['local_minima_indices']

    # --- 2. グラフ構造の構築 (Fruchterman-Reingold配置のため) ---
    # 特許にある通り、状態間の遷移（ハミング距離1）に基づくグラフを作成します
    print("Building adjacency graph...")
    
    # 全状態のインデックスとビット列の対応
    # (data['states']があればそれを使いますが、ここではインデックスから生成も可能)
    # 高速化のため、ビット演算で隣接リストを作成
    adj_list = []
    for i in range(num_states):
        for b in range(n_bits):
            # i の b番目のビットを反転させたインデックスを計算
            neighbor = i ^ (1 << b)
            if neighbor < num_states:
                adj_list.append((i, neighbor))
    
    G = nx.Graph()
    G.add_nodes_from(range(num_states))
    G.add_edges_from(adj_list)

    # --- 3. 配置計算 (Fruchterman-Reingold) ---
    # 特許文献  に基づく力学モデル配置
    print("Calculating Fruchterman-Reingold layout...")
    # seed固定で再現性を確保
    pos = nx.spring_layout(G, seed=42, iterations=50)

    # 座標とエネルギーの抽出
    pos_array = np.array([pos[i] for i in range(num_states)])
    x = pos_array[:, 0]
    y = pos_array[:, 1]
    
    # --- 4. グリッド補間 (Surface Fitting) ---
    # 特許文献 [cite: 70, 126] に基づく滑らかな曲面補間
    # Scipyのcubic補間を使用
    grid_x, grid_y = np.mgrid[x.min():x.max():500j, y.min():y.max():500j]
    grid_z = griddata((x, y), energies, (grid_x, grid_y), method='cubic')

    # --- 5. 描画 ---
    fig, ax = plt.subplots(figsize=(15, 12))

    # A. 等高線の描画 (ノード・エッジは描画しない)
    # カラーマップはエネルギーの低い方が青（安定）、高い方が赤（不安定）となる 'RdYlBu_r' 推奨
    # 特許図面のような白黒に近いものが良ければ 'Greys' などに変更可
    levels = np.linspace(energies.min(), energies.max(), 100)
    contour = ax.contourf(
        grid_x,
        grid_y,
        grid_z,
        levels=levels,
        cmap="RdYlBu_r", # エネルギーが低い(青) -> 高い(赤)。逆なら _r を取る
        alpha=0.8
    )
    
    # 等高線そのものの線（薄く入れると地形が見やすい）
    ax.contour(
        grid_x, 
        grid_y, 
        grid_z, 
        levels=20, 
        colors='k', 
        linewidths=0.3, 
        alpha=0.5
    )

    # B. ローカルミニマ (LM) の強調描画
    lm_x = [pos[i][0] for i in lm_indices]
    lm_y = [pos[i][1] for i in lm_indices]
    lm_e = [energies[i] for i in lm_indices] # 色分け用

    # LMを星型でプロット
    scatter = ax.scatter(
        lm_x, 
        lm_y, 
        s=300,            # サイズ大きめ
        c=lm_e,           # エネルギー値で色付け（または単色 'red' でも可）
        cmap="RdYlBu_r",
        edgecolors='black', # 縁取り
        linewidths=1.5,
        marker='*',       # 星型
        zorder=10,        # 最前面に表示
        label="Local Minima"
    )

    # LMのインデックス番号を添える（オプション）
    for i, txt in enumerate(lm_indices):
        ax.annotate(
            f"LM {txt}", 
            (lm_x[i], lm_y[i]),
            xytext=(5, 5), 
            textcoords='offset points',
            fontsize=12,
            fontweight='bold',
            color='black'
        )

    # カラーバー
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Energy")

    # 装飾
    ax.set_title(title, fontsize=18)
    ax.axis('off') # 軸を消す（特許の図のように）

    if save_fig:
        output_dir = Path("./figs_for_paper")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{filename}.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_path}")

    plt.show()



def plot_ranked_landscape(
    data_in,
    title="Energy Landscape with Ranked Local Minima",
    save_fig=False,
    filename="ranked_landscape"
):
    # --- 1. データ展開 ---
    energies = data_in['energies']
    lm_indices = data_in['local_minima_indices']
    num_states = len(energies)
    n_bits = int(np.log2(num_states))

    # --- 2. グラフ構築と配置 ---
    # レイアウト計算用の全結合グラフ
    adj_list = []
    for i in range(num_states):
        for b in range(n_bits):
            neighbor = i ^ (1 << b)
            if neighbor < num_states:
                adj_list.append((i, neighbor))
    
    G_layout = nx.Graph()
    G_layout.add_nodes_from(range(num_states))
    G_layout.add_edges_from(adj_list)

    print("Calculating Layout...")
    # 再現性のためseed固定
    pos = nx.spring_layout(G_layout, seed=42, iterations=50)
    pos_array = np.array([pos[i] for i in range(num_states)])
    x = pos_array[:, 0]
    y = pos_array[:, 1]

    # --- 3. グリッド補間（背景用） ---
    grid_x, grid_y = np.mgrid[x.min():x.max():500j, y.min():y.max():500j]
    grid_z = griddata((x, y), energies, (grid_x, grid_y), method='cubic')

    # --- 4. LMのランク付け処理 ---
    # LMのインデックスとそのエネルギーのペアを作成
    lm_data = []
    for idx in lm_indices:
        lm_data.append({'index': idx, 'energy': energies[idx]})
    
    # エネルギーが低い順（昇順）にソート
    lm_data_sorted = sorted(lm_data, key=lambda x: x['energy'])

    # --- 5. 描画 ---
    fig, ax = plt.subplots(figsize=(18, 15))

    # A. 背景のエネルギー地形
    contour = ax.contourf(
        grid_x, 
        grid_y, 
        grid_z, 
        levels=100, 
        cmap="RdYlBu_r", # 青=安定(低エネルギー)
        alpha=0.6
    )

    # B. ノードの描画
    # 1. LM以外のノード (極小・半透明)
    non_lm_indices = [i for i in range(num_states) if i not in lm_indices]
    non_lm_x = [pos[i][0] for i in non_lm_indices]
    non_lm_y = [pos[i][1] for i in non_lm_indices]
    
    ax.scatter(
        non_lm_x, 
        non_lm_y, 
        s=40,             
        c='black', 
        alpha=0.2, 
        marker='o',
        linewidths=0,
        zorder=1
    )

    # 2. LMノード (星型・強調)
    lm_x = [pos[item['index']][0] for item in lm_data_sorted]
    lm_y = [pos[item['index']][1] for item in lm_data_sorted]
    lm_c = [item['energy'] for item in lm_data_sorted] # エネルギーで色付けも可能だが今回は視認性重視で統一色推奨

    ax.scatter(
        lm_x, 
        lm_y, 
        s=600,           
        c='gold', 
        edgecolors='black', 
        marker='*',      
        linewidths=1.5,
        zorder=10,
        label="Local Minima"
    )

    # C. ラベルの描画 (ランク順)
    for rank, item in enumerate(lm_data_sorted, start=1):
        idx = item['index']
        energy = item['energy']
        
        # ビット列表記
        binary_label = f"{idx:0{n_bits}b}"
        
        # ランク表記
        if rank == 1:
            rank_text = f"# LM{rank} (Global Min)"
            box_color = "#ffdddd" # 1位だけ少し赤みがかった背景
        else:
            rank_text = f"# LM{rank}"
            box_color = "white"

        # 表示テキスト作成 (ランク + ビット列)
        label_text = f"{rank_text}\n[{binary_label}]"

        ax.annotate(
            label_text, 
            (pos[idx][0], pos[idx][1]),
            xytext=(0, 20),       # 星の少し上に表示
            textcoords='offset points',
            fontsize=11,
            fontweight='bold',
            ha='center',
            color='black',
            bbox=dict(boxstyle="round,pad=0.3", fc=box_color, alpha=0.8, ec="gray")
        )

    # 仕上げ
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Energy")
    ax.set_title(title, fontsize=20)
    ax.axis('off')

    if save_fig:
        output_dir = Path("./figs_for_paper")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{filename}.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_path}")

    plt.show()


def plot_trajectory_with_loops_and_highlights(
    data_in,
    trajectory,
    title="Trajectory with Self-Loops & Highlighted Nodes",
    save_fig=False,
    filename="trajectory_loops_highlight"
):
    # --- 1. データ展開 ---
    energies = data_in['energies']
    lm_indices = data_in['local_minima_indices']
    num_states = len(energies)
    n_bits = int(np.log2(num_states))

    # --- 2. グラフ構築と配置 (背景用) ---
    adj_list = []
    for i in range(num_states):
        for b in range(n_bits):
            neighbor = i ^ (1 << b)
            if neighbor < num_states:
                adj_list.append((i, neighbor))
    
    G_layout = nx.Graph()
    G_layout.add_nodes_from(range(num_states))
    G_layout.add_edges_from(adj_list)

    print("Calculating Layout...")
    pos = nx.spring_layout(G_layout, seed=42, iterations=50)
    pos_array = np.array([pos[i] for i in range(num_states)])
    x = pos_array[:, 0]
    y = pos_array[:, 1]

    # --- 3. グリッド補間 (背景用) ---
    grid_x, grid_y = np.mgrid[x.min():x.max():500j, y.min():y.max():500j]
    grid_z = griddata((x, y), energies, (grid_x, grid_y), method='cubic')

    # --- 4. トラジェクトリのインデックス変換 ---
    # 注意: VEM_MEM.pyでは最初の次元（index 0）がLSBなので、配列を逆順にしてから文字列に変換
    traj_indices = []
    for state in trajectory:
        # 状態配列を逆順にしてから文字列に変換（LSB-firstに合わせる）
        state_reversed = state[::-1]
        idx = int("".join(map(str, state_reversed)), 2)
        traj_indices.append(idx)
    
    # 通過したユニークなノードの集合
    visited_set = set(traj_indices)

    # --- 5. 描画 ---
    fig, ax = plt.subplots(figsize=(18, 15))

    # A. 背景 (等高線)
    contour = ax.contourf(
        grid_x, grid_y, grid_z, 
        levels=100, cmap="RdYlBu_r", alpha=0.5
    )

    # B. ノードの描画
    # 1. その他大勢 (visitedでもLMでもない) -> 極小・薄い黒
    bg_indices = [i for i in range(num_states) if i not in visited_set and i not in lm_indices]
    if bg_indices:
        ax.scatter(
            [pos[i][0] for i in bg_indices], 
            [pos[i][1] for i in bg_indices], 
            s=5, c='black', alpha=0.1, zorder=1
        )

    # 2. 通過したノード (Visited) -> 目立つ色 (シアン) の丸
    # LMであっても、まずは「通過した場所」として背景に色を置く
    visited_list = list(visited_set)
    ax.scatter(
        [pos[i][0] for i in visited_list],
        [pos[i][1] for i in visited_list],
        s=400,            # 少し大きめ
        c='cyan',         # ハイライト色
        edgecolors='blue',
        linewidths=2,
        marker='o',
        alpha=0.8,
        zorder=5,
        label="Visited States"
    )

    # 3. LMノード (星型) -> 最前面
    # LMデータの準備
    lm_data = [{'index': idx, 'energy': energies[idx]} for idx in lm_indices]
    lm_data_sorted = sorted(lm_data, key=lambda x: x['energy'])

    for rank, item in enumerate(lm_data_sorted, start=1):
        idx = item['index']
        # 星を描画
        ax.scatter(pos[idx][0], pos[idx][1], s=600, c='gold', edgecolors='black', marker='*', zorder=20)
        
        # ラベル
        label_text = f"# LM{rank}\n[{idx:0{n_bits}b}]"
        if rank == 1: label_text = f"# LM{rank} (Global)\n[{idx:0{n_bits}b}]"
        
        ax.annotate(
            label_text, (pos[idx][0], pos[idx][1]), xytext=(0, 25), 
            textcoords='offset points', fontsize=10, fontweight='bold', 
            ha='center', color='black',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="gray"),
            zorder=25
        )

    # C. 軌跡の描画 (矢印と自己ループ)
    traj_x = [pos[i][0] for i in traj_indices]
    traj_y = [pos[i][1] for i in traj_indices]

    # 線でつなぐ (同じ場所への遷移は線としては描画されないのでそのまま)
    ax.plot(traj_x, traj_y, color='magenta', linewidth=2, alpha=0.6, zorder=10, linestyle='--')

    for i in range(len(traj_indices) - 1):
        curr_idx = traj_indices[i]
        next_idx = traj_indices[i+1]
        
        x1, y1 = pos[curr_idx]
        x2, y2 = pos[next_idx]

        if curr_idx == next_idx:
            # === 自己ループ (Self-loop) ===
            # 同じ場所に留まる場合、くるっと回る矢印を描く
            # connectionstyle="arc3,rad=..." でループを作成
            # radの値を大きくするとループが大きくなる
            ax.annotate(
                "", 
                xy=(x1, y1), 
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->", 
                    color='magenta', 
                    lw=2.5,
                    connectionstyle="arc3,rad=2.5" # ここでループの大きさを調整
                ),
                zorder=15
            )
        else:
            # === 通常の遷移 (Transition) ===
            ax.annotate(
                "", 
                xy=(x2, y2), 
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->", 
                    color='magenta', 
                    lw=2.5,
                    shrinkA=10, # 始点の微調整(ノード被り防止)
                    shrinkB=10
                ),
                zorder=15
            )

    # 始点と終点のマーカー
    ax.scatter(traj_x[0], traj_y[0], s=150, c='lime', edgecolors='black', marker='D', zorder=30, label='Start')
    ax.scatter(traj_x[-1], traj_y[-1], s=150, c='red', edgecolors='black', marker='X', zorder=30, label='End')

    # 仕上げ
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Energy")
    ax.set_title(title, fontsize=20)
    ax.axis('off')
    # 凡例
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)

    if save_fig:
        output_dir = Path("./figs_for_paper")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{filename}.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_path}")

    plt.show()


def plot_weighted_landscape_with_nodes(
    data_in,
    trajectory,
    ax=None,  # <--- 描画先のAxesを受け取る
    title="Basin Regions (Tab20) with Energy Contours",
    hue_sep=False,
    is_group=False,
    trajectory_color="magenta",
    show_time_labels=True,
    save_fig=False,
    filename="basin_tab20_with_contours",
    *,
    grid_resolution=500,
    n_energy_contours=40,
    show_all_nodes=True,
    layout_iterations=100,
    layout_k=0.2,
    verbose=True,
    lightweight=False,
    save_dpi=300,
    save_format="pdf",
    attractor_label="State",
):
    # --- 1. Data Unpacking ---
    energies = data_in['energies']
    lm_indices = data_in['local_minima_indices']
    basin_mapping = data_in['basin_mapping']
    num_states = len(energies)
    n_bits = int(np.log2(num_states))

    if lightweight:
        grid_resolution = min(grid_resolution, 150)
        n_energy_contours = min(n_energy_contours, 10)
        show_all_nodes = False
        layout_iterations = min(layout_iterations, 30)
        save_dpi = min(save_dpi, 100)

    # --- 2. Weighted Graph (Cluster Layout) ---
    if verbose:
        print("Building Weighted Graph...")
    G = nx.Graph()
    G.add_nodes_from(range(num_states))
    for i in range(num_states):
        for b in range(n_bits):
            neighbor = i ^ (1 << b)
            if neighbor < num_states and i < neighbor:
                # 同じBasinなら強く、違うなら弱く
                w = 10.0 if basin_mapping[i] == basin_mapping[neighbor] else 0.5
                G.add_edge(i, neighbor, weight=w)

    if verbose:
        print("Calculating Layout...")
    pos = nx.spring_layout(
        G, seed=42, weight='weight', k=layout_k, iterations=layout_iterations
    )
    x = np.array([pos[i][0] for i in range(num_states)])
    y = np.array([pos[i][1] for i in range(num_states)])

    # --- 3. Interpolation ---
    grid_x, grid_y = np.mgrid[
        x.min()-0.1:x.max()+0.1:grid_resolution*1j,
        y.min()-0.1:y.max()+0.1:grid_resolution*1j,
    ]
    
    # Basin ID (領域用)
    grid_basin_raw = griddata((x, y), basin_mapping, (grid_x, grid_y), method='nearest')
    
    # Energy (等高線用)
    grid_energy = griddata((x, y), energies, (grid_x, grid_y), method='cubic')
    
    # --- 4. Color Mapping Setup ---
    # BA (basin attractor) をランク付け
    lm_data = [{'index': idx, 'energy': energies[idx]} for idx in lm_indices]
    lm_data_sorted = sorted(lm_data, key=lambda x: x['energy'])
    
    # Basin ID -> Rank
    basin_id_to_rank = {item['index']: i for i, item in enumerate(lm_data_sorted)}
    
    def get_rank(basin_id):
        return basin_id_to_rank.get(basin_id, -1)
    
    v_get_rank = np.vectorize(get_rank)
    grid_rank = v_get_rank(grid_basin_raw)

    # Colormap: group のときは 003F5C, 9A1B1B, 6A1B9A。それ以外は hue_sep または Tab20
    if is_group or hue_sep:
        # #003F5C Navy, #9A1B1B Dark Red, #6A1B9A Violet (RGBA 0-1)
        tab20 = [
            (0.0, 0.24705882352941178, 0.3607843137254902, 0.5),   # #003F5C
            (0.6039215686274509, 0.10588235294117647, 0.10588235294117647, 0.5),  # #9A1B1B
            (0.41568627450980394, 0.10588235294117647, 0.6039215686274509, 0.5)   # #6A1B9A
        ]
    else:
        tab20 = plt.cm.tab20.colors
    num_basins = len(lm_data_sorted)
    n_colors = len(tab20)
    used_colors = [tab20[i % n_colors] for i in range(num_basins)]
    custom_cmap = ListedColormap(used_colors)

    # --- 5. Trajectory Prep ---
    traj_arr = np.asarray(trajectory)
    if traj_arr.ndim == 2:
        traj_list = [traj_arr]
    elif traj_arr.ndim == 3:
        # (N, T, d): 被験者ごとの軌跡を分割して扱う（被験者間の誤接続を防ぐ）
        traj_list = [traj_arr[i] for i in range(traj_arr.shape[0])]
    else:
        raise ValueError(f"trajectory must be (T,d) or (N,T,d), got shape={traj_arr.shape}")

    traj_sequences = []
    node_to_times = {}
    for seq_idx, seq in enumerate(traj_list):
        seq_indices = []
        for t, state in enumerate(seq):
            # 状態配列を逆順にしてから文字列に変換（LSB-firstに合わせる）
            state_reversed = state[::-1]
            idx = int("".join(map(str, state_reversed)), 2)
            seq_indices.append(idx)
            if idx not in node_to_times:
                node_to_times[idx] = []
            node_to_times[idx].append((seq_idx, t))
        traj_sequences.append(seq_indices)

    visited_set = set(i for seq in traj_sequences for i in seq)
    is_multi_trajectory = len(traj_sequences) > 1
    if isinstance(trajectory_color, (list, tuple, np.ndarray)):
        if len(trajectory_color) == 0:
            traj_colors = ["magenta"] * len(traj_sequences)
        else:
            traj_colors = [trajectory_color[i % len(trajectory_color)] for i in range(len(traj_sequences))]
    else:
        traj_colors = [trajectory_color] * len(traj_sequences)

    # --- 6. Plotting ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 15))
    else:
        fig = ax.figure
    
    # A. 領域の塗りつぶし (Tab20)
    contour_levels = np.arange(num_basins + 1) - 0.5
    ax.contourf(
        grid_x, grid_y, grid_rank, 
        levels=contour_levels, 
        cmap=custom_cmap, 
        alpha=0.5
    )

    # B. エネルギー等高線 (Contours)
    if n_energy_contours > 0:
        ax.contour(
            grid_x, grid_y, grid_energy,
            levels=n_energy_contours, colors='black', linewidths=0.5
        )

    # C. 領域の境界線 (Basin Boundaries)
    ax.contour(
        grid_x, grid_y, grid_rank, 
        levels=np.arange(num_basins), 
        colors='black', linewidths=2.5, linestyles='solid'
    )
    
    # D. Nodes (optional: 全ノード描画。2^d が大きいときは show_all_nodes=False 推奨)
    if show_all_nodes:
        all_node_colors = []
        all_node_positions_x = []
        all_node_positions_y = []

        for node_idx in range(num_states):
            if node_idx in lm_indices:
                continue  # BAノードは後で描画
            basin_id = basin_mapping[node_idx]
            rank = basin_id_to_rank.get(basin_id, -1)
            if rank >= 0:
                color = used_colors[rank]
            else:
                color = 'gray'

            all_node_positions_x.append(pos[node_idx][0])
            all_node_positions_y.append(pos[node_idx][1])
            all_node_colors.append(color)

        if all_node_colors:
            ax.scatter(
                all_node_positions_x,
                all_node_positions_y,
                s=20,
                c=all_node_colors,
                alpha=0.6,
                edgecolors='black',
                linewidths=0.3,
                zorder=3,
                label="All Nodes (by Basin)"
            )
    
    # 2. Visited Nodes（より大きく、目立つように）
    visited_no_lm = [i for i in visited_set if i not in lm_indices]
    if visited_no_lm:
        if is_multi_trajectory:
            for seq_idx, traj_indices in enumerate(traj_sequences):
                color = traj_colors[seq_idx]
                seq_visited = [i for i in set(traj_indices) if i not in lm_indices]
                if len(seq_visited) == 0:
                    continue
                ax.scatter(
                    [pos[i][0] for i in seq_visited],
                    [pos[i][1] for i in seq_visited],
                    s=140,
                    c=[color],
                    edgecolors='black',
                    linewidths=1.2,
                    marker='o',
                    alpha=0.75,
                    zorder=10,
                )
        else:
            # 訪問ノードを大きく、目立つ色で描画
            ax.scatter(
                [pos[i][0] for i in visited_no_lm],
                [pos[i][1] for i in visited_no_lm],
                s=200,  # 大きく
                c=traj_colors[0],
                edgecolors='black',
                linewidths=2.0,
                marker='o',
                alpha=0.9,
                zorder=10,
                label="Visited"
            )

        # Text Labels for Timepoints (t <= 3 の時だけ表示、ノード文字列は表示しない)
        # 複数軌跡を重ねる場合はラベルが過密になるため非表示
        if show_time_labels and not is_multi_trajectory:
            for node_idx in visited_no_lm:
                hits = node_to_times[node_idx]
                times_filtered = [t for (_, t) in hits if t <= 3]
                if len(times_filtered) > 0:
                    t_str = ",".join(map(str, times_filtered))
                    ax.text(
                        pos[node_idx][0], pos[node_idx][1],
                        f"t={t_str}",
                        fontsize=12, color='darkblue', fontweight='bold',
                        ha='left', va='bottom',
                        zorder=25,
                        bbox=dict(boxstyle="square,pad=0.1", fc="white", alpha=0.8, ec=traj_colors[0], lw=1.5)
                    )

    # E. Trajectory Arrows (Curved) & Self-loops
    # 線での連結はやめて、矢印だけで遷移を表現する（曲線矢印と直線が混ざると見づらいため）
    # もし線も引きたい場合は、直線ではなく曲線補間が必要になるが、ここでは矢印のみで表現
    
    n_seq = len(traj_sequences)
    for seq_idx, traj_indices in enumerate(traj_sequences):
        line_color = traj_colors[seq_idx]
        # 同一軌跡が重なる場合、曲率をずらして見分ける
        rad_lane = 0.2 + 0.35 * (seq_idx - (n_seq - 1) / 2.0) if n_seq > 1 else 0.2
        loop_rad = 3.0 + 0.8 * (seq_idx - (n_seq - 1) / 2.0) if n_seq > 1 else 3.0
        for i in range(len(traj_indices) - 1):
            curr = traj_indices[i]
            next_node = traj_indices[i + 1]
            if curr == next_node:
                ax.annotate(
                    "",
                    xy=pos[curr],
                    xytext=pos[curr],
                    arrowprops=dict(
                        arrowstyle="->",
                        color=line_color,
                        lw=2.5,
                        connectionstyle=f"arc3,rad={loop_rad}",
                    ),
                    zorder=15 + seq_idx,
                )
            else:
                ax.annotate(
                    "",
                    xy=pos[next_node],
                    xytext=pos[curr],
                    arrowprops=dict(
                        arrowstyle="->",
                        color=line_color,
                        lw=2.5,
                        shrinkA=5,
                        shrinkB=5,
                        connectionstyle=f"arc3,rad={rad_lane}",
                    ),
                    zorder=15 + seq_idx,
                )

    # F. BA Nodes (Basin Attractors, Stars)
    for rank, item in enumerate(lm_data_sorted):
        idx = item['index']
        rank_idx = rank
        color = used_colors[rank_idx]
        
        # 星
        ax.scatter(
            pos[idx][0], pos[idx][1], 
            s=600, c=[color], edgecolors='black', linewidths=2.0, marker='*', zorder=20
        )
        
        # Label: attractor (State) のみ表示（ビット列は表示しない）
        display_rank = rank + 1
        if display_rank == 1:
            label_text = f"Global Min\n({attractor_label}{display_rank})"
        else:
            label_text = f"{attractor_label}{display_rank}"
        
        ax.annotate(
            label_text, (pos[idx][0], pos[idx][1]), xytext=(0, 20), 
            textcoords='offset points', fontsize=10, fontweight='bold', 
            ha='center', color='black',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray"),
            zorder=25
        )

    # Start/End Markers
    if is_multi_trajectory:
        for seq_idx, traj_indices in enumerate(traj_sequences):
            if len(traj_indices) == 0:
                continue
            color = traj_colors[seq_idx]
            ax.scatter(pos[traj_indices[0]][0], pos[traj_indices[0]][1], s=80, c=[color], edgecolors='black', marker='D', zorder=30)
            ax.scatter(pos[traj_indices[-1]][0], pos[traj_indices[-1]][1], s=80, c=[color], edgecolors='black', marker='X', zorder=30)
    else:
        traj_indices = traj_sequences[0]
        ax.scatter(pos[traj_indices[0]][0], pos[traj_indices[0]][1], s=200, c='lime', edgecolors='black', marker='D', zorder=30, label='Start')
        ax.scatter(pos[traj_indices[-1]][0], pos[traj_indices[-1]][1], s=200, c='red', edgecolors='black', marker='X', zorder=30, label='End')

    ax.set_title(title, fontsize=18)
    ax.axis('off')
    
    # 保存処理（axがNoneでない場合でも保存できるように修正）
    if save_fig:
        output_dir = Path("./figs_for_paper")
        output_dir.mkdir(parents=True, exist_ok=True)
        ext = save_format.lstrip(".")
        output_path = output_dir / f"{filename}.{ext}"
        fig.savefig(output_path, dpi=save_dpi, bbox_inches='tight')
        if verbose:
            print(f"Saved to {output_path}")
    
    # axがNoneの場合（この関数内で作成した場合）のみplt.show()を実行
    if ax is None:
        plt.show()


def _attractor_rank_map(land, rank_by="energy"):
    """attractor state_index -> rank 0,1,2,... (State1 = 0)"""
    lm = np.asarray(land["local_minima_indices"], dtype=int)
    if lm.size == 0:
        return {}, []
    if rank_by == "energy":
        order = np.argsort(land["energies"][lm])
    else:
        order = np.argsort(-np.asarray(land["occupation_times"], dtype=float))
    rank_map = {int(lm[order[i]]): int(i) for i in range(len(lm))}
    n = len(lm)
    colors = plt.cm.tab20(np.linspace(0, 1, max(n, 1)))[:n]
    return rank_map, colors


def _match_observation_to_state_index(obs_state, states=None, n_bits=None):
    """
    plot_weighted_landscape_with_nodes と同じ規則で state_index を求める。
    """
    obs = np.asarray(obs_state, dtype=int).reshape(-1)
    if n_bits is None:
        n_bits = obs.size
    state_reversed = obs[::-1]
    idx = int("".join(map(str, state_reversed.astype(int))), 2)
    if states is not None:
        states = np.asarray(states, dtype=int)
        if idx < states.shape[0] and np.all(states[idx] == obs):
            return idx
        match = np.all(states == obs, axis=1)
        if np.any(match):
            return int(np.where(match)[0][0])
        hd = np.sum(states != obs, axis=1)
        return int(np.argmin(hd))
    return idx


def _trajectory_state_ranks(traj, land, rank_by="energy", analyzer=None):
    """各時点がどの State (basin rank) にいたかを返す。"""
    rank_map, _ = _attractor_rank_map(land, rank_by=rank_by)
    basin_mapping = np.asarray(land["basin_mapping"], dtype=int)
    states = np.asarray(land["states"], dtype=int)
    traj = np.asarray(traj)
    T = traj.shape[0]

    if analyzer is not None and hasattr(analyzer, "_match_states_jax"):
        try:
            import jax.numpy as jnp
            obs_states_jax = jnp.array(traj.astype(int))
            sidx_arr = np.array(
                analyzer._match_states_jax(obs_states_jax, analyzer._states_jax)
            )
        except Exception:
            sidx_arr = np.array([
                _match_observation_to_state_index(traj[t], states) for t in range(T)
            ])
    else:
        sidx_arr = np.array([
            _match_observation_to_state_index(traj[t], states) for t in range(T)
        ])

    ranks = []
    for sidx in sidx_arr:
        attr = int(basin_mapping[int(sidx)])
        ranks.append(rank_map.get(attr, np.nan))
    return np.asarray(ranks, dtype=float)


def _n_states_in_landscape(land):
    """ランドスケープ上の局所最小（= State）の総数。"""
    lm = land.get("local_minima_indices", [])
    return int(len(np.asarray(lm)))


def _configure_state_yaxis(ax, n_states, ylabel, color, side="left", y_shift=0.0):
    """State rank 用の y 軸（全 State を表示）。y_shift で目盛位置を縦にずらす。"""
    n_states = int(n_states)
    y_shift = float(y_shift)
    if n_states > 0:
        ticks = np.arange(n_states) + y_shift
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"State{k + 1}" for k in range(n_states)])
        ax.set_ylim(-0.35 + y_shift, max(n_states - 1, 0) + 0.35 + y_shift)
    ax.set_ylabel(ylabel, color=color, fontsize=10, fontweight="bold")
    ax.tick_params(axis="y", labelcolor=color)
    if side == "right":
        ax.yaxis.label.set_color(color)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.2)


def _configure_time_xaxis(ax, T):
    """横軸: 時点を 1 刻みで表示。"""
    T = int(T)
    if T > 0:
        ax.set_xticks(np.arange(T))
        ax.set_xlim(-0.5, T - 0.5)
    ax.set_xlabel("Time", fontsize=10)


def _plot_state_line_on_axis(
    ax,
    ranks,
    line_color,
    zorder=3,
    marker_zorder=5,
    x_shift=0.0,
    linestyle="-",
    marker="o",
    label=None,
    markersize=11,
):
    """1本の State 折れ線（時点ノードを直線で結ぶ）。x_shift で横方向にずらして重なり回避。"""
    ranks = np.asarray(ranks, dtype=float)
    T = len(ranks)
    if T == 0:
        return
    t_axis = np.arange(T, dtype=float) + float(x_shift)
    valid = np.isfinite(ranks)
    if not np.any(valid):
        return

    ax.plot(
        t_axis[valid],
        ranks[valid],
        color=line_color,
        linewidth=2.0,
        alpha=0.95,
        zorder=zorder,
        solid_capstyle="round",
        linestyle=linestyle,
        marker=marker,
        markersize=markersize,
        markerfacecolor=line_color,
        markeredgecolor="black",
        markeredgewidth=0.9,
        label=label,
    )


def _lane_offsets_for_series(series_dict, lane_step=0.12):
    """
    同一軌跡（系列）をグループ化し、重なる系列に y 方向レーンオフセットを付与。
    Returns: {label: offset}
    """
    key_to_labels = {}
    for lbl, arr in series_dict.items():
        key = tuple(np.asarray(arr, dtype=float).tolist())
        key_to_labels.setdefault(key, []).append(lbl)
    offsets = {}
    for _key, lbls in key_to_labels.items():
        for j, lbl in enumerate(lbls):
            offsets[lbl] = j * lane_step
    return offsets


def _scatter_labels_with_jitter(ax, xs, ys, labels, colors, jitter=0.04):
    """重なった点にラベルが被らないようジッター付き散布。"""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    for k, lbl in enumerate(labels):
        ax.scatter(
            xs[k], ys[k], s=220, c=[colors[k]],
            edgecolors="black", linewidths=1.2, zorder=5,
        )
    # 重なり検出してオフセット
    placed = []
    for k, lbl in enumerate(labels):
        dx, dy = 8, 8
        for px, py, _ in placed:
            if abs(xs[k] - px) < jitter and abs(ys[k] - py) < jitter:
                dx += 14
                dy += 10
        ax.annotate(
            lbl, (xs[k], ys[k]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=13, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.85, ec=colors[k], lw=1.0),
        )
        placed.append((xs[k], ys[k], lbl))


def _plot_state_timeline(
    ax,
    ranks,
    state_colors=None,
    panel_label=None,
    title=None,
    line_color=None,
    n_states=None,
):
    """単一ランドスケープの State 折れ線（左軸のみ）。"""
    ranks = np.asarray(ranks, dtype=float)
    T = len(ranks)
    if T == 0:
        ax.axis("off")
        return

    lc = line_color if line_color is not None else "0.25"
    if n_states is None:
        n_states = int(np.nanmax(ranks)) + 1 if np.any(np.isfinite(ranks)) else 0
    _plot_state_line_on_axis(ax, ranks, lc)
    _configure_state_yaxis(ax, n_states, "State", lc, side="left")
    _configure_time_xaxis(ax, T)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    if panel_label is not None:
        ax.text(
            0.02, 0.98, str(panel_label), transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85, ec="none"),
        )
    if title:
        ax.set_title(title, fontsize=11, pad=6)


def _plot_state_timeline_dual(
    ax,
    ranks_individual,
    ranks_population,
    individual_color,
    population_color="red",
    panel_label=None,
    title=None,
    n_states_individual=None,
    n_states_population=None,
    y_jitter_population=0.25,
):
    """
    左軸: Individual の State、右軸: Population の State。
    折れ線・軸ラベルもそれぞれ individual_color / population_color。
    y 軸は各ランドスケープの全 State を表示。
    重なり回避は Time 方向ではなく Population 側を縦に y_jitter だけずらす。
    """
    ranks_individual = np.asarray(ranks_individual, dtype=float)
    ranks_population = np.asarray(ranks_population, dtype=float)
    T = len(ranks_individual)
    if T == 0:
        ax.axis("off")
        return

    if n_states_individual is None:
        n_states_individual = (
            int(np.nanmax(ranks_individual)) + 1
            if np.any(np.isfinite(ranks_individual)) else 0
        )
    if n_states_population is None:
        n_states_population = (
            int(np.nanmax(ranks_population)) + 1
            if np.any(np.isfinite(ranks_population)) else 0
        )

    _plot_state_line_on_axis(
        ax, ranks_individual, individual_color,
        zorder=3, marker_zorder=5, x_shift=0.0,
        linestyle="-", marker="o",
    )
    _configure_state_yaxis(
        ax, n_states_individual, "State (Individual)", individual_color, side="left",
    )
    for spine in ax.spines.values():
        spine.set_edgecolor(individual_color)
        spine.set_linewidth(1.0)

    ax2 = ax.twinx()
    jitter = float(y_jitter_population)
    ranks_pop_plot = ranks_population + jitter
    _plot_state_line_on_axis(
        ax2, ranks_pop_plot, population_color,
        zorder=4, marker_zorder=6, x_shift=0.0,
        linestyle="--", marker="D",
    )
    _configure_state_yaxis(
        ax2, n_states_population, "State (Population)", population_color,
        side="right", y_shift=jitter,
    )

    _configure_time_xaxis(ax, T)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)

    if panel_label is not None:
        ax.text(
            0.02, 0.98, str(panel_label), transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85, ec="none"),
        )
    if title:
        ax.set_title(title, fontsize=11, pad=6)


def plot_state_timeline_grid(
    analyzer,
    results,
    data_all,
    individual_indices,
    *,
    panel_labels=None,
    trajectory_colors=None,
    rank_by="energy",
    population_color="red",
    figsize=(12, 8),
    save_fig=False,
    filename="state_timelines_grid",
    save_dpi=100,
    save_format="pdf",
):
    """
    各 individual の State 時系列折れ線のみを 2x2 で描画。
    左軸=Individual、右軸=Population。
    """
    individual_indices = list(individual_indices)
    n = len(individual_indices)
    if n == 0:
        raise ValueError("individual_indices is empty")

    if panel_labels is None:
        panel_labels = [f"ID {sid}" for sid in individual_indices]
    if trajectory_colors is None:
        base = plt.cm.Set2(np.linspace(0, 1, max(n, 4)))
        trajectory_colors = [base[i % len(base)] for i in range(n)]

    mu_all = np.asarray(results["mu_all"])
    mu_agg = mu_all[:, 0, :] if mu_all.ndim == 3 else mu_all
    pop_land = analyzer.compute_energy_landscape(results["eta"], show_progress=False)

    nrows = 2 if n > 1 else 1
    ncols = 2 if n > 1 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for i, sid in enumerate(individual_indices[:4]):
        r, c = divmod(i, 2)
        ax = axes[r, c]
        theta_i = mu_agg[sid]
        ind_land = analyzer.compute_energy_landscape(theta_i, show_progress=False)
        ranks_ind = _trajectory_state_ranks(
            data_all[sid], ind_land, rank_by=rank_by, analyzer=analyzer,
        )
        ranks_pop = _trajectory_state_ranks(
            data_all[sid], pop_land, rank_by=rank_by, analyzer=analyzer,
        )
        _plot_state_timeline_dual(
            ax, ranks_ind, ranks_pop,
            individual_color=trajectory_colors[i % len(trajectory_colors)],
            population_color=population_color,
            panel_label=panel_labels[i] if i < len(panel_labels) else None,
            title=f"Subject {sid}",
            n_states_individual=_n_states_in_landscape(ind_land),
            n_states_population=_n_states_in_landscape(pop_land),
        )

    for j in range(n, nrows * ncols):
        r, c = divmod(j, 2)
        axes[r, c].axis("off")

    fig.suptitle("State trajectories", fontsize=14)
    plt.tight_layout()

    if save_fig:
        output_dir = Path("./figs_for_paper")
        output_dir.mkdir(parents=True, exist_ok=True)
        ext = save_format.lstrip(".")
        output_path = output_dir / f"{filename}.{ext}"
        fig.savefig(output_path, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved to {output_path}")

    return fig


def plot_individuals_with_population_center(
    analyzer,
    results,
    data_all,
    individual_indices,
    population_trajectory=None,
    population_index=None,
    population_overlay_all=False,
    trajectory_colors=None,
    population_trajectory_color="black",
    population_timeline_color="red",
    panel_labels=None,
    figsize=(22, 18),
    save_fig=False,
    filename="individuals_with_population_center",
    *,
    show_bottom_panels=True,
    rank_by="energy",
    lightweight=False,
    grid_resolution=500,
    n_energy_contours=40,
    show_all_nodes=True,
    layout_iterations=100,
    verbose=True,
    save_dpi=100,
    save_format="pdf",
):
    """
    4 Individual + 中央 Population のクロスレイアウト。
    各 Individual パネルは [State時系列 | ランドスケープ] を横並びに配置。

    添付スケッチに合わせ、下部に (左) 個人間の配置 (右) Population State1 からの距離 dis。
    """
    plot_kw = dict(
        grid_resolution=grid_resolution,
        n_energy_contours=n_energy_contours,
        show_all_nodes=show_all_nodes,
        layout_iterations=layout_iterations,
        verbose=verbose,
        lightweight=lightweight,
        save_dpi=save_dpi,
        save_format=save_format,
        attractor_label="State",
    )
    individual_indices = list(individual_indices)
    if len(individual_indices) != 4:
        raise ValueError(f"individual_indices must contain exactly 4 IDs, got {len(individual_indices)}")

    if panel_labels is None:
        panel_labels = ["A", "B", "C", "D"]
    if len(panel_labels) != 4:
        raise ValueError(f"panel_labels must have length 4, got {len(panel_labels)}")

    if trajectory_colors is None:
        base = plt.cm.Set2(np.linspace(0, 1, 4))
        trajectory_colors = [base[i] for i in range(4)]
    if len(trajectory_colors) != 4:
        raise ValueError(f"trajectory_colors must contain exactly 4 colors, got {len(trajectory_colors)}")

    if population_trajectory is None:
        if population_overlay_all:
            population_trajectory = data_all[individual_indices]
        else:
            if population_index is None:
                population_index = individual_indices[0]
            population_trajectory = data_all[population_index]

    eta = results["eta"]
    pop_land = analyzer.compute_energy_landscape(eta, show_progress=False)

    mu_all = np.asarray(results["mu_all"])
    mu_agg = mu_all[:, 0, :] if mu_all.ndim == 3 else mu_all

    pos_map = {0: (0, 0), 1: (0, 2), 2: (2, 0), 3: (2, 2), "population": (1, 1)}
    n_bottom = 1 if show_bottom_panels else 0
    height_ratios = [1, 1.25, 1] + ([0.55] if n_bottom else [])
    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(
        3 + n_bottom, 3,
        height_ratios=height_ratios,
        width_ratios=[1, 1.25, 1],
        wspace=0.08, hspace=0.08,
    )

    axes_land = {}
    axes_ts = {}
    for i in range(4):
        r, c = pos_map[i]
        gs_corner = outer[r, c].subgridspec(1, 2, width_ratios=[0.45, 0.55], wspace=0.08)
        axes_ts[i] = fig.add_subplot(gs_corner[0, 0])
        axes_land[i] = fig.add_subplot(gs_corner[0, 1])

    rg, cg = pos_map["population"]
    axes_land["population"] = fig.add_subplot(outer[rg, cg])

    for r, c in [(0, 1), (1, 0), (1, 2), (2, 1)]:
        ax_empty = fig.add_subplot(outer[r, c])
        ax_empty.axis("off")

    # Population reference State1 (global minimum on population landscape)
    pop_lm = np.asarray(pop_land["local_minima_indices"], dtype=int)
    pop_states = np.asarray(pop_land["states"], dtype=int)
    if pop_lm.size > 0:
        pop_ref_idx = int(pop_lm[np.argmin(pop_land["energies"][pop_lm])])
        pop_ref_bits = pop_states[pop_ref_idx]
    else:
        pop_ref_bits = None

    dis_series = {}
    for i, sid in enumerate(individual_indices):
        theta_i = mu_agg[sid]
        ind_land = analyzer.compute_energy_landscape(theta_i, show_progress=False)
        ranks_ind = _trajectory_state_ranks(
            data_all[sid], ind_land, rank_by=rank_by, analyzer=analyzer,
        )
        ranks_pop = _trajectory_state_ranks(
            data_all[sid], pop_land, rank_by=rank_by, analyzer=analyzer,
        )

        _plot_state_timeline_dual(
            axes_ts[i], ranks_ind, ranks_pop,
            individual_color=trajectory_colors[i],
            population_color=population_timeline_color,
            panel_label=panel_labels[i],
            title=f"ID {sid}",
            n_states_individual=_n_states_in_landscape(ind_land),
            n_states_population=_n_states_in_landscape(pop_land),
        )
        plot_weighted_landscape_with_nodes(
            ind_land,
            data_all[sid],
            ax=axes_land[i],
            title=f"Individual ID:{sid}",
            is_group=False,
            trajectory_color=trajectory_colors[i],
            show_time_labels=False,
            save_fig=False,
            **plot_kw,
        )

        if pop_ref_bits is not None:
            dis = [
                int(np.sum(np.asarray(data_all[sid][t], dtype=int) != pop_ref_bits))
                for t in range(data_all[sid].shape[0])
            ]
            dis_series[panel_labels[i]] = np.asarray(dis, dtype=float)

    pop_color_arg = trajectory_colors if population_overlay_all else population_trajectory_color
    plot_weighted_landscape_with_nodes(
        pop_land,
        population_trajectory,
        ax=axes_land["population"],
        title="Population Landscape (All Trajectories)" if population_overlay_all else "Population Landscape",
        is_group=True,
        trajectory_color=pop_color_arg,
        show_time_labels=not population_overlay_all,
        save_fig=False,
        **plot_kw,
    )

    if show_bottom_panels and dis_series:
        gs_bot = outer[3, :].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.12)
        ax_embed = fig.add_subplot(gs_bot[0, 0])
        ax_dis = fig.add_subplot(gs_bot[0, 1])

        # 左: 個人ラベル A–D の 2D 配置（dis の平均ベクトルから 2D 射影）
        labels = list(dis_series.keys())
        D = np.array([dis_series[lbl] for lbl in labels], dtype=float)
        mean_dis = D.mean(axis=1)
        if D.shape[1] >= 2:
            U, S, Vt = np.linalg.svd(D - D.mean(axis=0, keepdims=True), full_matrices=False)
            coords = U[:, :2] * S[:2]
        else:
            coords = np.c_[mean_dis, np.zeros(len(labels))]

        _scatter_labels_with_jitter(
            ax_embed, coords[:, 0], coords[:, 1], labels, trajectory_colors,
        )
        ax_embed.set_title("Individuals (by trajectory dis)", fontsize=12)
        ax_embed.set_xlabel("axis 1")
        ax_embed.set_ylabel("axis 2")
        ax_embed.grid(True, alpha=0.25)

        # 右: dis 時系列（同一軌跡はレーンずらし + 破線で区別）
        lane_offsets = _lane_offsets_for_series(dis_series, lane_step=0.15)
        linestyle_cycle = ["-", "--", "-.", ":"]
        for k, lbl in enumerate(labels):
            y = dis_series[lbl] + lane_offsets.get(lbl, 0.0)
            dup_note = ""
            if lane_offsets.get(lbl, 0.0) > 0:
                dup_note = " (dup)"
            ax_dis.plot(
                y, color=trajectory_colors[k],
                linewidth=1.8, alpha=0.9,
                linestyle=linestyle_cycle[lane_offsets.get(lbl, 0.0) % len(linestyle_cycle)],
                marker="o", markersize=5,
                label=f"{lbl}{dup_note}",
            )
        T_dis = max(len(v) for v in dis_series.values()) if dis_series else 0
        if T_dis > 0:
            ax_dis.set_xticks(np.arange(T_dis))
            ax_dis.set_xlim(-0.5, T_dis - 0.5)
        ax_dis.set_title("Distance to Population State1", fontsize=12)
        ax_dis.set_xlabel("Time")
        ax_dis.set_ylabel("dis (Hamming) + lane offset")
        ax_dis.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax_dis.grid(True, alpha=0.25, linestyle="--")

    fig.suptitle("4 Individuals + Population (Center)", fontsize=18)
    fig.subplots_adjust(top=0.94)

    if save_fig:
        output_dir = Path("./figs_for_paper")
        output_dir.mkdir(parents=True, exist_ok=True)
        ext = save_format.lstrip(".")
        output_path = output_dir / f"{filename}.{ext}"
        fig.savefig(output_path, dpi=save_dpi, bbox_inches="tight")
        if verbose:
            print(f"Saved to {output_path}")

    return fig


def plot_individuals_with_group_center(
    analyzer,
    results,
    data_all,
    individual_indices,
    group_trajectory=None,
    group_index=None,
    group_overlay_all=False,
    trajectory_colors=None,
    group_trajectory_color="black",
    ax=None,
    figsize=(16, 16),
    save_fig=False,
    filename="individuals_with_group_center",
    *,
    lightweight=False,
    grid_resolution=500,
    n_energy_contours=40,
    show_all_nodes=True,
    layout_iterations=100,
    verbose=True,
    save_dpi=100,
    save_format="pdf",
):
    """
    Individualを4枚、Groupを中央1枚に配置して描画する。
    レイアウト:
      □   □
        □
      □   □

    Args:
        analyzer: LandscapeAnalyzer
        results: VEMMEM.fit()の返り値（eta, mu_all を含む想定）
        data_all: (N, T, d) の2値時系列データ
        individual_indices: 個人描画対象のsubject_id 4件
        group_trajectory: Groupパネルに重ねる軌跡 (T, d) または (N, T, d)。Noneなら下記で自動決定
        group_index: group_trajectory未指定時に使用するsubject_id
        group_overlay_all: TrueならGroup中央は全被験者の軌跡を重ね描き
        trajectory_colors: Individual 4件の軌跡色リスト
        group_trajectory_color: Groupパネルの軌跡色
        ax: 既存Axesを使う場合に指定。dict または (3,3) のAxes配列を受け付ける。
        lightweight: True で軽量描画プリセット（格子150, 等高線10, 全ノード非表示, dpi=100）
        grid_resolution: 補間格子の解像度（小さいほどファイル軽量）
        n_energy_contours: エネルギー等高線の本数（0で非表示）
        show_all_nodes: False で 2^d 個の全ノード散布を省略（効果大）
        layout_iterations: spring_layout の反復回数
        save_dpi / save_format: 保存時の解像度・形式（png も可）
    """
    plot_kw = dict(
        grid_resolution=grid_resolution,
        n_energy_contours=n_energy_contours,
        show_all_nodes=show_all_nodes,
        layout_iterations=layout_iterations,
        verbose=verbose,
        lightweight=lightweight,
        save_dpi=save_dpi,
        save_format=save_format,
    )
    individual_indices = list(individual_indices)
    if len(individual_indices) != 4:
        raise ValueError(f"individual_indices must contain exactly 4 IDs, got {len(individual_indices)}")

    if trajectory_colors is None:
        base = plt.cm.Set2(np.linspace(0, 1, 4))
        trajectory_colors = [base[i] for i in range(4)]
    if len(trajectory_colors) != 4:
        raise ValueError(f"trajectory_colors must contain exactly 4 colors, got {len(trajectory_colors)}")

    if group_trajectory is None:
        if group_overlay_all:
            # Group中央には、指定した individual_indices のみを重ねる
            group_trajectory = data_all[individual_indices]  # (K, T, d), K=len(individual_indices)
        else:
            if group_index is None:
                group_index = individual_indices[0]
            group_trajectory = data_all[group_index]

    # ランドスケープ計算
    eta = results["eta"]
    group_land = analyzer.compute_energy_landscape(eta)

    mu_all = np.asarray(results["mu_all"])
    if mu_all.ndim == 3:
        mu_agg = mu_all[:, 0, :]
    else:
        mu_agg = mu_all

    # Figure and cross-like layout
    pos_map = {
        0: (0, 0),
        1: (0, 2),
        2: (2, 0),
        3: (2, 2),
        "group": (1, 1),
    }
    if ax is None:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, wspace=0.02, hspace=0.02)
        axes = {}
        for i in range(4):
            r, c = pos_map[i]
            axes[i] = fig.add_subplot(gs[r, c])
        rg, cg = pos_map["group"]
        axes["group"] = fig.add_subplot(gs[rg, cg])
        empty_cells = [(0, 1), (1, 0), (1, 2), (2, 1)]
        for r, c in empty_cells:
            ax_empty = fig.add_subplot(gs[r, c])
            ax_empty.axis("off")
    else:
        if isinstance(ax, dict):
            required_keys = {0, 1, 2, 3, "group"}
            if not required_keys.issubset(set(ax.keys())):
                raise ValueError("ax dict must include keys: 0, 1, 2, 3, 'group'")
            axes = ax
            fig = axes["group"].figure
        elif isinstance(ax, np.ndarray) and ax.ndim == 2 and ax.shape[0] >= 3 and ax.shape[1] >= 3:
            axes = {
                0: ax[0, 0],
                1: ax[0, 2],
                2: ax[2, 0],
                3: ax[2, 2],
                "group": ax[1, 1],
            }
            fig = axes["group"].figure
            for r, c in [(0, 1), (1, 0), (1, 2), (2, 1)]:
                ax[r, c].axis("off")
        else:
            raise ValueError("ax must be None, dict of axes, or a (>=3,>=3) axes ndarray")

    # Individual 4 panels
    for i, sid in enumerate(individual_indices):
        theta_i = mu_agg[sid]
        ind_land = analyzer.compute_energy_landscape(theta_i)
        plot_weighted_landscape_with_nodes(
            ind_land,
            data_all[sid],
            ax=axes[i],
            title=f"Individual ID:{sid}",
            is_group=False,
            trajectory_color=trajectory_colors[i],
            save_fig=False,
            attractor_label="State",
            **plot_kw,
        )

    # Population center panel (palette different by is_group=True)
    group_color_arg = trajectory_colors if group_overlay_all else group_trajectory_color
    plot_weighted_landscape_with_nodes(
        group_land,
        group_trajectory,
        ax=axes["group"],
        title="Population Landscape (All Trajectories)" if group_overlay_all else "Population Landscape",
        is_group=True,
        trajectory_color=group_color_arg,
        show_time_labels=not group_overlay_all,
        save_fig=False,
        attractor_label="State",
        **plot_kw,
    )

    fig.suptitle("4 Individuals + 1 Population (Center)", fontsize=18)
    plt.tight_layout()

    if save_fig:
        output_dir = Path("./figs_for_paper")
        output_dir.mkdir(parents=True, exist_ok=True)
        ext = save_format.lstrip(".")
        output_path = output_dir / f"{filename}.{ext}"
        fig.savefig(output_path, dpi=save_dpi, bbox_inches='tight')
        if verbose:
            print(f"Saved to {output_path}")

    return fig


def plot_landscapes_by_score(
    analyzer,
    results,
    data_all,
    score_df,
    target_score,
    score_col='score',
    subject_col='subject_id',
    figsize=(18, 15),
    save_fig=False,
    filename="landscapes_by_score"
):
    """
    指定スコアに一致する被験者全員の全時点を、1つのGroup levelランドスケープ上に重ねて表示。
    
    Args:
        analyzer: LandscapeAnalyzer インスタンス
        results: VEMMEM.fit() の返り値
        data_all: (N, T, d) のデータ
        score_df: スコア情報を含むDataFrame (subject_id, score列)
        target_score: 可視化したいスコア（単一の値）
        score_col: スコア列名
        subject_col: 被験者ID列名
        figsize: 図のサイズ
        save_fig: 保存するか
        filename: 保存ファイル名
    """
    import jax.numpy as jnp
    
    # Group level（eta）のランドスケープを計算（全員共通）
    eta = jnp.array(results['eta'])
    land_group = analyzer.compute_energy_landscape(eta)
    
    N, T, d = data_all.shape
    
    # スコアと被験者IDのマッピング
    score_map = dict(zip(score_df[subject_col], score_df[score_col]))
    
    # 指定スコアに一致する被験者を全て取得
    matching_subjects = []
    for sid in range(N):
        if sid in score_map and score_map[sid] == target_score:
            matching_subjects.append(sid)
    
    if len(matching_subjects) == 0:
        print(f"No subjects found with score={target_score}")
        return None
    
    # 全被験者・全時点の状態を収集（メタデータ付き）
    all_states = []
    metadata = []  # (subject_id, time_idx) のリスト
    
    for sid in matching_subjects:
        traj = data_all[sid]
        T_subj = traj.shape[0]
        for t in range(T_subj):
            all_states.append(traj[t])
            metadata.append((sid, t))
    
    all_states = np.array(all_states)
    
    # ノードごとに、どの被験者のどの時点が来ているかを記録
    num_states = len(land_group['energies'])
    n_bits = int(np.log2(num_states))
    node_to_metadata = {}  # node_idx -> [(subject_id, time_idx), ...]
    
    for state, (sid, t) in zip(all_states, metadata):
        # 状態配列を逆順にしてから文字列に変換（LSB-firstに合わせる）
        state_reversed = state[::-1]
        state_str = "".join(map(str, state_reversed.astype(int)))
        node_idx = int(state_str, 2)
        if node_idx not in node_to_metadata:
            node_to_metadata[node_idx] = []
        node_to_metadata[node_idx].append((sid, t))
    
    # グラフ構築とレイアウト計算（plot_weighted_landscape_with_nodesと同じロジック）
    G = nx.Graph()
    G.add_nodes_from(range(num_states))
    for i in range(num_states):
        for b in range(n_bits):
            neighbor = i ^ (1 << b)
            if neighbor < num_states and i < neighbor:
                w = 10.0 if land_group['basin_mapping'][i] == land_group['basin_mapping'][neighbor] else 0.5
                G.add_edge(i, neighbor, weight=w)
    
    pos = nx.spring_layout(G, seed=42, weight='weight', k=0.2, iterations=100)
    x = np.array([pos[i][0] for i in range(num_states)])
    y = np.array([pos[i][1] for i in range(num_states)])
    
    # 補間
    grid_x, grid_y = np.mgrid[x.min()-0.1:x.max()+0.1:500j, y.min()-0.1:y.max()+0.1:500j]
    grid_basin_raw = griddata((x, y), land_group['basin_mapping'], (grid_x, grid_y), method='nearest')
    grid_energy = griddata((x, y), land_group['energies'], (grid_x, grid_y), method='cubic')
    
    # カラーマッピング
    lm_indices = land_group['local_minima_indices']
    lm_data = [{'index': idx, 'energy': land_group['energies'][idx]} for idx in lm_indices]
    lm_data_sorted = sorted(lm_data, key=lambda x: x['energy'])
    basin_id_to_rank = {item['index']: i for i, item in enumerate(lm_data_sorted)}
    
    def get_rank(basin_id):
        return basin_id_to_rank.get(basin_id, -1)
    
    v_get_rank = np.vectorize(get_rank)
    grid_rank = v_get_rank(grid_basin_raw)
    
    tab20 = plt.cm.tab20.colors
    num_basins = len(lm_data_sorted)
    used_colors = [tab20[i % 20] for i in range(num_basins)]
    custom_cmap = ListedColormap(used_colors)
    
    # プロット
    fig, ax = plt.subplots(figsize=figsize)
    
    # A. 領域の塗りつぶし
    contour_levels = np.arange(num_basins + 1) - 0.5
    ax.contourf(grid_x, grid_y, grid_rank, levels=contour_levels, cmap=custom_cmap, alpha=0.5)
    
    # B. エネルギー等高線
    ax.contour(grid_x, grid_y, grid_energy, levels=40, colors='black', linewidths=0.5)
    
    # C. 領域の境界線
    ax.contour(grid_x, grid_y, grid_rank, levels=np.arange(num_basins), colors='black', linewidths=2.5, linestyles='solid')
    
    # D. 背景ノード
    visited_nodes = set(node_to_metadata.keys())
    bg_indices = [i for i in range(num_states) if i not in visited_nodes and i not in lm_indices]
    if bg_indices:
        ax.scatter([pos[i][0] for i in bg_indices], [pos[i][1] for i in bg_indices], 
                   s=15, c='black', alpha=0.15, zorder=2)
    
    # E. 訪問ノード（複数の被験者・時点が重なる可能性あり）
    visited_no_lm = [i for i in visited_nodes if i not in lm_indices]
    if visited_no_lm:
        ax.scatter([pos[i][0] for i in visited_no_lm], [pos[i][1] for i in visited_no_lm],
                   s=150, c='cyan', edgecolors='blue', linewidths=1.5, marker='o', alpha=1.0, zorder=5, label="Visited")
        
        # ラベル：各ノードに来ている全ての被験者・時点を表示
        for node_idx in visited_no_lm:
            meta_list = node_to_metadata[node_idx]
            # ラベル文字列を作成（例: "id=0,t=0,3; id=10,t=2"）
            label_parts = []
            for sid, t in meta_list:
                label_parts.append(f"id={sid},t={t}")
            label_text = "\n".join(label_parts)
            
            ax.text(pos[node_idx][0], pos[node_idx][1], label_text,
                   fontsize=8, color='darkblue', fontweight='bold',
                   ha='left', va='bottom', zorder=25,
                   bbox=dict(boxstyle="square,pad=0.2", fc="white", alpha=0.8, ec="blue", lw=1))
    
    # F. LM Nodes
    for rank, item in enumerate(lm_data_sorted):
        idx = item['index']
        color = used_colors[rank]
        ax.scatter(pos[idx][0], pos[idx][1], s=600, c=[color], edgecolors='black', 
                   linewidths=2.0, marker='*', zorder=20)
        display_rank = rank + 1
        label_text = f"LM{display_rank}\n[{idx:0{n_bits}b}]"
        if display_rank == 1:
            label_text = f"Global Min\n(LM{display_rank})\n[{idx:0{n_bits}b}]"
        ax.annotate(label_text, (pos[idx][0], pos[idx][1]), xytext=(0, 20),
                   textcoords='offset points', fontsize=10, fontweight='bold',
                   ha='center', color='black',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray"),
                   zorder=25)
    
    ax.set_title(f"Group Level Landscape: Score={target_score} (n={len(matching_subjects)} subjects, {len(all_states)} timepoints)", 
                fontsize=18)
    ax.axis('off')
    
    if save_fig:
        output_dir = Path("./figs_for_paper")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{filename}_score{target_score}.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_path}")
    plt.show()
    
    return fig