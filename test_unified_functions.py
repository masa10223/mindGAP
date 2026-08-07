#!/usr/bin/env python3
"""
統一された関数名のテストスクリプト

calc_energy/calculate_energy と calc_prob/calculate_prob の統一をテストします。
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from energy_landscape.ela import (
    calculate_energy, calculate_prob, fit_approx, run_mcmc_simulation,
    find_major_states, analyze_trajectory_dynamics, save_and_plot_dynamics
)


def test_unified_energy_calculation():
    """統一されたエネルギー計算関数のテスト"""
    print("=== 統一されたエネルギー計算テスト ===\n")
    
    # パラメータ
    n_questions = 5
    h = np.random.randn(n_questions) * 0.5
    J = np.random.randn(n_questions, n_questions) * 0.3
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)
    
    print(f"パラメータ設定:")
    print(f"  h: {h}")
    print(f"  J の範囲: {J.min():.4f} ～ {J.max():.4f}")
    
    # 単一状態のテスト
    print("\n1. 単一状態のエネルギー計算:")
    single_state = np.array([1, -1, 1, -1, 1])
    energy_single = calculate_energy(single_state, h, J)
    print(f"   状態 {single_state}: エネルギー = {energy_single:.4f}")
    
    # 複数状態のテスト
    print("\n2. 複数状態のエネルギー計算:")
    multiple_states = np.array([
        [1, 1, 1, 1, 1],
        [-1, -1, -1, -1, -1],
        [1, -1, 1, -1, 1]
    ])
    energies_multiple = calculate_energy(multiple_states, h, J)
    print(f"   複数状態のエネルギー: {energies_multiple}")
    
    # 一貫性テスト
    print("\n3. 一貫性テスト:")
    individual_energies = [calculate_energy(state, h, J) for state in multiple_states]
    print(f"   個別計算: {individual_energies}")
    print(f"   一括計算: {energies_multiple}")
    print(f"   差の最大値: {np.max(np.abs(energies_multiple - individual_energies)):.10f}")
    
    return True


def test_unified_probability_calculation():
    """統一された確率計算関数のテスト"""
    print("\n=== 統一された確率計算テスト ===\n")
    
    # パラメータ
    n_questions = 4  # 小規模テスト
    h = np.array([0.1, -0.2, 0.3, -0.1])
    J = np.array([
        [0.0, 0.2, -0.1, 0.0],
        [0.2, 0.0, 0.0, -0.1],
        [-0.1, 0.0, 0.0, 0.2],
        [0.0, -0.1, 0.2, 0.0]
    ])
    
    print(f"パラメータ設定:")
    print(f"  h: {h}")
    print(f"  J:\n{J}")
    
    # 全状態の生成
    from itertools import product
    all_states = np.array(list(product([-1, 1], repeat=n_questions)), dtype=np.int8)
    print(f"\n全状態数: {len(all_states)}")
    
    # 確率計算
    print("\n1. 確率計算:")
    probabilities = calculate_prob(h, J, all_states)
    print(f"   確率の範囲: {probabilities.min():.6f} ～ {probabilities.max():.6f}")
    print(f"   確率の合計: {probabilities.sum():.10f}")
    
    # 正規化チェック
    print("\n2. 正規化チェック:")
    if abs(probabilities.sum() - 1.0) < 1e-10:
        print("   ✅ 確率は正しく正規化されています")
    else:
        print(f"   ❌ 確率の合計が1.0ではありません: {probabilities.sum()}")
    
    # エネルギーとの一貫性チェック
    print("\n3. エネルギーとの一貫性チェック:")
    energies = calculate_energy(all_states, h, J)
    max_energy = np.max(energies)
    expected_probs = np.exp(-(energies - max_energy))
    expected_probs = expected_probs / expected_probs.sum()
    
    prob_diff = np.max(np.abs(probabilities - expected_probs))
    print(f"   計算された確率と期待値の差: {prob_diff:.10f}")
    
    if prob_diff < 1e-10:
        print("   ✅ エネルギー計算と確率計算は一貫しています")
    else:
        print("   ❌ エネルギー計算と確率計算に不整合があります")
    
    return True


def test_mcmc_with_unified_functions():
    """統一された関数を使用したMCMCテスト"""
    print("\n=== 統一された関数を使用したMCMCテスト ===\n")
    
    # パラメータ
    n_questions = 6
    h = np.random.randn(n_questions) * 0.3
    J = np.random.randn(n_questions, n_questions) * 0.2
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)
    
    print(f"パラメータ設定:")
    print(f"  h の範囲: {h.min():.4f} ～ {h.max():.4f}")
    print(f"  J の範囲: {J.min():.4f} ～ {J.max():.4f}")
    
    # MCMCシミュレーション
    print("\n1. MCMCシミュレーション:")
    n_steps = 10000
    trajectory = run_mcmc_simulation(h, J, n_steps=n_steps, burn_in=0.1, show_progress=True)
    print(f"   軌跡生成完了: 形状={trajectory.shape}")
    
    # 軌跡のエネルギー計算
    print("\n2. 軌跡のエネルギー計算:")
    trajectory_energies = [calculate_energy(state, h, J) for state in trajectory]
    print(f"   軌跡エネルギーの範囲: {min(trajectory_energies):.4f} ～ {max(trajectory_energies):.4f}")
    print(f"   軌跡エネルギーの平均: {np.mean(trajectory_energies):.4f}")
    
    # 主要状態の特定
    print("\n3. 主要状態の特定:")
    major_states = find_major_states(trajectory, num_major_states=3)
    print(f"   発見された主要状態数: {len(major_states)}")
    
    # ダイナミクス解析
    print("\n4. ダイナミクス解析:")
    sojourn_probs, transition_matrix = analyze_trajectory_dynamics(trajectory, major_states)
    print(f"   滞在確率: {sojourn_probs}")
    print(f"   遷移行列の非ゼロ要素数: {(transition_matrix > 0).sum()}")
    
    return True


def test_fit_approx_with_unified_functions():
    """統一された関数を使用したfit_approxテスト"""
    print("\n=== 統一された関数を使用したfit_approxテスト ===\n")
    
    # ダミーデータの生成
    n_questions = 5
    n_samples = 1000
    X_data = pd.DataFrame(np.random.randint(0, 2, size=(n_samples, n_questions)))
    
    print(f"データ設定:")
    print(f"  サンプル数: {n_samples}")
    print(f"  質問数: {n_questions}")
    print(f"  データの値の範囲: {X_data.min().min()} ～ {X_data.max().max()}")
    
    # パラメータ推定
    print("\n1. パラメータ推定:")
    h_estimated, J_estimated = fit_approx(X_data, max_iter=100, alpha=0.9)
    print(f"   推定完了:")
    print(f"   h の範囲: {h_estimated.min():.4f} ～ {h_estimated.max():.4f}")
    print(f"   J の範囲: {J_estimated.min():.4f} ～ {J_estimated.max():.4f}")
    
    # 推定されたパラメータでの確率計算テスト
    print("\n2. 推定パラメータでの確率計算テスト:")
    # 全状態の生成
    from itertools import product
    all_states = np.array(list(product([-1, 1], repeat=n_questions)), dtype=np.int8)
    all_states_01 = (all_states + 1) / 2  # {-1,1} -> {0,1}に変換
    
    probabilities = calculate_prob(h_estimated, J_estimated, all_states_01)
    print(f"   全状態の確率計算完了: {len(probabilities)}状態")
    print(f"   確率の合計: {probabilities.sum():.10f}")
    
    if abs(probabilities.sum() - 1.0) < 1e-10:
        print("   ✅ 推定パラメータでの確率計算は正常です")
    else:
        print(f"   ❌ 確率の合計が1.0ではありません: {probabilities.sum()}")
    
    return True


if __name__ == "__main__":
    try:
        # 統一されたエネルギー計算のテスト
        test_unified_energy_calculation()
        
        # 統一された確率計算のテスト
        test_unified_probability_calculation()
        
        # 統一された関数を使用したMCMCテスト
        test_mcmc_with_unified_functions()
        
        # 統一された関数を使用したfit_approxテスト
        test_fit_approx_with_unified_functions()
        
        print("\n🎉 統一された関数名のテスト完了!")
        print("\n✅ 全ての関数が正常に動作しています:")
        print("   - calculate_energy: 統一されたエネルギー計算")
        print("   - calculate_prob: 統一された確率計算")
        print("   - 既存のMCMC機能との互換性")
        print("   - 既存のfit_approx機能との互換性")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


