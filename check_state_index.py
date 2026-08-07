import numpy as np
import jax.numpy as jnp

# VEM_MEM.pyと同じ方法で状態を生成
d = 9
n_states = 1 << d
s = (jnp.arange(n_states)[:, None] & (1 << jnp.arange(d))) > 0
states = np.array(s.astype(jnp.int8))

# '111111100'を配列に変換（左から右に読む）
obs_str = '111111100'
obs_array = np.array([int(c) for c in obs_str])

print(f'観測状態 (文字列): {obs_str}')
print(f'観測状態 (配列): {obs_array}')

# 完全一致を探す
matches = np.all(states == obs_array, axis=1)
if np.any(matches):
    state_idx = np.where(matches)[0][0]
    print(f'\n完全一致が見つかりました: state_index = {state_idx}')
    print(f'対応する状態配列: {states[state_idx]}')
else:
    print('\n完全一致が見つかりませんでした')
    # ハミング距離が最小のものを探す
    hamming_dists = np.sum(states != obs_array, axis=1)
    min_idx = np.argmin(hamming_dists)
    print(f'ハミング距離最小: state_index = {min_idx}, 距離 = {hamming_dists[min_idx]}')
    print(f'対応する状態配列: {states[min_idx]}')

# 参考: 2進数として解釈した場合
binary_as_int = int(obs_str, 2)
print(f'\n文字列を2進数として解釈した場合: {binary_as_int}')
print(f'その状態配列: {states[binary_as_int]}')

# 状態生成の仕組みを確認
print(f'\n状態生成の仕組み確認:')
print(f'state_index 0: {states[0]}')
print(f'state_index 1: {states[1]}')
print(f'state_index 2: {states[2]}')
print(f'state_index 3: {states[3]}')
print(f'state_index 127: {states[127]}')
print(f'state_index 508: {states[508]}')
