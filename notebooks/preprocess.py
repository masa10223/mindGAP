import numpy as np
import pandas as pd

def phq9_binarize(x, thr=2):  # thr 引数を追加（デフォルト値は2）
    if pd.isna(x):
        return np.nan
    try:
        # 文字列の場合は数値に変換
        x = float(x)
        if x >= thr:  # 2 を thr に変更
            return 1
        elif x == 1:
            return 0
        else:
            return np.nan
    except (ValueError, TypeError):
        return np.nan

def hq25_binarize(x, thr = 2):
    if pd.isna(x):
        return np.nan
    try:
        # 文字列の場合は数値に変換
        x = float(x)
        if x >= thr:
            return 1
        elif 1 <= x < thr:
            return 0
        else:
            return np.nan
    except (ValueError, TypeError):
        return np.nan