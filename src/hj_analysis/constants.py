"""h/J 解析で共有する定数定義."""

from typing import Iterable, Tuple

DEFAULT_TARGET_ITEMS_0B: Tuple[int, ...] = (6, 7, 8)
DEFAULT_BACKGROUND_ITEMS_0B: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)


def to_one_based_items(items_0b: Iterable[int]) -> Tuple[int, ...]:
    """0-based item index列を1-basedへ変換する."""
    return tuple(int(i) + 1 for i in items_0b)
