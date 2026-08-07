"""Load questionnaire outcome scores (07_PHQ9_analysis.ipynb patterns)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "original"


def _read_excel(name: str, **kwargs) -> pd.DataFrame:
    path = DATA_DIR / name
    return pd.read_excel(path, engine="openpyxl", header=2, index_col=0, **kwargs)


def load_phq9_score_sum() -> tuple[pd.Index, np.ndarray]:
    phq9_df = _read_excel("PHQ-9.xlsx").dropna(axis=0)
    values = np.asarray(phq9_df - 1, dtype=float)
    score_sum = values.reshape(phq9_df.shape[0], 5, 9).sum(axis=2).sum(axis=1)
    return phq9_df.index, score_sum


def load_hq25_scores() -> dict[str, np.ndarray]:
    hq25_df = _read_excel("HQ-25+4.xlsx").dropna(axis=0)

    hq25_df.iloc[:, 3::29] = 6 - hq25_df.iloc[:, 3::29]
    hq25_df.iloc[:, 6::29] = 6 - hq25_df.iloc[:, 6::29]
    hq25_df.iloc[:, 9::29] = 6 - hq25_df.iloc[:, 9::29]
    hq25_df.iloc[:, 14::29] = 6 - hq25_df.iloc[:, 14::29]
    hq25_df.iloc[:, 20::29] = 6 - hq25_df.iloc[:, 20::29]
    hq25_df.iloc[:, 21::29] = 6 - hq25_df.iloc[:, 21::29]
    hq25_df.iloc[:, 24::29] = 6 - hq25_df.iloc[:, 24::29]

    social = np.array([0, 1, 3, 5, 7, 10, 12, 14, 17, 19, 24])
    isolation = np.array([4, 8, 11, 15, 18, 21, 22, 23])
    emotional = np.array([6, 9, 13, 16, 20])
    set_start_indices = np.array([0, 29, 29 * 2, 29 * 3, 29 * 4])

    def _subscale_sum(item_indices: np.ndarray) -> np.ndarray:
        cols = (set_start_indices[:, np.newaxis] + item_indices).flatten()
        arr = np.asarray(hq25_df.iloc[:, cols], dtype=float) - 1
        n_items = item_indices.shape[0]
        return arr.reshape(hq25_df.shape[0], 5, n_items).sum(axis=2).sum(axis=1)

    total_minus1 = np.asarray(hq25_df - 1, dtype=float)
    total_sum = total_minus1.reshape(hq25_df.shape[0], 5, 29).sum(axis=2).sum(axis=1)

    return {
        "index": hq25_df.index,
        "HQ-25 (total)": total_sum,
        "HQ-25 social": _subscale_sum(social),
        "HQ-25 emotional": _subscale_sum(emotional),
        "HQ-25 isolation": _subscale_sum(isolation),
    }


def load_iat_score_sum(subject_index: pd.Index) -> np.ndarray:
    iat_df = _read_excel("IAT.xlsx")
    iat_minus_1 = (iat_df - 1).loc[subject_index]
    n_waves = 5
    n_items = int(iat_minus_1.shape[1] / n_waves)
    arr = np.asarray(iat_minus_1, dtype=float)
    return arr.reshape(iat_minus_1.shape[0], n_waves, n_items).sum(axis=2).sum(axis=1)


def load_ips22_score_sum(subject_index: pd.Index) -> np.ndarray:
    ips22_df = _read_excel("IPS-22.xlsx")
    ips22_df = ips22_df[ips22_df.columns[:-1]]
    ips22_minus_1 = (ips22_df - 1).loc[subject_index]
    n_waves = 4
    n_items = int(ips22_minus_1.shape[1] / n_waves)
    arr = np.asarray(ips22_minus_1, dtype=float)
    return arr.reshape(ips22_minus_1.shape[0], n_waves, n_items).sum(axis=2).sum(axis=1)


def load_tacs22_score_sum(subject_index: pd.Index) -> np.ndarray:
    tacs22_df = _read_excel("TACS-22.xlsx")
    tacs22_df = tacs22_df[tacs22_df.columns[:-1]]
    tacs22_minus_1 = tacs22_df - 1
    tacs22_minus_1.iloc[:, 3::22] = 4 - tacs22_minus_1.iloc[:, 3::22]
    tacs22_minus_1.iloc[:, 8::22] = 4 - tacs22_minus_1.iloc[:, 8::22]
    tacs22_minus_1 = tacs22_minus_1.loc[subject_index]
    n_waves = 4
    n_items = int(tacs22_minus_1.shape[1] / n_waves)
    arr = np.asarray(tacs22_minus_1, dtype=float)
    return arr.reshape(tacs22_minus_1.shape[0], n_waves, n_items).sum(axis=2).sum(axis=1)


def load_all_outcome_scores() -> dict[str, np.ndarray]:
    """Return outcome label -> (n_subjects,) total score arrays (248 subjects)."""
    phq_index, phq9_sum = load_phq9_score_sum()
    hq25 = load_hq25_scores()
    hq_index = hq25["index"]

    if not phq_index.equals(hq_index):
        raise ValueError("PHQ-9 and HQ-25 subject indices do not match")

    outcomes = {
        "PHQ-9": phq9_sum,
        "HQ-25 (total)": hq25["HQ-25 (total)"],
        "HQ-25 social": hq25["HQ-25 social"],
        "HQ-25 emotional": hq25["HQ-25 emotional"],
        "HQ-25 isolation": hq25["HQ-25 isolation"],
        "IAT": load_iat_score_sum(hq_index),
        "IPS-22": load_ips22_score_sum(hq_index),
        "TACS-22": load_tacs22_score_sum(hq_index),
    }

    n = len(phq_index)
    for name, arr in outcomes.items():
        if len(arr) != n:
            raise ValueError(f"{name}: expected {n} subjects, got {len(arr)}")

    return outcomes


def load_all_outcome_means() -> dict[str, np.ndarray]:
    """Return outcome label -> wave-averaged score (sum / n_waves)."""
    sums = load_all_outcome_scores()
    wave_counts = {
        "PHQ-9": 5,
        "HQ-25 (total)": 5,
        "HQ-25 social": 5,
        "HQ-25 emotional": 5,
        "HQ-25 isolation": 5,
        "IAT": 5,
        "IPS-22": 4,
        "TACS-22": 4,
    }
    return {name: sums[name] / wave_counts[name] for name in sums}


def load_sa_score_means() -> pd.DataFrame:
    """Per-subject mean scores for ScoreAssociation (07_PHQ9 patterns)."""
    phq9_df = _read_excel("PHQ-9.xlsx").dropna(axis=0)
    n_subj = phq9_df.shape[0]
    subject_id = np.arange(n_subj, dtype=int)

    phq9_values = np.asarray(phq9_df - 1, dtype=float)
    score_mean = phq9_values.reshape(n_subj, 5, 9).sum(axis=2).mean(axis=1)

    hq25_df = _read_excel("HQ-25+4.xlsx").dropna(axis=0)
    hq25_df.iloc[:, 3::29] = 6 - hq25_df.iloc[:, 3::29]
    hq25_df.iloc[:, 6::29] = 6 - hq25_df.iloc[:, 6::29]
    hq25_df.iloc[:, 9::29] = 6 - hq25_df.iloc[:, 9::29]
    hq25_df.iloc[:, 14::29] = 6 - hq25_df.iloc[:, 14::29]
    hq25_df.iloc[:, 20::29] = 6 - hq25_df.iloc[:, 20::29]
    hq25_df.iloc[:, 21::29] = 6 - hq25_df.iloc[:, 21::29]
    hq25_df.iloc[:, 24::29] = 6 - hq25_df.iloc[:, 24::29]
    hq25_mean = (
        np.asarray(hq25_df - 1, dtype=float)
        .reshape(n_subj, 5, 29)
        .sum(axis=2)
        .mean(axis=1)
    )

    ext_index = hq25_df.index

    iat_df = _read_excel("IAT.xlsx")
    iat_minus_1 = (iat_df - 1).loc[ext_index]
    iat_mean = (
        np.asarray(iat_minus_1, dtype=float)
        .reshape(n_subj, 5, int(iat_minus_1.shape[1] / 5))
        .sum(axis=2)
        .mean(axis=1)
    )

    ips22_df = _read_excel("IPS-22.xlsx")
    ips22_df = ips22_df[ips22_df.columns[:-1]]
    ips22_minus_1 = (ips22_df - 1).loc[ext_index]
    ips22_mean = (
        np.asarray(ips22_minus_1, dtype=float)
        .reshape(n_subj, 4, int(ips22_minus_1.shape[1] / 4))
        .sum(axis=2)
        .mean(axis=1)
    )

    tacs22_df = _read_excel("TACS-22.xlsx")
    tacs22_df = tacs22_df[tacs22_df.columns[:-1]]
    tacs22_minus_1 = tacs22_df - 1
    tacs22_minus_1.iloc[:, 3::22] = 4 - tacs22_minus_1.iloc[:, 3::22]
    tacs22_minus_1.iloc[:, 8::22] = 4 - tacs22_minus_1.iloc[:, 8::22]
    tacs22_minus_1 = tacs22_minus_1.loc[ext_index]
    tacs22_mean = (
        np.asarray(tacs22_minus_1, dtype=float)
        .reshape(n_subj, 4, int(tacs22_minus_1.shape[1] / 4))
        .sum(axis=2)
        .mean(axis=1)
    )

    return pd.DataFrame(
        {
            "subject_id": subject_id,
            "score_mean": score_mean,
            "ips22_mean": ips22_mean,
            "iat_mean": iat_mean,
            "tacs22_mean": tacs22_mean,
            "hq25_mean": hq25_mean,
        }
    )


FIG6_PANEL_SPECS: list[dict[str, str]] = [
    {"label": "A", "outcome": "PHQ-9", "sa_col": "score_mean"},
    {"label": "B", "outcome": "IPS-22", "sa_col": "ips22_mean"},
    {"label": "C", "outcome": "IAT", "sa_col": "iat_mean"},
    {"label": "D", "outcome": "TACS-22", "sa_col": "tacs22_mean"},
    {"label": "E", "outcome": "HQ-25 (total)", "sa_col": "hq25_mean"},
]
