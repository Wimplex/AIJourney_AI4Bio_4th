from typing import Dict, Any, Iterable

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from utils.base import get_top_n_indices, min_max_scale_2d, save_model
from constants import MODELS_DIR


def build_peak_features(df: pd.DataFrame, by: str = "Intens.", n_peaks: int = 15, scaling_params: Dict[str, Any] = None):
    """ Extracts peak features """

    xs, ys = [], []
    for index in df.index:
        y = df.loc[index, by]
        x = df.loc[index, "m/z"]

        # Interpolate over specified column column
        y = interp1d(x, y)
        mi, ma = np.min(x), np.max(x)
        step_len = 10
        num = int((ma - mi) * step_len)
        xs_new = np.linspace(np.min(x), np.max(x), num)
        ys_new = y(xs_new)

        # Find peaks ids
        peaks_ids, properties = find_peaks(ys_new, height=0.1, prominence=0.01)

        # Filter peaks
        peaks_ids = peaks_ids[get_top_n_indices(ys_new[peaks_ids], n=n_peaks)]

        # Get peaks
        peaks_ys, peaks_xs = ys_new[peaks_ids], xs_new[peaks_ids]
        sorted_by_x_ids = np.argsort(peaks_xs)
        peaks_xs = peaks_xs[sorted_by_x_ids]
        peaks_ys = peaks_ys[sorted_by_x_ids]

        pad = n_peaks - len(peaks_xs)
        if pad > 0:
            zeros = np.zeros(pad)
            peaks_xs = np.append(peaks_xs, zeros)
            peaks_ys = np.append(peaks_ys, zeros)

        xs.append(peaks_xs)
        ys.append(peaks_ys)
    
    xs = np.array(xs)
    ys = np.array(ys)

    if scaling_params is None:
        xs, min_x, max_x = min_max_scale_2d(xs)
        ys, min_y, max_y = min_max_scale_2d(ys)
        save_model(
            {
                "x": {"min": min_x, "max": max_x},
                "y": {"min": min_y, "max": max_y}
            },
            MODELS_DIR / "scaling_params.pkl"
        )
    else:
        xs, _, _ = min_max_scale_2d(xs, scaling_params["x"]["min"], scaling_params["x"]["max"])
        ys, _, _ = min_max_scale_2d(xs, scaling_params["y"]["min"], scaling_params["y"]["max"])

    return xs, ys


def build_hist_features(df: pd.DataFrame, by: str = "Rel. Intens.", new_feat_name: str = None):
    """ Standartize feature vectors """

    def create_spec_vector(
        mz: Iterable, 
        feature: Iterable, 
        min_mz: int = 200, 
        max_mz: int = 2000
    ):
        spec = []
        for curr_mz in range(min_mz, max_mz):
            if curr_mz in mz:
                spec.append(feature[mz.index(curr_mz)])
            else: 
                spec.append(0)
        return np.array(spec)

    df["mz_rounded"] = df["m/z"].apply(lambda x: [int(x_ // 10) for x_ in x])
    feats = df.apply(lambda x: create_spec_vector(x["mz_rounded"], x[by]), axis=1)
    if new_feat_name:
        df[new_feat_name] = feats
    else:
        df[f"spec_{by}"] = feats
    return df


def prepare_data(train_df: pd.DataFrame) -> np.ndarray:
    X = train_df["spec"].values.tolist()
    X = np.stack(X)
    return X