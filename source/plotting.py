from functools import partial
from itertools import product
from typing import Iterable, Dict

import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve


from utils.base import check_dir
from metrics import eer_precise
from constants import TARGET_LABEL, VIS_DIR


plt.rcParams["axes.axisbelow"] = True


def plot_spectrum(train_df, by: str = "Intens."):
    """ Plots spectrum across each class with peaks markers """

    for target in tqdm.tqdm(train_df[TARGET_LABEL].unique()):
        df = train_df[train_df[TARGET_LABEL] == target]

        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(10, 6)
        for index in df.index:
            xticks = np.round(np.array(df.loc[index, "m/z"]), 0)
            ys = df.loc[index, by]
            ax.plot(xticks, ys, c="b", alpha=0.1)
            ax.set_xticks(xticks, xticks, rotation=90)
            ax.tick_params(axis="both", which="major", labelsize=8)

            if "peaks_x" in df.columns:
                xs, ys = df.loc[index, "peaks_x"], df.loc[index, "peaks_y"]
                ax.scatter(xs, ys, c="r", alpha=0.1)

        ax.set_title(target)
        plt.savefig(f"../plots/spec/{target}.png", dpi=150)
        plt.close()


def plot_spectrum_in_feature_space(
    peaks_x: np.ndarray, 
    peaks_y: np.ndarray, 
    targets: Iterable,
    n_peaks: int,
    plot_subdir: str = "feat_spec"
):
    """ Does the same but for extracted peak features """

    if isinstance(targets, list):
        targets = np.array(targets)

    for tar in tqdm.tqdm(np.unique(targets)):
        ids = np.argwhere(targets == tar)
        p_xs, p_ys = peaks_x[ids].squeeze(), peaks_y[ids].squeeze()
        p_xs, p_ys = p_xs.reshape((-1, 1)), p_ys.reshape((-1, 1))
        
        # Clustering
        cluster = KMeans(n_peaks)
        cl_label = cluster.fit_predict(p_xs)
        cl_label = cl_label / np.max(cl_label)

        # Plotting
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(10, 6)
        cmap = plt.cm.hsv
        ax.scatter(p_xs, p_ys, c=cmap(cl_label), alpha=0.5)
        ax.set_title(tar)
        plt.ylim(0.0, 1.0)
        plt.xlim(0.0, 1.0)
        plt.tight_layout()

        output_dir = VIS_DIR / plot_subdir
        check_dir(output_dir)
        plt.savefig(output_dir / f"{tar}.png", dpi=150)
        plt.close()


def plot_specific_points(points: Dict[str, float]):
    colors = plt.cm.Dark2
    for i, (name, point) in enumerate(points.items()):
        plt.axvline(x=point, c=colors(i / len(points)), linestyle="dotted", label=f"{name}={np.round(point, 3)}")


def plot_scores_hist(
    scores_dict: Dict[str, Iterable],
    title: str, 
    save_path: str, 
    specific_points: Dict[str, float] = None
):
    for name, scores in scores_dict.items():
        plt.hist(scores, bins=100, alpha=0.5, label=name)

    if specific_points: plot_specific_points(specific_points)
    plt.title(title)
    plt.legend()
    plt.grid(True, "both")
    plt.xlabel("Score")
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_fr_fa_curve(
    trues: Iterable, 
    probs: Iterable, 
    title: str, 
    save_path: str, 
    specific_points: Dict[str, float] = None
):
    fpr, tpr, thr = roc_curve(trues, probs)
    fnr = 1 - tpr
    plt.plot(thr, fpr, c="b", label="FA")
    plt.plot(thr, fnr, c="g", label="FR")
    if specific_points: plot_specific_points(specific_points)
    plt.title(title)
    plt.legend()
    plt.grid(True, "both")
    plt.xlabel("Threshold")
    plt.xlim(specific_points["eer"] - 0.2, specific_points["eer"] + 0.2)
    plt.ylim(0.0, 1.0)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_features(df: pd.DataFrame, df_aug: pd.DataFrame = None, type: str = "scatter", polar: bool = False, output_subdir: str = None):

    plotting_types = {
        "scatter": partial(plt.scatter, s=2),
        "line": plt.plot,
    }
    plot_funct = plotting_types[type]

    for tar in tqdm.tqdm(df[TARGET_LABEL].unique(), desc="Plotting"):
        curr_df = df[df[TARGET_LABEL] == tar]

        if df_aug is not None:
            curr_df_aug = df_aug[df_aug[TARGET_LABEL] == tar]
        
        if output_subdir:
            output_dir = VIS_DIR / output_subdir
        else:
            output_dir = VIS_DIR / "plots_hist_features"
        check_dir(output_dir)
        plt.figure(figsize=(12, 8))

        if polar:
            plt.axes(projection="polar")

        if df_aug is not None:
            for spec in curr_df_aug["spec"].values:
                xs = np.arange(spec.shape[0])
                if polar:

                    plot_funct(2 * np.pi * xs / len(spec), spec, color="g", alpha=0.6)
                else:
                    plot_funct(xs, spec, color="g", alpha=0.6)

        for spec in curr_df["spec"].values:
            xs = np.arange(spec.shape[0])
            if polar:
                plot_funct(2 * np.pi * xs / len(spec), spec, color="r", alpha=0.6)
            else:
                plot_funct(xs, spec, color="r", alpha=0.6)

        plt.title(tar)
        plt.savefig(output_dir / f"{tar}_spec.png", dpi=200)
        plt.close()


def plot_tsne(points: np.ndarray, labels: Iterable[str], output_dir: str):
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    uniques = np.unique(labels)
    plt.figure(figsize=(8, 6))
    colors = [plt.cm.Dark2(i) for i in range(9)]
    markers = "ovsD"
    combs = list(product(colors, markers))

    for i, l in enumerate(uniques):
        ids = np.argwhere(labels == l)
        curr_points = points[ids].squeeze()
        plt.scatter(
            curr_points[:, 0], 
            curr_points[:, 1], 
            c=combs[i][0], 
            s=1.9, 
            marker=combs[i][1], 
            linewidth=0.15,
            edgecolor="black",
            label=l,
        )
    plt.title("tSNE")
    plt.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))
    plt.tight_layout()
    plt.savefig(output_dir / "tsne.png", dpi=400)
    plt.close()