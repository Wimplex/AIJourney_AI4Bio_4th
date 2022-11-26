
import time
import json
from typing import Iterable, Dict, Any

import tqdm
import numpy as np
import pandas as pd

from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split, cross_val_score

from utils.base import check_dir
from feature_building import prepare_data
from constants import REPORTS_DIR, TARGET_LABEL
from metrics import eer_precise, classification_metrics
from plotting import plot_scores_hist, plot_fr_fa_curve


def __oos_detection_report(
    target_scores: Iterable, 
    imposter_scores: Iterable, 
    plot_eer: bool, 
    output_dir: str,
):
    check_dir(output_dir)
    trues = [1.0] * len(target_scores) + [0.0] * len(imposter_scores)
    probs = target_scores + imposter_scores
    plot_scores_hist(
        target_scores, 
        imposter_scores, 
        title="in-set and oos scores distribution",
        save_path=output_dir / "scores.png", 
        plot_eer_point=plot_eer
    )
    plot_fr_fa_curve(
        trues,
        probs,
        title="FR-FA curves on different thresholds",
        save_path=output_dir / "fr_fa.png",
        plot_eer_point=plot_eer
    )
    eer, eer_thr = eer_precise(trues, probs)
    with open(output_dir / "eer.txt", "w") as file:
        file.write(f"eer\t=\t{eer}\n")
        file.write(f"eer_thr\t=\t{eer_thr}\n")


def __train_and_predict_scores(
    estimator: ClassifierMixin, 
    in_set_df: pd.DataFrame, 
    oos_df: pd.DataFrame
):
    """ Fits model on in-set data and computes probability scores for in-set and oos data """

    X_in_set = prepare_data(in_set_df)
    X_oos = prepare_data(oos_df)
    X_in_set_train, X_in_set_val, y_in_set_train, _ = train_test_split(
            X_in_set, in_set_df[TARGET_LABEL], train_size=0.8, stratify=in_set_df[TARGET_LABEL])
    estimator.fit(X_in_set_train, y_in_set_train)
    in_set_scores = np.max(estimator.predict_proba(X_in_set_val), axis=1).tolist()
    oos_scores = np.max(estimator.predict_proba(X_oos), axis=1).tolist()
    return in_set_scores, oos_scores


def evaluation_report(
    estimator: ClassifierMixin,
    train_df: pd.DataFrame,
    report_oos_detection: bool = True,
    report_cv: bool = True,
    oos_df: pd.DataFrame = None,
    oos_num_iterations: int = 10, 
    oos_num_classes: int = 2,
    f1_cv_num_folds: int = 3
):

    dir_name = time.strftime("%Y-%m-%d %H_%M")
    output_dir = REPORTS_DIR / dir_name
    check_dir(output_dir)

    # OOS detection report
    if report_oos_detection:
        print("OOS detection evaluation")

        in_set_scores_overall, oos_scores_overall = [], []
        if oos_df is None:
            for _ in tqdm.tqdm(range(oos_num_iterations), desc="Confidence threshold computation"):
                oos_labels = np.random.choice(train_df[TARGET_LABEL].unique(), oos_num_classes)
                df_in_set = train_df[~train_df[TARGET_LABEL].isin(oos_labels)]
                df_oos = train_df[train_df[TARGET_LABEL].isin(oos_labels)]
                in_set_scores, oos_scores = __train_and_predict_scores(estimator, df_in_set, df_oos)
                in_set_scores_overall += in_set_scores
                oos_scores_overall += oos_scores
        else:
            in_set_scores, oos_scores = __train_and_predict_scores(estimator, train_df, oos_df)
            in_set_scores_overall += in_set_scores
            oos_scores_overall += oos_scores

        __oos_detection_report(in_set_scores_overall, oos_scores_overall, plot_eer=True, output_dir=output_dir)

    if report_cv:
        print("Cross-validation")
        X = prepare_data(train_df)
        score = cross_val_score(estimator, X, train_df[TARGET_LABEL], cv=f1_cv_num_folds, scoring="f1_macro", verbose=2)
        metrics = {"f1_score": score.tolist() }
    else:
        X = prepare_data(train_df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, train_df[TARGET_LABEL], stratify=train_df[TARGET_LABEL], train_size=0.8)
        estimator.fit(X_train, y_train)
        preds = estimator.predict(X_test)
        metrics = classification_metrics(y_test, preds)

    with open(output_dir / "metrics.json", "w") as file:
        json.dump(metrics, file)

    return output_dir