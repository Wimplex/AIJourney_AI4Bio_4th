import warnings
warnings.filterwarnings("ignore")

import time
import json
import argparse
from typing import Dict, Any, Iterable

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

import torch
from torch.utils.data import DataLoader

from train import infer, build_features
from nn.dataset import BioDataset
from nn.nn_blocks.zoo.resnet import AMSotmaxHeadModel, LinearHeadModel

import generation as gen
import augmentation as aug
from constants import INPUT_DIR, MODELS_DIR, REPORTS_DIR, TARGET_LABEL
from utils.base import seed_everything, read_data, check_dir, softmax
from plotting import plot_scores_hist, plot_fr_fa_curve, plot_tsne
from metrics import classification_metrics, eer_precise, fr_point, fa_point


def model_target_data(df: pd.DataFrame, features: Iterable[str], aug_params: Dict[str, Any]) -> pd.DataFrame:
    new_df = aug.augment_data(df, features, aug_params)
    return new_df


def model_imposter_data(df: pd.DataFrame, features: Iterable[str], gen_params: Dict[str, Any], aug_params: Dict[str, any]) -> pd.DataFrame:
    new_df = gen.generate_data(df, features, gen_params)
    new_df = aug.augment_data(new_df, features, aug_params)
    return new_df


def save_df(df: pd.DataFrame, logits: np.ndarray, thr: float, label2id: Dict[str, int], file_path: str):
    id2label = {v: k for k, v in label2id.items()}
    scores = [np.max(l) for l in logits]
    preds = [id2label[np.argmax(l)] for l in logits]
    for i in range(len(preds)):
        if scores[i] < thr:
            preds[i] = "new"
    df["predicted_class"] = preds
    df["predicted_score"] = scores
    df["correct"] = df["predicted_class"] == df[TARGET_LABEL]
    df[["predicted_class", TARGET_LABEL, "predicted_score", "correct"]].to_csv(file_path, index=True, index_label="id")


def oos_report(
    real_data_logits: np.ndarray,
    target_logits: np.ndarray, 
    imposter_logits: np.ndarray, 
    specific_points: Dict[str, float], 
    report_dir: str
) -> None:
    real_scores = [np.max(l) for l in real_data_logits]
    tar_scores = [np.max(l) for l in target_logits]
    imp_scores = [np.max(l) for l in imposter_logits]

    # Plot tar-imp scores hist
    scores = {"target_fake": tar_scores, "target_real": real_scores, "imposter": imp_scores}
    plot_scores_hist(
        scores,
        title="Target (real) VS Impostor (synthetic) scores",
        save_path=report_dir / "scores.png",
        specific_points=specific_points
    )

    # Plot FR-FA curve
    trues = [1.0] * (len(tar_scores) + len(real_scores)) + [0.0] * len(imp_scores)
    probs = real_scores + tar_scores + imp_scores
    plot_fr_fa_curve(
        trues, 
        probs, 
        title="FR-FA curves on different thresholds",
        save_path=report_dir / "fr_fa.png",
        specific_points=specific_points
    )


def classification_report(
    trues: Iterable, 
    probs: Iterable, 
    label2id: Dict[str, int], 
    report_dir: str,
    specific_points: Dict[str, float],
):
    id2label = {v: k for k, v in label2id.items()}
    preds = np.array([id2label[np.argmax(p)] for p in probs])
    results = {}

    # Compute in-class metrics
    if not isinstance(trues, np.ndarray): trues = np.array(trues)
    ids = np.argwhere(trues != "new")
    metrics = classification_metrics(trues[ids].squeeze(), preds[ids].squeeze(), probs[ids].squeeze())
    results["in-set"] = metrics

    new_thrs = np.linspace(specific_points["eer"] - 0.1, specific_points["eer"] + 0.1, 20).tolist()
    names = list(specific_points.keys()) + ["common"] * len(new_thrs)
    thresholds = list(specific_points.values()) + new_thrs
    for name, thr in zip(names, thresholds):
        preds_ = preds.copy()
        for i in range(len(probs)):
            if np.max(probs[i]) < thr:
                preds_[i] = "new"
        metrics = classification_metrics(trues, preds_)
        results[f"out-of-set: {name} = {thr}"] = metrics
    
    with open(report_dir / "classification.json", "w") as file:
        json.dump(results, file, indent="\t")

    with open(report_dir / "points.json", "w") as file:
        json.dump(specific_points, file, indent="\t")


def run(params: Dict[str, Any]):
    seed_everything(params["seed"])

    # Make reports dir
    dir_name = time.strftime("%Y-%m-%d %H_%M")
    output_dir = REPORTS_DIR / dir_name
    check_dir(output_dir)

    print(f"Testing results will be saved in {output_dir}")

    # Load and preprocess data
    df = read_data(INPUT_DIR / params["json_data_name"])
    df = build_features(df, params["features"])

    # Prepare target and imposter dfs
    feats = [f"spec_{f}" for f in params["features"]]
    tar_df = model_target_data(df, feats, params["aug"])
    tar_df["synth"] = True
    df["synth"] = False
    tar_df = pd.concat([df, tar_df])
    imp_df = model_imposter_data(df, feats, params["gen"], params["aug"])

    # Get ids for real and fake target data
    col = tar_df["synth"].values
    real_ids = np.argwhere(~col)
    fake_ids = np.argwhere(col)

    # Load checkpoint
    state_dict = torch.load(MODELS_DIR / params["checkpoint_name"], map_location="cpu")

    # Make dataloaders
    transforms = {"spec_Intens.": [aug.Normalize(p=1.0)]}
    # transforms = None
    tar_dset = BioDataset(tar_df, transforms, True, state_dict["label2id"])
    imp_dset = BioDataset(imp_df, transforms, True, state_dict["label2id"])
    tar_loader = DataLoader(tar_dset, 32)
    imp_loader = DataLoader(imp_dset, 32)

    # Initialize model
    model = AMSotmaxHeadModel(1800, len(params["features"]), 26, weights_path=None)
    # model = LinearHeadModel(1800, len(params["features"]), 26, weights_path=None)
    model.load_state_dict(state_dict["model"])

    # Infer target and imposters data
    print("Extracting logits:")
    tar_logits = infer(model, tar_loader, device=params["device"])
    imp_logits = infer(model, imp_loader, device=params["device"])
    tar_logits = np.array([softmax(l) for l in tar_logits])
    imp_logits = np.array([softmax(l) for l in imp_logits])
    tar_scores = np.max(tar_logits, axis=1)
    imp_scores = np.max(imp_logits, axis=1)

    # Compute specific points
    pfa, pfr = 0.2, 0.2
    trues = [0.0] * len(imp_scores) + [1.0] * len(tar_scores)
    scores = np.concatenate([imp_scores, tar_scores])
    eer, eer_thr = eer_precise(trues, scores)
    fr_thr = fr_point(trues, scores, pfr)
    fa_thr = fa_point(trues, scores, pfa)
    points = {
        "eer": eer_thr, 
        f"fr_{pfr}": fr_thr, 
        f"fa_{pfa}": fa_thr,
    }
    with open(output_dir / "eer.txt", "w") as file:
        file.write(f"{eer}")

    # Make reports
    real_tar_scores, fake_tar_scores = tar_scores[real_ids], tar_scores[fake_ids]
    oos_report(real_tar_scores, fake_tar_scores, imp_scores, points, output_dir)
    labels = np.array(tar_df[TARGET_LABEL].tolist() + ["new"] * len(imp_logits))
    classification_report(
        trues=labels, 
        probs=np.vstack([tar_logits, imp_logits]), 
        label2id=state_dict["label2id"], 
        report_dir=output_dir, 
        specific_points=points
    )

    # tSNE of training data
    print("Extracting embeddings:")
    tar_embs = infer(model, tar_loader, True, params["device"])
    imp_embs = infer(model, imp_loader, True, params["device"])
    tsne = TSNE()
    components = tsne.fit_transform(np.vstack([tar_embs, imp_embs]))
    labels = tar_df[TARGET_LABEL].tolist() + ["new"] * len(imp_embs)
    plot_tsne(components, labels, output_dir)

    # Save scores
    save_df(tar_df, tar_logits, points["eer"], state_dict["label2id"], output_dir / "targets.csv")
    save_df(imp_df, imp_logits, points["eer"], state_dict["label2id"], output_dir / "imposters.csv")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, help="Checkpoint of trained model to test (see 'source/models' dir)")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    params = {
        "seed": 1997,
        "json_data_name": "train.json",
        "checkpoint_name": args.checkpoint,
        # "features": ["Intens.", "FWHM", "Res."],
        # "features": ["Intens.", "Res."],
        "features": ["Intens."],
        "device": "cuda:0",

        "aug": {
            "num_augs_per_class": 100,
            "blocks": {
                "spec_Intens.": [
                    aug.SegmentsCombination(num_segments=100, p=1.0),
                    aug.PeakVariation(fraction_of_peaks=0.8, std_factor=0.8, p=0.8),
                    aug.NoisePerturbation(noise_std=0.01, p=0.7),
                    aug.SpectrumShift(shift_std=0.015, p=0.85),
                    # aug.PeakElimination(min_peaks_frac=0.095, max_peaks_frac=0.3, p=0.8),
                ],
                "spec_FWHM": [
                    aug.SegmentsCombination(num_segments=100, p=1.0)
                ],
                "spec_Res.": [
                    aug.SegmentsCombination(num_segments=100, p=1.0)
                ]
            },
        },

        "gen": {
            "num_new_classes": 26 * 5
        }
    }
    run(params)