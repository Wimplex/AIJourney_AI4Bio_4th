import warnings
from typing import Iterable

import torch
import numpy as np

from augmentation import Normalize
from constants import MODELS_DIR, OUTPUT_DIR, INPUT_DIR
from utils.base import read_data, check_dir, softmax, seed_everything

from train import infer as infer_nn, build_features
from nn.dataset import BioDataset
from nn.nn_blocks.zoo.resnet import AMSotmaxHeadModel, LinearHeadModel


warnings.simplefilter("ignore")


def run(json_data_name: str, features: Iterable[str], threshold: float = 0.5, checkpoint_name: str = "checkpoint_6"):
    seed_everything(1997)
    df = read_data(INPUT_DIR / json_data_name)
    df = build_features(df, features)

    state_dict = torch.load(MODELS_DIR / checkpoint_name, map_location="cpu")
    # transforms = None
    transforms = {"spec_Intens.": [Normalize(p=1.0)]}
    dset = BioDataset(df, transforms, True, state_dict["label2id"])
    loader = torch.utils.data.DataLoader(dset, 1)
    model = AMSotmaxHeadModel(input_dim=1800, in_channels=len(features), num_classes=26, weights_path=None)
    # model = LinearHeadModel(input_dim=1800, in_channels=len(features), num_classes=26, weights_path=None)
    model.load_state_dict(state_dict["model"])

    logits = infer_nn(model, loader)
    logits = np.array([softmax(l) for l in logits])
    scores = np.max(logits, axis=1)
    id2label = {v: k for k, v in state_dict["label2id"].items()}
    preds = np.array([id2label[np.argmax(l, axis=0)] for l in logits])

    for i in range(len(preds)):
        if np.max(scores[i]) < threshold:
            preds[i] = "new"

    df["target_class"] = preds
    
    check_dir(OUTPUT_DIR)
    df[["target_class"]].to_csv(OUTPUT_DIR / "results.csv", index=True, index_label="id")


if __name__ == "__main__":
    run(
        json_data_name="test.json",
        # features=["Intens.", "FWHM", "Res."],
        features=["Intens."],
        threshold=0.06849997755404827,
        # threshold=0.12632267616383425,
        checkpoint_name="report_on 2022-11-10 18_59_norm.pth"
    )