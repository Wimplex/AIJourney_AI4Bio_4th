import sys
sys.path.append('../source')

import warnings
warnings.filterwarnings("ignore")

import argparse
from typing import Dict, Any, Iterable

import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

import generation as gen
import augmentation as aug
from utils.base import read_data, seed_everything, softmax, invert_dict
from utils.pkf import read_pkf
from metrics import classification_metrics
from feature_building import build_hist_features
from constants import INPUT_DIR, MODELS_DIR, DATASETS_DIR, TARGET_LABEL

from nn.dataset import BioDataset
from nn.nn_blocks.zoo.resnet import LinearHeadModel, AMSotmaxHeadModel


def infer(model: nn.Module, loader: DataLoader, emb: bool = False, device: str = "cpu"):
    probs = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_X in tqdm.tqdm(loader):
            batch_X = batch_X.to(device)
            output = model(batch_X, emb=emb)
            probs.append(output)
    probs = torch.vstack(probs).detach().cpu().squeeze().numpy()
    return probs


def test(model: nn.Module, loader: DataLoader, criterion: nn.Module, device="cpu"):
    trues, preds = [], []
    model.eval()
    with torch.no_grad():
        losses = []
        for batch_X, batch_y in tqdm.tqdm(loader, desc="Test"):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            output = model(batch_X)

            if not model.head.curricular:
                loss = criterion(output, batch_y)
            else:
                loss = torch.Tensor([0.0])
            
            losses.append(loss.item())
            preds.append(F.softmax(output).argmax(dim=1))
            trues.append(batch_y)

    loss = np.mean(losses)
    preds = torch.concat(preds, dim=0).squeeze().detach().cpu().numpy()
    trues = torch.concat(trues, dim=0).squeeze().detach().cpu().numpy()

    if len(trues.shape) > 1 and trues.shape[-1] != 1:
        trues = np.argmax(trues, axis=-1)

    return classification_metrics(trues, preds), loss


def train(
    model: nn.Module, 
    train_loader: DataLoader, 
    test_loader: DataLoader, 
    optimizer: optim.Optimizer, 
    scheduler: optim.lr_scheduler._LRScheduler,
    criterion: nn.Module, 
    num_epochs: int, 
    device: str = "cpu", 
    log_interval: int = None,
    verbose: bool = True,
    sanity_check: bool = True,
    save_model_after_epoch: bool = True,
    epoch: int = None
):

    def do_test(m, epoch: int = None):
        metrics, loss = test(m, test_loader, criterion, device)
        if loss is not None:
            metrics["loss"] = loss
        print("Classification metrics on test-set:")
        print({k: np.round(v, 4) for k, v in metrics.items()})
        if save_model_after_epoch and epoch is not None:
            save_path = MODELS_DIR / f"checkpoint_{epoch}.pth"
            print(f"Saving model in {save_path}")
            if isinstance(train_loader.dataset, Subset):
                label2id = train_loader.dataset.dataset.label2id
            else:
                label2id = train_loader.dataset.label2id
            state_dict = {
                "model": m.state_dict(),
                "label2id": label2id,
            }
            if epoch is not None:
                torch.save(state_dict, save_path)
        m.train()
        return m
    
    model.to(device)

    if not model.training:
        model.train()

    if sanity_check:
        print("Sanity check")
        do_test(model)

    for ep in range(num_epochs):
        if verbose:
            print(f"\n===== Epoch {ep + 1}/{num_epochs} =====")

        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (batch_X, batch_y) in pbar:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            if model.head.curricular:
                loss = model(batch_X, batch_y)
            else:
                output = model(batch_X)
                loss = criterion(output, batch_y)

            loss.backward()
            optimizer.step()
            scheduler.step()

            if log_interval and i % log_interval == 0:
                desc = f"Train loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}"
                pbar.set_description(desc)

        do_test(model, epoch if epoch else ep)

    return model


def build_features(df: pd.DataFrame, features: Iterable[str]):
    for feat in features:
        df = build_hist_features(df, by=feat)
    return df


def run_pretraining(params: Dict):
    seed_everything(params["seed"])
    
    # Load pretraining dataset
    # .pkf-data from https://zenodo.org/record/1880975
    dataset_path = DATASETS_DIR / "zenodo" / "v3" / "181130_ZENODO_Peaklist_30Peaks_1.6.pkf"
    df = read_pkf(dataset_path, "full")
    df = build_features(df, params["features"])

    # Filter small classes (FOR PRETRAINING ONLY)
    classes_to_filter = [
        k for k, v in df.groupby(TARGET_LABEL)[TARGET_LABEL].count().iteritems() if v < 3]
    df = df[~df[TARGET_LABEL].isin(classes_to_filter)]
    print("Num classes:", len(df[TARGET_LABEL].unique()))

    # Generate real test data
    # test_real_df = aug.augment_data(df, params["augmentation"])

    # Prepare train loader
    train_dset = BioDataset(df, params["augmentation"]["blocks"], False)
    label2weight = {tar: 1.0 / df[df[TARGET_LABEL] == tar].shape[0] for tar in df[TARGET_LABEL].unique()}
    weights = np.array([label2weight[l] for l in df[TARGET_LABEL].values])
    weights = torch.from_numpy(weights)
    sampler = WeightedRandomSampler(weights.type("torch.DoubleTensor"), len(train_dset))
    train_loader = DataLoader(train_dset, batch_size=64, sampler=sampler, pin_memory=True, num_workers=6)

    # Prepare test loader
    transforms = {"spec_Intens.": [aug.Normalize(p=1.0)]}
    test_real_dset = BioDataset(df, transforms, label2id=train_dset.label2id)
    test_real_loader = DataLoader(test_real_dset, batch_size=32, shuffle=False, pin_memory=True)

    model = AMSotmaxHeadModel(
        input_dim=1800,
        in_channels=len(params["features"]),
        num_classes=len(df[TARGET_LABEL].unique()),
        head_params={"m": 0.35, "s": 10},
    )
    optimizer = optim.Adam(
        model.parameters(), 
        lr=params["training"]["lr"],
        betas=(0.5, 0.999),
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=2.5e-3, 
        total_steps=len(train_loader) * 50,
        pct_start=0.3,
    )
    criterion = nn.CrossEntropyLoss()

    model = train(
        model=model, 
        train_loader=train_loader, 
        test_loader=test_real_loader, 
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion, 
        num_epochs=50,
        log_interval=10,
        device=params["training"]["device"]
    )


def run_finetuning(params: Dict[str, Any], df: pd.DataFrame = None):
    seed_everything(params["seed"])

    if df is None:
        df = read_data(INPUT_DIR / "train.json")
        df = build_features(df, params["features"])

        # Uncomment it if you want species pretrain
        # df[TARGET_LABEL] = df[TARGET_LABEL].apply(lambda x: x.split("_")[0])

    # Prepare train loader
    train_dset = BioDataset(
        df, params["augmentation"]["blocks"], False, num_iters=params["training"]["num_iters"], mixup_p=0.7)
    train_loader = DataLoader(train_dset, batch_size=128, shuffle=True, pin_memory=True, num_workers=6)

    # Prepare test loader
    transforms = {"spec_Intens.": [aug.Normalize(p=1.0)]}
    test_real_dset = BioDataset(df, transforms, label2id=train_dset.label2id)
    test_real_loader = DataLoader(test_real_dset, batch_size=32, shuffle=False, pin_memory=True)

    model = AMSotmaxHeadModel(
        input_dim=1800,
        weights_path="nn/pretrained/bacteria_ID_resnet.ckpt",
        in_channels=len(params["features"]),
        num_classes=len(df[TARGET_LABEL].unique()),
        head_params={"m": 0.2, "s": 50},
        freeze_encoder=True
    )
    optimizer = optim.Adam(
        model.parameters(), 
        lr=params["training"]["lr"],
        betas=(0.5, 0.999),
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-3, 
        total_steps=len(train_loader) * params["training"]["num_epochs"],
        pct_start=0.3,
    )
    criterion = nn.CrossEntropyLoss()

    train(
        model=model,
        train_loader=train_loader, 
        test_loader=test_real_loader, 
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion, 
        num_epochs=params["training"]["num_epochs"],
        log_interval=10,
        device=params["training"]["device"]
    )
    return model, train_dset.label2id


def run_mixup_training(params: Dict[str, Any]):

    # Load target dataset
    df = read_data(INPUT_DIR / "train.json")
    df = build_features(df, params["features"])
    
    # Load adapt dataset
    needed_targets = ["Acinetobacter baumannii", "Pseudomonas aeruginosa", "Staphylococcus aureus"]
    adapt_df_path = DATASETS_DIR / "zenodo" / "v3" / "181130_ZENODO_Peaklist_30Peaks_1.6.pkf"
    adapt_df = read_pkf(adapt_df_path)
    ids = []
    for i in range(len(adapt_df)):
        for t in needed_targets:
            if adapt_df.iloc[i][TARGET_LABEL].startswith(t):
                ids.append(i)
                break
    adapt_df = adapt_df.iloc[ids]
    adapt_df = build_features(adapt_df, params["features"])
    
    N = 10
    n_most = int(0.08 * len(adapt_df))
    final_num_epochs = params["training"]["num_epochs"]
    for i in range(N):
        print(f"Start training on {i + 1} iteration.")

        if i != N - 1:
            params["training"]["num_epochs"] = 4
            params["training"]["num_iters"] = 10000
        else:
            params["training"]["num_epochs"] = final_num_epochs
            params["training"]["num_iters"] = 60000

        # Retrain model
        model, label2id = run_finetuning(params, df)
        id2label = invert_dict(label2id)

        # Infer logits and select rows with highest confidence
        adapt_dset = BioDataset(adapt_df, None, True)
        adapt_loader = DataLoader(adapt_dset, 32)
        logits = infer(model, adapt_loader, device=params["training"]["device"])
        labels = [id2label[np.argmax(softmax(l))] for l in logits]
        adapt_df[TARGET_LABEL] = labels
        ids = np.argsort([np.max(softmax(l)) for l in logits])[-n_most:]
        not_ids = np.argsort([np.max(softmax(l)) for l in logits])[:n_most]
        selected_rows = adapt_df.iloc[ids, :]
        adapt_df = adapt_df.iloc[not_ids]

        # Add them to main train_df
        print("Train df shape before data mix:", df.shape)
        df = pd.concat([df, selected_rows])
        print("Train df shape after data mix:", df.shape)

    
def run_curricular_training(params: Dict[str, Any]):
    seed_everything(params["seed"])

    # Load target dataset
    df = read_data(INPUT_DIR / "train.json")
    df = build_features(df, params["features"])

    # Prepare datasets
    train_dset = BioDataset(df, params["augmentation"]["blocks"], mixup_p=0.7)
    test_dset = BioDataset(df, label2id=train_dset.label2id)
    test_loader = DataLoader(test_dset, 1)

    # num_iters = np.ceil(params["training"]["num_iters"] / params["training"]["batch_size"]).astype("int")
    num_iters = np.ceil(len(train_dset) / params["training"]["batch_size"])
    total_steps = int(num_iters * params["training"]["num_epochs"])

    # Init model, optimizer, lr_scheduler and criterion
    model = AMSotmaxHeadModel(
        input_dim=1800,
        weights_path="nn/pretrained/pretrained_on_zenodo.pth",
        in_channels=len(params["features"]),
        num_classes=len(df[TARGET_LABEL].unique()),
        head_params={"m": 0.2, "s": 50},
        freeze_encoder=True
    )
    optimizer = optim.Adam(
        model.parameters(), 
        lr=params["training"]["lr"],
        betas=(0.5, 0.999),
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-3, 
        total_steps=total_steps,
        pct_start=0.3,
    )
    criterion = nn.CrossEntropyLoss()

    weights = np.array([1.0] * len(train_dset))
    for i in range(params["training"]["num_epochs"]):
        print(f"{i + 1}-th epoch started.")

        # Prepare train_loader with weighted sampling
        ids = np.argsort(weights)
        rearranged_dset = Subset(train_dset, ids)
        train_loader = DataLoader(rearranged_dset, batch_size=64)

        save_model = (i + 1) % 40 == 0
        # save_model = True

        # Run one epoch loop
        model = train(
            model=model,
            train_loader=train_loader, 
            test_loader=test_loader, 
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion, 
            num_epochs=1,
            log_interval=10,
            device=params["training"]["device"],
            verbose=False,
            sanity_check=False,
            save_model_after_epoch=save_model,
            epoch=i + 1
        )

        # Reweight samples
        device = params["training"]["device"]
        losses = []
        for batch in tqdm.tqdm(test_loader, desc="Reweighing"):
            X, y = batch[0], batch[1]
            X = X.to(device)
            y = y.to(device)
            loss = model(X, y)
            losses.append(loss.item())
        weights = np.array(losses)
        print("Overall train loss:", np.mean(weights))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", type=str, choices=["pretraining", "finetuning"])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    params = {
        "seed": 1997,
        # "features": ["Intens.", "FWHM", "Res."],
        "features": ["Intens."],

        "training": {
            "num_iters": 60000,
            "num_epochs": 7,
            "lr": 1e-3,
            "max_lr": 1e-2,
            "device": "cuda:0",
            "batch_size": 128
        },

        "augmentation": {
            "num_augs_per_class": 10,
            "blocks": {
                "spec_Intens.": [
                    aug.SegmentsCombination(
                        num_segments=100, p=1.0),
                    aug.PeakVariation(
                        fraction_of_peaks=0.8, std_factor=1.2, p=0.9),
                    aug.NoisePerturbation(
                        noise_std=0.07, p=0.7),
                    aug.SpectrumShift(
                        shift_std=0.02, p=0.95),
                    aug.PeakElimination(
                        min_peaks_frac=0.095, max_peaks_frac=0.3, inpute_val=0.0, p=0.7),       # Nullifies random peaks
                    aug.PeakElimination(
                        min_peaks_frac=0.001, max_peaks_frac=0.006, inpute_val="random", p=0.5), # Inputes random values to small amount of peaks
                    aug.Normalize(
                        p=1.0),
                ],
                "spec_FWHM": [
                    aug.SegmentsCombination(num_segments=100, p=1.0),
                    aug.SpectrumShift(shift_std=0.012, p=0.75),
                ],
                "spec_Res.": [
                    aug.SegmentsCombination(num_segments=100, p=1.0),
                ]
            },
        },
    }

    args = parse_args()

    if args.stage == "pretraining":
        # Pretraining on ZenodoV3 data.
        # After the training process:
        # 1. Choose best checkpoint from 'models' directory and replace it to 'nn/pretrained'.
        # 2. Rename it to 'pretrained_on_zenodo.pth'.
        run_pretraining(params)

    elif args.stage == "finetuning":
        # Finetuning on target AI4Bio dataset.
        # Uses 'pretrained_on_zenodo.pth' as starting weights state.
        run_finetuning(params)

    
    # Did not work
    # run_mixup_training(params)
    # run_curricular_training(params)