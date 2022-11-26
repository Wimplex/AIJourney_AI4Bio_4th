from typing import Dict, Any

import torch
import torch.nn as nn

from nn.nn_blocks.base import BaseModel
from nn.nn_blocks.heads import LinearHead, AMSoftmaxHead
from nn.nn_blocks.encoders import ResNet1D 


def LinearHeadModel(
    input_dim: int = 1800,
    in_channels: int = 1, 
    num_classes: int = 26, 
    weights_path: str = None,
    freeze_encoder: bool = True,
) -> nn.Module:

    encoder = ResNet1D(
        hidden_sizes=[100] * 6,
        num_blocks=[2] * 6,
        input_dim=input_dim,
        in_channels=in_channels,
        n_classes=num_classes
    )
    if weights_path:
        ignoring = ["linear.weight", "linear.bias"]
        if in_channels != 1:
            ignoring += [
                "conv1.weight", 
                "bn1.weight", 
                "bn1.bias", 
                "bn1.running_mean", 
                "bn1.running_var"
            ] 
        encoder.load_state_dict_part(
            torch.load(weights_path, map_location="cpu"), ignoring)
    if freeze_encoder:
        for name, param in encoder.named_parameters():
            if name.startswith("conv1") or name.startswith("bn1"): continue
            param.requires_grad = False

    head = LinearHead(encoder.get_encoding_size(in_channels), num_classes)
    model = BaseModel(encoder, head)
    return model


def AMSotmaxHeadModel(
    input_dim: int = 1800, 
    in_channels: int = 1,
    num_classes: int = 26, 
    weights_path: str = None,
    head_params: Dict[str, Any] = {"emb_size": 100, "s": 30, "m": 0.4},
    freeze_encoder: bool = True,
) -> nn.Module:

    encoder = ResNet1D(
        hidden_sizes=[100] * 6,
        num_blocks=[2] * 6,
        input_dim=input_dim,
        in_channels=in_channels,
        n_classes=num_classes
    )

    if weights_path:
        ignoring = ["linear.weight", "linear.bias"]
        if in_channels != 1:
            ignoring += [
                "conv1.weight", 
                "bn1.weight", 
                "bn1.bias", 
                "bn1.running_mean", 
                "bn1.running_var"
            ]
        
        state_dict = torch.load(weights_path, map_location="cpu")
        if "model" in list(state_dict.keys()):
            state_dict = state_dict["model"]
        encoder.load_state_dict_part(state_dict, ignoring)

    if freeze_encoder:
        for name, param in encoder.named_parameters():
            if in_channels == 1 and (name.startswith("conv1") or name.startswith("bn1")): 
                continue
            param.requires_grad = False

    head = AMSoftmaxHead(encoder.get_encoding_size(in_channels), num_classes, **head_params)
    model = BaseModel(encoder, head)
    return model
