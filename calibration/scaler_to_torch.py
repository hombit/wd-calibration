import torch
from sklearn.preprocessing import StandardScaler
from torch import nn as nn


def scaler_to_input_layer(scaler: StandardScaler, dtype=torch.float32) -> nn.Linear:
    layer = nn.Linear(scaler.n_features_in_, scaler.n_features_in_, bias=True)
    layer.requires_grad = False
    layer.weight = nn.Parameter(torch.diag(torch.tensor(1.0 / scaler.scale_, dtype=dtype)))
    layer.bias = nn.Parameter(torch.tensor(-scaler.mean_ / scaler.scale_, dtype=dtype))
    return layer


def scaler_to_output_layer(scaler: StandardScaler, dtype=torch.float32) -> nn.Linear:
    layer = nn.Linear(1, 1, bias=True)
    layer.requires_grad = False
    layer.weight = nn.Parameter(torch.diag(torch.tensor(scaler.scale_, dtype=dtype)))
    layer.bias = nn.Parameter(torch.tensor(scaler.mean_, dtype=dtype))
    return layer
