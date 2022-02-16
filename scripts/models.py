import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from typing import List
from torch import Tensor, Type
import torchvision
from utils import mse_loss, mae_loss, dice, dice_loss
from openfl.utilities import TensorKey
import segmentation_models_pytorch as smp
import numpy as np

class WideResNet50Dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.wide_resnet50_2(pretrained=True)
        self.backbone.fc = nn.Linear(2048, 1)
    
    def forward(self, x):
        return self.backbone(x)

class FederatedUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation="sigmoid")
    
    def forward(self, x):
        return self.backbone(x)
    
    def validate(self, col_name, round_num, input_tensor_dict, use_tqdm=False, **kwargs):
        """ Validate. Redifine function from PyTorchTaskRunner, to use our validation"""
        self.rebuild_model(round_num, input_tensor_dict, validation=True)
        self.eval()
        self.to(self.device)
        total_loss = 0
        total_dice_coef = 0
        total_samples = 0

        loader = self.data_loader.get_valid_loader()

        with torch.no_grad():
            for data, target in loader:
                samples = target.shape[0]
                total_samples += samples
                data, target = (
                    torch.tensor(data).to(self.device),
                    torch.tensor(target).to(self.device),
                )
                output = self(data)
                coef = dice(output, target)
                loss = dice_loss(output, target)
                total_loss += samples*loss.cpu().item()
                total_dice_coef += samples*coef.cpu().item()
        total_loss = total_loss / total_samples
        total_dice_coef = total_dice_coef / total_samples
        origin = col_name
        suffix = "validate"
        if kwargs["apply"] == "local":
            suffix += "_local"
        else:
            suffix += "_agg"
        tags = ("metric", suffix)
        output_tensor_dict = {
            TensorKey("dice_loss", origin, round_num, True, tags): np.array(total_loss),
            TensorKey("dice_coef", origin, round_num, True, tags): np.array(total_dice_coef)
        }
        return output_tensor_dict, {}

class FederatedWideResNet50Dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.wide_resnet50_2(pretrained=True)
        self.backbone.fc = nn.Linear(2048, 1)
    
    def forward(self, x):
        return self.backbone(x)
    
    def validate(self, col_name, round_num, input_tensor_dict, use_tqdm=False, **kwargs):
        """ Validate. Redifine function from PyTorchTaskRunner, to use our validation"""
        self.rebuild_model(round_num, input_tensor_dict, validation=True)
        self.eval()
        self.to(self.device)
        total_mse = 0
        total_mae = 0
        total_samples = 0

        loader = self.data_loader.get_valid_loader()

        with torch.no_grad():
            for data, target in loader:
                samples = target.shape[0]
                total_samples += samples
                data, target = (
                    torch.tensor(data).to(self.device),
                    torch.tensor(target).to(self.device),
                )
                output = self(data)
                val = mse_loss(output, target)
                mae = mae_loss(output, target)
                total_mse += samples*val.cpu().item()
                total_mae += samples*mae.cpu().item()
        total_mse = total_mse / total_samples
        total_mae = total_mae / total_samples
        total_rmse = np.sqrt(total_mse)
        origin = col_name
        suffix = "validate"
        if kwargs["apply"] == "local":
            suffix += "_local"
        else:
            suffix += "_agg"
        tags = ("metric", suffix)
        output_tensor_dict = {
            TensorKey("mse", origin, round_num, True, tags): np.array(total_mse),
            TensorKey("rmse", origin, round_num, True, tags): np.array(total_rmse),
            TensorKey("mae", origin, round_num, True, tags): np.array(total_mae)
        }
        return output_tensor_dict, {}
