# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import torchvision
import segmentation_models_pytorch as smp

from openfl.federated import PyTorchTaskRunner
from openfl.utilities import TensorKey
from utils import mse_loss, mae_loss, dice, dice_loss

class FederatedDensityNet(PyTorchTaskRunner):

    def __init__(self, device='cpu', **kwargs):
        super().__init__(device=device, **kwargs)
        self.init_network(device=self.device, **kwargs)
        self._init_optimizer()
        self.initialize_tensorkeys_for_functions()
        self.loss_fn = mse_loss

    def _init_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def init_network(self,device,**kwargs):
        self.backbone = torchvision.models.wide_resnet50_2(pretrained=True)
        self.backbone.fc = nn.Linear(2048, 1)
        self.to(device)

    def forward(self, x):
        return self.backbone(x)

    def validate(self, col_name, round_num, input_tensor_dict, use_tqdm=True, **kwargs):
        self.rebuild_model(round_num, input_tensor_dict, validation=True)
        self.eval()
        self.to(self.device)
        total_mse = 0
        total_mae = 0
        total_samples = 0

        loader = self.data_loader.get_valid_loader()
        if use_tqdm:
            loader = tqdm.tqdm(loader, desc='validate')

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

    def reset_opt_vars(self):
        self._init_optimizer()

class FederatedUNet(PyTorchTaskRunner):

    def __init__(self, device='cpu', **kwargs):
        super().__init__(device=device, **kwargs)
        self.init_network(device=self.device, **kwargs)
        self._init_optimizer()
        self.initialize_tensorkeys_for_functions()
        self.loss_fn = dice_loss

    def _init_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def init_network(self,device,**kwargs):
        self.backbone = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation="sigmoid")
        self.to(device)

    def forward(self, x):
        return self.backbone(x)

    def validate(self, col_name, round_num, input_tensor_dict, use_tqdm=True, **kwargs):
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

    def reset_opt_vars(self):
        self._init_optimizer()
