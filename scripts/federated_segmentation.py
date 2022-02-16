import openfl.native as fx
import imageio
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from data_loading import FederatedSegmentationDataset

from openfl.federated import FederatedModel
from utils import dice_loss
from models import FederatedUNet

#fx.init(col_names=['one', 'two'])
fx.setup_logging(level='INFO', log_file=None)

batch_size = 16
epochs = 30

IMG_DIR_DS1 = sys.argv[1]
MASK_DIR_DS1 = sys.argv[2]
IMG_DIR_DS2 = sys.argv[3]
MASK_DIR_DS2 = sys.argv[4]
IMG_DIRS = [IMG_DIR_DS1, IMG_DIR_DS2]
MASK_DIRS = [MASK_DIR_DS1, MASK_DIR_DS2]
SAVE_DIR = sys.argv[5]

def optimizer(x): return optim.Adam(x, lr=1e-4)

fl_data = FederatedSegmentationDataset(IMG_DIRS, MASK_DIRS, batch_size=batch_size)

fl_model = FederatedModel(build_model=FederatedUNet, optimizer=optimizer, loss_fn=dice_loss, data_loader=fl_data)
collaborator_models = fl_model.setup(num_collaborators=2)
collaborators = {'one': collaborator_models[0], 'two': collaborator_models[1]}

print('Federated Learning Plan:')
print(fx.get_plan())

final_fl_model = fx.run_experiment(collaborators, override_config={'aggregator.settings.rounds_to_train': epochs})
final_fl_model.save_native(f'{SAVE_DIR}/final_aggregated_model.pth')
