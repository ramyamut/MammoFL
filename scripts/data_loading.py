import imageio
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from skimage import io
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import glob
import os
import random
from sklearn.model_selection import train_test_split

from openfl.federated import FederatedDataSet
from openfl.federated import PyTorchDataLoader

DATA_PATH = 'deep_libra/'

def process_data_3a(csv_path, data_path):
    data_df = pd.read_csv(csv_path)
    labels_l = data_df['cumulus_proc_lcc_pd']
    labels_r = data_df['cumulus_proc_rcc_pd']
    subs = data_df['encrypted']

    lcc = glob.glob(f'{data_path}/*_L_CC.png')
    rcc = glob.glob(f'{data_path}/*_R_CC.png')
    lcc = [i.split('/')[-1] for i in lcc]
    lcc = [i.split('_')[0] for i in lcc]
    rcc = [i.split('/')[-1] for i in rcc]
    rcc = [i.split('_')[0] for i in rcc]

    labels = []
    paths = []
    for i in range(len(subs)):
        if subs[i] in lcc:
            labels.append(labels_l[i])
            paths.append(f'{data_path}/{subs[i]}_L_CC.png')
        if subs[i] in rcc:
            labels.append(labels_r[i])
            paths.append(f'{data_path}/{subs[i]}_R_CC.png')
    
    return paths, labels

def process_data_3b(csv_path, data_path):
    data_df = pd.read_csv(csv_path)
    paths = data_df['File Analyzed'].apply(lambda x: f'{data_path}/{x}.png').tolist()
    labels = data_df['BreastDensity(%)'].tolist()
    
    return paths, labels

def process_data(csv_path, data_path, ds):
    if ds == 'a':
        return process_data_3a(csv_path, data_path)
    return process_data_3b(csv_path, data_path)

def load_paths(IMG_DIR, MASK_DIR):
    mask_paths = glob.glob(f'{MASK_DIR}/*.png')
    img_paths = []
    for path in mask_paths:
        sub = path.split('/')[-1]
        sub = sub.split('.png')[0]
        img_path = os.path.join(IMG_DIR, f'{sub}.png')
        img_paths.append(img_path)
    return img_paths, mask_paths

class DensityDataset(Dataset):
    def __init__(self, data_path, collaborator_count, collaborator_num, is_validation):
        super().__init__()
        self.data_path = data_path
        self.collaborator_count = collaborator_count
        if collaborator_num == 0:
            ds = 'a'
        else:
            ds = 'b'
        self.path_to_data = f'/cbica/home/muthukrr/comp_space/senior_design/deep_libra/ds3{ds}/output/final_images/image'
        self.csv_path = f'/cbica/home/muthukrr/comp_space/senior_design/data/train_3{ds}.csv'
        self.paths, self.labels = process_data(self.csv_path, self.path_to_data, ds)
        X_train, X_test, y_train, y_test = train_test_split(self.paths, self.labels, test_size=0.2, random_state=42)
        self.X, self.y = (X_test, y_test) if is_validation else (X_train, y_train)
        if is_validation:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), transforms.RandomVerticalFlip(), transforms.RandomHorizontalFlip(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    
    def __getitem__(self, index):
        img_path = self.X[index]
        lab = self.y[index]
        img = imageio.imread(img_path) / 255
        img_transformed = self.transform(img)
        return img_transformed.float(), torch.tensor([lab]).float()

    def __len__(self):
        return len(self.y)

class SegmentationDataset(Dataset):
    def __init__(self, img_paths, mask_paths, collaborator_count, collaborator_num, is_validation):
        super().__init__()
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.collaborator_count = collaborator_count
        self.collaborator_num = collaborator_num
        self.is_validation = is_validation
        self.img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.mask_transform =  transforms.Compose([transforms.ToTensor()])
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        img = imageio.imread(img_path) / 255
        mask = imageio.imread(mask_path) / 255

        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
            img = np.concatenate([img]*3, axis=2)
        img_transformed = self.img_transform(img)
        mask_transformed = self.mask_transform(mask)

        if not self.is_validation:
            if random.random() > 0.5:
                img_transformed = TF.hflip(img_transformed)
                mask_transformed = TF.hflip(mask_transformed)

            if random.random() > 0.5:
                img_transformed = TF.vflip(img_transformed)
                mask_transformed = TF.vflip(mask_transformed)
        return img_transformed.float(), mask_transformed

    def __len__(self):
        return len(self.mask_paths)

class FederatedSegmentationDataset(FederatedDataSet):
    def __init__(self, img_dirs, mask_dirs, collaborator_count=2, collaborator_num=0, batch_size=1, **kwargs):
        """Instantiate the data object
        Args:
            collaborator_count: total number of collaborators
            collaborator_num: number of current collaborator
            batch_size:  the batch size of the data loader
            **kwargs: additional arguments, passed to super init
        """
        super().__init__([], [], [], [], batch_size, num_classes=1, **kwargs)

        self.collaborator_num = int(collaborator_num)

        self.collaborator_count = collaborator_count

        self.batch_size = batch_size

        self.img_dirs = img_dirs
        self.mask_dirs = mask_dirs

        img_paths, mask_paths = load_paths(self.img_dirs[self.collaborator_num], self.mask_dirs[self.collaborator_num])
        X_train, X_test, y_train, y_test = train_test_split(img_paths, mask_paths, test_size=0.2, random_state=42)

        self.training_set = SegmentationDataset(
            X_train, y_train, collaborator_count, collaborator_num, is_validation=False
        )
        self.valid_set = SegmentationDataset(
            X_test, y_test, collaborator_count, collaborator_num, is_validation=True
        )

        self.train_loader = self.get_train_loader()
        self.val_loader = self.get_valid_loader()

    def get_valid_loader(self, num_batches=None):
        return DataLoader(self.valid_set, batch_size=self.batch_size)

    def get_train_loader(self, num_batches=None):
        return DataLoader(
            self.training_set, batch_size=self.batch_size, shuffle=True
        )

    def get_train_data_size(self):
        return len(self.training_set)

    def get_valid_data_size(self):
        return len(self.valid_set)

    def get_feature_shape(self):
        return self.valid_set[0][0].shape

    def split(self, collaborator_count, shuffle=True, equally=True):
        return [
            FederatedSegmentationDataset(self.img_dirs, self.mask_dirs,
                           collaborator_count, collaborator_num, self.batch_size)
            for collaborator_num in range(collaborator_count)
        ]

class DensityFederatedDataset(FederatedDataSet):
    def __init__(self, collaborator_count=2, collaborator_num=0, batch_size=1, **kwargs):
        """Instantiate the data object
        Args:
            collaborator_count: total number of collaborators
            collaborator_num: number of current collaborator
            batch_size:  the batch size of the data loader
            **kwargs: additional arguments, passed to super init
        """
        super().__init__([], [], [], [], batch_size, num_classes=2, **kwargs)

        self.collaborator_num = int(collaborator_num)

        self.batch_size = batch_size

        self.training_set = DensityDataset(
            DATA_PATH, collaborator_count, collaborator_num, is_validation=False
        )
        self.valid_set = DensityDataset(
            DATA_PATH, collaborator_count, collaborator_num, is_validation=True
        )

        self.train_loader = self.get_train_loader()
        self.val_loader = self.get_valid_loader()

    def get_valid_loader(self, num_batches=None):
        return DataLoader(self.valid_set, batch_size=self.batch_size)

    def get_train_loader(self, num_batches=None):
        return DataLoader(
            self.training_set, batch_size=self.batch_size, shuffle=True
        )

    def get_train_data_size(self):
        return len(self.training_set)

    def get_valid_data_size(self):
        return len(self.valid_set)

    def get_feature_shape(self):
        return self.valid_set[0][0].shape

    def split(self, collaborator_count, shuffle=True, equally=True):
        return [
            DensityFederatedDataset(collaborator_count,
                           collaborator_num, self.batch_size)
            for collaborator_num in range(collaborator_count)
        ]

class PyTorchSegmentationDataLoader(PyTorchDataLoader):

    def __init__(self, data_path, batch_size, img_dirs, mask_dirs, **kwargs):
        """Instantiate the data object.

        Args:
            data_path: The file path to the data
            batch_size: The batch size of the data loader
            **kwargs: Additional arguments, passed to super
             init and load_mnist_shard
        """
        super().__init__(batch_size, **kwargs)

        img_paths, mask_paths = load_paths(img_dirs[int(data_path)], mask_dirs[int(data_path)])
        X_train, X_test, y_train, y_test = train_test_split(img_paths, mask_paths, test_size=0.2, random_state=42)

        self.training_set = SegmentationDataset(
            X_train, y_train, 2, int(data_path), is_validation=False
        )
        self.valid_set = SegmentationDataset(
            X_test, y_test, 2, int(data_path), is_validation=True
        )

        self.train_loader = self.get_train_loader()
        self.val_loader = self.get_valid_loader()
        self.batch_size = batch_size

    def get_valid_loader(self, num_batches=None):
        """Return validation dataloader."""
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def get_train_loader(self, num_batches=None):
        """Return train dataloader."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def get_train_data_size(self):
        """Return size of train dataset."""
        return len(self.train_dataset)

    def get_valid_data_size(self):
        """Return size of validation dataset."""
        return len(self.valid_dataset)

    def get_feature_shape(self):
        """Return data shape."""
        return self.valid_dataset[0][0].shape

class PyTorchDensityDataLoader(PyTorchDataLoader):

    def __init__(self, data_path, batch_size, **kwargs):
        """Instantiate the data object.

        Args:
            data_path: The file path to the data
            batch_size: The batch size of the data loader
            **kwargs: Additional arguments, passed to super
             init and load_mnist_shard
        """
        super().__init__(batch_size, **kwargs)

        self.valid_dataset = DensityDataset(DATA_PATH, collaborator_count=2, collaborator_num=int(data_path), is_validation=True)
        self.train_dataset = DensityDataset(DATA_PATH, collaborator_count=2, collaborator_num=int(data_path), is_validation=False)
        self.train_loader = self.get_train_loader()
        self.val_loader = self.get_valid_loader()
        self.batch_size = batch_size

    def get_valid_loader(self, num_batches=None):
        """Return validation dataloader."""
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def get_train_loader(self, num_batches=None):
        """Return train dataloader."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def get_train_data_size(self):
        """Return size of train dataset."""
        return len(self.train_dataset)

    def get_valid_data_size(self):
        """Return size of validation dataset."""
        return len(self.valid_dataset)

    def get_feature_shape(self):
        """Return data shape."""
        return self.valid_dataset[0][0].shape