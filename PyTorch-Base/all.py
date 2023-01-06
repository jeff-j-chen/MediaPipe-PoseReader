from operator import itemgetter

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import MultiStepLR

from trainer import Trainer, hooks, configuration
from trainer.utils import setup_system, patch_configs
from trainer.metrics import AccuracyEstimator
from trainer.tensorboard_visualizer import TensorBoardVisualizer

import os
import pandas as pd
from skimage import io
from torch.utils.data import Dataset, DataLoader


class FaceDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        label = self.annotations.iloc[index, 1]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data(batch_size, data_root, num_workers):

    universal_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=5, translate=(0.15, 0.15), scale=(0.9, 1.1), shear=5, resample=False, fillcolor=0),
        transforms.Normalize((0.4450, ), (0.3000, )),
    ])

    dataset = FaceDataset(csv_file='./face.csv', root_dir='./data/', transform=universal_transforms)

    train_size = int(0.75 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, test_loader


class FaceDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self._body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=3),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
        )
        self._head = nn.Sequential(
            nn.Linear(in_features=576, out_features=576),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(576),

            nn.Linear(in_features=576, out_features=576),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(576),

            nn.Linear(in_features=576, out_features=576),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(576),

            nn.Linear(in_features=576, out_features=576),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(576),

            nn.Linear(in_features=576, out_features=288),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(288),

            nn.Linear(in_features=288, out_features=144),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(144),

            nn.Linear(in_features=144, out_features=72),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(72),

            nn.Linear(in_features=72, out_features=4),
        )

    def forward(self, x):
        x = self._body(x)
        x = x.view(x.size()[0], -1)
        x = self._head(x)
        return x


class Experiment:
    def __init__(
        self,
        system_config: configuration.SystemConfig = configuration.SystemConfig(),
        dataset_config: configuration.DatasetConfig = configuration.DatasetConfig(),
        dataloader_config: configuration.DataloaderConfig = configuration.DataloaderConfig(),
        optimizer_config: configuration.OptimizerConfig = configuration.OptimizerConfig()
    ):
        self.loader_train, self.loader_test = get_data(
            batch_size=dataloader_config.batch_size,
            num_workers=dataloader_config.num_workers,
            data_root=dataset_config.root_dir
        )

        setup_system(system_config)

        self.model = FaceDetector()
        self.loss_fn = nn.CrossEntropyLoss()
        self.metric_fn = AccuracyEstimator(topk=(1, ))
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
            momentum=optimizer_config.momentum
        )
        self.lr_scheduler = MultiStepLR(
            self.optimizer, milestones=optimizer_config.lr_step_milestones, gamma=optimizer_config.lr_gamma
        )
        self.visualizer = TensorBoardVisualizer()

    def run(self, trainer_config: configuration.TrainerConfig) -> dict:

        device = torch.device(trainer_config.device)
        self.model = self.model.to(device)
        self.loss_fn = self.loss_fn.to(device)

        model_trainer = Trainer(
            model=self.model,
            loader_train=self.loader_train,
            loader_test=self.loader_test,
            loss_fn=self.loss_fn,
            metric_fn=self.metric_fn,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            device=device,
            data_getter=itemgetter(0),
            target_getter=itemgetter(1),
            stage_progress=trainer_config.progress_bar,
            get_key_metric=itemgetter("top1"),
            visualizer=self.visualizer,
            model_saving_frequency=trainer_config.model_saving_frequency,
            save_dir=trainer_config.model_dir
        )

        # model_trainer.register_hook("end_epoch", hooks.end_epoch_hook_classification)
        self.metrics = model_trainer.fit(trainer_config.epoch_num)
        return self.metrics


# dataloader_config, trainer_config = patch_configs()
# dataset_config = configuration.DatasetConfig(root_dir="data")
# experiment = Experiment(dataset_config=dataset_config, dataloader_config=dataloader_config)
# results = experiment.run(trainer_config)


import cv2
import numpy as np
model = FaceDetector()
checkpoint = torch.load("./model_best.pt")
state_dict = model.state_dict()
for k1, k2 in zip(state_dict.keys(), checkpoint.keys()):
    state_dict[k1] = checkpoint[k2]
model.load_state_dict(state_dict)
model.eval()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(count_parameters(model))