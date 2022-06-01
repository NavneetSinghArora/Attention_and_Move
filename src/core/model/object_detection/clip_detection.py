
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from src.core.data.simulator_dataset import SimulatorDataset
from os.path import join
import pandas as pd
import torch
import clip
import os
import numpy as np

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class ClipObjectDetection:
    def __init__(self, global_properties, simulator_properties):
        self.global_properties = global_properties
        self.simulator_properties = simulator_properties
        torch.cuda.set_device(0)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        self.training_data_path = ''
        self.validation_data_path = ''
        self.testing_data_path = ''
        self.learning_rate = 1e-2
        self.momentum = 0.2
        self.criterion = CrossEntropyLoss()

    def get_optimiser(self, optimiser='SGD'):
        if optimiser == 'SGD':
            return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        if optimiser == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def get_scheduler(self, optimiser, number_of_epochs, dataloader):
        return torch.optim.lr_scheduler.OneCycleLR(optimiser, 1e-2, total_steps=number_of_epochs * (2 * len(dataloader) - 1), base_momentum=0.0, max_momentum=0.5, pct_start=0.1, div_factor=1e2, final_div_factor=1e4)

    def get_dataloader(self):
        object_classes = (self.simulator_properties['object_classes']).split(',')

        training_images = os.listdir(
            '/export2/scratch/cv_proj_team1/Attention_and_Move/output/dataset/training/images/')
        training_annotations = os.listdir(
            '/export2/scratch/cv_proj_team1/Attention_and_Move/output/dataset/training/annotations/')
        training_dataset = SimulatorDataset(object_classes, training_images, training_annotations, self.preprocess, 'training')
        training_dataloader = DataLoader(training_dataset, batch_size=16, shuffle=True)

        validation_images = os.listdir(
            '/export2/scratch/cv_proj_team1/Attention_and_Move/output/dataset/validation/images/')
        validation_annotations = os.listdir(
            '/export2/scratch/cv_proj_team1/Attention_and_Move/output/dataset/validation/annotations/')
        validation_dataset = SimulatorDataset(object_classes, validation_images, validation_annotations, self.preprocess, 'validation')
        validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

        testing_images = os.listdir(
            '/export2/scratch/cv_proj_team1/Attention_and_Move/output/dataset/testing/images/')
        testing_annotations = os.listdir(
            '/export2/scratch/cv_proj_team1/Attention_and_Move/output/dataset/testing/annotations/')
        testing_dataset = SimulatorDataset(object_classes, testing_images, testing_annotations, self.preprocess, 'testing')
        testing_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=True)

        return training_dataloader

    def train(self):
        training_dataloader = self.get_dataloader()
        number_of_epochs = 1
        optimiser = self.get_optimiser()
        scheduler = self.get_scheduler(optimiser, number_of_epochs, training_dataloader)
        for epoch in range(1):
            with tqdm(total=len(training_dataloader) - 1) as bar:
                # loss_mean = RollingMean()
                for images, targets, texts in training_dataloader:
                    images_features = self.model.encode_image(images.to(self.device))
                    for i in range(len(targets)):
                        target = targets[i].reshape(1, 1).to(self.device)
                        texts_features = self.model.encode_text(texts[i].reshape(1,77).to(self.device))

                        optimiser.zero_grad()

                        # Join train and test features
                        features = torch.hstack([images_features, texts_features])

                        # L2-normalize features
                        features = features / features.norm(2, dim=1, keepdim=True)

                        # Apply Cross Entropy Loss SemiHardLoss
                        loss = self.criterion(features, np.squeeze(target).reshape(1))

                        loss.backward()
                    optimiser.step()
                    scheduler.step()

                    # Update metric and progress bar
                    # loss_mean.update(loss.item())
                    bar.update()
                    bar.set_description('{:.4f}'.format(loss.result()))


#
#
# class RollingMean():
#     def __init__(self):
#         self.n = 0
#         self.mean = 0
#
#     def update(self, value):
#         self.mean = (self.mean * self.n + value) / (self.n + 1)
#         self.n += 1
#
#     def result(self):
#         return self.mean
#
#
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, 1e-2, total_steps=n_epochs * (2*len(dltrain)-1),
#                                                base_momentum=0.0, max_momentum=0.5, pct_start=0.1, div_factor=1e2, final_div_factor=1e4)
#
# for epoch in range(n_epochs):
#     with tqdm(total=2 * len(dltrain) - 1) as bar:
#         loss_mean = RollingMean()
#         for images, texts, targets in dltrain:
#             targets = targets.to(device)
#
#             # Generate train and text features
#             # images_features = model.encode_image(images.to(device))
#             # texts_features = model.encode_text(texts.to(device))
#             #
#             # optim.zero_grad()
#             #
#             # # Join train and test features
#             # features = torch.hstack([images_features, texts_features])
#             #
#             # # L2-normalize features
#             # features = features / features.norm(2, dim=1, keepdim=True)
#             #
#             # # Apply Triplet SemiHardLoss
#             # loss = criterion(features, targets)
#             #
#             # loss.backward()
#             # optim.step()
#             # scheduler.step()
#             #
#             # # Update metric and progress bar
#             # loss_mean.update(loss.item())
#             # bar.update()
#             # bar.set_description('{:.4f}'.format(loss_mean.result()))