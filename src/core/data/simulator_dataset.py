from traceback import print_tb
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import torch


class SimulatorDataset(Dataset):

    def __init__(self,  df, preprocess, dataset_type):
        super(Dataset, self).__init__()
        self.df = df
        self.preprocess = preprocess
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.preprocess(Image.open('/export2/scratch/cv_proj_team1/Attention_and_Move/output/dataset/' + self.dataset_type + '/images/' + self.df['images'][idx]))
        target = self.df['targets'][idx]
        text = self.df['texts'][idx]
        return image, target, text
