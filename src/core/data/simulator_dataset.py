from torch.utils.data import Dataset
from clip.simple_tokenizer import SimpleTokenizer
from PIL import Image
import numpy as np
import pandas as pd
import torch


class SimulatorDataset(Dataset):

    def __init__(self, object_classes, images, annotations, preprocess, dataset_type):
        super(Dataset, self).__init__()
        self.object_classes = object_classes
        self.images = images
        self.annotations = annotations
        self.preprocess = preprocess
        self.dataset_type = dataset_type
        self.tokenizer = SimpleTokenizer()
        pass

    def tokenise(self, object_classes, context_length: int = 77):
        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.tokenizer.encode(object_class) + [eot_token] for object_class in object_classes]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            n = min(len(tokens), context_length)
            result[i, :n] = torch.tensor(tokens)[:n]
            if len(tokens) > context_length:
                result[i, -1] = tokens[-1]
        return result

    def get_annotaion_label(self, annotation_file):
        texts = []
        targets = []
        data = pd.read_csv(annotation_file, names=['label', 'size','center', 'cornerPoints'])
        data_classes = data['label']
        labels = np.arange(0, len(self.object_classes))
        for object_class in data_classes:
            if object_class.lower() in self.object_classes:
                texts = np.append(texts, object_class.lower())
                index = self.object_classes.index(object_class.lower())
                # label[index] = 1
                targets.append(int(labels[index]))

        return targets, texts

    def __len__(self):
        return len(self.images)
        pass

    def __getitem__(self, idx):
        image = self.preprocess(Image.open('/export2/scratch/cv_proj_team1/Attention_and_Move/output/dataset/' + self.dataset_type + '/images/' + self.images[idx]))
        targets, texts = self.get_annotaion_label('/export2/scratch/cv_proj_team1/Attention_and_Move/output/dataset/' + self.dataset_type + '/annotations/' + self.annotations[idx])
        tokenized_texts = self.tokenise(texts)

        return image, targets, tokenized_texts[0]
