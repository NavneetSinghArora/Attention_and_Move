from tqdm.auto import tqdm
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from src.core.data.simulator_dataset import SimulatorDataset
from clip.simple_tokenizer import SimpleTokenizer
from os.path import join
# from src.core.services.clip import custom_clip_accuracy
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
        self.object_classes = None
        self.best_validation_loss = None
        self.scheduler = None
        self.optimiser = None
        self.global_properties = global_properties
        self.simulator_properties = simulator_properties
        torch.cuda.set_device(int(self.global_properties['gpu']))
        self.device = "cuda:"+self.global_properties['gpu'] if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        self.training_data_path = ''
        self.validation_data_path = ''
        self.testing_data_path = ''
        self.learning_rate = 1e-2
        self.momentum = 0.2
        self.criterion = CrossEntropyLoss()
        self.tokenizer = SimpleTokenizer()
        self.n = 0
        self.mean = 0
        self.batch_size = 32
        self.number_of_epochs = 20
        self.freeze_layers = self.global_properties['frozen']

    def reset_rolling_mean(self):
        self.n = 0
        self.mean = 0

    def update_mean(self, value):
        self.mean = (self.mean * self.n + value) / (self.n + 1)
        self.n += 1

    def mean_result(self):
        return self.mean

    def get_optimiser(self, optimiser='SGD'):
        if optimiser == 'SGD':
            return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        if optimiser == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def get_scheduler(self, optimiser, number_of_epochs, dataloader):
        return torch.optim.lr_scheduler.OneCycleLR(optimiser, 1e-2, total_steps=number_of_epochs * (2 * len(dataloader) - 1), base_momentum=0.0, max_momentum=0.5, pct_start=0.1, div_factor=1e2, final_div_factor=1e4)

    def tokenize(self, object_class, context_length: int = 77):
        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.tokenizer.encode(object_class) + [eot_token]]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            n = min(len(tokens), context_length)
            result[i, :n] = torch.tensor(tokens)[:n]
            if len(tokens) > context_length:
                result[i, -1] = tokens[-1]
        return result

    def get_annotation_label(self, annotation_file, object_classes):
        texts = []
        targets = []
        data = pd.read_csv(annotation_file, names=['label', 'size', 'center', 'cornerPoints'])
        data_classes = data['label']
        labels = np.arange(0, len(object_classes))
        ind = 0
        for object_class in data_classes:
            if object_class.lower() in object_classes:
                texts.append(object_class.lower())
                index = object_classes.index(object_class.lower())
                targets.append(int(labels[index]))
            ind += 1
        
        return targets, texts

    def create_dataframe(self, images, annotations, object_classes, dataset_type):
        print(f"Creating {dataset_type} DataFrame")
        df = pd.DataFrame()
        image_count = 0
        for i in range(20):
            if i % 6 == 0:
                image_count += 1
                if image_count % 1000 == 0:
                    print(f"Processing {dataset_type} Image : {image_count}")
                targets, texts = self.get_annotation_label('/export2/scratch/cv_proj_team1/Attention_and_Move/output/dataset/' + dataset_type + '/annotations/' + annotations[i], object_classes)
                for j in range(len(targets)):
                    df = df.append({'images': images[i], 'targets': targets[j], 'texts': self.tokenize(texts[j])}, ignore_index=True)

        return df

    def get_dataloader(self):
        print('Creating DataLoader')
        self.object_classes = (self.simulator_properties['object_classes']).split(',')

        training_images = sorted(os.listdir(
            '/export2/scratch/cv_proj_team1/Attention_and_Move/output/dataset/training/images/'))
        training_annotations = sorted(os.listdir(
            '/export2/scratch/cv_proj_team1/Attention_and_Move/output/dataset/training/annotations/'))
        training_df = self.create_dataframe(training_images, training_annotations, self.object_classes, 'training')

        training_dataset = SimulatorDataset(training_df, self.preprocess, 'training')
        training_dataloader = DataLoader(training_dataset, batch_size=self.batch_size, shuffle=True)

        validation_images = sorted(os.listdir(
            '/export2/scratch/cv_proj_team1/Attention_and_Move/output/dataset/validation/images/'))
        validation_annotations = sorted(os.listdir(
            '/export2/scratch/cv_proj_team1/Attention_and_Move/output/dataset/validation/annotations/'))
        validation_df = self.create_dataframe(validation_images, validation_annotations, self.object_classes, 'validation')
        validation_dataset = SimulatorDataset(validation_df, self.preprocess, 'validation')
        validation_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=True)

        testing_images = sorted(os.listdir(
            '/export2/scratch/cv_proj_team1/Attention_and_Move/output/dataset/testing/images/'))
        testing_annotations = sorted(os.listdir(
            '/export2/scratch/cv_proj_team1/Attention_and_Move/output/dataset/testing/annotations/'))
        testing_df = self.create_dataframe(testing_images, testing_annotations, self.object_classes, 'testing')
        testing_dataset = SimulatorDataset(testing_df, self.preprocess, 'testing')
        testing_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=True)

        return training_dataloader, validation_dataloader, testing_dataloader

    def train(self, training_dataloader):
        with tqdm(total=len(training_dataloader) - 1) as bar:
            self.reset_rolling_mean()
            self.model.train()
            for batch_images, batch_targets, batch_texts in training_dataloader:
                images_features = self.model.encode_image(batch_images.to(self.device))
                batch_targets = batch_targets.to(self.device)
                # text = torch.cat([self.tokenize({c}) for c in self.object_classes]).to(self.device)
                # texts_features = self.model.encode_text(text.to(self.device))
                texts_features = self.model.encode_text(batch_texts.reshape(batch_texts.shape[0], 77).to(self.device))

                if self.freeze_layers:
                    # Freeze all but last layer of first 6 blocks
                    for name, param in self.model.named_parameters():
                        split_names = name.split('.')
                        if len(split_names) < 2:
                            continue
                        elif (split_names[1] == 'transformer' and int(split_names[3]) <6) or (split_names[0] == 'transformer' and int(split_names[2]) <6):
                            if not 'fc' in name:
                                param.requires_grad = False

                self.optimiser.zero_grad()

                # Join train and test features
                features = torch.hstack([images_features, texts_features])

                # L2-normalize features
                features = features / features.norm(2, dim=1, keepdim=True)

                # Apply Cross Entropy Loss SemiHardLoss
                # print(torch.nn.Softmax(features).to(self.device))
                loss = self.criterion(features, batch_targets)
                loss.backward()
                self.optimiser.step()
                self.scheduler.step()

                # custom_clip_accuracy(self.model, images_features, texts_features, batch_targets)

                # Update metric and progress bar
                self.update_mean(loss.item())
                bar.update()
                bar.set_description('{:.4f}'.format(self.mean_result()))
        return self.mean_result()

    def validate(self, validation_dataloader):
        with tqdm(total=len(validation_dataloader) - 1) as bar:
            self.reset_rolling_mean()
            self.model.eval()
            for batch_images, batch_targets, batch_texts in validation_dataloader:

                with torch.no_grad():
                    images_features = self.model.encode_image(batch_images.to(self.device))
                    batch_targets = batch_targets.to(self.device)
                    texts_features = self.model.encode_text(batch_texts.reshape(batch_texts.shape[0], 77).to(self.device))

                # Join train and test features
                features = torch.hstack([images_features, texts_features])

                # L2-normalize features
                features = features / features.norm(2, dim=1, keepdim=True)

                # Apply Cross Entropy Loss SemiHardLoss
                loss = self.criterion(features, batch_targets)

                # custom_clip_accuracy(self.model, images_features, texts_features, batch_targets)

                # Update metric and progress bar
                self.update_mean(loss.item())
                bar.update()
                bar.set_description('{:.4f}'.format(self.mean_result()))
        return self.mean_result()

    def test(self):
        checkpoint = torch.load(join(self.global_properties['root_directory'], 'checkpoints/clip_model'
                                                                               '/best_checkpoint/unfrozen_clip_best.ckpt'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        training_dataloader, validation_dataloader, testing_dataloader = self.get_dataloader()
        for batch_images, batch_targets, batch_texts in testing_dataloader:
            with torch.no_grad():
                image_features = self.model.encode_image(batch_images.to(self.device))
                batch_targets = batch_targets.to(self.device)
                text = torch.cat([self.tokenize('television')]).to(self.device)
                text_features = self.model.encode_text(text.to(self.device))
                # text = torch.cat([self.tokenize({f'a {c}'}) for c in self.object_classes]).to(self.device)
                # text_features = self.model.encode_text(text.to(self.device))

            similarity = (100.0 * image_features @ text_features.T)
            print(similarity)
            similarity = similarity.softmax(dim=-1)
            print(similarity)
            values, indices = similarity[0].topk(1, sorted=True)
            target_found = False
            for value, index in zip(values, indices):
                # found = ''
                # found = ' - found ' + target_object + '!'
                # target_found = True
                print(f"{self.object_classes[index]:>16s}: {100 * value.item():.2f}%")

    def save_best_checkpoint(self, epoch, validation_loss):
        save_flag = False

        if self.best_validation_loss is None:
            self.best_validation_loss = validation_loss
            save_flag = True
        elif validation_loss < self.best_validation_loss:
            self.best_validation_loss = validation_loss
            save_flag = True

        if self.freeze_layers:
            initial = "frozen"
        else:
            initial = "unfrozen"

        if save_flag:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimiser.state_dict(),
                'loss': self.best_validation_loss,
            }, join(self.global_properties['root_directory'], 'checkpoints/clip_model/epoch_checkpoints/' + initial + '_clip_' + str(epoch) + '.ckpt'))
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimiser.state_dict(),
                'loss': self.best_validation_loss,
            }, join(self.global_properties['root_directory'],
                    'checkpoints/clip_model/best_checkpoint/' + initial + '_clip_best.ckpt'))

    def run(self):
        print('Starting the run')
        training_dataloader, validation_dataloader, testing_dataloader = self.get_dataloader()
        self.optimiser = self.get_optimiser()
        self.scheduler = self.get_scheduler(self.optimiser, self.number_of_epochs, training_dataloader)
        for epoch in range(50):
            self.train(training_dataloader)
            validation_loss = self.validate(validation_dataloader)
            self.save_best_checkpoint(epoch, validation_loss)

