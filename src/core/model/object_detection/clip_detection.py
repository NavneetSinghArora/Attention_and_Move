from tqdm.auto import tqdm
from torch.nn import CrossEntropyLoss, TripletMarginWithDistanceLoss
from torch.utils.data import Dataset, DataLoader
from src.core.data.simulator_dataset import SimulatorDataset
from clip.simple_tokenizer import SimpleTokenizer
from os.path import join
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import pandas as pd
import clip
import os
import numpy as np
import torch
import torchvision

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

writer = SummaryWriter()
tb = SummaryWriter()


def get_metric_results(similarity, batch_targets, mean_accuracy, mean_precision, mean_recall, f1_score):
    for i in range(len(similarity)):
        accuracy = 0
        vector = similarity[i]
        target_vector = batch_targets[i]

        true_positive = 0
        false_positive = 0
        false_negative = 0
        for j in range(len(vector)):
            if vector[j] == target_vector[j] or (vector[j] > 0 and target_vector[j] > 0):
                accuracy += 1
                true_positive += 1
            if vector[j] > 0 and target_vector[j] == 0:
                false_positive += 1
            if target_vector[j] > 0 and vector[j] == 0:
                false_negative += 1

        accuracy = (accuracy / len(vector)) * 100
        mean_accuracy = (mean_accuracy + accuracy) / 2

        precision = (true_positive / (true_positive + false_positive)) * 100
        mean_precision = (mean_precision + precision) / 2

        recall = (true_positive / (true_positive + false_negative)) * 100
        mean_recall = (mean_recall + recall) / 2

        f1_score = 2 * ((mean_precision * mean_recall) / (mean_precision + mean_recall))

    return mean_accuracy, mean_precision, mean_recall, f1_score


def get_object_found_percentage(similarity, object_classes):
    values, indices = similarity[0].topk(len(object_classes), sorted=True)
    for value, index in zip(values, indices):
        print(f"{object_classes[index]:>16s}: {100 * value.item():.2f}%")


class ClipObjectDetection:

    __instance = None
    __instance_created = False

    def __init__(self, global_properties, simulator_properties):
        if not self.__instance_created:
            self.object_classes = simulator_properties['object_classes']
            self.best_validation_loss = None
            self.scheduler = None
            self.optimiser = None
            self.global_properties = global_properties
            self.simulator_properties = simulator_properties
            torch.cuda.set_device(int(self.global_properties['gpu']))
            self.device = "cuda:" + self.global_properties['gpu'] if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
            self.training_data_path = ''
            self.validation_data_path = ''
            self.testing_data_path = ''
            self.learning_rate = 1e-2
            self.momentum = 0.2
            self.criterion = CrossEntropyLoss()
            # self.criterion = TripletMarginWithDistanceLoss()
            self.tokenizer = SimpleTokenizer()
            self.n = 0
            self.mean = 0
            self.batch_size = 32
            self.number_of_epochs = 30
            self.freeze_layers = self.global_properties['frozen']
            self.__instance_created = True
            self.text = None
            self.texts_features = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls)
        return cls.__instance

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
        if optimiser == 'ADAM':
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def get_scheduler(self, optimiser, number_of_epochs, dataloader):
        return torch.optim.lr_scheduler.OneCycleLR(optimiser, 1e-2,
                                                   total_steps=number_of_epochs * (2 * len(dataloader) - 1),
                                                   base_momentum=0.0, max_momentum=0.5, pct_start=0.1, div_factor=1e2,
                                                   final_div_factor=1e4)

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
        with tqdm(total=len(images) - 1) as bar:
            df = pd.DataFrame()
            image_count = 0
            for i in range(len(images)):
                # if i % 6 == 0:
                image_count += 1
                targets_encoding = np.zeros(len(self.object_classes))
                # if image_count % 1000 == 0:
                #     print(f"Processing {dataset_type} Image : {image_count}")
                targets, texts = self.get_annotation_label(
                    '/export2/scratch/cv_proj_team1/Attention_and_Move/output/dataset/' + dataset_type + '/annotations/' +
                    annotations[i], object_classes)
                text = torch.cat([self.tokenize(c) for c in self.object_classes])
                for j in range(len(targets)):
                    targets_encoding[targets[j]] = 1
                #     df = df.append({'images': images[i], 'targets': targets[j], 'texts': self.tokenize(texts[j])}, ignore_index=True)
                df = df.append({'images': images[i], 'targets': targets_encoding, 'texts': text}, ignore_index=True)

                bar.update()

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
        validation_df = self.create_dataframe(validation_images, validation_annotations, self.object_classes,
                                              'validation')
        validation_dataset = SimulatorDataset(validation_df, self.preprocess, 'validation')
        validation_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=True)

        testing_images = sorted(os.listdir(
            '/export2/scratch/cv_proj_team1/Attention_and_Move/output/dataset/testing/images/'))
        testing_annotations = sorted(os.listdir(
            '/export2/scratch/cv_proj_team1/Attention_and_Move/output/dataset/testing/annotations/'))
        testing_df = self.create_dataframe(testing_images, testing_annotations, self.object_classes, 'testing')
        testing_dataset = SimulatorDataset(testing_df, self.preprocess, 'testing')
        testing_dataloader = DataLoader(testing_dataset, batch_size=self.batch_size, shuffle=True)

        # return training_dataloader, validation_dataloader, testing_dataloader
        return testing_dataloader

    def train(self, training_dataloader, epoch):
        if self.global_properties['checkpoint'] is not None:
            checkpoint = torch.load(join(self.global_properties['root_directory'], 'checkpoints/clip_model'
                                                                                   '/best_checkpoint/' + self.global_properties['checkpoint']))
            self.model.load_state_dict(checkpoint['model_state_dict'])
        mean_accuracy = 100
        mean_recall = 100
        mean_precision = 100
        f1_score = 100

        images, labels, texts = next(iter(training_dataloader))
        grid = torchvision.utils.make_grid(images)
        tb.add_image("Training Images", grid)

        with tqdm(total=len(training_dataloader) - 1) as bar:
            self.reset_rolling_mean()
            self.model.train()
            index = 0
            for batch_images, batch_targets, batch_texts in training_dataloader:
                index += 1
                text_features = []
                images_features = self.model.encode_image(batch_images.to(self.device))
                batch_targets = batch_targets.to(self.device)
                text = torch.cat([self.tokenize(c) for c in self.object_classes]).to(self.device)
                texts_features = self.model.encode_text(text.to(self.device))
                for i in range(len(batch_images)):
                    text_features.append(texts_features)

                if self.freeze_layers:
                    # Freeze all but last layer of first 6 blocks
                    for name, param in self.model.named_parameters():
                        split_names = name.split('.')
                        if len(split_names) < 2:
                            continue
                        elif (split_names[1] == 'transformer' and int(split_names[3]) < 6) or (
                                split_names[0] == 'transformer' and int(split_names[2]) < 6):
                            if not 'fc' in name:
                                param.requires_grad = False

                self.optimiser.zero_grad()

                features = (images_features @ texts_features.T)
                similarity = (100.0 * images_features @ texts_features.T)
                similarity = similarity.softmax(dim=-1)

                # Apply Cross Entropy Loss SemiHardLoss
                loss = self.criterion(features, batch_targets.softmax(dim=1))
                loss.backward()

                # This is a function call to get the percentage values of object presence in the frame
                # get_object_found_percentage(similarity, self.object_classes)

                mean_accuracy, mean_precision, mean_recall, f1_score = get_metric_results(similarity, batch_targets,
                                                                                          mean_accuracy, mean_precision,
                                                                                          mean_recall, f1_score)

                self.optimiser.step()
                self.scheduler.step()

                # Update metric and progress bar
                self.update_mean(loss.item())
                bar.update()
                # bar.set_description('{:.4f}'.format(self.mean_result()))
                bar.set_description(' Loss: {:.4f} - Accuracy: {:.4f} - Precision: {:.4f} - Recall: {:.4f} - F1-Score: {:.4f}'.format(loss, mean_accuracy, mean_precision, mean_recall, f1_score))

            tb.add_scalar("Training Loss", loss, epoch)
            tb.add_scalar("Training Accuracy", mean_accuracy, epoch)
            tb.add_scalar("Training Precision", mean_precision, epoch)
            tb.add_scalar("Training Recall", mean_recall, epoch)
            tb.add_scalar("Training F1-Score", f1_score, epoch)
            tb.close()

        return self.mean_result()

    def validate(self, validation_dataloader, epoch):
        mean_accuracy = 100
        mean_recall = 100
        mean_precision = 100
        f1_score = 100

        images, labels, texts = next(iter(validation_dataloader))
        grid = torchvision.utils.make_grid(images)
        tb.add_image("Validation Images", grid)

        with tqdm(total=len(validation_dataloader) - 1) as bar:
            self.reset_rolling_mean()
            self.model.eval()
            for batch_images, batch_targets, batch_texts in validation_dataloader:
                with torch.no_grad():
                    images_features = self.model.encode_image(batch_images.to(self.device))
                    batch_targets = batch_targets.to(self.device)
                    text = torch.cat([self.tokenize(c) for c in self.object_classes]).to(self.device)
                    texts_features = self.model.encode_text(text.to(self.device))

                features = (images_features @ texts_features.T)
                similarity = (100.0 * images_features @ texts_features.T)
                similarity = similarity.softmax(dim=-1)

                # Apply Cross Entropy Loss SemiHardLoss
                loss = self.criterion(features, batch_targets.softmax(dim=1))

                # Update metric and progress bar
                self.update_mean(loss.item())
                bar.update()
                bar.set_description('{:.4f}'.format(self.mean_result()))

                # This is a function call to get the percentage values of object presence in the frame
                # get_object_found_percentage(similarity, self.object_classes)

                mean_accuracy, mean_precision, mean_recall, f1_score = get_metric_results(similarity, batch_targets,
                                                                                          mean_accuracy, mean_precision,
                                                                                          mean_recall, f1_score)
            tb.add_scalar("Validation Loss", loss, epoch)
            tb.add_scalar("Validation Accuracy", mean_accuracy, epoch)
            tb.add_scalar("Validation Precision", mean_precision, epoch)
            tb.add_scalar("Validation Recall", mean_recall, epoch)
            tb.add_scalar("Validation F1-Score", f1_score, epoch)
            tb.close()

        return self.mean_result()

    def test(self, testing_dataloader=None):
        checkpoint = torch.load(join(self.global_properties['root_directory'], 'checkpoints/clip_model'
                                                                               '/best_checkpoint/frozen_clip_best.ckpt'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        mean_accuracy = 100
        mean_recall = 100
        mean_precision = 100
        f1_score = 100

        images, labels, texts = next(iter(testing_dataloader))
        grid = torchvision.utils.make_grid(images)
        tb.add_image("Testing Images", grid)

        # training_dataloader, validation_dataloader, testing_dataloader = self.get_dataloader()

        with tqdm(total=len(testing_dataloader) - 1) as bar:
            self.reset_rolling_mean()
            for batch_images, batch_targets, batch_texts in testing_dataloader:
                with torch.no_grad():
                    image_features = self.model.encode_image(batch_images.to(self.device))
                    batch_targets = batch_targets.to(self.device)
                    text = torch.cat([self.tokenize(c) for c in self.object_classes]).to(self.device)
                    text_features = self.model.encode_text(text.to(self.device))

                bar.update()

                similarity = (100.0 * image_features @ text_features.T)
                similarity = similarity.softmax(dim=-1)

                # This is a function call to get the percentage values of object presence in the frame
                # get_object_found_percentage(similarity, self.object_classes)

                mean_accuracy, mean_precision, mean_recall, f1_score = get_metric_results(similarity, batch_targets,
                                                                                          mean_accuracy, mean_precision,
                                                                                          mean_recall, f1_score)

            print(f'Accuracy: {round(mean_accuracy, 2)}%')
            print(f'Precision: {round(mean_precision, 2)}%')
            print(f'Recall: {round(mean_recall, 2)}%')
            print(f'F1-Score: {round(f1_score, 2)}%')

            tb.add_scalar("Testing Accuracy", mean_accuracy)
            tb.add_scalar("Testing Precision", mean_precision)
            tb.add_scalar("Testing Recall", mean_recall)
            tb.add_scalar("Testing F1-Score", f1_score)
            tb.close()


    def save_best_checkpoint(self, epoch, validation_loss):
        save_flag = False

        if self.best_validation_loss is None:
            self.best_validation_loss = validation_loss
            save_flag = True
        elif validation_loss < self.best_validation_loss:
            self.best_validation_loss = validation_loss
            save_flag = True

        if self.freeze_layers:
            initial = "frozen_" + self.global_properties['optimiser'] + '_' + str(self.learning_rate)
        else:
            initial = "unfrozen_" + self.global_properties['optimiser'] + '_' + str(self.learning_rate)

        if save_flag:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimiser.state_dict(),
                'loss': self.best_validation_loss,
            }, join(self.global_properties['root_directory'],
                    'checkpoints/clip_model/epoch_checkpoints/' + initial + '_clip_' + str(epoch) + '.ckpt'))
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
        # testing_dataloader = self.get_dataloader()
        if self.global_properties['optimiser'] is not None:
            self.optimiser = self.get_optimiser(self.global_properties['optimiser'])
        if self.global_properties['learningrate'] is not None:
            self.learning_rate = self.global_properties['learningrate']
        self.scheduler = self.get_scheduler(self.optimiser, self.number_of_epochs, training_dataloader)

        self.number_of_epochs = self.global_properties['epochs']
        for epoch in range(self.number_of_epochs):
            self.train(training_dataloader, epoch)
            validation_loss = self.validate(validation_dataloader, epoch)
            self.save_best_checkpoint(epoch, validation_loss)

        self.test(testing_dataloader)

    def get_encoding(self, images, use_text_features=False):
        with torch.no_grad():
            images_features = self.model.encode_image(images.to(self.device))
            if self.text is None or self.texts_features is None:
                self.text = torch.cat([self.tokenize(c) for c in self.object_classes]).to(self.device)
                self.texts_features = self.model.encode_text(self.text.to(self.device))

        if use_text_features == 'True':
            return images_features @ self.texts_features.T
        else:
            return images_features
