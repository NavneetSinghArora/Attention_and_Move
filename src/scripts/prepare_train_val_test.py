from random import shuffle
from os.path import exists
import shutil
import os

# Data split ratio
split_ratio = '60:20:20'

# Main directory for the images
images = os.listdir('/export2/scratch/cv_proj_team1/Attention_and_Move/output/dataset/images')
shuffle(images)

total_images = len(images)
total_training_images = int((total_images * int(split_ratio.split(':')[0]))/100)
total_validation_images = int((total_images * int(split_ratio.split(':')[1]))/100)
total_testing_images = int((total_images * int(split_ratio.split(':')[2]))/100)

directory = '/export2/scratch/cv_proj_team1/Attention_and_Move/output/dataset/'
print("Working with Training Images")
for i in range(0, total_training_images):
    source_image = directory + 'images/' + images[i]
    target_image = directory + 'training/images/' + images[i]
    source_annotation = directory + 'annotations/' + images[i].replace('png', 'csv')
    target_annotation = directory + 'training/annotations/' + images[i].replace('png', 'csv')
    if exists(source_image) and exists(source_annotation):
        shutil.copyfile(source_image, target_image)
        shutil.copyfile(source_annotation, target_annotation)

print("Working with Validation Images")
for i in range(total_training_images, total_training_images + total_validation_images):
    source_image = directory + 'images/' + images[i]
    target_image = directory + 'validation/images/' + images[i]
    source_annotation = directory + 'annotations/' + images[i].replace('png', 'csv')
    target_annotation = directory + 'validation/annotations/' + images[i].replace('png', 'csv')
    if exists(source_image) and exists(source_annotation):
        shutil.copyfile(source_image, target_image)
        shutil.copyfile(source_annotation, target_annotation)

print("Working with Testing Images")
for i in range(total_training_images + total_validation_images, total_training_images + total_validation_images + total_testing_images):
    source_image = directory + 'images/' + images[i]
    target_image = directory + 'testing/images/' + images[i]
    source_annotation = directory + 'annotations/' + images[i].replace('png', 'csv')
    target_annotation = directory + 'testing/annotations/' + images[i].replace('png', 'csv')
    if exists(source_image) and exists(source_annotation):
        shutil.copyfile(source_image, target_image)
        shutil.copyfile(source_annotation, target_annotation)

print("We are all done!!!")