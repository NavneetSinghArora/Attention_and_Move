import clip
import torch
from os.path import join
from PIL import Image
from torchvision.datasets import CIFAR100


def predict_clip(frame, target_object, target_object_threshold, rootDirectory, simulator_properties):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load('ViT-L/14@336px', device=device, download_root=join(rootDirectory, 'data/external/clip/models/'))

    obj_classes = simulator_properties['object_classes'].split(',')

    # cifar100 = CIFAR100(root=join(rootDirectory, 'data/external/torchvision/datasets/'), download=True, train=False)

    image = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
    text = torch.cat([clip.tokenize(f'a {c}') for c in obj_classes]).to(device)


    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    # print('image features:', image_features.shape)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    # print('text features:', text_features.shape)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    values, indices = similarity[0].topk(10)

    # Print the result
    target_found = False
    for value, index in zip(values, indices):
        # found = ''
        if obj_classes[index] == target_object.lower() and 100 * value.item() >= int(target_object_threshold):
            # found = ' - found ' + target_object + '!'
            target_found = True
        # print(f"{obj_classes[index]:>16s}: {100 * value.item():.2f}%{found}")

    return (image_features, text_features, target_found, similarity)

# def custom_clip_accuracy(model, image_features, text_features, object_classes):
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)
#
#     similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
#     values = similarity[0]
#     print()
