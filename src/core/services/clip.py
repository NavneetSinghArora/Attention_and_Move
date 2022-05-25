import clip
import torch
from os.path import join
from PIL import Image
from torchvision.datasets import CIFAR100


def predict(img, idx, target_object, rootDirectory):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load('ViT-L/14@336px', device=device, download_root=join(rootDirectory, 'data/external/clip/models/'))

    # Download the dataset
    cifar100 = CIFAR100(root=join(rootDirectory, 'data/external/torchvision/datasets/'), download=True, train=False)

    image = preprocess(Image.fromarray(img)).unsqueeze(0).to(device)
    text = torch.cat([clip.tokenize(f'a {c}') for c in cifar100.classes]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(10)

    # Print the result
    print('Agent ' + str(idx) + ':')
    for value, index in zip(values, indices):
        found = ''
        if cifar100.classes[index] == target_object.lower():
            found = ' - found ' + target_object + '!'
        print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%{found}")