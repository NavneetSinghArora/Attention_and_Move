
import clip
import os
import torch

from PIL import Image
from torchvision.datasets import CIFAR100


# follow installation instructions on https://github.com/openai/CLIP
def predict(img, idx, target):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Download the dataset
    cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=False, train=False)

    image = preprocess(Image.fromarray(img)).unsqueeze(0).to(device)
    text = torch.cat([clip.tokenize(f"a {c}") for c in cifar100.classes]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(10)

    # Print the result
    print("Agent " + str(idx) + ":")
    for value, index in zip(values, indices):
        print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")