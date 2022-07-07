import clip
import torch
from os.path import join
from PIL import Image
from pathlib import Path
from clip.simple_tokenizer import SimpleTokenizer

from src.core.utils.constants import PROJECT_ROOT_DIR
from src.scripts import prepare_dataset
from src.core.utils.simulator.simulator_variables import SimulatorVariables
from src.core.utils.properties.global_variables import GlobalVariables


def get_package_root():
    """
    This function fetches the path to the parent directory and returns the path as a string.

    Returns:
        path ():    Path to the root directory of the project as a string.
    """

    return str(Path(__file__).parent.resolve().parent.resolve().parent.resolve().parent.resolve())


kwargs = {'package_root': get_package_root()}
global_variables = GlobalVariables(**kwargs)
global_properties = global_variables.global_properties
global_properties['gpu'] = "0"
global_properties['frozen'] = True
global_properties['learningrate'] = 0.01
global_properties['epochs'] = 20
global_properties['optimiser'] = 'SGD'
simulator_variables = SimulatorVariables(global_properties)
simulator_variables.load_configuration_properties()
simulator_properties = simulator_variables.simulator_properties

tokenizer = SimpleTokenizer()

checkpoint = torch.load(join(global_properties['root_directory'], 'checkpoints/clip_model'
                                                                  '/best_checkpoint/frozen_clip_best.ckpt'))

object_classes = (simulator_properties['object_classes']).split(',')
torch.cuda.set_device(int(global_properties['gpu']))
device = "cuda:" + global_properties['gpu'] if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model.load_state_dict(checkpoint['model_state_dict'])

def tokenize(object_class, context_length: int = 77):
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(object_class) + [eot_token]]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        n = min(len(tokens), context_length)
        result[i, :n] = torch.tensor(tokens)[:n]
        if len(tokens) > context_length:
            result[i, -1] = tokens[-1]
    return result


def get_clip_encoding(image):


    with torch.no_grad():
        images_features = model.encode_image(image)
        text = torch.cat([tokenize(c) for c in object_classes]).to(device)
        texts_features = model.encode_text(text.to(device))

    features = (images_features @ texts_features.T)
    # similarity = (100.0 * images_features @ texts_features.T)
    # similarity = similarity.softmax(dim=-1)

    return features

