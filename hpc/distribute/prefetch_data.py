import clip
import os

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from os.path import join
from torchvision.datasets import CIFAR100

# =======================================================================
# NOTE: This script prefetches data that is required to run this
#       project on hummel. It will be executed by hummel.sh!
# =======================================================================

dirname = os.path.dirname(__file__)                                                                                 # get path of data.py
rootDirectory = os.path.join(dirname, '../../')                                                                     # specify project root

clip.load('ViT-L/14@336px', device='cpu', download_root=join(rootDirectory, 'data/external/clip/models/'))          # load models
CIFAR100(root=join(rootDirectory, 'data/external/torchvision/datasets/'), download=True, train=False)               # load dataset

try:
    Controller(platform=CloudRendering)                                                                             # load environment
except:
    pass                                                                                                            # no need to show exceptions