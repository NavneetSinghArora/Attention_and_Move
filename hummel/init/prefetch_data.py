from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from clip import load
from os.path import expanduser, join
from shutil import rmtree
from src.core.utils.constants import PROJECT_ROOT_DIR

# =======================================================================
# NOTE: This script prefetches data that is required to run this
#       project on Hummel. It will be executed by hummel.sh!
# =======================================================================

load('ViT-L/14@336px', device='cpu', download_root=join(PROJECT_ROOT_DIR, 'data/external/clip/models/'))            # load clip model

try:
    rmtree(expanduser('~') + '/.ai2thor/')                                                                          # deletes .ai2thor directory (makes sure only necessary file are transfered)
    controller = Controller(platform=CloudRendering)                                                                # download environment, i.e. thor-CloudRendering-[a-z0-9]*.zip
    controller.stop()                                                                                               # close controller
except:
    pass                                                                                                            # no need to show exceptions