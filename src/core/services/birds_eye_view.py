import matplotlib.pyplot as plt
from PIL import Image


class BirdsEyeView(object):

    _instance = None
    _plt = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BirdsEyeView, cls).__new__(cls)
            cls._plt = plt.ion()                                            # switch on interactive mode for pyplot
        
        return cls._instance                                                # singleton

    @classmethod
    # show a birds eye view of each given event (could also be run in a separate thread to not slow down the training)
    def update(self, e):                                                    # input: any MultiAgentEvent with a top-down camera bound to agent 0
        image = Image.fromarray(e.third_party_camera_frames[0])             # extracts the bird view image from the given event
        plt.imshow(image)                                                   # add the image to the pyplot
        plt.show()                                                          # shows the image in the pyplot
        plt.pause(0.001)                                                    # needed to refresh the pyplot UI (to show the changed image, else the UI will stay black)
        plt.clf()