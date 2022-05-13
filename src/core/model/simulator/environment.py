"""

"""

# Importing python libraries for required processing
import ai2thor.server
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt


class Environment:
    """
    This class is used to initialize all the basic environment for the simulation in AI2Thor which will be applicable to the entire project.
    """

    __instance = None
    __instance_created = False
    __controller: Controller = None

    def __init__(self, simulator_properties):
        """
        This method makes sure that the properties are initialized only once in lifetime of this object.
        """

        self.simulator_properties = simulator_properties
        self._started = False

        if not self.__instance_created:
            __controller = Controller(platform=CloudRendering,

                                      agentCount=int(self.simulator_properties['number_of_agents']),
                                      agentMode=self.simulator_properties['agent_mode'],
                                      visibilityDistance=float(self.simulator_properties['visibility_distance']),
                                      scene=self.simulator_properties['floor_scene'],

                                      gridSize=float(self.simulator_properties['grid_size']),
                                      snapToGrid=True,
                                      rotateStepDegrees=90,

                                      renderDepthImage=bool(self.simulator_properties['render_depth_image']),
                                      renderInstanceSegmentation=bool(self.simulator_properties['render_image_segmentation']),

                                      width=300,
                                      height=300,
                                      fieldOfView=int(self.simulator_properties['field_of_view']))

            self.__instance_created = True

    def __new__(cls, *args, **kwargs):
        """
        This is a class method.
        This method makes sure that the class follows Singleton Pattern.
        """
        if cls.__instance is None:
            cls.__instance = object.__new__(cls)

        return cls.__instance

    def start(self) -> None:
        self.__controller.start()
        self._started = True

    def stop(self) -> None:
        self.__controller.stop()
        self._started = False

    def reset(self):
        return self.__controller.reset()

    def get_controller(self):
        return self.__controller

    @property
    def get_scene_name(self) -> str:
        return self.__controller.last_event.metadata["sceneName"]

    @property
    def get_total_number_of_agents(self) -> int:
        return len(self.__controller.last_event.events)
