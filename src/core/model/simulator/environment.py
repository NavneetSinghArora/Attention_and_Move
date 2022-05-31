"""

"""

# Importing python libraries for required processing
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering, Linux64, OSXIntel64
from src.core.services.clip import predict
from src.core.services.viewer import Viewer
from src.core.services.common_services import visualize_frames
from src.core.model.object_detection.clip_detection import ClipObjectDetection
import random
import sys

class Environment:
    """
    This class is used to initialize all the basic environment for the simulation in AI2Thor which will be applicable to the entire project.
    """

    __instance = None
    __instance_created = False
    __controller: Controller = None

    def __init__(self, global_properties, simulator_properties):
        """
        This method makes sure that the properties are initialized only once in lifetime of this object.
        """

        self.global_properties = global_properties
        self.simulator_properties = simulator_properties

        # get global properties
        self.rootDirectory = self.global_properties['root_directory']

        # get simulator properties
        self.agentCount = int(self.simulator_properties['number_of_agents'])

        self._started = False

        if not self.__instance_created:
            self.__controller = Controller(platform=getattr(sys.modules[__name__], self.simulator_properties['platform']),

                                           agentCount=self.agentCount,
                                           agentMode=self.simulator_properties['agent_mode'],
                                           visibilityDistance=float(self.simulator_properties['visibility_distance']),
                                           scene=self.simulator_properties['floor_scene'],

                                           gridSize=float(self.simulator_properties['grid_size']),
                                           snapToGrid=True,
                                           rotateStepDegrees=30,

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

    def randomize_agents(self) -> None:
        all_possible_positions = self.__controller.step(action="GetReachablePositions").metadata["actionReturn"]
        agent_positions = random.choices(all_possible_positions, k=2)
        self.__controller.step(action="Teleport", position=agent_positions[0], agentId=0)
        self.__controller.step(action="Teleport", position=agent_positions[1], agentId=1)

    def start(self) -> None:
        """
        First assign the agents a random position
        Then, pull out frame and locate the object inside that frame
        If found, then skip to next agent
        else, rotate the agent to get the next frame
        """
        # Randomizing the position of the agents before starting with any episode
        self.randomize_agents()
        print('Agent Positions Randomized')

        print('Starting with the Agent Movements')

        target_object = self.simulator_properties['target_object']
        clip_object_detection = ClipObjectDetection(self.global_properties, self.simulator_properties).train()

        self._started = True

    def stop(self):
        return self.__controller.step(action="Done")

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
