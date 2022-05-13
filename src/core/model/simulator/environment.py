"""

"""

# Importing python libraries for required processing
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from src.core.services.common_services import visualize_frames
import random


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
        self._started = False

        if not self.__instance_created:
            self.__controller = Controller(platform=CloudRendering,

                                           agentCount=int(self.simulator_properties['number_of_agents']),
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
        # Randomizing the position of the agents before starting with any episode
        self.randomize_agents()
        print('Agent Positions Randomized')

        print('Staring with the Agent Movements')
        for i in range(0, 12):
            print(f"Making Agent Movement: {i}")
            event_0 = self.__controller.step('RotateLeft', agentId=0)
            event_1 = self.__controller.step('RotateRight', agentId=1)

            for j, multi_agent_event in enumerate([event_0, event_1]):
                rgb_frames = [event.frame for event in multi_agent_event.events]
                visualize_frames(rgb_frames, (8, 8), i, self.global_properties['root_directory'])

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
