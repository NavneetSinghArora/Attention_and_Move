"""
This file is used to set up and initialize all the global variable required in the entire package.
"""

# Importing required python libraries for processing
from src.core.utils.simulator.property_loader import LoadProperties


class SimulatorVariables:
    """
    This class is used to initialize all the simulator properties which will be required by the entire project.
    """
    __instance = None
    __instance_created = False

    def __init__(self, global_properties):
        """
        This method makes sure that the properties are initialized only once in lifetime of this object.
        """

        if not self.__instance_created:
            self.simulator_properties = None
            self.load_properties = LoadProperties(global_properties)

            self.__instance_created = True

    def __new__(cls, *args, **kwargs):
        """
        This is a class method.
        This method makes sure that the class follows Singleton Pattern.
        """
        if cls.__instance is None:
            cls.__instance = object.__new__(cls)

        return cls.__instance

    def create_scenes(self):
        """
        This method creates the different scene names available in iTHOR.
        These names are then converted to the train/val/test split in the ratio of 20:5:5.
        """

        # kitchens = [f"FloorPlan{i}" for i in range(1, 31)]
        # bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
        # bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]
        living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]

        training_scenes = living_rooms[:20]
        self.simulator_properties['training_scenes'] = training_scenes

        validation_scenes = living_rooms[20:25]
        self.simulator_properties['validation_scenes'] = validation_scenes

        testing_scenes = living_rooms[25:]
        self.simulator_properties['testing_scenes'] = testing_scenes

    def load_configuration_properties(self):
        self.simulator_properties = self.load_properties.fetch_properties()
        self.create_scenes()
