"""
This file is used to set up and initialize all the runtime variable required in the entire package.
"""

# Importing required python libraries for processing
from os.path import join

from src.core.utils.constants import PROJECT_ROOT_DIR
from src.core.utils.runtime.property_loader import LoadProperties


class RuntimeVariables:
    """
    This class is used to initialize all the runtime properties which will be required by the entire project.
    """
    __instance = None
    __instance_created = False

    def __init__(self, **kwargs):
        """
        This method makes sure that the properties are initialized only once in lifetime of this object.
        """

        if not self.__instance_created:
            self.runtime_properties = {'root_directory': PROJECT_ROOT_DIR}
            self.runtime_properties['runtime_configurations'] = join(PROJECT_ROOT_DIR, 'resources/project/runtime_args.properties')

            self.load_properties = LoadProperties(self.runtime_properties).fetch_properties()

            self.__instance_created = True

    def __new__(cls, *args, **kwargs):
        """
        This is a class method.
        This method makes sure that the class follows Singleton Pattern.
        """

        if cls.__instance is None:
            cls.__instance = object.__new__(cls)

        return cls.__instance

    def load_configuration_properties(self):
        self.runtime_properties = self.load_properties.fetch_properties()
