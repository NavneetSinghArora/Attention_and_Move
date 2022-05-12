"""
This file is used to set up and initialize all the global variable required in the entire package.
"""

# Importing required python libraries for processing
from os.path import join
from src.core.utils.property_loader import LoadProperties


class GlobalVariables:
    """
    This class is used to initialize all the global properties which will be required by the entire project.
    """
    __instance = None
    __instance_created = False

    def __init__(self, **kwargs):
        """
        This method makes sure that the properties are initialized only once in lifetime of this object.
        """

        if not self.__instance_created:
            self.global_properties = {'root_directory': kwargs['package_root']}
            self.global_properties['global_configurations'] = join(self.global_properties['root_directory'], 'resources/model/configuration.properties')

            self.load_properties = LoadProperties(self.global_properties)

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
        self.global_properties = self.load_properties.fetch_properties()
