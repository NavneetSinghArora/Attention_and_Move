"""
This file fetches the properties from 'configuration.properties' and returns them.
These are the system-wide properties loaded only once in the entire project.
"""

# Importing required python libraries for processing
from jproperties import Properties
from os.path import join


class LoadProperties:
    """
    This is the class to load all the properties and make them available to the entire project.
    """
    def __init__(self, global_properties):
        """
        This method creates an object for the properties class.
        """
        self.configs = Properties()
        self.simulator_properties = {}
        self.property_file = join(global_properties['root_directory'], global_properties['simulator_configurations'])

    def fetch_properties(self):
        """
        This method opens up the properties file, reads the file and fetches all the parameters.
        """

        with open(self.property_file, 'rb') as file:
            self.configs.load(file)

        properties = self.configs.items()

        for item in properties:
            key = item[0]
            value = item[1].data
            self.simulator_properties[key] = value

        return self.simulator_properties
