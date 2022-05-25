"""
This script offers the CLI interface for the entire project.
"""

# Importing python libraries for required processing
from src.core.utils.global_variables import GlobalVariables
from src.core.utils.simulator.simulator_variables import SimulatorVariables
from src.core.model.simulator.environment import Environment
from src.data.prepare_dataset import PrepareDataset
from pathlib import Path
import click
# import pandas as pd
# import numpy as np

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


def get_package_root():
    """
    This function fetches the path to the parent directory and returns the path as a string.

    :return path:   Path to the root directory of the project as a string.
    :rtype:         str
    """

    return str(Path(__file__).parent.resolve().parent.resolve().parent.resolve())


def initialise_simulator(global_properties):
    """
    This function creates the SimulatorVariables class and helps load the simulator related properties.

    :param global_properties:       It contains the global properties for the entire project.
    :type global_properties:        dict
    :return simulator_properties:   It contains the properties specific to the simulator.
    :rtype:                         dict
    """

    simulator_variables = SimulatorVariables(global_properties)
    simulator_variables.load_configuration_properties()
    simulator_properties = simulator_variables.simulator_properties

    return simulator_properties


def initialise_environment(global_properties, simulator_properties):
    """
    This function initialises the AI2Thor environment to be used throughout the project.

    :param global_properties:       It contains the global properties for the entire project.
    :type global_properties:        dict
    :param simulator_properties:    It contains the properties specific to the simulator.
    :type simulator_properties:     dict
    :return environment:            This is the object of the environment containing the agent configuration among other information.
    :rtype:                         Environment
    """

    environment = Environment(global_properties, simulator_properties)
    return environment


def _init(**kwargs):
    """
    This function initialises the global system properties by creating the GlobalVariables class.
    """

    # Initializing the parent root directory for path configuration.
    kwargs['package_root'] = get_package_root()

    # Initializing the Global Variables which will be available throughout the project.
    global_variables = GlobalVariables(**kwargs)
    global_variables.load_configuration_properties()

    return global_variables


# Command line interface (CLI) main
@click.group(chain=True, help='Command line tool for pyetl.', invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
def cli(**kwargs):
    """
    This is the entry point function for the project. It initialises the project with passed CLI arguments.
    """

    _init(**kwargs)

    # Entry point for the command line interface
    click.echo("\nThis is the command line interface of Attention_and_Move. Type 'attention_and_move --help' for details.\n")


@cli.command('training', context_settings=CONTEXT_SETTINGS)
@click.option('-p', '--platform', required=True, is_flag=False, default='CloudRendering', show_default=True, help='Choose between CloudRendering, Linux64, OSXIntel64')
@click.option('-s', '--start', required=True, is_flag=True, help='This is a flag to start the training and agent movements')
def start_environment(**kwargs):

    global_variables = GlobalVariables(**kwargs)
    global_properties = global_variables.global_properties
    print('Project Properties Initialized')

    simulator_properties = initialise_simulator(global_properties)
    print('Simulator Properties Initialized')

    if kwargs['platform']:
        print("Platform = " + kwargs['platform'])
        simulator_properties['platform'] = kwargs['platform']

    environment = initialise_environment(global_properties, simulator_properties)
    print('Simulator Environment Initialized')

    if kwargs['start']:
        print('Staring the environment')
        environment.start()


@cli.command('dataset', context_settings=CONTEXT_SETTINGS)
@click.option('-c', '--create', required=True, is_flag=True, help='This is a flag to start the dataset creation')
@click.option('-s', '--scene', required=True, is_flag=False, type=click.Choice(['LivingRoom', 'Kitchen', 'Bedroom', 'Bathroom']),
              help='This is to choose the various AI2Thor scenes to create the dataset', multiple=True)
def prepare_dataset(**kwargs):
    """
    Prepare the dataset using the AI2Thor egocentric agent views using MultiAgent configuration.
    """
    print('Staring with Dataset Creation')

    global_variables = GlobalVariables(**kwargs)
    global_properties = global_variables.global_properties

    dataset_scenes = kwargs['scene']
    PrepareDataset(global_properties, dataset_scenes).collect_dataset()


if __name__ == '__main__':
    cli()
