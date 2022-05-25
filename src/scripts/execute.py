"""
This script offers the CLI interface for the entire project.
"""

# Importing python libraries for required processing
from src.core.utils.global_variables import GlobalVariables
from src.core.utils.simulator.simulator_variables import SimulatorVariables
from src.core.model.simulator.environment import Environment
from pathlib import Path
import click
# import pandas as pd
# import numpy as np

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


def get_package_root():
    """
    Thi function fetches the path to the parent directory and returns the path as a string.

    Returns:
        path ():    Path to the root directory of the project as a string.
    """

    return str(Path(__file__).parent.resolve().parent.resolve().parent.resolve())


def initialize_simulator(global_properties):
    """

    :param global_properties:
    :type global_properties:
    :return:
    :rtype:
    """
    simulator_variables = SimulatorVariables(global_properties)
    simulator_variables.load_configuration_properties()
    simulator_properties = simulator_variables.simulator_properties

    return simulator_properties


def initialize_environment(global_properties, simulator_properties):
    """

    :param global_properties:
    :type global_properties:
    :param simulator_properties:
    :type simulator_properties:
    :return:
    :rtype:
    """
    environment = Environment(global_properties, simulator_properties)
    return environment


def _init(**kwargs):
    """
    Initialize the global system properties.
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
    global_variables = _init(**kwargs)
    global_properties = global_variables.global_properties

    """Entry point for the command line interface."""
    click.echo("\nThis is the command line interface of Attention_and_Move. Type 'attention_and_move --help' for details.\n")


@cli.command('training', context_settings=CONTEXT_SETTINGS)
@click.option('-p', '--platform', is_flag=False, default='CloudRendering', show_default=True, help='Choose between CloudRendering, Linux64, OSXIntel64')
@click.option('-s', '--start', is_flag=True, help='This is a flag to start the training and agent movements')
def start_environment(**kwargs):
    global_variables = GlobalVariables(**kwargs)
    global_properties = global_variables.global_properties
    print('Project Properties Initialized')

    simulator_properties = initialize_simulator(global_properties)
    print('Simulator Properties Initialized')

    if kwargs['platform']:
        print("Platform = " + kwargs['platform'])
        simulator_properties['platform'] = kwargs['platform']

    environment = initialize_environment(global_properties, simulator_properties)
    print('Simulator Environment Initialized')

    if kwargs['start']:
        print('Staring the environment')
        environment.start()
    
if __name__ == '__main__':
    cli()
