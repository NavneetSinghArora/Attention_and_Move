"""
This script offers the CLI interface for the entire project.
"""

# Importing python libraries for required processing
from src.core.utils.global_variables import GlobalVariables
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

    return str(Path(__file__).parent)


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
@click.option('-v', '--version', is_flag=True, help='Prints version number and exits.')
def cli(**kwargs):
    global_variables = _init(**kwargs)
    global_properties = global_variables.global_properties

    """Entry point for the command line interface."""
    click.echo("\nThis is the command line interface of Attention_and_Move. Type 'attention_and_move --help' for details.\n")


if __name__ == '__main__':
    cli()
