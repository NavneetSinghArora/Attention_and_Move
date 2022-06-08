"""
This script offers the CLI interface for the entire project.
"""
import click

from hummel.commands import Hummel
from src.core.model.simulator.environment import Environment
from src.core.utils.constants import CONTEXT_SETTINGS, PROJECT_ROOT_DIR
from src.core.utils.properties.global_variables import GlobalVariables
from src.core.utils.simulator.simulator_variables import SimulatorVariables


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
    
    # Initializing the Global Variables which will be available throughout the project.
    global_variables = GlobalVariables(**kwargs)
    global_variables.load_configuration_properties()

    return global_variables


# Command line interface (CLI) main
@click.group(chain=True, help='Command line tool for pyetl.', invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
def cli1(**kwargs):
    pass


@click.group(chain=True, help='Command line tool for pyetl.', invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
def cli2(**kwargs):
    pass


@cli1.command('training', context_settings=CONTEXT_SETTINGS)
@click.option('-p', '--platform', is_flag=False, default='CloudRendering', show_default=True, help='Choose between CloudRendering, Linux64, OSXIntel64')
@click.option('-s', '--start', is_flag=True, help='Start training')
def training(**kwargs):

    """Entry point for the command line interface."""
    click.echo("\nThis is the command line interface of Attention_and_Move. Type 'attention_and_move --help' for details.\n")

    """Start training on local machine"""
    global_variables = _init(**kwargs)
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
        print('Staring environment')
        environment.start()


@cli2.command('hummel', context_settings=CONTEXT_SETTINGS)
@click.option('-u', '--user', is_flag=False, help='UHH username, i.e. ba*####')
@click.option('-i', '--init', is_flag=True, help='Initialize AAM on Hummel')
@click.option('-m', '--mnist', is_flag=True, help='Start MNIST-Test')
@click.option('-t', '--train', is_flag=True, help='Start training on Hummel')
@click.option('-m', '--mnist', is_flag=True, help='Start MNIST-Test')
@click.option('-p', '--platform', is_flag=False, default='CloudRendering', show_default=True, help='Choose between CloudRendering, Linux64, OSXIntel64') # only needed for local debugging
@click.option('-l', '--lr', is_flag=False, type=float, default=0.0001, show_default=True, help='Learning rate')
@click.option('-s', '--seed', is_flag=False, type=int, default=1, show_default=True, help='Random seed')
@click.option('-w', '--workers',is_flag=False, type=int, default=1, show_default=True, help='Number of training processes')
@click.option('-n', '--num_steps', is_flag=False, type=int, default=50, show_default=True, help='Number of forward steps in A3C')
@click.option('-f', '--save_freq', is_flag=False, type=int, default=1e6, show_default=True, help='Number of training episodes till save')
@click.option('-d', '--checkpoints_dir', is_flag=False, type=str, default='output/checkpoints/', show_default=True, help='Folder for trained checkpoints')
@click.option('-c', '--use_checkpoint', is_flag=False, type=str, default='', show_default=True, help='Checkpoint to resume training from')
@click.option('-e', '--max_ep', is_flag=False, type=float, default='inf', show_default=True, help='Maximum number of episodes')
@click.option('-v', '--visualize_test_agent', is_flag=False, type=bool, default=False, show_default=True, help='Create plots and graphics for test agent')
@click.option('-q', '--use_episode_init_queue', is_flag=False, type=bool, default=False, show_default=True, help='Necessary when evaluating models on fixed datasets')
def hummel(**kwargs):
    """Run AAM on Hummel"""

    if kwargs['init']:
        if kwargs['user']:
            print("Starting to initialize AAM on Hummel with user={}".format(kwargs['user']))
            Hummel.init(username=kwargs['user'])
        else:
            print('Please provide your UHH username with the option --user=username and retry again!')

    if kwargs['mnist']:
        print("Start MNIST-Test")
        Hummel.mnist()

    if kwargs['train']:
        # remove all parameters that may not be forwarded to main.py (all parameters not contained in arguments.py must be omitted!)
        kwargs.pop('user')
        kwargs.pop('init')
        kwargs.pop('mnist')
        kwargs.pop('train')
        kwargs.pop('mnist')

        print("Start training on Hummel")
        Hummel.train(kwargs)

    if kwargs['mnist']:
        print("Start MNIST-Test")
        Hummel.mnist()

cli = click.CommandCollection(sources=[cli1, cli2])

if __name__ == '__main__':
    cli()
