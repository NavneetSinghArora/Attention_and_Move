"""
This script offers the CLI interface for the entire project.
"""

from hummel.commands import Hummel
from src.core.model.simulator.environment import Environment
from src.core.utils.properties.global_variables import GlobalVariables
from src.core.utils.simulator.simulator_variables import SimulatorVariables
from pathlib import Path
import click

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


def get_package_root():
    """
    This function fetches the path to the parent directory and returns the path as a string.

    Returns:
        path ():    Path to the root directory of the project as a string.
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
    Initialise the global system properties.
    """

    # Initializing the parent root directory for path configuration.
    kwargs['package_root'] = get_package_root()

    # Initializing the Global Variables which will be available throughout the project.
    global_variables = GlobalVariables(**kwargs)
    global_variables.load_configuration_properties()

    return global_variables


# Command line interface (CLI) main
@click.group(chain=True, help='Command line tool for pyetl.', invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
def cli1(**kwargs):
    _init(**kwargs)

    click.echo("\nEntry point '1' for Command Line Interface. Type 'attention_and_move --help' for details.\n")


@click.group(chain=True, help='Command line tool for pyetl.', invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
def cli2(**kwargs):
    _init(**kwargs)

    click.echo("\nEntry point '2' for Command Line Interface. Type 'attention_and_move --help' for details.\n")


@click.group(chain=True, help='Command line tool for pyetl.', invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
def cli3(**kwargs):
    _init(**kwargs)

    click.echo("\nEntry point '3' for Command Line Interface. Type 'attention_and_move --help' for details.\n")


@cli1.command('training', context_settings=CONTEXT_SETTINGS)
@click.option('-p', '--platform', is_flag=False, default='CloudRendering', show_default=True, help='Choose between CloudRendering, Linux64, OSXIntel64')
@click.option('-s', '--start', is_flag=True, help='Start training')
@click.option('-g', '--gpu', help='Give GPU options as 0 or 1')
@click.option('-f', '--frozen', is_flag=True, help='Layers to be frozen or not')
@click.option('-c', '--checkpoint', help='Training on top of a checkpoint')
@click.option('-o', '--optimiser', help='Training with specific optimiser')
@click.option('-l', '--learningrate', help='Choose specific learning rate')
@click.option('-e', '--epochs', help='Choose specific learning rate')
def start_local_training(**kwargs):
    _init(**kwargs)

    """Start training on local machine"""
    global_variables = _init(**kwargs)
    global_properties = global_variables.global_properties
    print('Project Properties Initialized')

    simulator_properties = initialise_simulator(global_properties)
    print('Simulator Properties Initialized')

    if kwargs['platform']:
        print("Platform = " + kwargs['platform'])
        simulator_properties['platform'] = kwargs['platform']

    if kwargs['gpu']:
        print("GPU = " + kwargs['gpu'])
        global_properties['gpu'] = kwargs['gpu']
    else:
        print("GPU = " + "1")
        global_properties['gpu'] = "1"

    if kwargs['frozen']:
        print("Layers Frozen = True")
        global_properties['frozen'] = True
    else:
        print("Layers Frozen = False")
        global_properties['frozen'] = False

    if kwargs['checkpoint']:
        global_properties['checkpoint'] = kwargs['checkpoint']
    else:
        global_properties['checkpoint'] = None

    if kwargs['optimiser']:
        global_properties['optimiser'] = kwargs['optimiser']
    else:
        global_properties['optimiser'] = 'SGD'

    if kwargs['learningrate']:
        global_properties['learningrate'] = kwargs['learningrate']
    else:
        global_properties['learningrate'] = 0.01

    if kwargs['epochs']:
        global_properties['epochs'] = int(kwargs['epochs'])
    else:
        global_properties['epochs'] = 20

    environment = initialise_environment(global_properties, simulator_properties)
    print('Simulator Environment Initialized')

    if kwargs['start']:
        print('Staring environment')
        environment.start()


@cli2.command('hummel', context_settings=CONTEXT_SETTINGS)
@click.option('-u', '--user', is_flag=False, help='UHH username, i.e. ba*####')
@click.option('-i', '--init', is_flag=True, help='Initialize AAM on Hummel')
@click.option('-m', '--mnist', is_flag=True, help='Start MNIST-Test')
@click.option('-t', '--train', is_flag=True, help='Start training on Hummel')
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

        print("Start training on Hummel")
        Hummel.train(kwargs)


@cli3.command('dataset', context_settings=CONTEXT_SETTINGS)
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
    # PrepareDataset(global_properties, dataset_scenes).collect_dataset()



cli = click.CommandCollection(sources=[cli1, cli3])

if __name__ == '__main__':
    cli()
