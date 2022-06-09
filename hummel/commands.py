from os import system
from shlex import split
from subprocess import call

from src.core.utils.constants import PROJECT_ROOT_DIR


class Hummel:

    @staticmethod
    def init(username):
        call(split(f'sh {PROJECT_ROOT_DIR}/hummel/init/hummel.sh {username}'))

    @staticmethod
    def train(kwargs):
        parameters = ' '.join(["--" + key + "=" + str(value) for key, value in kwargs.items()])
        system(f'python {PROJECT_ROOT_DIR}/src/core/main.py {parameters}')