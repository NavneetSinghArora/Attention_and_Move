import subprocess
import shlex
import os

class Hummel:

    @staticmethod
    def init(username):
        subprocess.call(shlex.split('sh {}/init/hummel.sh {}'.format(os.path.dirname(os.path.realpath(__file__)), username)))

    @staticmethod
    def train(kwargs):
        parameters = ' '.join(["--" + key + "=" + str(value) for key, value in kwargs.items()])
        os.system('python {}/../src/core/main.py {}'.format(os.path.dirname(os.path.realpath(__file__)), parameters))