import subprocess
import shlex
import os

class Hummel:

    @staticmethod
    def init(username):
        subprocess.call(shlex.split('sh {}/hummel.sh {}'.format(os.path.dirname(os.path.realpath(__file__)),username)))

    @staticmethod
    def train():
        os.system('python {}/../src/core/main.py'.format(os.path.dirname(os.path.realpath(__file__))))