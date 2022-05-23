import subprocess
import shlex
import os

class Hummel:

    @staticmethod
    def init(username):
        subprocess.call(shlex.split('sh {}/hummel.sh {}'.format(os.path.dirname(os.path.realpath(__file__)),username)))

    @staticmethod
    def train(username):
        # TODO: bash: sbatch: command not found - because modules will only be loaded in interactive shell! Find solution!
        subprocess.call(shlex.split('ssh hummel1 "cd /work/{0}/logs/ && sbatch /home/{0}/jobs/start.sh"'.format(username)))