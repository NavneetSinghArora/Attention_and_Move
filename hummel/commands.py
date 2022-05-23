import subprocess
import shlex
import os

class Hummel:

    @staticmethod
    def init(username):
        subprocess.call(shlex.split('sh {}/hummel.sh {}'.format(os.path.dirname(os.path.realpath(__file__)),username)))

    @staticmethod
    def train(username):
        subprocess.call(shlex.split('ssh hummel1 "cd \$WORK/logs/ && sbatch \$HOME/jobs/start.sh"'.format(os.path.dirname(os.path.realpath(__file__)),username)))