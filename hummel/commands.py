import subprocess
import shlex
import os

class Hummel:

    @staticmethod
    def init(username):
        subprocess.call(shlex.split('sh {}/hummel.sh {}'.format(os.path.dirname(os.path.realpath(__file__)),username)))

    @staticmethod
    def train():
        pass
        
        # add path to the python script that shall be executed locally on hummel, please keep '{}/../' as part of the path!
        # os.system('python3 {}/../your_path.py'.format(os.path.dirname(os.path.realpath(__file__))))