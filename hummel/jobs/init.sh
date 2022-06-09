#!/bin/bash

# make sure the singularity module is loaded on the interactive bash shell, before running this script!
# to load the module use the command: 'module load singularity'

cd /usw/$USER                                                                   # change to user software folder
singularity build --sandbox container_sandbox container.sif                     # create singularity sandbox from container

# give advice on how to run a batch job
echo "You may now start your first batch job by running 'cd \$WORK/logs/ && sbatch \$HOME/jobs/start.sh'"

exit