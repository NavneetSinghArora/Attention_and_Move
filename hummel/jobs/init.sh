#!/bin/bash

module load singularity                                                         # load singularity module
cd /usw/$USER                                                                   # change to user software folder
singularity build --sandbox container_sandbox container.sif                     # create singularity sandbox from container

# give advice on how to run a batch job
echo "You may now start your first batch job by running 'cd \$WORK/logs/ && sbatch \$HOME/jobs/start.sh'"

exit