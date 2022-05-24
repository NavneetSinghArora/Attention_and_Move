#!/bin/bash
#SBATCH --job-name=AAM                                                          # Attention and Move
#SBATCH --nodes=2
#SBATCH --partition=gpu
#SBATCH --tasks-per-node=1
#SBATCH --time=00:03:00
##SBATCH --mail-user=                                                           # add your email here
##SBATCH --mail-type=END,FAIL                                                   # don't forget to uncomment ;)
#SBATCH --export=NONE

set -e                                                                          # stop on first error
source /sw/batch/init.sh                                                        # init clean environment
module load singularity                                                         # load singularity module

export MASTER_ADDR=$(hostname -s)                                               # set master ip
export MASTER_PORT=23456                                                        # set master port
export SINGULARITY_TMPDIR=$RRZ_LOCAL_TMPDIR                                     # set singularity temp dir
export SINGULARITY_CACHEDIR=$TMPDIR/SINGULARITY_CACHE                           # set singularity cache dir

cp -r $WORK/{.ai2thor,.config,data} $RRZ_GLOBAL_TMPDIR                          # copy data to global temp dir

# print some information
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NODELIST=${SLURM_NODELIST}"
echo "Contents of ${RRZ_GLOBAL_TMPDIR}:"
echo "$(ls -la $RRZ_GLOBAL_TMPDIR)"

# run on each assigned node
srun singularity exec --home $RRZ_GLOBAL_TMPDIR --bind $HOME/jobs:/jobs,$RRZ_GLOBAL_TMPDIR/data:/Attention_and_Move/data,$WORK/output:/Attention_and_Move/output --nv --userns /usw/$USER/container_sandbox bash /jobs/node.sh

exit