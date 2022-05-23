#!/bin/bash

# ==========================================================================
#   
#   This script prefetches, creates and distributes all files needed to 
#   run this project on Hummel.
#   
#   1. make sure that python>=3.8 and singularity==3.9.1 are available
#   2. make sure ~/.ssh/config is configured with hummel1
#   3. make sure you are in the UHH or Informatik VPN
#   4. enter your password when asked
#   5. be patient, initialization may take some time
#   
#   TODO: make sure only thor-CloudRendering environment is transfered
#   
# ==========================================================================

# prefetch all data sources (.ai2thor & data)
python prefetch_data.py

# create singularity container
sudo singularity build container.sif container.def

# copy data from local to Hummel via sftp
sftp hummel1 <<EOF
    mkdir /home/$1/jobs
    mkdir /work/$1/.ai2thor
    mkdir /work/$1/.config
    put -r ../jobs /home/$1
    put -r $HOME/.ai2thor /work/$1
    put -r $HOME/.config/unity3d /work/$1/.config
    put -r ../../data /work/$1
    put container.sif /usw/$1
    bye
EOF

# give advice on how to finalize initialization
echo "Please finish initialization by running 'sh \$HOME/jobs/init.sh' on interactive Hummel shell."