#!/bin/bash

# ==========================================================================
#   
#   This script prefetches, creates and distributes all files needed to 
#   run this project on Hummel.
#   
#   1. make sure that python>=3.8 and singularity==3.9.1 (installation steps below) are available locally
#   2. make sure that ssh-agent is used and ~/.ssh/config is configured with hummel1 (see https://www.rrz.uni-hamburg.de/services/hpc/grundwissen/ssh-keys.html)
#   3. make sure you are in the UHH or Informatik VPN
#   4. enter your passwords (for sudo and UHH username) when asked
#   5. be patient, initialization may take some time
#   
#   Steps to install SingularityCE 3.9.1 locally (modified from https://sylabs.io/guides/3.9/user-guide/quick_start.html):
#   
#   sudo apt-get update && sudo apt-get install -y build-essential uuid-dev libgpgme-dev squashfs-tools libseccomp-dev wget pkg-config git cryptsetup-bin
#   export VERSION=1.18.2 OS=linux ARCH=amd64 && wget https://dl.google.com/go/go$VERSION.$OS-$ARCH.tar.gz && sudo tar -C /usr/local -xzvf go$VERSION.$OS-$ARCH.tar.gz && rm go$VERSION.$OS-$ARCH.tar.gz
#   echo 'export GOPATH=${HOME}/go' >> ~/.bashrc && echo 'export PATH=/usr/local/go/bin:${PATH}:${GOPATH}/bin' >> ~/.bashrc && source ~/.bashrc
#   export VERSION=3.9.1 && wget https://github.com/sylabs/singularity/releases/download/v${VERSION}/singularity-ce-${VERSION}.tar.gz && tar -xzf singularity-ce-${VERSION}.tar.gz && cd singularity-ce-${VERSION}
#   ./mconfig && make -C ./builddir && sudo make -C ./builddir install
#   
# ==========================================================================

# get current paths
PROJECT_ROOT_DIR=$(python -c "from src.core.utils.constants import PROJECT_ROOT_DIR; print(PROJECT_ROOT_DIR)")
SCRIPT_PATH=$(dirname "$0")

# prefetch all data sources (.ai2thor & data)
python $SCRIPT_PATH/prefetch_data.py

# create singularity container (requires singularity installed)
sudo singularity build $SCRIPT_PATH/container.sif $SCRIPT_PATH/container.def

# copy data from local to Hummel via sftp
sftp hummel1 <<EOF
    mkdir /home/$1/jobs
    mkdir /work/$1/.ai2thor
    mkdir /work/$1/.config
    put -r $SCRIPT_PATH/../jobs /home/$1
    put -r $HOME/.ai2thor /work/$1
    put -r $HOME/.config/unity3d /work/$1/.config
    put -r $PROJECT_ROOT_DIR/data /work/$1
    put $SCRIPT_PATH/container.sif /usw/$1
    bye
EOF

# give advice on how to finalize initialization
echo "Please finish initialization by running 'sh \$HOME/jobs/init.sh' on interactive Hummel shell."