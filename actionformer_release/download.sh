#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -cwd

cd /groups/gag51400/users/rtanaka/github/ActionFormer
source .venv/bin/activate
cd actionformer_release
#ego4d --output_directory="/home/ubuntu/ego4d_data" --datasets omnivore_video_swinl -y
ego4d --output_directory="/groups/gag51400/datasets/ego4d/v2" --datasets omnivore_video_swinl -y
#egoexo -o "/groups/gag51400/datasets/ego4d/v2" --parts features/omnivore_video
# curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
# unzip awscliv2.zip
# ./aws/install --install-dir ./aws-cli --bin-dir ./bin
# export PATH=$PATH:./aws-cli/v2/current/bin
# aws --version
# cd actionformer_release
# pip install ego4d
# ego4d --output_directory="/groups/gag51400/datasets/ego4d/v2" --datasets omnivore_video_swinl

pip install awscli
