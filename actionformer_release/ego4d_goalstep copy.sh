#!/bin/bash


#$ -l rt_F=1
#$ -l h_rt=18:00:00
#$ -j y
#$ -cwd

cd /home/ubuntu/slocal/ActionFormer
source .venv/bin/activate
which python
cd actionformer_release/libs/utils
unset PYTHONPATH
#export PYTHONPATH=/home/acg16955mc/.local/lib/python3.8/site-packages/$PYTHONPATH
export PYTHONPATH=/home/ubuntu/.local/lib/python3.8/site-packages/$PYTHONPATH
python setup.py install --user
cd ../..
python ./train.py ./configs/ego_4d_goalstep.yaml --output ego4d_goalstep_train_1

