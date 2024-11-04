#!/bin/bash

cd /home/ubuntu/slocal/ActionFormer
source .venv/bin/activate
which python
cd actionformer_release/libs/utils
unset PYTHONPATH
#export PYTHONPATH=/home/acg16955mc/.local/lib/python3.8/site-packages/$PYTHONPATH
export PYTHONPATH=/home/ubuntu/.local/lib/python3.8/site-packages/$PYTHONPATH
python setup.py install --user
cd ../..
export CUDA_VISIBLE_DEVICES=1
python ./train.py ./configs/ego_4d_goalstep.yaml --output ego4d_goalstep_train_1

