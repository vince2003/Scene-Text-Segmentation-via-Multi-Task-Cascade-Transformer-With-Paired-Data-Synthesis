#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-18539}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --eval-options efficient_test=True ${@:3}
