#!/usr/bin/env bash

CONFIG=n01kt4/best.py

# path of the model to be tested
WORKDIR=exp/nus/bevmap/bevmap-bs1-lr05-aug10-01/
# to store results in a difference path
SAVEDIR=exp/nus/results/bevmap-bs1-lr05-aug10/
# checkpoint to use, latest by default
CKPT=latest.pth

PY_ARGS=${@:1}
python tools/test.py $CONFIG \
    ${WORKDIR}${CKPT}\
    --format-only \
    --eval-options jsonfile_prefix=$SAVEDIR
#    --eval bbox