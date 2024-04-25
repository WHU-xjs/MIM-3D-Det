#!/usr/bin/env bash

T=`date +%m%d%H%M`

CONFIG=n01kt4/mim_tiny_maps.py
GPUID=0
SAVEDIR=exp/nus/mim_map_test/mim-tiny-rsz-lay6-attn/
# LOADDIR=exp/nus/bevdet_tiny_map/bevdet-tiny-mcds-mem-bam-lay-01/epoch_8.pth
TEMP=A

if [ ${SAVEDIR:0-3:2} == '00' ] # left 2 chars of last 3
then LOG=/dev/null # 00 are debug-only exps, clean manually
else LOG=n01kt4/logs/trains.log
fi

echo -e "\n\">--- train.$T.$TEMP.log ---<\"" >> $LOG
echo -e "  config: $CONFIG \n  saved in $SAVEDIR" >> $LOG
if [ $LOADDIR ]
then echo -e "\twith loaded checkpoint: $LOADDIR\n" >> $LOG
fi

PY_ARGS=${@:1}
python tools/train.py \
    $CONFIG \
    --gpu-ids $GPUID \
    --work-dir $SAVEDIR \
    --seed 0 \
    --deterministic \
    ${PY_ARGS} \
    2>&1 | tee n01kt4/logs/train.temp.$TEMP.log

echo -e "\nresult for \"$TEMP\"" >> $LOG
tail -n 21 n01kt4/logs/train.temp.$TEMP.log >> $LOG
rm n01kt4/logs/train.temp.$TEMP.log

# more args, put into middle if use
    # --gpus=$GPUS \
    # --resume-from $LOADDIR \
# train log file auto-gen in $SAVEDIR