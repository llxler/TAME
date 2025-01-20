#!/bin/bash

TASK_NAME="tabddpm_cardio_-2_90w"
DATA_NAME="cardio_train"

source ~/miniconda3/etc/profile.d/conda.sh &&

conda activate tabsyn &&

python main.py --dataname $DATA_NAME --method tabddpm --mode sample --save_path sample_end_csv/${TASK_NAME}.csv --task_name $TASK_NAME --eval_flag False --gpu 3&&

conda activate synthcity &&

# 运行 eval/bash_quality.py 脚本
python -m eval.bash_quality --task_name $TASK_NAME