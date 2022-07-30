#!/bin/sh
#CUDA_VISIBLE_DEVICES=$1 python main.py --method $2 --dataset office_home --source Real --target Clipart --net $3 --save_check
CUDA_VISIBLE_DEVICES=$1 python train_strong_data_augment.py --method MME --dataset multi --source real --target sketch --net resnet34 --save_check